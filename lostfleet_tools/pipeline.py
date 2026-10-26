"""Image-space masks and lossless exports, independent of the UI."""

from io import BytesIO
import json
import math
import warnings
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = 16_000_000
PART_NAMES = (
    "head", "body", "left_upper_arm", "right_upper_arm", "left_forearm",
    "right_forearm", "left_hand", "right_hand", "left_thigh", "right_thigh",
    "left_shin", "right_shin", "left_foot", "right_foot", "weapon", "backpack",
)


def load_image(data: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format not in ("PNG", "JPEG"):
                    raise ValueError("仅支持 PNG 和 JPG 图片")
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError("图片最多支持 1600 万像素")
                return ImageOps.exif_transpose(source).convert("RGBA")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("无法读取图片，或图片过大") from exc


def number(value, low, high, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必须为数值")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} 超出范围 {low}..{high}")
    return float(value)


def point(value, width, height):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("坐标应为 [x, y]")
    return [number(value[0], 0, width - 1, "x"), number(value[1], 0, height - 1, "y")]


def white_background_alpha(image: Image.Image, mode: str, threshold: int) -> np.ndarray:
    """Remove near-white pixels explicitly; border mode preserves enclosed whites."""
    rgba = np.asarray(image)
    if mode == "keep":
        return np.full(rgba.shape[:2], 255, dtype=np.uint8)
    near_white = np.all(rgba[:, :, :3] >= threshold, axis=2)
    if mode == "all_white":
        return np.where(near_white, 0, 255).astype(np.uint8)
    if mode != "border_white":
        raise ValueError("未知背景模式")
    # Transparent pixels also connect disjoint white regions to the image border.
    candidates = (near_white | (rgba[:, :, 3] == 0)).astype(np.uint8)
    _, labels = cv2.connectedComponents(candidates, connectivity=8)
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    border = border[border != 0]
    removed = np.isin(labels, border)
    return np.where(removed, 0, 255).astype(np.uint8)


def render_part(image: Image.Image, part: dict, background: dict | None = None):
    if not isinstance(part, dict) or part.get("name") not in PART_NAMES:
        raise ValueError("未知部件名称")
    width, height = image.size
    regions = part.get("regions", [])
    if not isinstance(regions, list) or not 1 <= len(regions) <= 100:
        raise ValueError("每个部件需要 1–100 个区域")
    mask = np.zeros((height, width), np.uint8)
    erase = np.zeros_like(mask)
    normalized = []
    additive_points = []
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError("区域格式错误")
        points = region.get("points", [])
        if not isinstance(points, list) or not 3 <= len(points) <= 1000:
            raise ValueError("多边形需要 3–1000 个点")
        points = [point(p, width, height) for p in points]
        polygon = np.rint(points).astype(np.int32)
        if cv2.contourArea(polygon) < 1:
            raise ValueError("选区面积太小，请重新绘制")
        operation = region.get("operation", "add")
        if operation not in ("add", "subtract"):
            raise ValueError("未知区域操作")
        cv2.fillPoly(mask if operation == "add" else erase, [polygon], 255)
        if operation == "add":
            additive_points.extend(points)
        normalized.append({"operation": operation, "points": points})
    if not additive_points:
        raise ValueError("部件至少需要一个添加区域")
    pts = np.asarray(additive_points)
    region_bounds = [float(pts[:, 0].min()), float(pts[:, 1].min()),
                     float(pts[:, 0].max()), float(pts[:, 1].max())]
    pivot = point(part.get("pivot") or [(region_bounds[0] + region_bounds[2]) / 2,
                                        (region_bounds[1] + region_bounds[3]) / 2], width, height)
    edge = int(number(part.get("edge", 0), -32, 32, "边缘扩缩"))
    feather = number(part.get("feather", 0), 0, 16, "羽化")
    overlap = int(number(part.get("overlap", 0), 0, 128, "关节重叠"))
    if edge:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(edge) * 2 + 1,) * 2)
        mask = (cv2.dilate if edge > 0 else cv2.erode)(mask, kernel,
                    borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if overlap:
        cv2.circle(mask, tuple(np.rint(pivot).astype(int)), overlap, 255, -1)
    if feather:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather)
    mask[erase > 0] = 0
    background = background or {}
    threshold = int(number(background.get("threshold", 245), 0, 255, "白色阈值"))
    bg_alpha = white_background_alpha(image, background.get("mode", "border_white"), threshold)
    pixels = np.array(image)
    alpha = (pixels[:, :, 3].astype(np.float32) * (mask / 255.0) * (bg_alpha / 255.0))
    pixels[:, :, 3] = np.rint(alpha).astype(np.uint8)
    pixels[pixels[:, :, 3] == 0, :3] = 0
    full = Image.fromarray(pixels)
    bounds = full.getbbox()
    if bounds is None:
        raise ValueError(f"{part['name']} 选区没有可见像素，请检查选区和去白设置")
    x0, y0, x1, y1 = bounds
    metadata = {
        "name": part["name"], "regions": normalized,
        "original_region_bounds": region_bounds,
        "canvas_origin": [0, 0], "crop_bounds": list(bounds),
        "crop_size": [x1 - x0, y1 - y0], "crop_offset": [x0, y0],
        "pivot": pivot, "pivot_in_crop": [pivot[0] - x0, pivot[1] - y0],
        "pivot_spine": [pivot[0] - width / 2, height / 2 - pivot[1]],
        "adjustments": {"edge": edge, "feather": feather, "overlap": overlap},
    }
    return full, full.crop(bounds), metadata


def png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def export_zip(image: Image.Image, parts: list, background: dict | None = None) -> bytes:
    if not isinstance(parts, list) or not 1 <= len(parts) <= len(PART_NAMES):
        raise ValueError("请选择 1–16 个部件导出")
    names = [p.get("name") for p in parts if isinstance(p, dict)]
    if len(names) != len(parts) or len(set(names)) != len(names):
        raise ValueError("部件格式错误或名称重复")
    buffer = BytesIO()
    metadata = {"schema_version": 1, "canvas_size": list(image.size),
                "coordinates": "top-left origin; x right, y down; pixels; crop bounds right/bottom exclusive",
                "spine_coordinates": "canvas center origin; x right, y up",
                "background": background or {"mode": "border_white", "threshold": 245},
                "occlusion_reconstructed": False, "parts": []}
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for part in parts:
            full, cropped, entry = render_part(image, part, background)
            name = entry["name"]
            entry["files"] = {"full": f"full/{name}.png", "cropped": f"cropped/{name}.png"}
            archive.writestr(entry["files"]["full"], png_bytes(full))
            archive.writestr(entry["files"]["cropped"], png_bytes(cropped))
            metadata["parts"].append(entry)
        archive.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        archive.writestr("README.txt", "Manual segmentation assets. full/: common canvas; cropped/: tight alpha bounds.\n"
                          "metadata.json describes offsets and pivots, not a native Spine skeleton.\n"
                          "Import PNG attachments into Spine and manually create bones/slots.\n"
                          "No hidden anatomy is reconstructed.\n")
    return buffer.getvalue()
