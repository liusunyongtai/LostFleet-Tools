from io import BytesIO
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
from PIL import Image

from lostfleet_tools.pipeline import PART_NAMES, export_zip, load_image, png_bytes, render_part
from lostfleet_tools.server import EditorServer


def rectangle(x0, y0, x1, y1, operation="add"):
    return {"operation": operation, "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def part(**overrides):
    value = {"name": "head", "regions": [rectangle(4, 4, 25, 25)], "pivot": [12, 10]}
    value.update(overrides)
    return value


def character():
    image = Image.new("RGBA", (32, 32), "white")
    image.paste((180, 20, 30, 255), (8, 8, 24, 24))
    return image


class PipelineTests(unittest.TestCase):
    def test_png_and_jpeg_uploads(self):
        for format in ("PNG", "JPEG"):
            with self.subTest(format=format):
                buffer = BytesIO()
                character().convert("RGB").save(buffer, format=format)
                image = load_image(buffer.getvalue())
                self.assertEqual(image.mode, "RGBA")
                self.assertEqual(image.size, (32, 32))

    def test_jpeg_exif_orientation(self):
        image = Image.new("RGB", (20, 10), "red")
        exif = image.getexif()
        exif[274] = 6
        buffer = BytesIO()
        image.save(buffer, "JPEG", exif=exif)
        self.assertEqual(load_image(buffer.getvalue()).size, (10, 20))

    def test_jpeg_background_removed_after_decode(self):
        buffer = BytesIO()
        character().convert("RGB").save(buffer, "JPEG", quality=95)
        image = load_image(buffer.getvalue())
        full, _, _ = render_part(image, part())
        self.assertEqual(full.getpixel((4, 4))[3], 0)
        self.assertEqual(full.getpixel((16, 16))[3], 255)

    def test_all_sixteen_supported_parts_export(self):
        with ZipFile(BytesIO(export_zip(character(), [part(name=name) for name in PART_NAMES]))) as archive:
            meta = json.loads(archive.read("metadata.json"))
            self.assertEqual({p["name"] for p in meta["parts"]}, set(PART_NAMES))
            self.assertEqual(len(meta["parts"]), 16)

    def test_invalid_and_unsupported_images(self):
        with self.assertRaises(ValueError):
            load_image(b"not an image")
        buffer = BytesIO()
        character().save(buffer, "GIF")
        with self.assertRaises(ValueError):
            load_image(buffer.getvalue())

    def test_white_background_and_tight_crop(self):
        full, crop, meta = render_part(character(), part())
        self.assertEqual(full.mode, "RGBA")
        self.assertEqual(full.size, (32, 32))
        self.assertEqual(full.getpixel((5, 5)), (0, 0, 0, 0))
        self.assertEqual(full.getpixel((10, 10)), (180, 20, 30, 255))
        self.assertEqual(crop.size, (16, 16))
        self.assertEqual(meta["crop_bounds"], [8, 8, 24, 24])
        self.assertEqual(meta["pivot_in_crop"], [4, 2])
        self.assertEqual(meta["pivot_spine"], [-4, 6])
        self.assertEqual(meta["original_region_bounds"], [4, 4, 25, 25])

    def test_source_alpha_is_preserved_not_squared(self):
        image = Image.new("RGBA", (32, 32), (100, 50, 10, 128))
        full, _, _ = render_part(image, part(), {"mode": "keep"})
        self.assertEqual(full.getpixel((10, 10))[3], 128)
        self.assertEqual(full.getpixel((0, 0))[3], 0)

    def test_enclosed_white_is_preserved_unless_requested(self):
        image = character()
        image.paste("white", (12, 12, 18, 18))
        full, _, _ = render_part(image, part())
        self.assertEqual(full.getpixel((14, 14))[3], 255)
        full, _, _ = render_part(image, part(), {"mode": "all_white"})
        self.assertEqual(full.getpixel((14, 14))[3], 0)

    def test_subtract_polygon_overrides_feather_and_overlap(self):
        image = Image.new("RGBA", (32, 32), "red")
        full, _, _ = render_part(image, part(regions=[rectangle(4, 4, 25, 25),
                                  rectangle(10, 8, 15, 14, "subtract")], feather=2, overlap=5), {"mode": "keep"})
        self.assertEqual(full.getpixel((12, 10))[3], 0)
        self.assertGreater(full.getpixel((20, 20))[3], 0)

    def test_polygon_excludes_rectangle_corner(self):
        image = Image.new("RGBA", (32, 32), "red")
        triangle = {"points": [[4, 4], [25, 4], [4, 25]]}
        full, _, _ = render_part(image, part(regions=[triangle]), {"mode": "keep"})
        self.assertEqual(full.getpixel((23, 23))[3], 0)
        self.assertEqual(full.getpixel((6, 6))[3], 255)

    def test_edge_expansion_erosion_and_feather(self):
        image = Image.new("RGBA", (32, 32), "red")
        base, _, _ = render_part(image, part(), {"mode": "keep"})
        expanded, _, _ = render_part(image, part(edge=2), {"mode": "keep"})
        eroded, _, _ = render_part(image, part(edge=-2), {"mode": "keep"})
        count = lambda im: np.count_nonzero(np.asarray(im)[:, :, 3])
        self.assertGreater(count(expanded), count(base))
        self.assertLess(count(eroded), count(base))
        feathered, _, _ = render_part(image, part(feather=1), {"mode": "keep"})
        alpha = np.asarray(feathered)[:, :, 3]
        self.assertTrue(np.any((alpha > 0) & (alpha < 255)))

    def test_joint_overlap_adds_only_existing_pixels(self):
        image = character()
        small = part(regions=[rectangle(10, 10, 13, 13)], pivot=[14, 14])
        plain, _, _ = render_part(image, small)
        full, _, _ = render_part(image, {**small, "overlap": 5})
        self.assertEqual(plain.getpixel((17, 14))[3], 0)
        self.assertEqual(full.getpixel((17, 14))[3], 255)
        self.assertEqual(full.getpixel((0, 0))[3], 0)
        self.assertEqual(full.getpixel((17, 14))[:3], image.getpixel((17, 14))[:3])

    def test_invalid_regions_and_parameters(self):
        cases = [part(name="../../bad"), part(regions=[]), part(edge=100),
                 part(feather=float("nan")), part(pivot=[-1, 5]),
                 part(regions=[{"points": [[1, 1], [1, 1], [1, 1]]}]),
                 part(regions=[rectangle(1, 1, 100, 100)]),
                 part(regions=[rectangle(1, 1, 5, 5, "subtract")])]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                render_part(character(), value)

    def test_empty_visible_region_rejected(self):
        with self.assertRaisesRegex(ValueError, "没有可见像素"):
            render_part(character(), part(regions=[rectangle(0, 0, 3, 3)]))

    def test_zip_contains_rgba_assets_and_reconstructable_coordinates(self):
        body = part(name="body", regions=[rectangle(9, 9, 22, 22)])
        with ZipFile(BytesIO(export_zip(character(), [part(), body]))) as archive:
            self.assertEqual(len(archive.namelist()), 6)
            metadata = json.loads(archive.read("metadata.json"))
            self.assertFalse(metadata["occlusion_reconstructed"])
            self.assertEqual(len(metadata["parts"]), 2)
            for entry in metadata["parts"]:
                full = Image.open(BytesIO(archive.read(entry["files"]["full"])))
                crop = Image.open(BytesIO(archive.read(entry["files"]["cropped"])))
                self.assertEqual(full.mode, "RGBA")
                self.assertEqual(crop.mode, "RGBA")
                reconstructed = Image.new("RGBA", full.size)
                reconstructed.paste(crop, tuple(entry["crop_offset"]))
                np.testing.assert_array_equal(np.asarray(full), np.asarray(reconstructed))

    def test_no_parts_or_duplicate_names_rejected(self):
        for parts in ([], [part(), part()]):
            with self.assertRaises(ValueError):
                export_zip(character(), parts)


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = EditorServer(("127.0.0.1", 0))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def post(self, path, body, raw=False, headers=None):
        data = body if raw else json.dumps(body).encode()
        request = Request(self.url + path, data=data, headers=headers or {})
        return urlopen(request, timeout=10)

    def upload(self):
        with self.post("/api/upload", png_bytes(character()), True) as response:
            return json.load(response)["image_id"]

    def test_pages_are_served(self):
        for path, expected in (("/", b"canvas"), ("/index.html", b"canvas"),
                               ("/app.js", b"pointerdown"), ("/style.css?v=3", b"background")):
            with self.subTest(path=path), urlopen(self.url + path) as response:
                self.assertEqual(response.status, 200)
                self.assertIn(expected, response.read())

    def test_editor_document_bundles_latest_assets(self):
        from hashlib import sha256
        from lostfleet_tools.server import WEB
        html = (WEB / "index.html").read_text(encoding="utf-8")
        css = (WEB / "style.css").read_text(encoding="utf-8")
        script = (WEB / "app.js").read_text(encoding="utf-8")
        expected_build = sha256((html + css + script).encode()).hexdigest()[:16]
        with urlopen(self.url + "/") as response:
            document = response.read().decode()
            self.assertEqual(response.headers["X-LostFleet-Build"], expected_build)
            self.assertEqual(response.headers["X-LostFleet-Document"], "self-contained")
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertIn('<style id="editor-styles">' + css, document)
            self.assertIn('<script id="editor-script">', document)
            self.assertIn('document.documentElement.dataset.editorReady', document)
            self.assertNotIn('href="./style.css', document)
            self.assertNotIn('src="./app.js', document)
            self.assertLess(document.index('<main class="app-layout">'),
                            document.index('<script id="editor-script">'))

    def test_upload_preview_export_end_to_end(self):
        image_id = self.upload()
        with urlopen(self.url + "/api/image/" + image_id) as response:
            self.assertEqual(Image.open(BytesIO(response.read())).size, (32, 32))
        payload = {"image_id": image_id, "part": part(), "parts": [part()]}
        with self.post("/api/preview", payload) as response:
            preview = Image.open(BytesIO(response.read()))
            self.assertEqual(preview.getpixel((5, 5))[3], 0)
        with self.post("/api/export", payload) as response:
            self.assertEqual(response.headers["Content-Type"], "application/zip")
            with ZipFile(BytesIO(response.read())) as archive:
                self.assertIn("cropped/head.png", archive.namelist())

    def test_errors_are_actionable(self):
        image_id = self.upload()
        cases = [("/api/upload", b"bad image", True, 400),
                 ("/api/export", {"image_id": "missing", "parts": []}, False, 404),
                 ("/api/export", {"image_id": image_id, "parts": []}, False, 400),
                 ("/api/preview", {"image_id": image_id, "part": None}, False, 400),
                 ("/api/export", b"{bad json", True, 400)]
        for path, body, raw, code in cases:
            with self.subTest(path=path, body=body), self.assertRaises(HTTPError) as caught:
                self.post(path, body, raw)
            self.assertEqual(caught.exception.code, code)
            self.assertIn("error", json.load(caught.exception))

    def test_cross_origin_post_blocked(self):
        with self.assertRaises(HTTPError) as caught:
            self.post("/api/upload", png_bytes(character()), True, {"Origin": "https://evil.example"})
        self.assertEqual(caught.exception.code, 403)

    def test_unknown_route_does_not_expose_files(self):
        with self.assertRaises(HTTPError) as caught:
            urlopen(self.url + "/../requirements.txt")
        self.assertEqual(caught.exception.code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
