"""Local single-user HTTP editor. No third-party web framework required."""

import argparse
from hashlib import sha256
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
from threading import Thread
from urllib.parse import urlsplit
import webbrowser

from .pipeline import export_zip, load_image, png_bytes, render_part

MAX_BODY = 32 * 1024 * 1024
WEB = Path(__file__).parent / "web"


def editor_document():
    """Keep the editor usable when a preview proxy only forwards the document URL.

    Source CSS/JS remain separate for development and their existing endpoints
    remain available. The served page needs no secondary static requests.
    """
    html = (WEB / "index.html").read_text(encoding="utf-8")
    css = (WEB / "style.css").read_text(encoding="utf-8")
    script = (WEB / "app.js").read_text(encoding="utf-8")
    build = sha256((html + css + script).encode()).hexdigest()[:16]
    html = html.replace('<link rel="stylesheet" href="./style.css?v=3">',
                        '<style id="editor-styles">' + css + '</style>')
    html = html.replace('<script src="./app.js?v=3" defer></script>',
                        f'<meta name="lostfleet-build" content="{build}">')
    # Execute after the DOM, without depending on an external script URL.
    script = script.replace('</script', r'<\/script')
    html = html.replace('</body>', '<script id="editor-script">\n' + script + '\n</script></body>')
    return html.encode(), build


class EditorServer(HTTPServer):
    def __init__(self, address):
        super().__init__(address, Handler)
        self.images = OrderedDict()


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body, content_type="application/json; charset=utf-8", download=False, build=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if build:
            self.send_header("X-LostFleet-Build", build)
            self.send_header("X-LostFleet-Document", "self-contained")
        if download:
            self.send_header("Content-Disposition", 'attachment; filename="lostfleet-assets.zip"')
        self.end_headers()
        self.wfile.write(body)

    def error(self, status, message):
        self.reply(status, json.dumps({"error": message}, ensure_ascii=False).encode())

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            document, build = editor_document()
            self.reply(200, document, "text/html; charset=utf-8", build=build)
            return
        files = {"/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8")}
        if path in files:
            name, kind = files[path]
            self.reply(200, (WEB / name).read_bytes(), kind)
        elif path.startswith("/api/image/"):
            image = self.server.images.get(path.removeprefix("/api/image/"))
            if image is None:
                self.error(404, "图片不存在，请重新上传")
            else:
                self.reply(200, png_bytes(image), "image/png")
        else:
            self.error(404, "页面不存在")

    def do_POST(self):
        # Prevent other websites from issuing requests to this local editor.
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self.error(403, "不允许跨站请求")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                self.close_connection = True
                self.error(413, "请求为空或超过 32 MiB")
                return
            data = self.rfile.read(length)
            path = urlsplit(self.path).path
            if path == "/api/upload":
                image = load_image(data)
                image_id = secrets.token_urlsafe(18)
                # Single-user tool: retain only the latest source to bound memory.
                self.server.images.clear()
                self.server.images[image_id] = image
                self.reply(200, json.dumps({"image_id": image_id, "width": image.width,
                                           "height": image.height}).encode())
                return
            if path not in ("/api/preview", "/api/export"):
                self.error(404, "接口不存在")
                return
            body = json.loads(data)
            if not isinstance(body, dict):
                raise ValueError("请求必须为 JSON 对象")
            image = self.server.images.get(body.get("image_id"))
            if image is None:
                self.error(404, "图片不存在或已被替换，请重新上传")
                return
            background = body.get("background", {})
            if not isinstance(background, dict):
                raise ValueError("背景参数格式错误")
            if path == "/api/preview":
                full, _, _ = render_part(image, body.get("part"), background)
                self.reply(200, png_bytes(full), "image/png")
            else:
                self.reply(200, export_zip(image, body.get("parts"), background), "application/zip", True)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            self.error(400, str(exc))


def main():
    parser = argparse.ArgumentParser(description="LostFleet manual sprite splitter")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open-browser", action="store_true", help="Open the editor in the default browser after binding the port")
    args = parser.parse_args()
    with EditorServer((args.host, args.port)) as server:
        print(f"LostFleet editor running on {args.host}:{server.server_port}", flush=True)
        if args.open_browser:
            browser_host = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
            Thread(target=webbrowser.open,
                   args=(f"http://{browser_host}:{server.server_port}/",), daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
