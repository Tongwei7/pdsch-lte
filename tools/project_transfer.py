"""临时项目文件传输服务，通过已有 VS Code SSH 端口转发访问。

只监听服务器回环地址，并检查临时令牌和路径边界；没有 shell 执行接口。
该服务用于部署，不参与 LTE 仿真。启动一小时后自动停止服务循环。
"""

import hashlib
import io
import json
import os
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path("/home/luorongwei/pdsch_lte").resolve()
DATA = Path("/data/luorongwei/pdsch_lte").resolve()
TOKEN = os.environ["PDSCH_TRANSFER_TOKEN"]


class Handler(BaseHTTPRequestHandler):
    """仅处理授权项目目录内的文件读取和代码压缩包上传。"""

    def authenticated(self):
        """要求请求携带匹配的项目临时令牌。"""
        return self.headers.get("X-Project-Token") == TOKEN

    def do_GET(self):
        """读取代码目录文件，或 data/ 前缀对应的数据目录文件。"""
        if not self.authenticated():
            self.send_error(403)
            return
        part = self.path.lstrip("/")
        parent = DATA if part.startswith("data/") else ROOT
        if part.startswith("data/"):
            part = part[5:]
        path = (parent / part).resolve()
        if not path.is_relative_to(parent) or not path.is_file():
            self.send_error(404)
            return
        if path.stat().st_size > 50 * 1024 * 1024:
            self.send_error(413)
            return
        raw = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_PUT(self):
        """核对压缩包大小、摘要和全部路径后，再逐文件替换项目代码。"""
        if not self.authenticated() or self.path != "/bundle":
            self.send_error(403)
            return
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= 10 * 1024 * 1024:
            self.send_error(413)
            return
        raw = self.rfile.read(size)
        if hashlib.sha256(raw).hexdigest() != self.headers.get("X-Bundle-SHA256"):
            self.send_error(400)
            return
        archive = zipfile.ZipFile(io.BytesIO(raw))
        members = archive.infolist()
        if sum(m.file_size for m in members) > 50 * 1024 * 1024:
            self.send_error(413)
            return
        # 必须先验证所有成员，拒绝目录越界和符号链接，再开始写文件。
        for member in members:
            destination = (ROOT / member.filename).resolve()
            if not destination.is_relative_to(ROOT) or member.is_dir():
                self.send_error(400)
                return
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                self.send_error(400)
                return
        for member in members:
            destination = (ROOT / member.filename).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".uploading")
            temporary.write_bytes(archive.read(member))
            temporary.replace(destination)
        releases = ROOT / "releases"
        releases.mkdir(exist_ok=True)
        (releases / (hashlib.sha256(raw).hexdigest() + ".zip")).write_bytes(raw)
        reply = json.dumps(
            {"files": len(members), "sha256": hashlib.sha256(raw).hexdigest()}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 8877), Handler)
    timer = threading.Timer(3600, server.shutdown)
    timer.daemon = True
    timer.start()
    print(
        "Project transfer listening on http://127.0.0.1:8877 (loopback only; expires in 1 hour)",
        flush=True,
    )
    server.serve_forever()
