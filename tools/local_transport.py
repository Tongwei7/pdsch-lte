"""本机部署客户端：经临时 VS Code 转发上传代码或读取项目结果。

连接配置保存在本机 .runtime/lte_server_deploy，不是数据集的一部分。
该脚本不会生成无线信号，也不会启动服务器上的生成进程。
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT.parent / ".runtime/lte_server_deploy"


def request(path, body=None):
    """无 body 时读取文件；有 body 时上传压缩包，并附带 SHA-256 摘要。"""
    state = json.loads((LOCAL / "transport.json").read_text())
    headers = {"X-Project-Token": state["token"]}
    if body is not None:
        headers["X-Bundle-SHA256"] = hashlib.sha256(body).hexdigest()
    req = urllib.request.Request(
        state["url"] + "/" + path.lstrip("/"),
        data=body,
        headers=headers,
        method="PUT" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["upload", "get"])
    parser.add_argument("path", nargs="?")
    parser.add_argument("--output")
    args = parser.parse_args()
    data = (
        request("bundle", (LOCAL / "project.zip").read_bytes())
        if args.action == "upload"
        else request(args.path)
    )
    if args.output:
        Path(args.output).write_bytes(data)
        print(json.dumps({"saved": args.output, "bytes": len(data)}))
    else:
        print(data.decode("utf-8", errors="replace"))
