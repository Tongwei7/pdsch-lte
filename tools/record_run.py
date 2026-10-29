"""启动实验前记录 Git 提交和计划/清单摘要，不会启动计算任务。"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*arguments):
    """只读取当前工程 Git 信息，不切换分支或修改源码。"""
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def describe_file(path):
    """流式计算摘要，不复制清单及其引用的数据集。"""
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(source), "sha256": digest.hexdigest(), "bytes": source.stat().st_size}


def main():
    """默认要求已提交状态，并拒绝覆盖已有运行记录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan", help="本次实际使用的 generation_plan.json")
    parser.add_argument("--manifest", help="本次使用的 manifest.csv")
    parser.add_argument("--allow-dirty", action="store_true", help="明确记录未提交状态，仅供调试")
    args = parser.parse_args()
    destination = Path(args.output).expanduser().resolve()
    if destination.is_relative_to(ROOT):
        raise ValueError("Save run records outside the code directory, normally under /data")
    status = git("status", "--porcelain", "--untracked-files=normal")
    if status and not args.allow_dirty:
        raise RuntimeError("Working tree has uncommitted changes; commit first or use --allow-dirty")
    versions = {}
    for package in ("numpy", "scipy", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_dirty": bool(status),
        "git_status": status.splitlines(),
        "python": platform.python_version(),
        "packages": versions,
        "note": "Pre-launch observation; cannot prove what an earlier process loaded.",
    }
    for key in ("plan", "manifest"):
        source = getattr(args, key)
        if source:
            result[key] = describe_file(source)
    if args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        result["generator_revision"] = plan.get("generator_revision")
        result["generator_source_sha256"] = plan.get("generator_source_sha256")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(destination), "commit": result["git_commit"], "dirty": bool(status)}))


if __name__ == "__main__":
    main()
