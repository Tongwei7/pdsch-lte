"""按需导出已提交工程；不再创建源码副本或临时传输入口。"""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export_project(output):
    """使用 git archive，只导出干净提交，拒绝覆盖已有文件。"""
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    if destination.is_relative_to(ROOT):
        raise ValueError("Export outside the source directory")
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=ROOT, text=True
    ).strip()
    changes = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True
    )
    if changes.strip():
        raise RuntimeError("Commit or resolve working-tree changes before exporting")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 提交树不包含 .git、未跟踪数据及忽略的依赖，不复制另一套网络源码。
    subprocess.run(
        ["git", "archive", "--format=zip", "--output", str(destination), revision],
        cwd=ROOT,
        check=True,
    )
    return {"commit": revision, "output": str(destination), "bytes": destination.stat().st_size}


def main():
    """必须明确指定输出路径；普通 Git 维护不会创建额外压缩包。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(export_project(args.output)))


if __name__ == "__main__":
    main()
