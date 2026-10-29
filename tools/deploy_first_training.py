"""由用户部署脚本调用：核对服务器基线、更新唯一源码、测试、提交并后台启动。"""

import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path("/home/luorongwei/pdsch_lte")
DATA = Path("/data/luorongwei/pdsch_lte")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    archive_path = Path(sys.argv[1])
    if len(sys.argv) != 3 or sha(archive_path.read_bytes()) != sys.argv[2]:
        raise RuntimeError("Deployment archive checksum mismatch")
    if git("rev-parse", "--show-toplevel") != str(ROOT):
        raise RuntimeError("Unexpected server repository root")
    # 用户若仍在修改或运行，保留现场，拒绝覆盖；不擅自停止任务。
    if git("status", "--porcelain"):
        raise RuntimeError("Server working tree is dirty; commit/review existing edits first")
    commands = subprocess.check_output(
        ["ps", "-u", str(os.getuid()), "-o", "pid=,args="], text=True
    )
    for command in commands.splitlines():
        pid = command.strip().split(maxsplit=1)[0]
        try:
            project_cwd = (Path("/proc") / pid / "cwd").resolve() == ROOT
        except OSError:
            project_cwd = False
        if (str(ROOT) in command or project_cwd) and any(
            s in command
            for s in (
                "lte_dataset_generate(",
                "tools/run_server.py",
                "tools/run_first_training.py",
                "training_compat/run.py",
            )
        ):
            raise RuntimeError("Project process is still running; wait for it before deploying")
    changed, original = [], {}
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("update_manifest.json"))
        for name, info in manifest["files"].items():
            path = ROOT / name
            if not path.resolve().is_relative_to(ROOT.resolve()) or path.is_symlink():
                raise ValueError("Unsafe source destination")
            raw = archive.read(name)
            if sha(raw) != info["sha256"]:
                raise ValueError(f"Bad update file: {name}")
            old = path.read_bytes() if path.exists() else None
            if old == raw:
                continue
            if info["base_sha256"] is None:
                if old is not None:
                    raise RuntimeError(f"New source already exists with different contents: {name}")
            elif old is None or sha(old) != info["base_sha256"]:
                raise RuntimeError(f"Server file differs from inspected baseline: {name}")
            if name.endswith(".py"):
                compile(raw, name, "exec")
            changed.append(name)
            original[name] = old
        env = dict(
            os.environ, PYTHONPATH=str(ROOT / ".deps") + os.pathsep + str(ROOT / "training_compat")
        )
        try:
            for name in changed:
                path = ROOT / name
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name + ".installing")
                temporary.write_bytes(archive.read(name))
                temporary.replace(path)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-p",
                    "test_batch_plan.py",
                    "-v",
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )
        except BaseException:
            for name, old in original.items():
                path = ROOT / name
                if old is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(old)
            raise
    if changed:
        subprocess.run(["git", "add", "--", *changed], cwd=ROOT, check=True)
        subprocess.run(
            ["git", "commit", "-m", "首轮训练改为2000份，冻结批次与任务参数并复用已有数据"],
            cwd=ROOT,
            check=True,
        )
    # 这里不依赖服务器访问 GitHub，也不会自动推送任何实测资料。
    log_root = ROOT / "logs"
    log_root.mkdir(exist_ok=True)
    log_path = log_root / f"first_training_{time.strftime('%Y%m%d_%H%M%S')}.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "tools/run_first_training.py"),
                "--captures",
                "2000",
                "--workers",
                "4",
                "--epochs",
                "60",
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    # 短暂观察能立即发现环境/认证/导入失败；长期进度由状态文件记录。
    time.sleep(2)
    code = process.poll()
    result = {
        "commit": git("rev-parse", "HEAD"),
        "launcher_pid": process.pid,
        "log": str(log_path),
        "workflow_status": str(DATA / "training_runs/first_train_2000/workflow_status.json"),
        "exit_code_after_two_seconds": code,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if code is not None and code != 0:
        raise RuntimeError(f"Workflow exited immediately; read {log_path}")


if __name__ == "__main__":
    main()
