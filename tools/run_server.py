"""服务器批量生成入口：检查前置条件、启动 MATLAB、验收并衔接下一阶段。

从服务器 bash 运行；所有数据写入 DATA，代码和日志位于 ROOT。
本脚本负责调度进程，LTE 信号和标签由 matlab/lte_dataset_generate.m 生成。
"""

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data/luorongwei/pdsch_lte")


def write(path, value):
    """先写临时 JSON 再替换正式文件，减少监控端读到半份进度的机会。"""
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def resolve_generator_revision(source_revision):
    """仅为已核验的纯注释版本保留原批次标识；其他源码变化仍产生新版本。

    映射键是全部 MATLAB 源码和系统信息资产的完整内容摘要。因此修改
    任意 MATLAB 算法、参数或资产后，都不会匹配这次注释迁移的映射。
    """
    path = ROOT / "reference/generator_revision_compatibility.json"
    if not path.exists():
        return source_revision
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("kind") != "verified_comment_only":
        raise RuntimeError("Unrecognized generator revision compatibility record")
    return record["revisions"].get(source_revision, source_revision)


def main():
    """执行一个批次；启用 --chain 后，前置验收通过才进入下一阶段。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["acceptance", "pilot", "production"], default="acceptance"
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chain", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    state = DATA / "pipeline_status.json"
    stage = args.stage
    # 同一阶段只允许一个启动器持有锁，防止重复进程写同一批输出。
    lock = (ROOT / f"logs/{stage}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # acceptance 校验物理链路和 AI 接口；pilot 再检查较大批数据的可读取性。
    if stage != "acceptance":
        acceptance = json.loads((DATA / "acceptance/acceptance_report.json").read_text())
        if (
            acceptance.get("completed", 0) < 22
            or acceptance.get("failures")
            or not acceptance.get("current_model_forward_backward")
        ):
            raise RuntimeError(
                "The full physical-link/current-reader acceptance gate has not passed"
            )
    if stage == "production":
        pilot = json.loads((DATA / "pilot/acceptance_report.json").read_text())
        if pilot.get("completed", 0) < 400 or pilot.get("current_reader_validated_shards", 0) < 400:
            raise RuntimeError("The pilot data-read gate has not passed")
    subprocess.run([sys.executable, str(ROOT / "tools/make_assets.py"), "--plans-only"], check=True)
    config_path = ROOT / f"configs/{stage}.json"
    config = json.loads(config_path.read_text())
    # 保存真实源码摘要；仅对已验证执行逻辑相同的本次注释迁移保留旧版本号。
    # 后续算法或资产变化仍须重新进行 acceptance/pilot 验收。
    signature = hashlib.sha256()
    for path in sorted(
        list((ROOT / "matlab").glob("*.m")) + list((ROOT / "reference").glob("synthetic*.mat"))
    ):
        signature.update(path.name.encode())
        signature.update(path.read_bytes())
    config["generator_source_sha256"] = signature.hexdigest()
    config["generator_revision"] = resolve_generator_revision(signature.hexdigest())
    if (
        stage != "acceptance"
        and acceptance.get("generator_revision") != config["generator_revision"]
    ):
        raise RuntimeError("Acceptance belongs to another generator revision; re-run acceptance")
    if stage == "production" and pilot.get("generator_revision") != config["generator_revision"]:
        raise RuntimeError("Pilot belongs to another generator revision; re-run pilot")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    (DATA / stage).mkdir(exist_ok=True)
    shutil.copyfile(config_path, DATA / stage / "generation_plan.json")
    # 将 NumPy/SciPy 放入项目私有目录；沿用服务器现有的 PyTorch。
    deps = ROOT / ".deps"
    env = dict(
        os.environ,
        PYTHONPATH=str(deps)
        + os.pathsep
        + str(ROOT / "training_compat")
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
    )
    if importlib.util.find_spec("scipy") is None and not (deps / "scipy").exists():
        write(state, dict(stage=stage, state="preparing_dependencies", started_at=time.time()))
        with (ROOT / "logs/dependencies.log").open("ab") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--target",
                    str(deps),
                    "numpy>=1.24,<2.3",
                    "scipy>=1.13,<1.16",
                ],
                stdout=log,
                stderr=log,
                check=True,
            )
    # 空间预算单位为字节；正式阶段用 pilot 实际落盘大小外推并留 25% 余量。
    required = (5 if stage == "acceptance" else 40 if stage == "pilot" else 700) * 2**30
    if stage == "production":
        pilot_root = DATA / "pilot"
        real_bytes = sum(p.stat().st_size for p in pilot_root.rglob("*") if p.is_file())
        pilot_windows = pilot["total_windows"]
        required = max(
            100 * 2**30, real_bytes / max(pilot_windows, 1) * config["requested_windows"] * 1.25
        )
    if shutil.disk_usage(DATA).free < required:
        raise RuntimeError("Insufficient free data-disk budget for this generation stage")
    started = time.time()
    processes = []
    if stage == "acceptance":
        with (ROOT / "logs/physical_unit_validation.log").open("ab") as log:
            subprocess.run(
                [
                    "/usr/local/bin/matlab",
                    "-batch",
                    f"addpath('{ROOT.as_posix()}/matlab'); r=validate_link; assert(r.passed);",
                ],
                cwd=ROOT,
                stdout=log,
                stderr=log,
                env=env,
                check=True,
            )
    # 每个 MATLAB 进程只处理自己的任务编号，并写独立日志/捕获文件。
    # 限制每进程计算线程数，避免多进程叠加后过度占用 CPU。
    for worker in range(args.workers):
        code = f"addpath('{ROOT.as_posix()}/matlab'); maxNumCompThreads(2); lte_dataset_generate('{ROOT.as_posix()}/configs/{stage}.json',{worker},{args.workers});"
        logfile = (ROOT / f"logs/{stage}_worker_{worker:02d}.log").open("ab")
        process = subprocess.Popen(
            ["/usr/local/bin/matlab", "-batch", code],
            cwd=ROOT,
            stdout=logfile,
            stderr=logfile,
            stdin=subprocess.DEVNULL,
            env=env,
        )
        logfile.close()
        processes.append(process)
    write(
        DATA / f"{stage}_pids.json",
        {"launcher": os.getpid(), "workers": [p.pid for p in processes]},
    )
    # 只统计本版本记录；complete 是捕获生成完成，后面仍需整体数据验收。
    while True:
        codes = [p.poll() for p in processes]
        records = []
        folder = DATA / stage / "status"
        if folder.exists():
            for path in folder.glob("*.json"):
                try:
                    record = json.loads(path.read_text())
                    if record.get("generator_revision") == config["generator_revision"]:
                        records.append(record)
                except (OSError, ValueError):
                    continue
        counts = {
            name: sum(r.get("state") == name for r in records)
            for name in ("complete", "failed", "rejected")
        }
        keep = json.loads((ROOT / f"configs/{stage}.json").read_text())["keep_subframes"]
        write(
            state,
            dict(
                stage=stage,
                state="generating" if any(c is None for c in codes) else "checking",
                counts=counts,
                windows=counts["complete"] * (keep // 10),
                elapsed_seconds=time.time() - started,
                worker_pids=[p.pid for p in processes],
                exit_codes=codes,
            ),
        )
        if all(c is not None for c in codes):
            break
        time.sleep(10)
    # 调用当前训练读取器检查标签、坐标和形状；验收阶段另做网络梯度检查。
    with (ROOT / f"logs/{stage}_acceptance.log").open("ab") as log:
        command = [
            sys.executable,
            str(ROOT / "tools/accept_dataset.py"),
            str(DATA / stage),
            "--training-code",
            str(ROOT / "training_compat"),
        ]
        if stage == "acceptance":
            command += ["--model"]
        check = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=log)
    if check.returncode == 0 and stage in ("acceptance", "pilot"):
        with (ROOT / f"logs/{stage}_figures.log").open("ab") as log:
            subprocess.run(
                [
                    "/usr/local/bin/matlab",
                    "-batch",
                    f"addpath('{ROOT.as_posix()}/matlab'); render_dataset_report('{(DATA/stage).as_posix()}');",
                ],
                cwd=ROOT,
                stdout=log,
                stderr=log,
                env=env,
                check=True,
            )
    write(
        state,
        dict(
            stage=stage,
            state=(
                "complete"
                if check.returncode == 0 and all(c == 0 for c in codes)
                else "needs_attention"
            ),
            counts=counts,
            windows=counts["complete"] * (keep // 10),
            elapsed_seconds=time.time() - started,
            worker_exit_codes=codes,
            check_exit_code=check.returncode,
        ),
    )
    # 释放本阶段锁后再启动下一阶段；下一启动器还会核对数量及版本门槛。
    if args.chain and check.returncode == 0 and all(c == 0 for c in codes):
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        next_stage = {"acceptance": "pilot", "pilot": "production"}.get(stage)
        if next_stage:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/run_server.py"),
                    "--stage",
                    next_stage,
                    "--workers",
                    str(args.workers),
                    "--chain",
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        DATA.mkdir(exist_ok=True)
        write(DATA / "pipeline_status.json", {"state": "failed", "error": str(exc)})
        raise
