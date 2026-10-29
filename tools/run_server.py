"""服务器批次入口：冻结计划、按任务参数续跑、验收后才衔接下一阶段。"""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from batch_plan import (
    DEFAULT_BATCH,
    freeze_batch,
    freeze_plan,
    matching_record,
    require_gate,
    source_identity,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data/luorongwei/pdsch_lte")


def environment():
    """仅向项目私有目录补充缺失依赖，保持服务器现有 PyTorch。"""
    deps = ROOT / ".deps"
    sys.path.insert(0, str(deps))
    if any(importlib.util.find_spec(p) is None for p in ("numpy", "scipy")):
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
            check=True,
        )
    return dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [str(deps), str(ROOT / "training_compat"), os.environ.get("PYTHONPATH", "")]
        ),
    )


def runtime_identity(matlab, env):
    """记录真实 MATLAB/LTE Toolbox 和数值库版本，升级环境时使用新批次。"""
    code = (
        "v=ver('lte'); assert(~isempty(v)); "
        "fprintf('PDSCH_ENV:%s\\n',jsonencode(struct('matlab',version('-release'),"
        "'lte',v.Version))); "
    )
    output = subprocess.check_output([matlab, "-batch", code], env=env, text=True)
    line = next(
        (s.split("PDSCH_ENV:", 1)[1] for s in output.splitlines() if "PDSCH_ENV:" in s), None
    )
    if line is None:
        raise RuntimeError("Could not determine MATLAB and LTE Toolbox versions")
    result = json.loads(line)
    result.update(
        python=sys.version.split()[0],
        packages={p: importlib.metadata.version(p) for p in ("numpy", "scipy")},
    )
    return result


def read_records(folder, plan):
    records = []
    expected = {j["id"]: j for j in plan["jobs"]}
    for path in (folder / "status").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        job = expected.get(record.get("id"))
        if (
            not job
            or record.get("generator_revision") != plan["generator_revision"]
            or record.get("task_sha256") != job["task_sha256"]
            or any(record.get(k) != job[k] for k in ("split", "group_id"))
        ):
            raise RuntimeError(
                f"Task parameters/revision mismatch: {path}; outputs were not overwritten"
            )
        if record.get("state") == "complete":
            if not record.get("files") or any(not (folder / p).is_file() for p in record["files"]):
                raise RuntimeError(f"Completed task has missing files: {path}")
        records.append(record)
    return records


def run_stage(args, batch_root, identity, env):
    stage = args.stage
    config_path, config = freeze_plan(batch_root, stage, identity)
    folder = batch_root / stage
    for prerequisite in {
        "acceptance": [],
        "pilot": ["acceptance"],
        "production": ["acceptance", "pilot"],
    }[stage]:
        _, previous = freeze_plan(batch_root, prerequisite, identity)
        report = json.loads((batch_root / prerequisite / "acceptance_report.json").read_text())
        if report.get("plan_sha256") != previous["plan_sha256"]:
            raise RuntimeError("Prerequisite report does not belong to this frozen plan")
        require_gate(report, prerequisite)
    records = read_records(folder, config)
    pending = len(config["jobs"]) - sum(r.get("state") == "complete" for r in records)
    # 只为尚未完成的任务预留空间；不再为 2,000 份硬性要求 700 GiB。
    bytes_per_capture = 45 * 2**20
    if stage == "production":
        pilot_files = list((batch_root / "pilot/shards").glob("*.mat"))
        if pilot_files:
            bytes_per_capture = sum(p.stat().st_size for p in pilot_files) / len(pilot_files)
    required = pending * bytes_per_capture * 1.3 + 2 * 2**30
    if shutil.disk_usage(batch_root).free < required:
        raise RuntimeError(f"Insufficient data space: estimated {required / 2**30:.1f} GiB needed")
    logs = ROOT / "logs" / args.batch
    logs.mkdir(parents=True, exist_ok=True)
    state_path = batch_root / "pipeline_status.json"
    write_json(
        batch_root / f"{stage}_invocation.json",
        {
            "recorded_at": time.time(),
            "source": source_identity(ROOT),
            "plan_sha256": config["plan_sha256"],
            "runtime": identity["runtime"],
        },
    )
    if stage == "acceptance":
        with (logs / "physical_unit_validation.log").open("ab") as log:
            subprocess.run(
                [
                    args.matlab,
                    "-batch",
                    f"addpath('{ROOT.as_posix()}/matlab'); r=validate_link; assert(r.passed);",
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=log,
                check=True,
            )
    started, processes = time.time(), []
    try:
        for worker in range(args.workers):
            code = (
                f"addpath('{ROOT.as_posix()}/matlab'); maxNumCompThreads(2); "
                f"lte_dataset_generate('{config_path.as_posix()}',{worker},{args.workers});"
            )
            with (logs / f"{stage}_worker_{worker:02d}.log").open("ab") as log:
                processes.append(
                    subprocess.Popen(
                        [args.matlab, "-batch", code],
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=log,
                        stdin=subprocess.DEVNULL,
                    )
                )
        write_json(
            batch_root / f"{stage}_pids.json",
            {"launcher": os.getpid(), "workers": [p.pid for p in processes]},
        )
        while True:
            codes = [p.poll() for p in processes]
            records = read_records(folder, config)
            counts = {
                s: sum(r["state"] == s for r in records) for s in ("complete", "failed", "rejected")
            }
            write_json(
                state_path,
                dict(
                    batch=args.batch,
                    stage=stage,
                    state="generating",
                    counts=counts,
                    requested_captures=len(config["jobs"]),
                    windows=counts["complete"] * (config["keep_subframes"] // 10),
                    elapsed_seconds=time.time() - started,
                    worker_exit_codes=codes,
                ),
            )
            if all(c is not None for c in codes):
                break
            time.sleep(10)
    except BaseException:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            p.wait()
        raise
    if source_identity(ROOT)["revision"] != identity["generator_revision"]:
        raise RuntimeError("Generation source changed while workers were running")
    freeze_plan(batch_root, stage, identity)
    command = [
        sys.executable,
        str(ROOT / "tools/accept_dataset.py"),
        str(folder),
        "--training-code",
        str(ROOT / "training_compat"),
    ]
    if stage == "acceptance":
        command += ["--model"]
    with (logs / f"{stage}_acceptance.log").open("ab") as log:
        check = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=log)
    report_path = folder / "acceptance_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    passed = check.returncode == 0 and all(c == 0 for c in codes) and report.get("passed", False)
    write_json(
        state_path,
        dict(
            batch=args.batch,
            stage=stage,
            state="complete" if passed else "needs_attention",
            counts=counts,
            windows=report.get("total_windows", 0),
            worker_exit_codes=codes,
            check_exit_code=check.returncode,
            elapsed_seconds=time.time() - started,
        ),
    )
    if not passed:
        raise RuntimeError(f"{stage} did not pass; see {logs / (stage + '_acceptance.log')}")
    require_gate(report, stage)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["acceptance", "pilot", "production"], default="acceptance"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chain", action="store_true")
    parser.add_argument("--batch", default=DEFAULT_BATCH)
    parser.add_argument(
        "--captures",
        type=int,
        default=None,
        help="New batch defaults to 2000; resume keeps the frozen count",
    )
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--matlab", default="/usr/local/bin/matlab")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.batch):
        parser.error("Invalid --batch")
    import fcntl

    args.data_root.mkdir(parents=True, exist_ok=True)
    with (args.data_root / f".{args.batch}.pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = environment()
        identity = source_identity(ROOT)
        batch_root, batch = freeze_batch(
            args.data_root, args.batch, args.captures, identity, runtime_identity(args.matlab, env)
        )
        try:
            while True:
                run_stage(args, batch_root, batch, env)
                following = {"acceptance": "pilot", "pilot": "production"}.get(args.stage)
                if not args.chain or following is None:
                    break
                args.stage = following
        except Exception as exc:
            write_json(
                batch_root / "pipeline_status.json",
                {"batch": args.batch, "stage": args.stage, "state": "failed", "error": str(exc)},
            )
            raise


if __name__ == "__main__":
    main()
