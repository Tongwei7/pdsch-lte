"""首轮实验：优先选取已有数据，否则生成 2,000 份新捕获，再单卡训练和测试。"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

from batch_plan import DEFAULT_BATCH, DEFAULT_CAPTURES, write_json

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data/luorongwei/pdsch_lte")
SCENARIOS = {
    "clean",
    "epa",
    "eva",
    "etu",
    "deep_contiguous",
    "true_gap",
    "cochannel",
    "adjacent",
    "target_absent",
    "spatial_mux",
    "distributed",
}


def select_rows(rows, captures):
    """按已有 split 保留整个 group，轮流覆盖场景和带宽，最多选 captures 行。"""
    groups, seen = defaultdict(list), set()
    for row in rows:
        if row.get("alignment_valid") != "1":
            continue
        if row.get("schema") != "pdsch-occupancy-proposal-v1" or not row.get("group_id"):
            raise ValueError("Source manifest has no supported schema/group_id")
        if row.get("split") not in ("train", "validation", "test"):
            raise ValueError("Invalid source split")
        if row["path"] in seen:
            raise ValueError("Duplicate source shard path")
        seen.add(row["path"])
        groups[row["group_id"]].append(row)
    for group in groups.values():
        if len({r["split"] for r in group}) != 1:
            raise ValueError("Same-origin captures cross splits")
    available = sum(map(len, groups.values()))
    if available < captures * 0.8:
        raise ValueError(
            f"Only {available} usable source captures; need at least {int(captures * .8)}"
        )
    limit = min(captures, available)
    quotas = {"train": int(limit * 0.7), "validation": int(limit * 0.15)}
    quotas["test"] = limit - sum(quotas.values())
    selected = []
    for split, quota in quotas.items():
        buckets = defaultdict(list)
        for key, group in groups.items():
            if group[0]["split"] == split:
                bucket = (tuple(sorted({r["scenario"] for r in group})), group[0].get("nrb", ""))
                buckets[bucket].append((hashlib.sha256(key.encode()).hexdigest(), group))
        queues = [deque(group for _, group in sorted(buckets[key])) for key in sorted(buckets)]
        count = 0
        while any(queues):
            for queue in queues:
                if queue:
                    group = queue.popleft()
                    if count + len(group) <= quota:
                        selected.extend(group)
                        count += len(group)
        if count < quota - 2:
            raise ValueError(f"Insufficient {split} groups for requested subset")
    if SCENARIOS - {r["scenario"] for r in selected}:
        raise ValueError("Training subset does not cover all 11 scenarios")
    for split in quotas:
        if {int(r["nrb"]) for r in selected if r["split"] == split} != {50, 75, 100}:
            raise ValueError(f"Subset {split} does not cover all three bandwidths")
    return selected


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_manifest(source, destination, captures, training_code):
    """逐文件校验所选数据；只生成新清单，不复制或改写原 MAT 数据。"""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = select_rows(list(csv.DictReader(stream)), captures)
    sys.path.insert(0, str(training_code))
    from pdsch_dl.data import read_arrays

    counts, scenarios = defaultdict(int), defaultdict(int)
    for row in rows:
        path = (source.parent / row["path"]).resolve()
        if not path.is_relative_to((source.parent / "shards").resolve()):
            raise ValueError("Source manifest path escapes its shards directory")
        if not row.get("sha256") or file_digest(path) != row["sha256"]:
            raise ValueError(f"Source shard checksum mismatch: {path}")
        arrays = read_arrays(path)
        shape = arrays["rx_real"].shape
        if shape[0] // 12 != int(row["nrb"]):
            raise ValueError("Source bandwidth disagrees with manifest")
        if shape[-1] < 10:
            raise ValueError("Source shard has no full 10 ms window")
        indices = arrays["subframe_index"]
        if not any(
            all(indices[i + j + 1] == indices[i + j] + 1 for j in range(9))
            for i in range(len(indices) - 9)
        ):
            raise ValueError("Source shard has no contiguous 10 ms window")
        row["path"] = str(path)  # 新清单在别处，绝对路径仍引用原文件。
        counts[row["split"]] += 1
        scenarios[row["scenario"]] += 1
    if destination.exists():
        with destination.open(encoding="utf-8", newline="") as stream:
            if list(csv.DictReader(stream)) != rows:
                raise RuntimeError("Existing training selection differs; choose a new --experiment")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    provenance = {
        p: file_digest(source.parent / p)
        for p in ("generation_plan.json", "acceptance_report.json")
        if (source.parent / p).exists()
    }
    return {
        "source_manifest": str(source),
        "source_manifest_sha256": file_digest(source),
        "selected_captures": len(rows),
        "split_captures": dict(counts),
        "source_provenance_sha256": provenance,
        "scenario_counts": dict(scenarios),
        "copies_of_mat_files_created": 0,
    }


def launch(args, env):
    """训练完成后用独立测试集评价；阈值与最佳模型由验证集选择。"""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; first full training requires a server GPU")
    directory = args.data_root / "training_runs" / args.experiment
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "workflow_status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    if previous.get("state") == "complete":
        print(json.dumps(previous, ensure_ascii=False))
        return
    manifest = directory / "manifest.csv"
    training_code = ROOT / "training_compat"
    if not (training_code / "run.py").exists():
        raise RuntimeError("Server training_compat snapshot is missing")
    sources = [
        args.data_root / "production/manifest.csv",
        args.data_root / "batches" / args.batch / "production/manifest.csv",
    ]
    if manifest.exists():
        selection = json.loads((directory / "selection.json").read_text())
        # 续训也核对原文件摘要，不在旧实验目录偷偷换一批数据。
        selected = prepare_manifest(
            Path(selection["source_manifest"]), manifest, args.captures, training_code
        )
    else:
        selected = None
        for source in sources:
            if not source.exists():
                continue
            try:
                selected = prepare_manifest(source, manifest, args.captures, training_code)
            except ValueError as exc:
                print(f"Source not usable: {source}: {exc}", flush=True)
                continue
            break
        if selected is None:
            write_json(status_path, {"state": "generating", "batch": args.batch})
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/run_server.py"),
                    "--chain",
                    "--batch",
                    args.batch,
                    "--captures",
                    str(args.captures),
                    "--workers",
                    str(args.workers),
                    "--data-root",
                    str(args.data_root),
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )
            selected = prepare_manifest(sources[1], manifest, args.captures, training_code)
        write_json(directory / "selection.json", selected)
    run_info = directory / "run_info.json"
    if not run_info.exists():
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/record_run.py"),
                "--manifest",
                str(manifest),
                "--output",
                str(run_info),
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
    model_root = directory / "model"
    model = previous.get("model_directory")
    if model:
        model_root = Path(model)
    if model_root.exists() and not (model_root / "history.json").exists():
        # 训练可能在写入 history 之前中断；只有可续训的检查点才能继续。
        if not (model_root / "last.pt").exists():
            raise RuntimeError(
                "Incomplete model directory without last.pt; choose a new --experiment"
            )
    phase = (
        previous.get("failed_phase") if previous.get("state") == "failed" else previous.get("state")
    )
    if phase != "evaluating":
        command = [
            sys.executable,
            str(training_code / "run.py"),
            "train",
            "--manifest",
            str(manifest),
            "--output",
            str(model_root),
            "--epochs",
            str(args.epochs),
            "--patience",
            "10",
            "--batch-size",
            "4",
            "--workers",
            "4",
            "--public",
            "--device",
            "cuda:0",
            "--amp",
            "auto",
            "--seed",
            "42",
        ]
        if model_root.exists():
            checkpoint = model_root / "last.pt"
            if not checkpoint.exists():
                raise RuntimeError("Existing model output has no last.pt")
            model_root = directory / f"model_resume_{time.time_ns()}"
            command[command.index("--output") + 1] = str(model_root)
            command += ["--resume", str(checkpoint)]
        write_json(status_path, dict(state="training", model_directory=str(model_root), **selected))
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    write_json(status_path, dict(state="evaluating", model_directory=str(model_root), **selected))
    subprocess.run(
        [
            sys.executable,
            str(training_code / "run.py"),
            "evaluate",
            "--manifest",
            str(manifest),
            "--checkpoint",
            str(model_root / "best.pt"),
            "--split",
            "test",
            "--output",
            str(directory / "test_metrics.json"),
            "--device",
            "cuda:0",
            "--batch-size",
            "4",
            "--workers",
            "4",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    write_json(status_path, dict(state="complete", model_directory=str(model_root), **selected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=int, default=DEFAULT_CAPTURES)
    parser.add_argument("--batch", default=DEFAULT_BATCH)
    parser.add_argument("--experiment", default="first_train_2000")
    parser.add_argument("--workers", type=int, default=4, help="MATLAB generation processes")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--data-root", type=Path, default=DATA)
    args = parser.parse_args()
    import re

    if any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", s)
        for s in (args.batch, args.experiment)
    ):
        parser.error("Use simple names for --batch and --experiment")
    if not 100 <= args.captures <= 100000 or args.epochs < 1 or not 1 <= args.workers <= 32:
        parser.error("Invalid captures/epochs/workers")
    from run_server import environment

    env = environment()
    # 首轮只用一张 GPU，足以训练这个小网络；保留其他卡给用户的其他任务。
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("PDSCH_TRAIN_GPU", "0")
    import fcntl

    args.data_root.mkdir(parents=True, exist_ok=True)
    with (args.data_root / f".{args.experiment}.training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            launch(args, env)
        except Exception as exc:
            path = args.data_root / "training_runs" / args.experiment / "workflow_status.json"
            old = json.loads(path.read_text()) if path.exists() else {}
            old.update(failed_phase=old.get("state"), state="failed", error=str(exc))
            write_json(path, old)
            raise


if __name__ == "__main__":
    main()
