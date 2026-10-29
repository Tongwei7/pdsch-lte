"""验证 MATLAB 分块与当前 AI 读取器的兼容性，并生成数据清单和检查报告。

该检查验证软件接口和标签一致性，不评价训练后的 PDSCH 检测精度。
真实分配只用于检查/监督，不会被拼接进网络输入。
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]


def finalize(directory, training_code=None, check_model=False):
    """汇总一个批次。

    directory 是 acceptance/pilot/production 数据目录；training_code 是冻结的
    AI 代码快照目录。check_model=True 时，每个已有集合各取一个窗口做前向
    和反向传播，不执行优化器更新，也不保存训练模型。
    """
    directory = Path(directory).resolve()
    records = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((directory / "status").glob("*.json"))
    ]
    # 过滤其他生成器版本的记录，防止续跑时把旧文件误计入新批次。
    plan_path = directory / "generation_plan.json"
    revision = (
        json.loads(plan_path.read_text()).get("generator_revision") if plan_path.exists() else None
    )
    stale = sum(r.get("generator_revision") != revision for r in records) if revision else 0
    if revision:
        records = [r for r in records if r.get("generator_revision") == revision]
    rows, stats, failures = [], [], []
    split_groups = {}
    for record in records:
        if record["state"] != "complete":
            failures.append(
                {
                    "id": record["id"],
                    "state": record["state"],
                    "reason": record.get("reason", record.get("message")),
                }
            )
            continue
        group, split = record["group_id"], record["split"]
        if group in split_groups and split_groups[group] != split:
            raise ValueError("Shared source group across splits")
        split_groups[group] = split
        for relative in record["files"]:
            path = directory / relative
            a = loadmat(path)
            re = a["label_re"]
            nrb, s = re.shape[0] // 12, re.shape[-1]
            # K=12*nrb；每个 PRB 的 12 个子载波和 14 个符号取并集。
            # 这与 MATLAB 发送端的 PRB 标签定义一致，且不依赖接收功率。
            expected = re.reshape(nrb, 12, 14, s).any(axis=(1, 2))
            assert re.dtype == np.uint8 and set(np.unique(re)).issubset({0, 1})
            assert np.array_equal(expected, a["label_prb"])
            # 正常 CP 下一个时隙有 7 个符号；分别核对两个时隙的占用。
            slot = np.stack(
                [
                    re[:, k * 7 : (k + 1) * 7].reshape(nrb, 12, 7, s).any(axis=(1, 2))
                    for k in range(2)
                ],
                axis=1,
            )
            assert np.array_equal(slot, a["label_prb_slot"])
            assert a["rx_real"].dtype == np.float32 and np.isfinite(a["rx_real"]).all()
            assert a["rx_imag"].dtype == np.float32 and np.isfinite(a["rx_imag"]).all()
            assert np.all(a["public_excluded"] <= a["public_valid"])
            # 公共图是接收侧估计，可能判断错误。这里只统计错误排除数，
            # 不用发送真值“修补”输入，否则会造成部署时无法获得的信息泄漏。
            wrong_public = int(np.sum((a["public_excluded"] > 0) & (re > 0)))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(
                dict(
                    schema="pdsch-occupancy-proposal-v1",
                    path=relative,
                    split=split,
                    group_id=group,
                    alignment_valid="1",
                    scenario=record["scenario"],
                    sha256=digest,
                    dataset_version="lte-pdsch-occupancy-v1",
                    nrb=nrb,
                    subframes=s,
                )
            )
            stats.append(
                dict(
                    id=record["id"],
                    nrb=nrb,
                    subframes=s,
                    positive_prb_fraction=float(expected.mean()),
                    receiver_public_false_exclusions=wrong_public,
                )
            )
    # 路径相对清单保存，便于数据集迁移；group_id 用来检查跨集合泄漏。
    manifest = directory / "manifest.csv"
    temporary = manifest.with_suffix(".partial.csv")
    if rows:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(manifest)
    report = dict(
        dataset_version="lte-pdsch-occupancy-v1",
        generator_revision=revision,
        stale_records_ignored=stale,
        completed=len(rows),
        failures=failures,
        total_subframes=sum(x["subframes"] for x in stats),
        total_windows=sum(x["subframes"] // 10 for x in stats),
        shards=stats,
        warning="Compatibility and mapping verification, not trained detection performance",
    )
    report["passed"] = bool(rows) and not failures
    # 直接使用现有读取器，而不是另写一个只为验收通过的简化读取器。
    if training_code and rows:
        sys.path.insert(0, str(Path(training_code).resolve()))
        from pdsch_dl.data import discover, GridDataset, read_arrays

        for row in rows:
            read_arrays(directory / row["path"])
        discovered = discover(manifest)
        report["current_reader_validated_shards"] = len(rows)
        report["split_windows"] = {}
        for split in ("train", "validation", "test"):
            if not any(r["split"] == split for r in rows):
                continue
            ds = GridDataset(discovered, split, public=True)
            report["split_windows"][split] = len(ds)
            if check_model:
                # 新建临时模型，核对两个输出头与标签的形状及梯度数值。
                import torch
                from pdsch_dl.model import PDSCHUNet

                torch.set_num_threads(2)
                sample = ds[0]
                x = sample["x"][None]
                model = PDSCHUNet(x.shape[1])
                out = model(x, int(sample["symbols"]))
                assert out["re"].shape == sample["re"][None].shape
                assert out["prb"].shape == sample["prb"][None].shape
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    out["prb"], sample["prb"][None]
                )
                loss += 0.3 * torch.nn.functional.binary_cross_entropy_with_logits(
                    out["re"], sample["re"][None]
                )
                loss.backward()
                assert all(
                    torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
                )
        report["current_model_forward_backward"] = bool(check_model)
    (directory / "acceptance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("shards", "failures")}, ensure_ascii=False
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--training-code")
    parser.add_argument("--model", action="store_true")
    args = parser.parse_args()
    result = finalize(args.directory, args.training_code, args.model)
    if not result["completed"]:
        raise SystemExit("No successfully generated shards to accept")
