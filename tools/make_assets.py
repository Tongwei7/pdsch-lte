"""准备合成系统信息资产，以及可复现的验收/试生成/正式生成参数计划。

assets() 只读取本地已保存的实测解码档案，不访问原始 Y 盘。
服务器通常使用 --plans-only，沿用已经上传的合成 SIB 比特资产。
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / ".runtime" / "lte_asn1"))
import numpy as np


def assets():
    """从已解析的 ASN.1 结构构造合成 SIB1/SIB2，并验证编码往返一致。"""
    from scipy.io import savemat
    from pycrate_asn1dir import RRCLTE

    source = PROJECT / "measured_reference/20261027/deep/system_information/decoded_rrc.json"
    records = json.loads(source.read_text(encoding="utf-8"))["messages"]
    original = next(
        m
        for m in records
        if m["message_type"] == "systemInformationBlockType1" and m["PER_roundtrip_prefix_matches"]
    )
    # 使用真实解码得到的结构作为语法模板，替换网络/小区标识为合成值。
    obj = RRCLTE.EUTRA_RRC_Definitions.BCCH_DL_SCH_Message
    raw_records = json.loads(
        (source.parent / "crc_valid_payloads.json").read_text(encoding="utf-8")
    )
    packet = next(
        bytes.fromhex(r["PayloadHex"])
        for r in raw_records
        if hashlib.sha256(bytes.fromhex(r["PayloadHex"])).hexdigest() == original["payload_sha256"]
    )
    obj.from_uper(packet)
    value = copy.deepcopy(obj.get_val())
    # get_val() 返回有类型的 ASN.1 CHOICE 元组，保留其层次后再修改字段。
    sib1 = value["message"][1][1]
    sib1["cellAccessRelatedInfo"]["plmn-IdentityList"] = [
        {
            "plmn-Identity": {"mcc": [0, 0, 1], "mnc": [0, 1]},
            "cellReservedForOperatorUse": "notReserved",
        }
    ]
    sib1["cellAccessRelatedInfo"]["trackingAreaCode"] = (1, 16)
    sib1["cellAccessRelatedInfo"]["cellIdentity"] = (1, 28)
    sib1["schedulingInfoList"] = [{"si-Periodicity": "rf16", "sib-MappingInfo": []}]
    sib1["si-WindowLength"] = "ms20"
    obj.set_val(value)
    payload = obj.to_uper()
    obj.from_uper(payload)
    assert obj.to_uper() == payload
    dest = ROOT / "reference"
    dest.mkdir(exist_ok=True)
    savemat(
        dest / "synthetic_sib1.mat",
        {"sib1_bits": np.unpackbits(np.frombuffer(payload, dtype=np.uint8)).astype(np.uint8)},
    )
    (dest / "synthetic_sib1.json").write_text(
        json.dumps(
            {
                "purpose": "Synthetic valid SIB1 ASN.1 payload; identifiers replaced",
                "hex": payload.hex(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "decoded": obj.get_val(),
                "source_template_evidence": "Repeated CRC valid and PER-roundtrip-valid SIB1",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    # SIB1 第一个 SI 调度项隐含包含 SIB2，因此也准备对应的 SIB2 资产。
    # 三个带宽各存一份；这不表示已经完整恢复了实测 SIB2 参数。
    si_record = next(m for m in records if m["message_type"] == "systemInformation")
    packet = next(
        bytes.fromhex(r["PayloadHex"])
        for r in raw_records
        if hashlib.sha256(bytes.fromhex(r["PayloadHex"])).hexdigest() == si_record["payload_sha256"]
    )
    obj.from_uper(packet)
    si_value = copy.deepcopy(obj.get_val())
    ies = si_value["message"][1][1]["criticalExtensions"][1]
    ies["sib-TypeAndInfo"] = [entry for entry in ies["sib-TypeAndInfo"] if entry[0] == "sib2"]
    assert len(ies["sib-TypeAndInfo"]) == 1
    sib2 = ies["sib-TypeAndInfo"][0][1]
    sib2["radioResourceConfigCommon"]["prach-Config"]["prach-ConfigInfo"]["prach-FreqOffset"] = 1
    for nrb in (50, 75, 100):
        sib2["freqInfo"]["ul-Bandwidth"] = f"n{nrb}"
        obj.set_val(si_value)
        si_payload = obj.to_uper()
        obj.from_uper(si_payload)
        assert obj.to_uper() == si_payload
        savemat(
            dest / f"synthetic_sib2_{nrb}.mat",
            {
                "sib2_bits": np.unpackbits(np.frombuffer(si_payload, dtype=np.uint8)).astype(
                    np.uint8
                )
            },
        )
    obj.set_val(si_value)
    si_payload = obj.to_uper()
    obj.from_uper(si_payload)
    assert obj.to_uper() == si_payload
    savemat(
        dest / "synthetic_sib2.mat",
        {"sib2_bits": np.unpackbits(np.frombuffer(si_payload, dtype=np.uint8)).astype(np.uint8)},
    )
    (dest / "synthetic_sib2.json").write_text(
        json.dumps(
            {
                "hex": si_payload.hex(),
                "decoded": obj.get_val(),
                "purpose": "Synthetic SIB2 template; not a verified measured SIB2 parameter inference",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def plan(name, groups, keep_sf):
    """生成确定性任务清单。

    name 是批次名；groups 是捕获任务数（不全等于独立传播实现数）；
    keep_sf 是每份捕获导出的子帧数，1 个子帧为 1 ms。
    pilot/production 使用相同基础种子序列，重叠部分不能重复计为独立样本。
    """
    scenarios = [
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
    ]
    jobs = []
    for i in range(groups):
        rng = np.random.default_rng(2026102800 + i)
        scenario = scenarios[i % len(scenarios)]
        nrb = [100, 50, 75][i % 3]
        # 在生成之前划分集合；后面的衰落/真实间隙配对继续沿用同一分组。
        partition = i % 20
        split = "train" if partition < 14 else "validation" if partition < 17 else "test"
        # 这些场景、SNR/SIR 和业务比例是覆盖设计，不是实测业务分布反演。
        channel = {
            "clean": "Off",
            "epa": "EPA",
            "eva": "EVA",
            "etu": "ETU",
            "deep_contiguous": "TwoRay",
            "true_gap": "TwoRay",
            "cochannel": "EPA",
            "adjacent": "EPA",
            "target_absent": "EPA",
            "spatial_mux": "EVA",
            "distributed": "EPA",
        }[scenario]
        jobs.append(
            dict(
                id=f"{name}_{i:06d}",
                group_id=f"{name}_independent_capture_{i:06d}",
                seed=int(2026102800 + i),
                split=split,
                scenario=scenario,
                nrb=nrb,
                pci=int(rng.integers(0, 504)),
                start_sfn=int(rng.integers(0, 990)),
                center_hz={100: 0, 50: 600000, 75: -500000}[nrb],
                cfo_hz=float(rng.uniform(-350, 350)),
                capture_cut_ms=float(rng.uniform(0.1, 9.5)),
                digital_gain=float(10 ** rng.uniform(-1, 1)),
                snr_db=35.0 if name == "acceptance" else float(rng.uniform(0, 30)),
                sir_db=15.0 if name == "acceptance" else float(rng.uniform(-5, 20)),
                cochannel=scenario in ("cochannel", "target_absent"),
                adjacent=scenario == "adjacent",
                channel=channel,
                interferer_delay_samples=int(rng.integers(0, 600)),
                interferer_cfo_hz=float(rng.uniform(-100, 100)),
                load_probability=float(rng.uniform(0.2, 0.95)),
                spatial_mux=scenario == "spatial_mux",
                distributed=scenario == "distributed",
                save_iq=name == "acceptance" or i % 50 == 0,
                verify_payload=name == "acceptance",
            )
        )
        if scenario == "true_gap":
            # 复用上一连续分配场景的传播参数、随机种子和捕获设置，改变分配形态。
            # 后续业务抽样和按平均功率设噪会随分配改变，不能视作逐样点完全配对。
            # 复制 split/group_id 使这对同源样本不会跨训练、验证、测试集合。
            paired = copy.deepcopy(jobs[-2])
            paired["id"] = jobs[-1]["id"]
            paired["scenario"] = "true_gap"
            jobs[-1] = paired
    config = dict(
        dataset_version="lte-pdsch-occupancy-v1",
        reader_schema="pdsch-occupancy-proposal-v1",
        output_root=f"/data/luorongwei/pdsch_lte/{name}",
        keep_subframes=keep_sf,
        requested_windows=groups * (keep_sf // 10),
        jobs=jobs,
        stop_on_failure=name == "acceptance",
        note="Scenario/MCS/traffic/SNR distributions are synthetic coverage assumptions; measured full scan incomplete.",
    )
    dest = ROOT / "configs"
    dest.mkdir(exist_ok=True)
    (dest / f"{name}.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans-only", action="store_true")
    args = parser.parse_args()
    if not args.plans_only:
        assets()
    plan("acceptance", 22, 30)
    plan("pilot", 500, 100)
    plan("production", 20000, 100)
