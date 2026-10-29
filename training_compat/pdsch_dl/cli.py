import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .data import GridDataset, discover, features, read_arrays
from .engine import device_for, evaluate, load_checkpoint, train, write_json
from .metrics import region_records
from .runtime import precision_for, autocast
from .deployment import export_bundle


def predict(args):
    device = device_for(args.device)
    model, ckpt, norm = load_checkpoint(args.checkpoint, device)
    precision = precision_for(getattr(args, "amp", "auto"), device)
    a = read_arrays(args.input, labels=False)
    opts = ckpt["options"]
    window = opts["window"]
    k, l, _, n = a["rx_real"].shape
    prb = np.zeros((k//12, n), np.float32)
    re = np.zeros((k, l, n), np.float32)
    counts = np.zeros(n, np.int32)
    model.eval()
    # Process each continuous run separately. Missing subframes are never bridged.
    boundaries = np.r_[0, np.flatnonzero(np.diff(a["subframe_index"]) != 1)+1, n]
    starts = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        if hi-lo >= window:
            local = list(range(int(lo), int(hi-window+1), window))
            if local[-1] != hi-window:
                local.append(int(hi-window))
            starts.extend(local)
    with torch.inference_mode():
        for start in starts:
            if not np.equal(np.diff(a["subframe_index"][start:start+window]), 1).all():
                continue
            x = norm(features(a, start, window, opts["mode"], opts["rx"], opts["public"]))
            with autocast(device, precision):
                out = model(torch.from_numpy(x[None]).to(device), l)
            prb[:, start:start+window] += out["prb"].float().sigmoid().cpu().numpy()[0, 0]
            fine = out["re"].float().sigmoid().cpu().numpy()[0, 0].reshape(k, window, l).transpose(0, 2, 1)
            re[:, :, start:start+window] += fine
            counts[start:start+window] += 1
    valid = counts > 0
    prb /= np.maximum(counts, 1)[None]
    re /= np.maximum(counts, 1)[None, None]
    records = []
    for i in np.flatnonzero(valid):
        records += region_records(prb[:, i:i+1], re[:, :, i], a["subframe_index"][i:i+1], l, ckpt["threshold"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"probabilities.npz", prb_probability=prb, re_probability=re,
                        valid_subframes=valid, subframe_index=a["subframe_index"])
    result = {"threshold": ckpt["threshold"], "valid_subframes": int(valid.sum()), "unprocessed_subframes": int((~valid).sum()),
              "regions": records, "coordinates": "zero-based active-subcarrier indices; DC omitted; frame ordinal is capture-relative, not SFN",
              "semantics": "contiguous occupancy regions, not user identities; probabilities are uncalibrated"}
    write_json(output/"regions.json", result)
    return {"output": str(output.resolve()), "valid_subframes": result["valid_subframes"], "regions": len(records)}


def main():
    parser = argparse.ArgumentParser(description="Lightweight LTE PDSCH occupancy network; new training-interface proposal")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Validate explicit new contract and splits")
    inspect.add_argument("--manifest", required=True)
    inspect.add_argument("--window", type=int, default=10)
    inspect.add_argument("--output", default="inspection.json")
    training = sub.add_parser("train")
    training.add_argument("--manifest", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--window", type=int, default=10)
    training.add_argument("--mode", choices=["iq_power", "power"], default="iq_power")
    training.add_argument("--rx", choices=["first", "all"], default="first")
    training.add_argument("--public", action="store_true")
    training.add_argument("--base", type=int, default=8)
    training.add_argument("--epochs", type=int, default=60)
    training.add_argument("--patience", type=int, default=10)
    training.add_argument("--learning-rate", type=float, default=1e-3)
    training.add_argument("--auxiliary-weight", type=float, default=.3)
    training.add_argument("--normalization-windows", type=int, default=256)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--max-batches", type=int, default=None, help="Debug-only truncation; checkpoint is marked limited_run")
    training.add_argument("--resume", default=None, help="Resume optimizer/scaler/epoch from last.pt into a new output directory")
    validation = sub.add_parser("evaluate")
    validation.add_argument("--manifest", required=True)
    validation.add_argument("--checkpoint", required=True)
    validation.add_argument("--split", choices=["validation", "test"], default="test")
    validation.add_argument("--output", required=True)
    inference = sub.add_parser("predict")
    inference.add_argument("--input", required=True, help="Receiver grid shard; no target labels required")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--output", required=True)
    export = sub.add_parser("export", help="CPU/FP32 portable inference bundle")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--subcarriers", type=int, default=1200)
    export.add_argument("--symbols", type=int, choices=[12, 14], default=14)
    for command in (training, validation):
        command.add_argument("--batch-size", type=int, default=4)
        command.add_argument("--workers", type=int, default=0)
    for command in (training, validation, inference):
        command.add_argument("--device", default="auto")
        command.add_argument("--amp", choices=["auto", "off", "bf16", "fp16"], default="auto")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "4")))
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and args.command != "train":
        raise ValueError("torchrun is supported for train only; evaluate/predict use one process")
    if args.command == "inspect":
        rows = discover(args.manifest)
        result = {"interface_status": "proposal_not_final_generator_format", "splits": [GridDataset(rows, s, window=args.window).summary for s in ("train", "validation", "test")]}
        write_json(args.output, result)
    elif args.command == "train":
        result = train(args)
    elif args.command == "evaluate":
        result = evaluate(args)
    elif args.command == "export":
        result = export_bundle(args.checkpoint, args.output, args.subcarriers, args.symbols)
    else:
        result = predict(args)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
