"""Standalone reference check for an exported inference directory.

Requires NumPy and PyTorch only. No training package, SciPy, NCCL or server paths.
This entry validates normalized network input; raw IQ still needs the specified
LTE frontend and exactly the preprocessing recorded in deployment.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=["weights", "scripted"], default="weights")
    args = parser.parse_args()
    root = Path(args.bundle)
    cfg = json.loads((root/"deployment.json").read_text(encoding="utf-8"))
    if cfg.get("artifact_schema") != "pdsch-inference-v1":
        raise ValueError("Unsupported inference artifact")
    device = torch.device(args.device)
    if args.backend == "weights":
        # model.py sits beside this script in the exported directory.
        from model import PDSCHUNet
        model = PDSCHUNet(**cfg["model_config"])
        model.load_state_dict(torch.load(root/"weights.pt", map_location="cpu", weights_only=True))
        model = model.to(device).eval()
    else:
        model = torch.jit.load(str(root/"model_scripted.pt"), map_location=device).eval()
    with np.load(root/"reference.npz", allow_pickle=False) as reference:
        x = torch.from_numpy(reference["normalized_input"]).to(device)
        with torch.inference_mode():
            out = model(x, cfg["reference_symbols_per_subframe"])
        errors = {}
        for key in ("re", "prb"):
            actual = out[key].float().cpu().numpy()
            expected = reference[key+"_logits"]
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
            errors[key] = float(np.max(np.abs(actual-expected)))
    print(json.dumps({"passed": True, "device": str(device), "backend": args.backend,
                      "max_logit_error": errors, "limited_training": cfg["limited_training"]}, indent=2))


if __name__ == "__main__":
    main()
