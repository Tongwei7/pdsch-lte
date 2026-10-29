"""Export CPU/FP32 inference artifacts without optimizer, DDP or server paths."""
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .data import SCHEMA
from .model import PDSCHUNet


def export_bundle(checkpoint, directory, subcarriers=1200, symbols=14):
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if source.get("contract") != SCHEMA:
        raise ValueError("Checkpoint contract unsupported")
    if subcarriers % 12 or symbols not in (12, 14):
        raise ValueError("Invalid reference input dimensions")
    model = PDSCHUNet(**source["model_config"])
    model.load_state_dict(source["model_state"])
    model = model.float().cpu().eval()
    # Scripting retains K/time padding and normal/extended-CP branches.
    scripted = torch.jit.script(model)
    config = {"artifact_schema": "pdsch-inference-v1", "input_contract": SCHEMA,
              "model_config": source["model_config"], "preprocessing": source["options"],
              "normalizer": source["normalizer"], "prb_threshold": source["threshold"],
              "re_threshold": .5, "reference_symbols_per_subframe": symbols,
              "network_input": "normalized float32 [B,C,K,T]; forward(x, symbols_per_subframe)",
              "channel_order": "I(all selected RX), Q(all selected RX), log10-power(all selected RX), optional public_excluded/public_valid",
              "power": "10*log10(I**2+Q**2+1e-12); digital units, not calibrated dBm",
              "outputs": "raw logits re [B,1,K,T], prb [B,1,K/12,T/L]; apply sigmoid once",
              "coordinates": "0-based active subcarrier indices, DC omitted; subframes relative to capture",
              "limited_training": source["limited_run"], "trained_epoch": source["epoch"],
              "export_torch": str(torch.__version__), "precision": "float32; no optimizer or DDP state",
              "scope": "occupancy regions, not RNTI or UE identity"}
    generator = torch.Generator().manual_seed(2026)
    x = torch.randn(1, model.in_channels, subcarriers, source["options"]["window"]*symbols, generator=generator)
    with torch.inference_mode():
        expected = model(x, symbols)
        actual = scripted(x, symbols)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), output/"weights.pt")
    scripted.save(str(output/"model_scripted.pt"))
    (output/"deployment.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(output/"reference.npz", normalized_input=x.numpy(),
                        re_logits=expected["re"].numpy(), prb_logits=expected["prb"].numpy())
    shutil.copyfile(Path(__file__).parent/"model.py", output/"model.py")
    portable = Path(__file__).resolve().parent.parent/"portable_inference.py"
    shutil.copyfile(portable, output/"portable_inference.py")
    (output/"requirements.txt").write_text("numpy>=1.26,<3\ntorch>=2.5,<3\n", encoding="utf-8")
    return {"output": str(output.resolve()), "parameters": sum(p.numel() for p in model.parameters()),
            "reference_checked": True, "limited_training": source["limited_run"]}
