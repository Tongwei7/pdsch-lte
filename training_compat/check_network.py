"""Run software tests and report size/cost; no LTE accuracy claim."""
import argparse
from datetime import datetime, timezone
import io
import platform
import time
import unittest
from pathlib import Path

import numpy as np
import scipy
import torch

from pdsch_dl.engine import write_json
from pdsch_dl.model import PDSCHUNet


def profile(k, channels=3, base=8):
    model = PDSCHUNet(channels, base=base).eval()
    multiply_accumulates = [0]
    def counter(layer, inputs, output):
        kernel = layer.kernel_size[0]*layer.kernel_size[1]
        multiply_accumulates[0] += output.numel()*kernel*(layer.in_channels//layer.groups)
    handles = [m.register_forward_hook(counter) for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    x = torch.zeros(1, channels, k, 140)
    with torch.inference_mode():
        output = model(x)
    for handle in handles:
        handle.remove()
    durations = []
    with torch.inference_mode():
        for _ in range(3):
            start = time.perf_counter()
            model(x)
            durations.append(time.perf_counter()-start)
    return {"base": base, "input_shape": list(x.shape), "output_shapes": {key: list(v.shape) for key, v in output.items()},
            "parameters": sum(p.numel() for p in model.parameters()), "conv_macs": multiply_accumulates[0],
            "mac_definition": "convolution multiply-accumulates only; excludes normalization, interpolation, pooling, activations, frontend and IO",
            "cpu_forward_median_seconds": float(np.median(durations)), "timing_repeats": 3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="software_validation.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent/"tests"))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue())
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "purpose": "software validation, not trained LTE detection performance",
              "interface_status": "proposal_not_final_generator_format", "passed": result.wasSuccessful(),
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "test_log": stream.getvalue(), "python": platform.python_version(), "torch": str(torch.__version__),
              "numpy": np.__version__, "scipy": scipy.__version__, "cuda_available": torch.cuda.is_available(),
              "cpu_threads": torch.get_num_threads()}
    if result.wasSuccessful():
        report["profiles"] = [profile(600), profile(1200), profile(1200, base=16)]
    write_json(args.output, report)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
