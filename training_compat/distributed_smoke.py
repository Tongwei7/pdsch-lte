"""Synthetic distributed software check. This does not measure LTE accuracy."""
import argparse
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from pdsch_dl.engine import loss_fn, write_json
from pdsch_dl.model import PDSCHUNet
from pdsch_dl.runtime import setup, precision_for, autocast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", default="auto")
    parser.add_argument("--output", default="distributed_validation.json")
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    runtime = setup(args.device)
    try:
        precision = precision_for(args.amp, runtime.device)
        torch.manual_seed(42)
        model = PDSCHUNet().to(runtime.device)
        wrapped = DistributedDataParallel(model, device_ids=[runtime.device.index] if runtime.device.type == "cuda" else None,
                                          broadcast_buffers=False) if runtime.world > 1 else model
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        torch.manual_seed(100+runtime.rank)
        started = time.perf_counter()
        for _ in range(2):
            x = torch.randn(1, 3, 72, 28, device=runtime.device)
            targets = {"re": torch.zeros(1, 1, 72, 28, device=runtime.device),
                       "prb": torch.zeros(1, 1, 6, 2, device=runtime.device)}
            optimizer.zero_grad(set_to_none=True)
            with autocast(runtime.device, precision):
                loss = loss_fn(wrapped(x), targets, .3)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        checksum = torch.stack([p.detach().double().sum() for p in model.parameters()]).sum().reshape(1)
        values = [torch.zeros_like(checksum) for _ in range(runtime.world)]
        if runtime.world > 1:
            dist.all_gather(values, checksum)
        else:
            values = [checksum]
        synchronized = all(torch.allclose(v, values[0], rtol=1e-8, atol=1e-8) for v in values)
        if not synchronized:
            raise RuntimeError("DDP parameter synchronization failed")
        if runtime.primary:
            report = {"passed": True, "world_size": runtime.world, "device": str(runtime.device),
                      "precision": precision, "parameters_synchronized": synchronized,
                      "backend": dist.get_backend() if runtime.world > 1 else None,
                      "seconds": time.perf_counter()-started, "purpose": "synthetic software check, not LTE accuracy"}
            write_json(args.output, report)
            print(report)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
