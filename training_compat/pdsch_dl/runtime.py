"""One-process-per-GPU runtime for torchrun; CPU/Gloo is a software-test path."""
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
import os

import torch
import torch.distributed as dist


@dataclass
class Runtime:
    device: torch.device
    rank: int = 0
    world: int = 1
    owned_group: bool = False

    @property
    def primary(self):
        return self.rank == 0

    def broadcast(self, value):
        if self.world == 1:
            return value
        items = [value if self.primary else None]
        dist.broadcast_object_list(items, src=0, device=self.device)
        return items[0]

    def sum(self, values):
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.world > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.cpu().tolist()

    def close(self):
        if self.owned_group and dist.is_initialized():
            dist.destroy_process_group()


def setup(request="auto"):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    cuda = torch.cuda.is_available()
    if request == "auto":
        device = torch.device("cuda", local if world > 1 else 0) if cuda else torch.device("cpu")
    else:
        device = torch.device(request)
    if device.type == "cuda":
        if not cuda:
            raise ValueError("CUDA unavailable; install a compatible CUDA-enabled PyTorch on the server")
        if world > 1:
            if device.index is not None and device.index != local:
                raise ValueError("torchrun assigns GPU via LOCAL_RANK; use --device cuda")
            device = torch.device("cuda", local)
        elif device.index is None:
            device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
    if world > 1:
        if not dist.is_available():
            raise ValueError("This PyTorch build has no distributed support")
        backend = "nccl" if device.type == "cuda" else "gloo"
        if backend == "nccl" and not dist.is_nccl_available():
            raise ValueError("Multi-GPU server training requires NCCL-enabled PyTorch")
        dist.init_process_group(backend, init_method=os.environ.get("PDSCH_DDP_INIT_METHOD", "env://"),
                                rank=rank, world_size=world, timeout=timedelta(minutes=30))
    return Runtime(device, rank, world, world > 1)


def precision_for(request, device):
    if request == "auto":
        return ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type == "cuda" else "off"
    if request not in ("off", "bf16", "fp16"):
        raise ValueError("precision must be auto/off/bf16/fp16")
    if request != "off" and device.type != "cuda":
        raise ValueError("Explicit bf16/fp16 mode requires CUDA; use --amp off on CPU")
    if request == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU/PyTorch combination does not support bf16")
    return request


def autocast(device, precision):
    if precision == "off":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
