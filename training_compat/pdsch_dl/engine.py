"""Train/validation/evaluation. Test labels never select thresholds or checkpoints."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from .data import GridDataset, Normalizer, ShapeBatches, DistributedShapeBatches, discover, SCHEMA
from .metrics import binary_counts, binary_scores, segment_counts, segment_scores
from .model import PDSCHUNet
from .runtime import setup, precision_for, autocast


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def device_for(request):
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable in this Python environment")
    return device


def loader(dataset, batch_size, shuffle=False, workers=0, seed=42, runtime=None):
    if runtime is not None and runtime.world > 1 and shuffle:
        sampler = DistributedShapeBatches(dataset, batch_size, runtime.rank, runtime.world, seed)
    else:
        sampler = ShapeBatches(dataset, batch_size, shuffle, seed)
    options = {"pin_memory": runtime is not None and runtime.device.type == "cuda"}
    if workers:
        options.update(persistent_workers=True, multiprocessing_context="spawn", prefetch_factor=2)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=workers, **options)


def loss_fn(out, batch, auxiliary_weight):
    return F.binary_cross_entropy_with_logits(out["prb"].float(), batch["prb"]) + auxiliary_weight*F.binary_cross_entropy_with_logits(out["re"].float(), batch["re"])


def data_options(options):
    return {k: options[k] for k in ("window", "mode", "rx", "public")}


@torch.inference_mode()
def score(model, batches, device, thresholds, auxiliary_weight=0.3, max_batches=None, precision="off"):
    model.eval()
    totals = [np.zeros(4, np.int64) for _ in thresholds]
    segments = [np.zeros(4, np.int64) for _ in thresholds]
    re_counts = np.zeros(4, np.int64)
    loss, samples, elapsed = 0., 0, 0.
    for step, batch in enumerate(batches):
        if max_batches and step >= max_batches:
            break
        symbols = int(batch["symbols"][0])
        x = batch["x"].to(device)
        targets = {key: batch[key].to(device) for key in ("re", "prb")}
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with autocast(device, precision):
            out = model(x, symbols)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter()-start
        b = len(x)
        loss += float(loss_fn(out, targets, auxiliary_weight))*b
        samples += b
        prob = out["prb"].float().sigmoid().cpu().numpy()[:, 0]
        truth = batch["prb"].numpy()[:, 0]
        for i, threshold in enumerate(thresholds):
            totals[i] += binary_counts(prob, truth, threshold)
            segments[i] += segment_counts(prob, truth, threshold)
        re_counts += binary_counts(out["re"].float().sigmoid().cpu().numpy(), batch["re"].numpy(), .5)
    if not samples:
        raise ValueError("No evaluation samples")
    return [{"threshold": float(t), "prb": binary_scores(c), "segments": segment_scores(s),
             "re_at_0_5": binary_scores(re_counts), "loss": loss/samples, "windows": samples,
             "model_seconds_per_window": elapsed/samples} for t, c, s in zip(thresholds, totals, segments)]


def load_checkpoint(path, device):
    # Checkpoints created here contain tensors and primitive state only.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("contract") != SCHEMA:
        raise ValueError("Unsupported checkpoint contract")
    model = PDSCHUNet(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt, Normalizer(**ckpt["normalizer"])


def train(args):
    if args.epochs < 1 or args.patience < 1 or args.auxiliary_weight < 0:
        raise ValueError("Invalid training hyperparameters")
    runtime = setup(args.device)
    try:
        return train_runtime(args, runtime)
    finally:
        runtime.close()


def train_runtime(args, runtime):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = runtime.device
    precision = precision_for(getattr(args, "amp", "auto"), device)
    rows = discover(args.manifest)
    opts = {"window": args.window, "mode": args.mode, "rx": args.rx, "public": args.public}
    training = GridDataset(rows, "train", **opts)
    previous = None
    if getattr(args, "resume", None):
        previous = torch.load(args.resume, map_location="cpu", weights_only=True)
        if previous.get("contract") != SCHEMA or "optimizer_state" not in previous:
            raise ValueError("Checkpoint is not a resumable server checkpoint")
        if previous["options"] != opts or previous["model_config"] != {"in_channels": training.channels, "base": args.base}:
            raise ValueError("Resume input options or model configuration differ")
        if previous["seed"] != args.seed:
            raise ValueError("Resume must use the saved seed")
    state = None
    if runtime.primary:
        state = previous["normalizer"] if previous else Normalizer.fit(training, args.normalization_windows).state()
    norm = Normalizer(**runtime.broadcast(state))
    training.normalizer = norm
    training.cache.clear()
    validation = GridDataset(rows, "validation", normalizer=norm, **opts) if runtime.primary else None
    if runtime.primary and training.channels != validation.channels:
        raise ValueError("Train/validation channels differ")
    model = PDSCHUNet(training.channels, args.base).to(device)
    if previous:
        model.load_state_dict(previous["model_state"])
    forward_model = model
    if runtime.world > 1:
        forward_model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                                broadcast_buffers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    if previous:
        optimizer.load_state_dict(previous["optimizer_state"])
        if previous.get("precision") == precision:
            scaler.load_state_dict(previous["scaler_state"])
    train_batches = loader(training, args.batch_size, True, args.workers, args.seed, runtime)
    val_batches = loader(validation, args.batch_size, workers=args.workers, runtime=runtime) if runtime.primary else None
    run = Path(args.output)
    if runtime.primary:
        run.mkdir(parents=True, exist_ok=False)
        if previous:
            best_path = Path(args.resume).parent/"best.pt"
            if not best_path.is_file():
                raise ValueError("Resume needs the matching best.pt beside the resumed checkpoint")
            best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
            if best_checkpoint["options"] != opts or best_checkpoint["model_config"] != model.configuration():
                raise ValueError("Resume sibling best.pt does not match this model/input")
            torch.save(best_checkpoint, run/"best.pt")
        manifest_copy = (Path(args.manifest).resolve()).read_text(encoding="utf-8-sig")
        (run/"manifest_used.csv").write_text(manifest_copy, encoding="utf-8")
        write_json(run/"configuration.json", {"arguments": vars(args), "data": [training.summary, validation.summary],
                   "normalizer": norm.state(), "parameters": sum(p.numel() for p in model.parameters()),
                   "torch": str(torch.__version__), "device": str(device), "world_size": runtime.world,
                   "precision": precision, "batch_per_rank": args.batch_size,
                   "global_batch": args.batch_size*runtime.world,
                   "train_padding_repeats_per_epoch": getattr(train_batches.batch_sampler, "repeated_examples", 0),
                   "validation_execution": "rank 0, all windows, no padding (except explicit debug truncation)",
                   "interface_status": "proposal_not_final_generator_format"})
    runtime.broadcast(True)
    best = previous["best_merit"] if previous else -float("inf")
    stale = previous["stale"] if previous else 0
    history = previous["history"] if previous else []
    start_epoch = previous["epoch"]+1 if previous else 1
    if start_epoch > args.epochs:
        raise ValueError("--epochs is the total target epoch; it must exceed the resumed epoch")
    train_batches.batch_sampler.epoch = start_epoch-1
    thresholds = np.linspace(.1, .9, 17).tolist()
    for epoch in range(start_epoch, args.epochs+1):
        forward_model.train()
        total, count = 0., 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_start = time.perf_counter()
        for step, batch in enumerate(train_batches):
            if args.max_batches and step >= args.max_batches:
                break
            symbols = int(batch["symbols"][0])
            x = batch["x"].to(device, non_blocking=True)
            target = {k: batch[k].to(device, non_blocking=True) for k in ("re", "prb")}
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, precision):
                out = forward_model(x, symbols)
                loss = loss_fn(out, target, args.auxiliary_weight)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())*len(x)
            count += len(x)
        if not count:
            raise ValueError("No training batches")
        total, count = runtime.sum([total, count])
        elapsed = time.perf_counter()-train_start
        stop = False
        if runtime.primary:
            # Use the bare module: other ranks wait, so DDP collectives cannot occur here.
            results = score(model, val_batches, device, thresholds, args.auxiliary_weight, args.max_batches, precision)
            selected = max(results, key=lambda m: (m["prb"]["f1"] if m["prb"]["f1"] is not None else -1., -m["prb"]["false_positive_rate"] if m["prb"]["false_positive_rate"] is not None else 0., m["threshold"]))
            merit = selected["prb"]["f1"]
            merit = float(merit) if merit is not None else -selected["loss"]
            improved = merit > best
            best, stale = (merit, 0) if improved else (best, stale+1)
            row = {"epoch": epoch, "train_loss": total/count, "train_windows_per_second_including_padding": count/elapsed,
                   "world_size": runtime.world, "precision": precision, "validation": selected}
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            checkpoint = {"contract": SCHEMA, "model_config": model.configuration(), "model_state": model.state_dict(),
                          "normalizer": norm.state(), "options": opts, "epoch": epoch, "seed": args.seed,
                          "threshold": selected["threshold"], "auxiliary_weight": args.auxiliary_weight,
                          "limited_run": bool(args.max_batches) or bool(previous and previous["limited_run"]),
                          "validation": selected, "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
                          "precision": precision, "best_merit": best, "stale": stale, "history": history,
                          "interface_status": "proposal_not_final_generator_format"}
            torch.save(checkpoint, run/"last.pt")
            if improved:
                torch.save(checkpoint, run/"best.pt")
                write_json(run/"validation_thresholds.json", results)
            write_json(run/"history.json", history)
            stop = stale >= args.patience
        if runtime.broadcast(stop):
            break
    if runtime.primary:
        return {"run": str(run.resolve()), "epoch": history[-1]["epoch"], "parameters": sum(p.numel() for p in model.parameters()),
                "limited_run": bool(args.max_batches), "world_size": runtime.world, "precision": precision}
    return None


def evaluate(args):
    device = device_for(args.device)
    model, ckpt, norm = load_checkpoint(args.checkpoint, device)
    dataset = GridDataset(discover(args.manifest), args.split, normalizer=norm, **data_options(ckpt["options"]))
    precision = precision_for(getattr(args, "amp", "auto"), device)
    results = score(model, loader(dataset, args.batch_size, workers=args.workers), device,
                    [ckpt["threshold"]], ckpt["auxiliary_weight"], precision=precision)[0]
    result = {"split": args.split, "threshold_source": "checkpoint_validation_only", "limited_training": ckpt["limited_run"],
              "data": dataset.summary, "metrics": results}
    write_json(args.output, result)
    return result
