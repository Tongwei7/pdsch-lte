"""Proposed explicit contract, independent of all deleted data generators.

The running dataset-analysis chat has NOT finalized its export schema. Accept
only the documented proposal; never guess a historical MATLAB layout.
"""
import csv
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, Sampler


SCHEMA = "pdsch-occupancy-proposal-v1"
INPUT_FIELDS = ["rx_real", "rx_imag", "subframe_index", "public_excluded", "public_valid"]
LABEL_FIELDS = ["label_re", "label_prb"]


def read_arrays(path, labels=True):
    path = Path(path)
    fields = INPUT_FIELDS + (LABEL_FIELDS if labels else [])
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key] for key in fields if key in source}
    elif path.suffix.lower() == ".mat":
        try:
            arrays = loadmat(path, variable_names=fields, squeeze_me=False)
        except NotImplementedError as exc:
            raise ValueError("MATLAB v7.3 needs an agreed HDF5 adapter; use -v7 for this proposal") from exc
    else:
        raise ValueError("Expected .npz or explicitly contracted MATLAB -v7 .mat")
    for key in ("rx_real", "rx_imag", "subframe_index"):
        if key not in arrays:
            raise ValueError(f"{path}: missing {key}; no legacy adaptation")
    re, im = np.asarray(arrays["rx_real"]), np.asarray(arrays["rx_imag"])
    if re.shape != im.shape or re.ndim not in (2, 3, 4):
        raise ValueError("rx_real/rx_imag must be [K,L,R,S]; trailing singleton axes may be omitted")
    re = re.reshape(re.shape + (1,) * (4 - re.ndim))
    im = im.reshape(re.shape)
    k, l, r, s = re.shape
    if k % 12 or l not in (12, 14) or r < 1 or s < 1:
        raise ValueError("Invalid LTE dimensions")
    if re.dtype != np.float32 or im.dtype != np.float32 or not np.isfinite(re).all() or not np.isfinite(im).all():
        raise ValueError("Receiver components must be finite float32")
    arrays["rx_real"], arrays["rx_imag"] = re, im
    raw = np.asarray(arrays["subframe_index"]).reshape(-1)
    if raw.size != s or not np.isfinite(raw).all() or (raw < 0).any() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError("Invalid subframe_index")
    ix = raw.astype(np.int64)
    if (np.diff(ix) <= 0).any():
        raise ValueError("subframe_index must be strictly increasing")
    arrays["subframe_index"] = ix
    for key in ("public_excluded", "public_valid") + (tuple(LABEL_FIELDS) if labels else ()):
        if key not in arrays:
            if key in LABEL_FIELDS:
                raise ValueError(f"Missing supervised target {key}")
            continue
        shape = (k // 12, s) if key == "label_prb" else (k, l, s)
        a = np.asarray(arrays[key])
        if tuple(v for v in a.shape if v != 1) != tuple(v for v in shape if v != 1) or not np.isin(a, [0, 1]).all():
            raise ValueError(f"Invalid axes or nonbinary values in {key}")
        arrays[key] = a.reshape(shape).astype(np.float32)
    if ("public_excluded" in arrays) != ("public_valid" in arrays):
        raise ValueError("Public prior and its validity must be supplied together")
    if "public_excluded" in arrays and (arrays["public_excluded"] > arrays["public_valid"]).any():
        raise ValueError("Unknown prior locations must not be marked excluded")
    if labels:
        derived = arrays["label_re"].reshape(k // 12, 12, l, s).any(axis=(1, 2))
        if not np.array_equal(derived, arrays["label_prb"].astype(bool)):
            raise ValueError("PRB occupancy disagrees with union of PDSCH data REs")
    return arrays


def features(a, start, count, mode="iq_power", rx="first", public=False):
    if mode not in ("iq_power", "power") or rx not in ("first", "all"):
        raise ValueError("Unknown feature selection")
    real, imag = a["rx_real"], a["rx_imag"]
    k, l, r, s = real.shape
    if not (0 <= start and 1 <= count <= s - start):
        raise ValueError("Window out of range")
    stop = start + count
    r = 1 if rx == "first" else r
    real, imag = real[:, :, :r, start:stop], imag[:, :, :r, start:stop]
    power = 10 * np.log10(real.astype(np.float64)**2 + imag.astype(np.float64)**2 + 1e-12)
    parts = [real, imag, power] if mode == "iq_power" else [power]
    # I(all RX), Q(all RX), power(all RX); S then L form the time axis.
    x = np.concatenate([p.transpose(2, 0, 3, 1).reshape(r, k, count*l) for p in parts], axis=0)
    if public:
        extra = []
        for key in ("public_excluded", "public_valid"):
            p = a.get(key, np.zeros((k, l, s), np.float32))
            extra.append(p[:, :, start:stop].transpose(0, 2, 1).reshape(1, k, count*l))
        x = np.concatenate((x, *extra), axis=0)
    return x.astype(np.float32)


def discover(manifest):
    manifest = Path(manifest).resolve()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    group_splits, paths = {}, set()
    for row in rows:
        if row.get("schema") != SCHEMA:
            raise ValueError(f"Explicit schema must be {SCHEMA}; final generator format is not agreed")
        if row.get("split") not in ("train", "validation", "test") or not row.get("group_id"):
            raise ValueError("Every shard needs split and group_id")
        if row["group_id"] in group_splits and group_splits[row["group_id"]] != row["split"]:
            raise ValueError("Capture/scenario group appears across splits")
        group_splits[row["group_id"]] = row["split"]
        path = (manifest.parent / row["path"]).resolve()
        if path in paths or not path.is_file():
            raise ValueError(f"Duplicate or missing shard: {path}")
        paths.add(path)
        row["path"] = str(path)
        if row.get("alignment_valid") not in ("0", "1"):
            raise ValueError("alignment_valid must explicitly be 0 or 1")
    return rows


class Normalizer:
    def __init__(self, mean, scale):
        self.mean, self.scale = np.asarray(mean, np.float32), np.asarray(scale, np.float32)
        if self.mean.ndim != 1 or self.mean.shape != self.scale.shape or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("Invalid normalization state")

    def __call__(self, x):
        if x.shape[0] != len(self.mean):
            raise ValueError("Channel mismatch with training")
        return (x - self.mean[:, None, None]) / self.scale[:, None, None]

    def state(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def fit(cls, dataset, max_windows=256):
        if dataset.split != "train" or dataset.normalizer is not None or max_windows < 1:
            raise ValueError("Fit normalization on raw train windows only")
        ids = np.unique(np.linspace(0, len(dataset)-1, min(max_windows, len(dataset))).astype(int))
        total = np.zeros(dataset.channels, np.float64)
        sq, pixels = total.copy(), 0
        for i in ids:
            x = dataset.raw(int(i))[0].astype(np.float64)
            total += x.sum(axis=(1, 2))
            sq += (x*x).sum(axis=(1, 2))
            pixels += x.shape[1] * x.shape[2]
        mean = total / pixels
        scale = np.sqrt(np.maximum(sq / pixels - mean**2, 1e-8))
        n = dataset.channels - 2*int(dataset.public)
        if dataset.mode == "iq_power":
            r = n // 3
            shared = np.sqrt(np.maximum((sq[:r] + sq[r:2*r]) / (2*pixels), 1e-8))
            mean[:2*r] = 0
            scale[:r], scale[r:2*r] = shared, shared
        if dataset.public:
            mean[-2:], scale[-2:] = 0, 1
        return cls(mean, scale)


class GridDataset(Dataset):
    def __init__(self, rows, split, window=10, stride=None, mode="iq_power", rx="first", public=False, normalizer=None, cache_size=2):
        if window < 1 or (stride is not None and stride < 1) or cache_size < 1:
            raise ValueError("window, stride, cache_size must be positive")
        self.split, self.window, self.stride = split, window, stride or window
        self.mode, self.rx, self.public = mode, rx, public
        self.normalizer, self.cache_size = normalizer, cache_size
        self.cache, self.records, self.windows, self.keys = OrderedDict(), [], [], []
        self.channels = None
        self.summary = {"split": split, "rejected_shards": 0, "valid_shards": 0, "subframes": 0, "used_subframes": 0}
        for row in rows:
            if row["split"] != split:
                continue
            if row["alignment_valid"] == "0":
                self.summary["rejected_shards"] += 1
                continue
            a = read_arrays(row["path"])
            k, l, r, s = a["rx_real"].shape
            c = features(a, 0, 1, mode, rx, public).shape[0]
            if self.channels is not None and c != self.channels:
                raise ValueError("Mixed RX counts: select rx=first or use separate models")
            self.channels = c
            ri = len(self.records)
            self.records.append(dict(row, shape=(k, l, r, s), indices=a["subframe_index"]))
            used = set()
            for start in range(0, s-window+1, self.stride):
                if not np.equal(np.diff(a["subframe_index"][start:start+window]), 1).all():
                    continue
                self.windows.append((ri, start))
                self.keys.append((k, l, c))
                used.update(range(start, start+window))
            self.summary["valid_shards"] += 1
            self.summary["subframes"] += s
            self.summary["used_subframes"] += len(used)
        self.summary["windows"] = len(self.windows)
        if not self.windows:
            raise ValueError(f"No contiguous {split} windows of length {window}")

    def __len__(self):
        return len(self.windows)

    def raw(self, index):
        ri, start = self.windows[index]
        if ri not in self.cache:
            self.cache[ri] = read_arrays(self.records[ri]["path"])
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(ri)
        a = self.cache[ri]
        x = features(a, start, self.window, self.mode, self.rx, self.public)
        k, l = a["rx_real"].shape[:2]
        stop = start+self.window
        re = a["label_re"][:, :, start:stop].transpose(0, 2, 1).reshape(k, self.window*l).copy()
        prb = a["label_prb"][:, start:stop].copy()
        return x, re, prb

    def __getitem__(self, index):
        x, re, prb = self.raw(index)
        if self.normalizer is not None:
            x = self.normalizer(x)
        return {"x": torch.from_numpy(x), "re": torch.from_numpy(re[None]), "prb": torch.from_numpy(prb[None]), "index": index, "symbols": self.keys[index][1]}

    def metadata(self, index):
        ri, start = self.windows[index]
        row = self.records[ri]
        return {"path": row["path"], "group_id": row["group_id"], "local_start": start, "subframe_index": row["indices"][start:start+self.window].tolist()}


class ShapeBatches(Sampler):
    def __init__(self, dataset, batch_size=4, shuffle=False, seed=42):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.groups = defaultdict(list)
        for i, key in enumerate(dataset.keys):
            self.groups[key].append(i)
        self.batch_size, self.shuffle, self.seed, self.epoch = batch_size, shuffle, seed, 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch)
        self.epoch += 1
        batches = []
        for ids in self.groups.values():
            order = np.array(ids)
            if self.shuffle:
                rng.shuffle(order)
            batches.extend(order[s:s+self.batch_size].tolist() for s in range(0, len(order), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self):
        return sum((len(v)+self.batch_size-1)//self.batch_size for v in self.groups.values())


class DistributedShapeBatches(ShapeBatches):
    """Deterministic same-shape global batches, equal local sizes and step counts.

Only training uses repetition padding. Validation is rank-0-only and never padded.
"""
    def __init__(self, dataset, batch_size, rank, world, seed=42):
        super().__init__(dataset, batch_size, True, seed)
        if world < 2 or not 0 <= rank < world:
            raise ValueError("Invalid distributed rank/world")
        self.rank, self.world = rank, world

    @property
    def repeated_examples(self):
        size = self.batch_size*self.world
        return sum(((len(v)+size-1)//size)*size-len(v) for v in self.groups.values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch)
        self.epoch += 1
        batches = []
        size = self.batch_size*self.world
        for ids in self.groups.values():
            order = np.array(ids)
            rng.shuffle(order)
            padded = np.resize(order, ((len(order)+size-1)//size)*size)
            batches.extend(padded[s:s+size] for s in range(0, len(padded), size))
        rng.shuffle(batches)
        for global_batch in batches:
            lo = self.rank*self.batch_size
            yield global_batch[lo:lo+self.batch_size].tolist()

    def __len__(self):
        size = self.batch_size*self.world
        return sum((len(v)+size-1)//size for v in self.groups.values())
