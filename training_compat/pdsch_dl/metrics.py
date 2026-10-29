"""Occupancy metrics and observable contiguous frequency runs (not UE IDs)."""
import numpy as np


def runs(bits):
    bits = np.asarray(bits, dtype=bool).reshape(-1)
    edges = np.diff(np.r_[False, bits, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), (np.flatnonzero(edges == -1)-1).tolist()))


def binary_counts(probability, target, threshold):
    pred, true = np.asarray(probability) >= threshold, np.asarray(target, dtype=bool)
    return np.array([np.sum(pred & true), np.sum(pred & ~true), np.sum(~pred & true), np.sum(~pred & ~true)], np.int64)


def binary_scores(counts):
    tp, fp, fn, tn = map(int, counts)
    def ratio(a, b):
        return a / b if b else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": ratio(tp, tp+fp), "recall": ratio(tp, tp+fn),
            "f1": ratio(2*tp, 2*tp+fp+fn), "iou": ratio(tp, tp+fp+fn),
            "false_positive_rate": ratio(fp, fp+tn)}


def segment_counts(prob, target, threshold):
    """Counts [true runs, predicted runs, over-split true runs, over-merged predicted runs]."""
    p, t = np.asarray(prob) >= threshold, np.asarray(target, dtype=bool)
    if p.ndim == 2:
        p, t = p[None], t[None]
    total = np.zeros(4, np.int64)
    for b in range(p.shape[0]):
        for sf in range(p.shape[2]):
            pr, tr = runs(p[b, :, sf]), runs(t[b, :, sf])
            overlap = np.array([[max(a, c) <= min(z, d) for c, d in pr] for a, z in tr], dtype=bool).reshape(len(tr), len(pr))
            total += [len(tr), len(pr), int((overlap.sum(axis=1) > 1).sum()), int((overlap.sum(axis=0) > 1).sum())]
    return total


def segment_scores(counts):
    nt, npred, split, merge = map(int, counts)
    return {"true_segments": nt, "predicted_segments": npred,
            "split_true_segments": split, "merged_predicted_segments": merge,
            "split_rate": split/nt if nt else None,
            "merge_rate": merge/npred if npred else None}


def region_records(prb_probability, re_probability, subframes, symbols, threshold, re_threshold=0.5):
    rows = []
    for s, number in enumerate(subframes):
        for lo, hi in runs(prb_probability[:, s] >= threshold):
            fine = re_probability[lo*12:(hi+1)*12, s*symbols:(s+1)*symbols]
            occupied_symbols = np.flatnonzero((fine >= re_threshold).any(axis=0))
            rows.append({"subframe_index": int(number), "frame_ordinal": int(number)//10,
                         "subframe_in_frame": int(number)%10,
                         "prb_start": lo, "prb_end": hi,
                         "active_subcarrier_start": lo*12, "active_subcarrier_end": (hi+1)*12-1,
                         "symbol_start_estimate": int(occupied_symbols[0]) if occupied_symbols.size else None,
                         "symbol_end_estimate": int(occupied_symbols[-1]) if occupied_symbols.size else None,
                         "mean_prb_probability": float(prb_probability[lo:hi+1, s].mean())})
    return rows
