"""Software checks only. Fixtures are not standards-compliant LTE simulations."""
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import savemat
import torch

from pdsch_dl.data import SCHEMA, GridDataset, Normalizer, ShapeBatches, discover, features, read_arrays
from pdsch_dl.engine import evaluate, train
from pdsch_dl.cli import predict
from pdsch_dl.metrics import binary_scores, region_records, runs, segment_counts
from pdsch_dl.model import PDSCHUNet


def fixture(k=72, l=14, r=1, s=4, seed=1):
    rng = np.random.default_rng(seed)
    mask = np.zeros((k, l, s), np.float32)
    mask[12:48, 3:, ::2] = 1
    mask[12:24, 3:, 1::2] = 1
    mask[36:60, 3:, 1::2] = 1
    real = rng.normal(size=(k, l, r, s)).astype(np.float32)
    imag = rng.normal(size=real.shape).astype(np.float32)
    return {"rx_real": real, "rx_imag": imag, "label_re": mask,
            "label_prb": mask.reshape(k//12, 12, l, s).any(axis=(1, 2)).astype(np.float32),
            "subframe_index": np.arange(s, dtype=np.int64)}


def write_fixture_set(root):
    manifest = root/"manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["schema", "path", "split", "group_id", "alignment_valid"])
        writer.writeheader()
        for split in ("train", "validation", "test"):
            for j, (k, l, r) in enumerate(((72, 14, 1), (180, 12, 2))):
                name = f"{split}_{j}.npz"
                np.savez(root/name, **fixture(k, l, r, seed=100*j+len(split)))
                writer.writerow({"schema": SCHEMA, "path": name, "split": split, "group_id": f"{split}-{j}", "alignment_valid": 1})
    return manifest


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_model_bandwidth_cp_and_gradients(self):
        model = PDSCHUNet(3)
        self.assertLess(sum(p.numel() for p in model.parameters()), 500000)
        for k, l in ((72, 14), (300, 12), (600, 14), (900, 12), (1200, 14)):
            with torch.no_grad():
                out = model(torch.randn(1, 3, k, l*2), l)
            self.assertEqual(tuple(out["re"].shape), (1, 1, k, l*2))
            self.assertEqual(tuple(out["prb"].shape), (1, 1, k//12, 2))
        out = model(torch.randn(1, 3, 72, 28))
        (out["prb"].square().mean()+out["re"].square().mean()).backward()
        self.assertGreater(model.prb_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.re_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.e0[0][0].weight.grad.abs().sum().item(), 0)

    def test_axis_order_matlab_and_target_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = fixture(r=2)
            a["rx_real"][7, 8, 1, 2] = 123
            np.savez(root/"a.npz", **a, true_rnti=np.array([42]))
            savemat(root/"a.mat", a)
            for name in ("a.npz", "a.mat"):
                loaded = read_arrays(root/name)
                x = features(loaded, 1, 2, rx="all")
                self.assertEqual(x[1, 7, 14+8], 123)
                self.assertEqual(x.shape, (6, 72, 28))
                np.testing.assert_array_equal(loaded["label_re"], a["label_re"])
            only = {k: v for k, v in a.items() if not k.startswith("label")}
            np.savez(root/"receiver_only.npz", **only)
            loaded = read_arrays(root/"receiver_only.npz", labels=False)
            self.assertNotIn("label_re", loaded)
            np.testing.assert_array_equal(features(loaded, 0, 2), features(a, 0, 2))
            with self.assertRaisesRegex(ValueError, "supervised"):
                read_arrays(root/"receiver_only.npz")

    def test_contract_rejects_bad_labels_and_unknown_layout(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = fixture()
            a["label_prb"][:] = 0
            np.savez(root/"bad.npz", **a)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                read_arrays(root/"bad.npz")
            np.savez(root/"unknown.npz", features=np.zeros((72, 14, 3), np.float32))
            with self.assertRaisesRegex(ValueError, "missing rx_real"):
                read_arrays(root/"unknown.npz")

    def test_split_leakage_gaps_and_train_only_normalizer(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = write_fixture_set(root)
            rows = discover(manifest)
            trainset = GridDataset(rows, "train", window=2)
            norm = Normalizer.fit(trainset)
            val = GridDataset(rows, "validation", window=2)
            with self.assertRaisesRegex(ValueError, "raw train"):
                Normalizer.fit(val)
            self.assertEqual(norm.mean.shape, (3,))
            self.assertEqual(norm.scale[0], norm.scale[1])
            for batch in ShapeBatches(trainset, 3):
                self.assertEqual(len({trainset.keys[i] for i in batch}), 1)
            text = manifest.read_text(encoding="utf-8")
            manifest.write_text(text.replace("validation-0", "train-0"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "across splits"):
                discover(manifest)

    def test_gap_windows_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = fixture()
            a["subframe_index"] = np.array([0, 1, 8, 9])
            np.savez(root/"gap.npz", **a)
            rows = [{"path": str(root/"gap.npz"), "split": "train", "group_id": "g", "alignment_valid": "1"}]
            with self.assertRaisesRegex(ValueError, "No contiguous"):
                GridDataset(rows, "train", window=3)
            self.assertEqual(len(GridDataset(rows, "train", window=2)), 2)

    def test_segments_do_not_bridge_empty_gaps(self):
        self.assertEqual(runs([0, 1, 1, 0, 1]), [(1, 2), (4, 4)])
        true = np.array([0, 1, 1, 1, 1, 0])[:, None]
        pred = np.array([0, 1, 0, 0, 1, 0])[:, None]
        np.testing.assert_array_equal(segment_counts(pred, true, .5), [1, 2, 1, 0])
        np.testing.assert_array_equal(segment_counts(true, pred, .5), [2, 1, 0, 1])
        empty = np.zeros((6, 1))
        np.testing.assert_array_equal(segment_counts(empty, empty, .5), [0, 0, 0, 0])
        rows = region_records(pred, np.ones((72, 14)), [7], 14, .5)
        self.assertEqual(rows[0]["active_subcarrier_start"], 12)
        self.assertEqual(rows[1]["active_subcarrier_end"], 59)
        self.assertIsNone(binary_scores([0, 0, 0, 10])["f1"])

    def test_train_checkpoint_evaluate_and_receiver_only_predict(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = write_fixture_set(root)
            args = SimpleNamespace(manifest=str(manifest), output=str(root/"run"), window=2,
                mode="iq_power", rx="first", public=False, base=8, epochs=2, patience=2,
                learning_rate=1e-3, auxiliary_weight=.3, normalization_windows=8, seed=7,
                max_batches=2, batch_size=2, workers=0, device="cpu")
            result = train(args)
            self.assertTrue(result["limited_run"])
            checkpoint = root/"run"/"best.pt"
            result = evaluate(SimpleNamespace(manifest=str(manifest), checkpoint=str(checkpoint),
                split="test", output=str(root/"evaluation.json"), batch_size=2, workers=0, device="cpu"))
            self.assertEqual(result["threshold_source"], "checkpoint_validation_only")
            a = fixture(s=5)
            del a["label_prb"], a["label_re"]
            np.savez(root/"receiver.npz", **a)
            result = predict(SimpleNamespace(input=str(root/"receiver.npz"), checkpoint=str(checkpoint),
                output=str(root/"prediction"), device="cpu"))
            self.assertEqual(result["valid_subframes"], 5)


if __name__ == "__main__":
    unittest.main()
