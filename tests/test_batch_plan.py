"""验证批次隔离、计划不可变、任务身份、验收门槛和同源子集选择。"""

import copy
import contextlib
import csv
import importlib.util
import json
import io
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from batch_plan import (
    digest,
    freeze_batch,
    freeze_plan,
    matching_record,
    require_gate,
    source_identity,
    write_json,
)
from make_assets import build_plan
from run_first_training import select_rows
from accept_dataset import finalize
import deploy_first_training as installer


class BatchPlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.source = {"revision": "generation-v2", "raw_files": {"source": "hash"}}
        self.runtime = {"matlab": "2025a", "lte": "25.1"}

    def batch(self, count=None):
        return freeze_batch(self.directory, "experiment", count, self.source, self.runtime)

    def test_default_2000_and_immutable_resume(self):
        root, identity = self.batch()
        self.assertEqual(identity["production_captures"], 2000)
        path, plan = freeze_plan(root, "production", identity)
        self.assertEqual(len(plan["jobs"]), 2000)
        self.assertEqual(plan["requested_windows"], 20000)
        before = path.read_bytes()
        self.assertEqual(freeze_plan(root, "production", identity)[1], plan)
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(RuntimeError):
            self.batch(5000)
        self.assertEqual(path.read_bytes(), before)

    def test_python_parameters_and_environment_cannot_mix(self):
        root, identity = self.batch()
        path, plan = freeze_plan(root, "production", identity)
        altered = copy.deepcopy(plan)
        altered["jobs"][0]["snr_db"] = -10
        write_json(path, altered)
        with self.assertRaises(RuntimeError):
            freeze_plan(root, "production", identity)
        with self.assertRaises(RuntimeError):
            freeze_batch(
                self.directory,
                "experiment",
                None,
                dict(self.source, revision="new-code"),
                self.runtime,
            )
        with self.assertRaises(RuntimeError):
            freeze_batch(
                self.directory, "experiment", None, self.source, dict(self.runtime, matlab="2026b")
            )

    def test_grouping_stage_seeds_and_compact_profile(self):
        root, identity = self.batch()
        plans = [freeze_plan(root, s, identity)[1] for s in ("acceptance", "pilot", "production")]
        seeds = [{j["seed"] for j in p["jobs"]} for p in plans]
        self.assertFalse(seeds[0] & seeds[1] or seeds[1] & seeds[2] or seeds[0] & seeds[2])
        for plan in plans:
            for i, job in enumerate(plan["jobs"]):
                if job["scenario"] == "true_gap":
                    self.assertEqual(job["group_id"], plan["jobs"][i - 1]["group_id"])
                    self.assertEqual(job["split"], plan["jobs"][i - 1]["split"])
        self.assertTrue(all(j["save_crs"] for j in plans[0]["jobs"]))
        self.assertEqual(sum(j["save_crs"] for j in plans[2]["jobs"]), 40)
        record = dict(
            id=plans[2]["jobs"][0]["id"],
            generator_revision=identity["generator_revision"],
            task_sha256=plans[2]["jobs"][0]["task_sha256"],
        )
        self.assertTrue(matching_record(record, plans[2]))
        record["task_sha256"] = "other-parameters"
        self.assertFalse(matching_record(record, plans[2]))

    def test_source_scope_and_non_executable_changes(self):
        root = self.directory / "code"
        for folder in ("tools", "matlab", "reference"):
            (root / folder).mkdir(parents=True)
        py = root / "tools/make_assets.py"
        matlab = root / "matlab/lte_dataset_generate.m"
        py.write_text('"""docs"""\nx = 1\n')
        (root / "tools/batch_plan.py").write_text("seed = 1\n")
        matlab.write_text("function x = f\nx = 1;\nend\n")
        (root / "reference/synthetic_sib1.mat").write_bytes(b"asset")
        first = source_identity(root)
        py.write_text('# comment\n"""new docs"""\nx = 1\n')
        matlab.write_text("% comment\nfunction x = f\nx = 1;\nend\n")
        (root / "matlab/render_dataset_report.m").write_text("changed plot")
        second = source_identity(root)
        self.assertEqual(first["revision"], second["revision"])
        self.assertNotEqual(first["raw_files"], second["raw_files"])
        py.write_text("x = 5\n")
        self.assertNotEqual(first["revision"], source_identity(root)["revision"])

    def test_source_folder_and_path_cannot_be_adopted(self):
        with self.assertRaises(ValueError):
            freeze_batch(self.directory, "../escape", None, self.source, self.runtime)
        path = self.directory / "batches/experiment"
        path.mkdir(parents=True)
        (path / "old_data.mat").write_bytes(b"keep")
        with self.assertRaises(RuntimeError):
            self.batch()
        self.assertEqual((path / "old_data.mat").read_bytes(), b"keep")

    def test_stratified_subset_preserves_entire_groups_and_splits(self):
        jobs = build_plan("production", 5000, 100)["jobs"]
        rows = [
            dict(
                schema="pdsch-occupancy-proposal-v1",
                alignment_valid="1",
                path=j["id"] + ".mat",
                split=j["split"],
                group_id=j["group_id"],
                scenario=j["scenario"],
                nrb=str(j["nrb"]),
            )
            for j in jobs
        ]
        selected = select_rows(rows, 2000)
        self.assertGreaterEqual(len(selected), 1994)
        self.assertLessEqual(len(selected), 2000)
        self.assertEqual(selected, select_rows(rows, 2000))
        selected_groups = {r["group_id"] for r in selected}
        self.assertEqual(len(selected), sum(r["group_id"] in selected_groups for r in rows))
        self.assertEqual({r["scenario"] for r in selected}, {j["scenario"] for j in jobs})
        broken = copy.deepcopy(rows)
        paired = next(i for i, r in enumerate(broken) if r["scenario"] == "true_gap")
        broken[paired]["split"] = "test" if broken[paired]["split"] != "test" else "train"
        with self.assertRaises(ValueError):
            select_rows(broken, 2000)

    def test_acceptance_rejects_missing_jobs_and_bad_parameters(self):
        root, identity = self.batch()
        path, plan = freeze_plan(root, "acceptance", identity)
        folder = path.parent
        (folder / "status").mkdir()
        (folder / "shards").mkdir()
        job = plan["jobs"][0]
        k, s = 12, 10
        re = np.ones((k, 14, s), np.uint8)
        savemat(
            folder / "shards/one.mat",
            {
                "rx_real": np.ones((k, 14, 1, s), np.float32),
                "rx_imag": np.zeros((k, 14, 1, s), np.float32),
                "label_re": re,
                "label_prb": np.ones((1, s), np.uint8),
                "label_prb_slot": np.ones((1, 2, s), np.uint8),
                "public_excluded": np.zeros_like(re),
                "public_valid": np.zeros_like(re),
            },
        )
        record = dict(
            id=job["id"],
            state="complete",
            split=job["split"],
            group_id=job["group_id"],
            scenario=job["scenario"],
            generator_revision=identity["generator_revision"],
            task_sha256=job["task_sha256"],
            files=["shards/one.mat"],
        )
        marker = folder / "status/one.json"
        write_json(marker, record)
        with contextlib.redirect_stdout(io.StringIO()):
            report = finalize(folder)
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["missing_jobs"]), 21)
        with self.assertRaises(RuntimeError):
            require_gate(report, "acceptance")
        record["task_sha256"] = "wrong"
        write_json(marker, record)
        with self.assertRaises(ValueError):
            finalize(folder)


class DeploymentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "code"
        (self.root / "tools").mkdir(parents=True)
        self.old = self.root / "tools/one.py"
        self.old.write_bytes(b"x = 1\n")
        self.archive = Path(self.tmp.name) / "update.zip"
        files = {"tools/one.py": b"x = 2\n", "tools/new.py": b"y = 3\n"}
        manifest = {
            "files": {
                p: {
                    "sha256": installer.sha(raw),
                    "base_sha256": (
                        installer.sha(self.old.read_bytes()) if p == "tools/one.py" else None
                    ),
                }
                for p, raw in files.items()
            }
        }
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("update_manifest.json", json.dumps(manifest))
            for name, raw in files.items():
                archive.writestr(name, raw)
        self.args = ["installer", str(self.archive), installer.sha(self.archive.read_bytes())]

    def run_installer(self, *, failure=False, busy=False):
        def git(*args):
            if args == ("rev-parse", "--show-toplevel"):
                return str(self.root)
            return "" if args[0] == "status" else "new-commit"

        process = Mock(pid=1234)
        process.poll.return_value = None
        error = subprocess.CalledProcessError(1, "unit tests") if failure else None
        with (
            patch.object(installer, "ROOT", self.root),
            patch.object(installer, "DATA", self.root.parent / "data"),
            patch.object(installer, "git", side_effect=git),
            patch.object(installer.os, "getuid", return_value=1000, create=True),
            patch.object(
                installer.subprocess,
                "check_output",
                return_value=(f"123 {self.root}/tools/run_server.py" if busy else ""),
            ),
            patch.object(installer.subprocess, "run", side_effect=error) as command,
            patch.object(installer.subprocess, "Popen", return_value=process) as start,
            patch.object(installer.time, "sleep"),
            patch.object(sys, "argv", self.args),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            installer.main()
            return command, start

    def test_success_updates_only_listed_sources_and_starts_background(self):
        command, start = self.run_installer()
        self.assertEqual(self.old.read_bytes(), b"x = 2\n")
        self.assertEqual((self.root / "tools/new.py").read_bytes(), b"y = 3\n")
        self.assertTrue(start.call_args.kwargs["start_new_session"])
        self.assertEqual(command.call_count, 3)  # 测试、精确暂存、提交。

    def test_failed_server_tests_roll_back_sources(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_installer(failure=True)
        self.assertEqual(self.old.read_bytes(), b"x = 1\n")
        self.assertFalse((self.root / "tools/new.py").exists())

    def test_busy_or_changed_server_is_not_overwritten(self):
        with self.assertRaises(RuntimeError):
            self.run_installer(busy=True)
        self.old.write_bytes(b"user edit\n")
        with self.assertRaises(RuntimeError):
            self.run_installer()
        self.assertEqual(self.old.read_bytes(), b"user edit\n")
        self.assertFalse((self.root / "tools/new.py").exists())


if __name__ == "__main__":
    unittest.main()
