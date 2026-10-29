"""使用临时仓库核验 Git 归档和运行追溯；不读取实际数据集。"""

import contextlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(shutil.which("git"), "Git is required")
class GitToolsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        self.data = self.base / "data"
        self.data.mkdir()
        for name in (".gitignore", ".gitattributes"):
            (self.repo / name).write_bytes((PROJECT / name).read_bytes())
        self.matlab = b"function x = example\r\nx = 1;\r\nend\r\n"
        (self.repo / "example.m").write_bytes(self.matlab)
        (self.repo / "reference").mkdir()
        (self.repo / "reference/synthetic_example.mat").write_bytes(bytes(range(256)))
        for name in (".deps", "logs", "configs"):
            (self.repo / name).mkdir()
            (self.repo / name / "ignored.json").write_text("{}")
        self.git("init", "-q")
        self.git("config", "core.autocrlf", "false")
        self.git("add", ".")
        self.git("-c", "user.name=Git tools test", "-c", "user.email=test@example.invalid",
                 "commit", "-qm", "Test fixture")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.repo, text=True).strip()

    def test_archive_excludes_generated_files_and_preserves_matlab_bytes(self):
        module = load_tool("package_project")
        module.ROOT = self.repo
        destination = self.data / "code.zip"
        module.export_project(destination)
        with zipfile.ZipFile(destination) as archive:
            self.assertEqual(archive.read("example.m"), self.matlab)
            self.assertIn("reference/synthetic_example.mat", archive.namelist())
            self.assertFalse(any(n.startswith((".git/", ".deps/", "logs/", "configs/"))
                                 for n in archive.namelist()))
        with self.assertRaises(FileExistsError):
            module.export_project(destination)
        (self.repo / "example.m").write_text("changed")
        with self.assertRaises(RuntimeError):
            module.export_project(self.data / "dirty.zip")

    def test_run_record_requires_clean_state_and_does_not_overwrite(self):
        module = load_tool("record_run")
        module.ROOT = self.repo
        manifest = self.data / "manifest.csv"
        manifest.write_text("schema,path,split\n")
        destination = self.data / "run_info.json"
        args = ["record_run", "--manifest", str(manifest), "--output", str(destination)]
        with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
            module.main()
        result = json.loads(destination.read_text())
        self.assertFalse(result["git_dirty"])
        self.assertEqual(result["git_commit"], self.git("rev-parse", "HEAD"))
        self.assertEqual(result["manifest"]["bytes"], manifest.stat().st_size)
        with patch.object(sys, "argv", args), self.assertRaises(FileExistsError):
            module.main()
        (self.repo / "example.m").write_text("changed")
        with patch.object(sys, "argv", args), self.assertRaises(RuntimeError):
            module.main()


if __name__ == "__main__":
    unittest.main()
