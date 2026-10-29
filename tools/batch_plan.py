"""冻结批次与任务身份。旧批次只读；改变生成条件时必须使用新的批次名。"""

import ast
import hashlib
import json
import re
from pathlib import Path

DEFAULT_CAPTURES = 2000
DEFAULT_BATCH = "first_train_2000"
SEEDS = {"acceptance": 2026102900, "pilot": 2027102900, "production": 2028102900}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class WithoutDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        node = super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
        return node


def source_identity(root):
    """Python 按 AST；MATLAB 忽略整行注释，其他差异保守处理。

    不把绘图、说明、验证器或网络变更算作信号生成变更。MATLAB 不进行
    危险的正则去空白/引号改写，行内注释和重排仍可能要求新批次。
    """
    root = Path(root)
    files = [
        root / "tools/make_assets.py",
        root / "tools/batch_plan.py",
        root / "matlab/lte_dataset_generate.m",
    ]
    files += sorted((root / "reference").glob("synthetic*.mat"))
    if len(files) < 4:
        raise RuntimeError("Synthetic SIB assets are missing")
    raw, executable = {}, {}
    for path in files:
        name = path.relative_to(root).as_posix()
        data = path.read_bytes()
        raw[name] = hashlib.sha256(data).hexdigest()
        if path.suffix == ".py":
            tree = WithoutDocstrings().visit(ast.parse(data.decode("utf-8-sig")))
            data = ast.dump(tree, include_attributes=False).encode()
        elif path.suffix == ".m":
            kept, depth = [], 0
            for line in data.decode("utf-8-sig").splitlines():
                stripped = line.strip()
                if stripped == "%{":
                    depth += 1
                    continue
                if stripped == "%}" and depth:
                    depth -= 1
                    continue
                if depth or not stripped or stripped.startswith("%"):
                    continue
                kept.append(line.rstrip())
            if depth:
                raise ValueError("Unterminated MATLAB block comment")
            data = "\n".join(kept).encode()
        executable[name] = hashlib.sha256(data).hexdigest()
    return {"revision": digest(executable), "raw_files": raw, "generation_files": executable}


def freeze_batch(data_root, batch, captures, source, runtime):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", batch):
        raise ValueError("Batch name must be a simple directory name")
    root = Path(data_root) / "batches" / batch
    path = root / "batch_identity.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    count = (
        captures
        if captures is not None
        else old["production_captures"] if old else DEFAULT_CAPTURES
    )
    if not 100 <= count <= 100000:
        raise ValueError("Production captures must be between 100 and 100000")
    identity = {
        "management_version": 2,
        "batch": batch,
        "production_captures": count,
        "generator_revision": source["revision"],
        "runtime": runtime,
    }
    if old and any(old.get(k) != v for k, v in identity.items()):
        raise RuntimeError(
            "Batch conditions changed; choose a NEW --batch. Existing data was not modified."
        )
    if not old:
        if root.exists() and any(root.iterdir()):
            raise RuntimeError("Refusing to adopt an existing nonempty unversioned batch directory")
        write_json(path, dict(identity, initial_source_files=source["raw_files"]))
    return root, old or identity


def freeze_plan(batch_root, stage, identity):
    from make_assets import build_plan

    folder = Path(batch_root) / stage
    path = folder / "generation_plan.json"
    if path.exists():
        plan = json.loads(path.read_text(encoding="utf-8"))
        checksum = plan.pop("plan_sha256", None)
        if digest(plan) != checksum or plan["generator_revision"] != identity["generator_revision"]:
            raise RuntimeError("Frozen generation plan was modified; refusing to resume")
        plan["plan_sha256"] = checksum
        return path, plan
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError("Refusing to adopt existing stage outputs without a frozen plan")
    count = {"acceptance": 22, "pilot": 500, "production": identity["production_captures"]}[stage]
    plan = build_plan(stage, count, 30 if stage == "acceptance" else 100, folder, SEEDS[stage])
    plan.update(
        management_version=2,
        batch=identity["batch"],
        generator_revision=identity["generator_revision"],
    )
    for job in plan["jobs"]:
        job["task_sha256"] = digest(job)
    plan["plan_sha256"] = digest(plan)
    write_json(path, plan)
    return path, plan


def matching_record(record, plan):
    expected = next((j for j in plan["jobs"] if j["id"] == record.get("id")), None)
    return bool(
        expected
        and record.get("generator_revision") == plan["generator_revision"]
        and record.get("task_sha256") == expected["task_sha256"]
    )


def require_gate(report, stage):
    """执行失败必须阻断；同步拒绝按预先保存的比例门槛计数。"""
    if not report.get("passed") or report.get("missing_jobs") or report.get("execution_failures"):
        raise RuntimeError(f"{stage} acceptance gate has not passed")
    if report.get("current_reader_validated_shards", 0) != report.get("completed", 0):
        raise RuntimeError(f"{stage} training-reader verification is incomplete")
    if stage == "acceptance" and not report.get("current_model_forward_backward"):
        raise RuntimeError("Model forward/backward acceptance is missing")
