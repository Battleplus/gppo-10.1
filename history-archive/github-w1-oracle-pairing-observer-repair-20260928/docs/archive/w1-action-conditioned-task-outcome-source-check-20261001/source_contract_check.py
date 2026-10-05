"""Check the frozen W1 service-mode label path without importing the environment."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[2] / "runs" / "w1-light-repaired-fair-rerun-v2-nativefs-once" / "native" / "gppo_world"
WORKSPACE = Path(__file__).resolve().parents[2]
COLLECTOR = WORKSPACE / "research-plans" / "w1-eawm-jepa-world-model-production-repair-v4-e2e-integration-auth-repair-v1" / "production_data.py"
HOOKS = COLLECTOR.with_name("runtime_hooks.py")
SOURCE_RUN_CONFIG = WORKSPACE / "runs" / "w1-light-repaired-fair-rerun-v2-nativefs-once" / "environment.json"


def _method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def _attribute(node: ast.AST, *names: str) -> bool:
    for name in reversed(names):
        if not isinstance(node, ast.Attribute) or node.attr != name:
            return False
        node = node.value
    return isinstance(node, ast.Name) and node.id == "self"


def check_source(root: Path = SOURCE_ROOT) -> dict[str, object]:
    paths = {name: root / name for name in ("m10_environment.py", "task_lifecycle.py")}
    sources = {name: path.read_bytes() for name, path in paths.items()}
    env = ast.parse(sources["m10_environment.py"], filename=str(paths["m10_environment.py"]))
    lifecycle = ast.parse(sources["task_lifecycle.py"], filename=str(paths["task_lifecycle.py"]))
    config = next(node for node in env.body if isinstance(node, ast.ClassDef) and node.name == "M10Config")
    mode = next(node for node in config.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "task_completion_mode")
    assert isinstance(mode.value, ast.Constant) and mode.value.value == "continuous_service_until_deadline", "default_mode_changed"

    emit = _method(env, "M10Environment", "_emit_completion_notices")
    statements = emit.body[1:] if isinstance(emit.body[0], ast.Expr) and isinstance(emit.body[0].value, ast.Constant) else emit.body
    guard = statements[0]
    assert isinstance(guard, ast.If) and isinstance(guard.test, ast.Compare), "completion_guard_missing"
    assert _attribute(guard.test.left, "config", "task_completion_mode"), "completion_guard_mode_changed"
    assert len(guard.test.ops) == 1 and isinstance(guard.test.ops[0], ast.NotEq), "completion_guard_operator_changed"
    assert len(guard.test.comparators) == 1 and isinstance(guard.test.comparators[0], ast.Constant) and guard.test.comparators[0].value == "arrival_to_region", "completion_guard_target_changed"
    assert len(guard.body) == 1 and isinstance(guard.body[0], ast.Return), "completion_guard_return_changed"

    deliver = _method(env, "M10Environment", "_deliver_observations")
    task_values = [node.value for node in ast.walk(deliver) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "values" for target in node.targets)
                   and isinstance(node.value, ast.Dict)
                   and any(isinstance(key, ast.Constant) and key.value == "remaining_service" for key in node.value.keys)]
    assert len(task_values) == 1, "task_telemetry_schema_changed"
    public_fields = {key.value for key in task_values[0].keys if isinstance(key, ast.Constant)}
    assert "pending" in public_fields and "remaining_service" in public_fields, "task_telemetry_fields_missing"
    assert "completion" not in public_fields and "completed_at" not in public_fields, "new_completion_telemetry_requires_review"

    service = _method(lifecycle, "TaskLifecycle", "provide_service")
    assert any(isinstance(node, ast.Assign) and any(_attribute(target, "completed_at") for target in node.targets)
               for node in ast.walk(service)), "physical_completion_timestamp_missing"
    step = _method(env, "M10Environment", "step")
    info = [node.value for node in ast.walk(step) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "info" for target in node.targets)
            and isinstance(node.value, ast.Dict)]
    assert len(info) == 1, "step_info_schema_changed"
    info_fields = {key.value for key in info[0].keys if isinstance(key, ast.Constant)}
    assert "tasks" in info_fields and "completion_records" in info_fields, "step_terminal_fields_missing"
    assert "completed_at" not in info_fields, "physical_timestamp_export_requires_review"
    collector_source = COLLECTOR.read_bytes()
    hooks_source = HOOKS.read_bytes()
    collector = ast.parse(collector_source, filename=str(COLLECTOR))
    hooks = ast.parse(hooks_source, filename=str(HOOKS))
    collect_unit = _method(collector, "ProductionDataCollector", "_collect_unit")
    assert any(isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "config" for target in node.targets)
               and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
               and _attribute(node.value.func, "M10Config") and not node.value.args and not node.value.keywords
               for node in ast.walk(collect_unit)), "collector_config_not_default"
    for hook_name in ("route_runner", "episode_runner"):
        method = _method(hooks, "NativeRuntimeHooks", hook_name)
        assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "m10_config"
                   and not node.args and not node.keywords for node in ast.walk(method)), f"{hook_name}_config_not_default"
    source_config_bytes = SOURCE_RUN_CONFIG.read_bytes()
    source_config = json.loads(source_config_bytes)
    return {
        "schema": "w1-action-conditioned-task-outcome-source-check/1.0.0",
        "source_sha256": {
            **{name: hashlib.sha256(content).hexdigest() for name, content in sources.items()},
            "production_data.py": hashlib.sha256(collector_source).hexdigest(),
            "runtime_hooks.py": hashlib.sha256(hooks_source).hexdigest(),
            "source_run_environment.json": hashlib.sha256(source_config_bytes).hexdigest(),
        },
        "default_mode": mode.value.value,
        "actual_collector_uses_default_config": True,
        "policy_and_task_hooks_use_default_config": True,
        "source_run_recorded_mode": source_config["task_completion_mode"],
        "source_run_recorded_deadline_basis": source_config["deadline_basis"],
        "physical_completed_at_in_lifecycle": True,
        "physical_completed_at_in_step_info": False,
        "task_public_telemetry_fields": sorted(public_fields),
        "completion_notice_enabled_in_default_mode": False,
        "host_confirmation_target_in_default_mode": "UNAVAILABLE",
    }


if __name__ == "__main__":
    print(json.dumps(check_source(), indent=2, sort_keys=True))
