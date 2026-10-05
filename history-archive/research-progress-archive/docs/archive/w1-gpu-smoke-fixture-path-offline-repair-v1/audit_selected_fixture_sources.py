"""Compile and inventory selected fixture imports without importing models or environments."""
import ast
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
package = root / 'package'
entry = ast.parse((package / 'acceptance_entry.py').read_text(encoding='utf-8'))
method = next(node for node in entry.body if isinstance(node, ast.FunctionDef) and node.name == '_run_bounded_tests')
cases = next(node.value for node in method.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == 'cases' for target in node.targets))
selected = []
for case in cases.elts:
    stage = ast.literal_eval(case.elts[0])
    reference = case.elts[1]
    module_name, class_name = reference.value.id, reference.attr
    function_name = ast.literal_eval(case.elts[2])
    path = package / (module_name + '.py')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    assert any(isinstance(node, ast.FunctionDef) and node.name == function_name for node in cls.body)
    selected.append({'phase': stage, 'module': module_name, 'class': class_name, 'method': function_name})

def resolve(name):
    for base in (package, package / 'native'):
        candidate = base.joinpath(*name.split('.'))
        for path in (candidate.with_suffix('.py'), candidate / '__init__.py'):
            if path.is_file(): return path
    return None

pending = [case['module'] for case in selected]
seen, origins, external, dynamic, relative = set(), {}, set(), [], []
while pending:
    name = pending.pop()
    path = resolve(name)
    if path is None:
        external.add(name.split('.')[0])
        continue
    if path in seen: continue
    seen.add(path)
    source = path.read_text(encoding='utf-8-sig')
    compile(source, str(path), 'exec')  # syntax only: never execute this code
    tree = ast.parse(source)
    rel = path.relative_to(package).as_posix()
    origins[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): pending.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Record unresolved relatives; don't imply a runtime import proof.
                relative.append({'file': rel, 'line': node.lineno, 'expression': ast.unparse(node)})
            elif node.module: pending.append(node.module)
        elif isinstance(node, ast.Call):
            expr = ast.unparse(node.func)
            if expr == '__import__' and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                pending.append(node.args[0].value)
            elif expr in ('runpy.run_path', 'importlib.import_module', '__import__'):
                dynamic.append({'file': rel, 'line': node.lineno, 'expression': ast.unparse(node)})

result = {'classification': 'GPU_SMOKE_TEST_OFFLINE_SOURCE_AUDIT', 'selected_production_entry_cases': selected,
          'syntax_checked_local_modules': len(origins), 'source_sha256': dict(sorted(origins.items())),
          'stdlib_dependencies': sorted(external & set(sys.stdlib_module_names)),
          'third_party_or_unresolved_absolute_imports': sorted(external - set(sys.stdlib_module_names)),
          'dynamic_load_expressions_requiring_runtime_validation': dynamic,
          'relative_imports_requiring_runtime_validation': relative,
          'full_runtime_import_validated': False, 'gpu_training_validated': False,
          'remote_calls': 0, 'model_calls': 0, 'environment_calls': 0,
          'old_pid_reuse': {'pid': 19428, 'current_process_name': 'LogonUI',
                           'original_controller_still_live': False},
          'new_attempt_created': False, 'automatic_retry': False}
(root / 'selected-fixture-source-audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k not in ('source_sha256', 'stdlib_dependencies', 'dynamic_load_expressions_requiring_runtime_validation', 'relative_imports_requiring_runtime_validation')}, indent=2))
