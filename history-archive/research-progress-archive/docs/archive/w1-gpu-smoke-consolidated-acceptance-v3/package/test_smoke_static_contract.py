"""No-model regressions for the exact production entry/fixture contract."""
import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent

class SmokeStaticContractTests(unittest.TestCase):
    def test_selected_request_methods_exist_and_gpu_device_is_canonical(self):
        request = json.loads((ROOT / 'SERVER_ACCEPTANCE_REQUEST.json').read_text(encoding='utf-8'))
        for scope in request['scope'].values():
            module, cls, method = scope['test_id'].split('.')
            tree = ast.parse((ROOT / (module + '.py')).read_text(encoding='utf-8'))
            declaration = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
            self.assertIn(method, [node.name for node in declaration.body if isinstance(node, ast.FunctionDef)])
        tree = ast.parse((ROOT / 'acceptance_entry.py').read_text(encoding='utf-8'))
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
        devices = [node.value.value for node in assignments if isinstance(node.value, ast.Constant)
            and any(isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                    and target.slice.value == 'W1_SYNTHETIC_TEST_DEVICE' for target in node.targets)]
        self.assertEqual(devices, [request['gpu']['logical_device']])

    def test_system_only_dispatch_has_no_model_or_cuda_initialization(self):
        tree = ast.parse((ROOT / 'acceptance_entry.py').read_text(encoding='utf-8'))
        branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute) and node.test.attr == 'system_only')
        calls = [ast.unparse(node.func) for statement in branch.body for node in ast.walk(statement) if isinstance(node, ast.Call)]
        self.assertIn('_bootstrap_runtime', calls)
        self.assertIn('run', calls)
        self.assertFalse(any('cuda' in name or 'model' in name for name in calls))

if __name__ == '__main__':
    unittest.main()
