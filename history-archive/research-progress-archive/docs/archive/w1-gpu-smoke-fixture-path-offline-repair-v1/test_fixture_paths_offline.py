"""Validate the actual fixture path expressions without models or environment."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import tempfile
import unittest

ROOT=Path(__file__).parent
PACKAGE=ROOT/'package'
OLD=ROOT.parent/'w1-gpu-smoke-test-bootstrap-repair-v1'/'package'

def lifecycle_globals(package):
    tree=ast.parse((package/'test_action_conditioned_task_outcomes.py').read_text(encoding='utf-8'))
    names={'PACKAGE','SOURCE_ROOT','LIFECYCLE','TaskLifecycle','TaskState'}
    statements=[node for node in tree.body if isinstance(node,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id in names for t in node.targets)]
    module=ast.Module(body=statements,type_ignores=[])
    scope={'__file__':str(package/'test_action_conditioned_task_outcomes.py'),
           'Path':Path,'runpy':runpy}
    # Execute the production fixture's own path/load statements, not substitutes.
    exec(compile(module,'fixture-path-statements','exec'),scope)
    return scope

class FixturePaths(unittest.TestCase):
    def test_actual_fixture_loads_under_foreign_directory_with_no_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            staged=Path(temporary)/'isolated'/'package'
            staged.mkdir(parents=True)
            shutil.copy2(PACKAGE/'test_action_conditioned_task_outcomes.py',staged)
            target=staged/'native'/'gppo_world'
            target.mkdir(parents=True)
            shutil.copy2(PACKAGE/'native/gppo_world/task_lifecycle.py',target)
            scope=lifecycle_globals(staged)
            self.assertEqual(scope['SOURCE_ROOT'],target)
            self.assertEqual(scope['TaskLifecycle'].__name__,'TaskLifecycle')
            self.assertFalse((Path(temporary)/'runs').exists())

    def test_missing_packaged_module_fails_without_workspace_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            staged=Path(temporary)/'package'
            staged.mkdir()
            shutil.copy2(PACKAGE/'test_action_conditioned_task_outcomes.py',staged)
            with self.assertRaises(FileNotFoundError) as caught:
                lifecycle_globals(staged)
            self.assertEqual(Path(caught.exception.filename),staged/'native/gppo_world/task_lifecycle.py')

    def test_original_module_bytes_and_code_unchanged(self):
        for name in ['native/gppo_world/task_lifecycle.py','native/gppo_world/m10_environment.py',
                     'production_world.py','joint_pipeline.py','production_data.py']:
            self.assertEqual((PACKAGE/name).read_bytes(),(OLD/name).read_bytes())

    def test_config_fixture_reads_packaged_source_expression(self):
        tree=ast.parse((PACKAGE/'test_runtime_config_contract.py').read_text(encoding='utf-8'))
        method=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)
                    and node.name=='test_frozen_config_matches_actual_m10config_default_fields')
        statement=method.body[0]
        scope={'PACKAGE':PACKAGE}
        exec(compile(ast.Module(body=[statement],type_ignores=[]),'source-path-expression','exec'),scope)
        self.assertEqual(scope['source'],PACKAGE/'native/gppo_world/m10_environment.py')
        self.assertTrue(scope['source'].is_file())

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(FixturePaths)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    evidence={'classification':'GPU_SMOKE_TEST_OFFLINE_REPAIR',
              'tests_run':result.testsRun,'passed':result.wasSuccessful(),
              'coverage':'actual fixture path/load statements and unchanged production-module bytes; not full fixture import or GPU pipeline',
              'model_initializations':0,'model_forwards':0,'optimizer_updates':0,
              'real_environment_calls':0,'remote_calls':0,'new_attempt_created':False,
              'changed_files':{name:hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest()
                               for name in ['test_action_conditioned_task_outcomes.py','test_runtime_config_contract.py']}}
    (ROOT/'offline-test-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
