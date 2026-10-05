"""Load the actual selected fixture graph, denying historical workspace access."""
import importlib
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
SELECTED = ('test_cpu_scope_contract', 'test_supervised_phase_integration',
            'test_joint_pipeline', 'test_sequence_world', 'test_runtime_config_contract')


def verify():
    forbidden = ('/home/runs/', '/mnt/e/', 'E:\\Z', 'E:/Z')
    reads = set()
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(args[0])
            if any(prefix in path for prefix in forbidden):
                raise RuntimeError('HISTORICAL_WORKSPACE_ACCESS_DENIED:' + path)
            if path.startswith(str(ROOT)): reads.add(path)
    sys.addaudithook(audit)
    sys.path.insert(0, str(ROOT / 'native'))
    modules = {name: importlib.import_module(name) for name in SELECTED}
    origins = {}
    for name, module in list(sys.modules.items()):
        origin = getattr(module, '__file__', None)
        if not origin: continue
        path = Path(origin).resolve()
        if path.is_relative_to(ROOT): origins[name] = str(path)
        elif name.startswith('gppo_world') or name in SELECTED:
            raise RuntimeError('FIXTURE_ORIGIN_OUTSIDE_ISOLATED_PACKAGE:' + name)
    stream = io.StringIO()
    # Pure source/config comparison: no environment construction or model calls.
    case = modules['test_runtime_config_contract'].RuntimeConfigContractTests(
        'test_frozen_config_matches_actual_m10config_default_fields')
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.TestSuite([case]))
    if not result.wasSuccessful() or result.testsRun != 1 or result.skipped:
        raise RuntimeError('ISOLATED_CONFIG_TEST_FAILED:' + stream.getvalue())
    lifecycle = ROOT / 'native/gppo_world/task_lifecycle.py'
    if str(lifecycle) not in reads:
        raise RuntimeError('PACKAGED_LIFECYCLE_NOT_ACTUALLY_READ')
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('DEPENDENCY_PROBE_UNEXPECTED_CUDA_INITIALIZATION')
    return {'status': 'isolated_fixture_import_pass', 'module_origins': origins,
            'package_reads': sorted(reads), 'historical_paths_denied': list(forbidden),
            'config_test': stream.getvalue(), 'selected_fixture_modules': list(SELECTED),
            'cuda_initialized': False, 'model_calls': 0, 'environment_calls': 0,
            'checkpoint_calls': 0, 'full_training_executed': False}
