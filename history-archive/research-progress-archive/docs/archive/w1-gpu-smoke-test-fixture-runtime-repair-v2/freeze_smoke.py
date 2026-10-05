"""Finalize this independently authorized engineering copy, then verify immutable bytes."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'package'
sys.path.insert(0, str(PACKAGE))
import launch_server_acceptance as launcher
from freeze_joint_package import freeze

NAME = 'w1-gpu-smoke-test-fixture-runtime-repair-v2'
ATTEMPT = NAME + '-once'
def read(name): return json.loads((PACKAGE / name).read_text(encoding='utf-8'))
def write(name, value): (PACKAGE / name).write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

request = read('SERVER_ACCEPTANCE_REQUEST.json')
request.update(acceptance_id=NAME, engineering_job_id=ATTEMPT,
               remote_execution_root='/home/user1/' + ATTEMPT,
               evidence_directory='/home/user1/' + NAME + '-evidence')
request['implementation_sha256'] = {name: launcher._sha256(PACKAGE / name) for name in sorted(launcher.IMPLEMENTATION_FILES)}
write('SERVER_ACCEPTANCE_REQUEST.json', request)
old = ROOT.parent / 'w1-gpu-smoke-test-bootstrap-repair-v1/package'
old_request = json.loads((old / 'SERVER_ACCEPTANCE_REQUEST.json').read_text())
assert request['limits'] == old_request['limits']
assert request['scope'] == old_request['scope']
for name in ('RESOURCE_REQUEST.json', 'experiment-matrix.json', 'parent-split.json', 'launch-contract.json'):
    obj = read(name)
    obj['attempt'] = ATTEMPT
    if name == 'launch-contract.json': obj['native_execution_root'] = request['remote_execution_root']
    write(name, obj)
inputs = read('runtime-inputs.json')
inputs['source_run']['wsl_root'] = request['remote_execution_root'] + '/native/source-evidence'
write('runtime-inputs.json', inputs)
for name in ('production_data.py', 'production_world.py', 'joint_pipeline.py', 'w1_graph_jepa.py', 'w1_training.py',
             'task_outcome_contract.py', 'sequence_data_contract.py', 'native/gppo_world/m10_environment.py',
             'native/gppo_world/task_lifecycle.py'):
    assert (PACKAGE / name).read_bytes() == (old / name).read_bytes(), name
freeze(PACKAGE)
checked = launcher.verify_structure()
identity = {'engineering_job_id': ATTEMPT, 'manifest_sha256': checked['identity']['manifest_sha256'],
            'hashes_sha256': checked['identity']['hashes_sha256'], 'server_acceptance_request_sha256': checked['request_sha256'],
            'resource_request_sha256': launcher._sha256(PACKAGE / 'RESOURCE_REQUEST.json'),
            'classification': 'GPU_SMOKE_TEST', 'old_evidence_preserved': True,
            'budgets_profile_model_and_training_sources_unchanged': True}
(ROOT / 'smoke-package-identity.json').write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
changes = {path.relative_to(PACKAGE).as_posix(): launcher._sha256(path) for path in PACKAGE.rglob('*')
           if path.is_file() and '__pycache__' not in path.parts
           and (not (old / path.relative_to(PACKAGE)).exists() or path.read_bytes() != (old / path.relative_to(PACKAGE)).read_bytes())}
(ROOT / 'source-differences.json').write_text(json.dumps(changes, indent=2) + '\n', encoding='utf-8')
print(json.dumps(identity))
