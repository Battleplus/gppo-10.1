"""Read-only checks of consumed evidence; writes new audit only in development directory."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
old = root.parent / 'w1-gpu-smoke-test-bootstrap-repair-v1'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
sys.path.insert(0, str(old / 'package'))
import launch_server_acceptance
checked = launch_server_acceptance.verify_structure()
old_manifest = json.loads((old / 'evidence-hashes.json').read_text(encoding='utf-8'))
for relative, identity in old_manifest['files'].items():
    path = old / relative
    assert path.stat().st_size == identity['bytes'] and sha(path) == identity['sha256'], relative
download = root.parent / 'runs/w1-gpu-smoke-test-bootstrap-repair-v1-once'
export = json.loads((download / 'acceptance-evidence-manifest.json').read_text())
assert sha(download / 'acceptance-evidence-manifest.json') == old_manifest['failure_export_manifest_sha256']
for relative, identity in export['files'].items():
    path = download / relative
    assert path.stat().st_size == identity['bytes'] and sha(path) == identity['sha256'], relative
changed = sorted(str(path.relative_to(root / 'package')).replace('\\', '/')
                 for path in (root / 'package').rglob('*') if path.is_file()
                 and '__pycache__' not in path.parts
                 and (not (old / 'package' / path.relative_to(root / 'package')).exists()
                      or sha(path) != sha(old / 'package' / path.relative_to(root / 'package'))))
expected = ['acceptance_entry.py', 'launch_joint_once.py', 'launch_server_acceptance.py',
            'test_action_conditioned_task_outcomes.py', 'test_runtime_config_contract.py',
            'test_smoke_transport_orchestration.py', 'test_transport_wall_budget.py',
            'transport_wall_budget.py']
assert changed == expected, changed
result = {'classification': 'GPU_SMOKE_TEST_OFFLINE_REPAIR',
          'consumed_package_identity_verified': True, 'consumed_evidence_unchanged': True,
          'verified_failure_export': True, 'new_dynamic_identity_created': False,
          'development_copy_frozen': False, 'runner_ready': False,
          'source_manifest_sha256': checked['identity']['manifest_sha256'],
          'source_hashes_sha256': checked['identity']['hashes_sha256'],
          'tests': {'fixture_path': 4, 'wall_authorization_orchestration': 34},
          'new_model_environment_remote_calls': 0,
          'changed_file_hashes': {relative: sha(root / 'package' / relative) for relative in changed}}
(root / 'repair-identity-audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
