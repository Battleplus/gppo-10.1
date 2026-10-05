"""Bind this explicit user approval outside the immutable research package."""
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'package'
EXPECTED = {
    'execution-manifest.json': '219cd82a046ebffd3feb3f1e2f9bc5b158f7ab040273d6baef5b866b01ff2895',
    'hashes.json': 'aea59645b889fda413fa9ef110453622d402019016132b0daf59d5779d6d465f',
    'RESOURCE_REQUEST.json': '85939a3d6211dc86c5db2a56a368b657c42975468b48dadc141e99b7a7e57e8a',
}
ATTEMPT = 'w1-g1-gppo-task-validation-local-v1-once'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    return {p.relative_to(PACKAGE).as_posix(): digest(p)
            for p in PACKAGE.rglob('*') if p.is_file()}


def main():
    before = inventory()
    delivery = json.loads((ROOT / 'final-delivery-evidence.json').read_text(encoding='utf-8'))
    if before != delivery['after']:
        raise RuntimeError('FROZEN_DELIVERY_CHANGED')
    if any(digest(PACKAGE / n) != h for n, h in EXPECTED.items()):
        raise RuntimeError('USER_APPROVED_DIGEST_MISMATCH')
    if any((ROOT / n).exists() for n in
           ['launch-intent.json', 'launch-evidence.json', 'external-token.txt', 'verified-export']):
        raise RuntimeError('PRIOR_LAUNCH_OR_TOKEN_EXISTS_NO_REUSE')
    sys.path.insert(0, str(PACKAGE))
    from manifest_contract import read_object, verify_external_authorization
    from worker_contract import verify_worker_contract
    from task_contract import verify_task_inputs
    verify_worker_contract(PACKAGE, ATTEMPT, EXPECTED['execution-manifest.json'], EXPECTED['hashes.json'])
    verify_task_inputs(PACKAGE)
    # These failed checks belong to the external synthetic driver, not the formal entry.
    forbidden = ['hidden_replay_observations_match_ledger_encode_charges',
                 'reduced_route_budgets', 'run_controlled_synthetic_pipeline']
    occurrences = [p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob('*.py')
                   if any(s in p.read_text(encoding='utf-8') for s in forbidden)]
    if occurrences:
        raise RuntimeError('SYNTHETIC_DRIVER_CHECK_IN_FORMAL_PACKAGE:' + repr(occurrences))
    auth_path = ROOT / 'external-authorization.json'
    template = read_object(auth_path, 'AUTHORIZATION')
    if template['status'] != 'NOT_APPROVED' or template['token_sha256'] is not None:
        raise RuntimeError('NEW_UNAPPROVED_TEMPLATE_REQUIRED')
    with (ROOT / 'external-authorization.before-approval.json').open('xb') as handle:
        handle.write(auth_path.read_bytes())
    token = secrets.token_urlsafe(48)
    token_path = ROOT / 'external-token.txt'
    with token_path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(token + '\n')
    principal = subprocess.run(['whoami'], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(['icacls', str(token_path), '/inheritance:r', '/grant:r', principal + ':(F)'],
                   capture_output=True, check=True)
    authorization = dict(template, status='APPROVED',
                         token_sha256=hashlib.sha256(token.encode('utf-8')).hexdigest())
    auth_path.write_text(json.dumps(authorization, ensure_ascii=False, sort_keys=True, indent=2) + '\n',
                         encoding='utf-8')
    loaded = read_object(auth_path, 'AUTHORIZATION')
    if loaded != authorization:
        raise RuntimeError('AUTHORIZATION_ROUNDTRIP_MISMATCH')
    verified = verify_external_authorization(PACKAGE, auth_path, token=token, preflight_only=False)
    if verified['request']['attempt'] != ATTEMPT:
        raise RuntimeError('AUTHORIZED_ATTEMPT_MISMATCH')
    command = ['C:/Python314/python.exe', '-B', str(PACKAGE / 'run_g1_task_validation.py'),
               '--preflight-only', '--authorization-file', str(auth_path)]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, timeout=60)
    (ROOT / 'approved-windows-preflight.stdout.raw').write_bytes(result.stdout)
    (ROOT / 'approved-windows-preflight.stderr.raw').write_bytes(result.stderr)
    (ROOT / 'approved-windows-preflight.stdout.txt').write_text(result.stdout.decode('utf-8', 'replace'), encoding='utf-8')
    (ROOT / 'approved-windows-preflight.stderr.txt').write_text(result.stderr.decode('utf-8', 'replace'), encoding='utf-8')
    evidence = {
        'attempt': ATTEMPT, 'approved_input_sha256': EXPECTED,
        'authorization_source': 'explicit user approval in this conversation, 2026-10-04',
        'authorization_standard_json_roundtrip': True,
        'authorization_fields_and_token_binding_validated': True,
        'secret_logged': False, 'token_file_owner_only_acl': True,
        'formal_package_contains_failed_synthetic_verifier': False,
        'preflight_command': command, 'preflight_exit_code': result.returncode,
        'preflight_wall_seconds': time.monotonic() - started,
        'frozen_package_unchanged': before == inventory(),
        'staging_started': False, 'worker_started': False, 'authorization_consumed': False,
    }
    (ROOT / 'approved-launch-preparation-evidence.json').write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    token = None
    print(json.dumps(evidence, ensure_ascii=False))
    if result.returncode or not evidence['frozen_package_unchanged']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
