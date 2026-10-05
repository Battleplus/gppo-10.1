import getpass
import hashlib
import json
import secrets
import sys
import time
import traceback
from pathlib import Path

root = Path(__file__).parent
sys.path.insert(0, str(root / 'package'))
import launch_server_acceptance as launcher


def main():
    import os
    lock_path = root / 'controller-start.lock'
    with lock_path.open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'executable': sys.executable,
                   'entry': str(Path(__file__).resolve()), 'automatic_retry': False}, stream)
    regression = json.loads((root / 'prior-transport-regression.json').read_text(encoding='utf-8'))
    if regression.get('status') != 'pass' or regression.get('duplicate_rejected') is not True:
        raise RuntimeError('TRANSPORT_REGRESSION_REQUIRED')
    checked = launcher.verify_structure()
    (root / 'interactive-status.json').write_text(json.dumps({'status': 'awaiting_local_password',
        'worker_started': False, 'test': 'GPU_SMOKE_TEST'}), encoding='utf-8')
    print('W1 GPU_SMOKE_TEST: enter SSH password locally; it will not be saved.', flush=True)
    password = getpass.getpass('SSH password (not archived): ')
    (root / 'interactive-status.json').write_text(json.dumps({'status': 'connecting',
        'worker_started': False}), encoding='utf-8')
    token = secrets.token_urlsafe(32)
    request = checked['request']
    authorization = {
        'schema': launcher.acceptance_entry.AUTH_SCHEMA, 'status': 'APPROVED',
        'engineering_job_id': request['engineering_job_id'],
        'remote_execution_root': request['remote_execution_root'],
        'evidence_directory': request['evidence_directory'],
        'server_acceptance_request_sha256': checked['request_sha256'],
        'execution_manifest_sha256': checked['identity']['manifest_sha256'],
        'hashes_sha256': checked['identity']['hashes_sha256'],
        'authorization_id': 'gpu-smoke-fixture-runtime-repair-v2-user-authorized',
        'one_time_token_sha256': hashlib.sha256(token.encode()).hexdigest(),
        'formal_research_attempt_created': False, 'automatic_retry': False,
    }
    path = root / 'external-authorization.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(authorization, stream, sort_keys=True, indent=2)
        stream.write('\n')
    launcher.acceptance_entry.validate_authorization(
        launcher.acceptance_entry._read_json(path), request=request,
        request_sha256=checked['request_sha256'], manifest_sha256=checked['identity']['manifest_sha256'],
        hashes_sha256=checked['identity']['hashes_sha256'], token=token)
    # Credentials reach the production entry through an in-memory stdin only.
    import io
    original_stdin = sys.stdin
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    try:
        sys.stdin = io.StringIO(password + '\n' + token + '\n')
        result = launcher.run_formal(checked, path, 'hzy', password_stdin=True)
    except BaseException:
        result = {'status': 'technical_stop', 'exception_chain': traceback.format_exc()}
    finally:
        sys.stdin.close()
        sys.stdin = original_stdin
    serialized = json.dumps(result, ensure_ascii=False, indent=2).replace(token, '<redacted>').replace(password, '<redacted>')
    (root / 'controller-result.json').write_text(serialized + '\n', encoding='utf-8')
    result = json.loads(serialized)
    wall = time.monotonic() - started_wall
    cpu = time.process_time() - started_cpu
    (root / 'controller-measurement.json').write_text(json.dumps({
        'classification': 'GPU_SMOKE_TEST', 'wall_seconds': wall,
        'controller_process_cpu_seconds': cpu, 'ssh_sftp_remote_cpu_seconds': None,
        'prior_transport_regression_reused_as_evidence': True,
        'cpu_scope': 'local controller process only; remote SSH/SFTP CPU not measured',
        'full_resource_acceptance': False,
    }, indent=2) + '\n', encoding='utf-8')
    token = password = None
    (root / 'interactive-status.json').write_text(json.dumps({'status': 'finished',
        'remote_exit_code': result.get('remote_exit_code'),
        'controller_result': str(root / 'controller-result.json')}), encoding='utf-8')
    print(json.dumps({'status': result.get('status'), 'remote_exit_code': result.get('remote_exit_code'),
                      'engineering_job_consumed': result.get('engineering_job_consumed'),
                      'result_path': str(root / 'controller-result.json'), 'wall_seconds': wall}))
    return 0 if result.get('status') in ('complete', 'gpu_smoke_test_pass') else 1


if __name__ == '__main__':
    try:
        exit_code = main()
    except BaseException:
        (root / 'controller-outer-error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        (root / 'interactive-status.json').write_text(json.dumps({'status': 'controller_error',
            'worker_started': False}), encoding='utf-8')
        exit_code = 1
    raise SystemExit(exit_code)
