"""Real controller orchestration, in-memory SSH/SFTP boundary; no remote/GPU calls."""
import hashlib
import io
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

import launch_server_acceptance as launcher
from test_server_acceptance_entry import request_fixture, authorization_fixture
from test_transport_wall_budget import Clock
from transport_wall_budget import WallBudget


class Channel:
    def __init__(self): self.timeout = None
    def settimeout(self, value): self.timeout = value
    def shutdown_write(self): pass
    def recv_exit_status(self): return 0


class Stream(io.BytesIO):
    def __init__(self, data=b'', *, commit=None):
        super().__init__(data)
        self.channel = Channel()
        self.commit = commit
    def write(self, data):
        return super().write(data.encode() if isinstance(data, str) else data)
    def close(self):
        if self.commit is not None and not self.closed: self.commit(self.getvalue())
        super().close()


class SFTP:
    def __init__(self, client): self.client = client
    def get_channel(self): return Channel()
    def stat(self, path):
        if self.client.closed: raise OSError('inspection transport unavailable')
        if path not in self.client.directories: raise FileNotFoundError(path)
        return object()
    def mkdir(self, path, mode=0o700):
        if path in self.client.directories: raise FileExistsError(path)
        self.client.directories.add(path)
    def put(self, local, remote):
        if self.client.fail_upload:
            self.client.closed = True
            raise RuntimeError('original upload failure')
        self.client.clock.now += self.client.upload_seconds
        self.client.upload_seconds = 0
        self.client.files[remote] = Path(local).read_bytes()
    def open(self, path, mode):
        if mode == 'wx':
            if path in self.client.files: raise FileExistsError(path)
            self.client.files[path] = b''
            return Stream(commit=lambda data: self.client.files.__setitem__(path, data))
        return Stream(self.client.files[path])
    def chmod(self, *_args): pass
    def get(self, remote, local): Path(local).write_bytes(self.client.files[remote])
    def close(self): pass


class SSH:
    def __init__(self, checked, clock, *, upload_seconds=2, fail_upload=False):
        self.checked, self.clock = checked, clock
        self.upload_seconds, self.fail_upload = upload_seconds, fail_upload
        self.files, self.directories, self.commands = {}, set(), []
        self.closed = False
    def close(self): self.closed = True
    def open_sftp(self):
        if self.closed: raise OSError('inspection transport unavailable')
        return SFTP(self)
    def exec_command(self, command, **kwargs):
        self.commands.append((command, kwargs['timeout']))
        self.clock.now += 1
        args = shlex.split(command)
        request = self.checked['request']
        if '--bootstrap-only' in args:
            payload = {'status': 'bootstrap_pass'}
        elif '--evidence-dir' in args:
            payload = {'status': 'complete', 'resources': {
                'wall_seconds': 1., 'complete_process_cpu_seconds': .1,
                'charged_upper_wall_seconds': 6.,
                'charged_upper_complete_process_cpu_seconds': 3.1}}
            body = b'{"synthetic_transport_fixture":true}\n'
            manifest = {'verified': True, 'engineering_job_id': request['engineering_job_id'],
                        'formal_research_attempt_created': False,
                        'files': {'acceptance-result.json': {'bytes': len(body),
                                  'sha256': hashlib.sha256(body).hexdigest()}}}
            manifest_bytes = json.dumps(manifest).encode()
            base = request['evidence_directory']
            self.files[base + '/acceptance-result.json'] = body
            self.files[base + '/acceptance-evidence-manifest.json'] = manifest_bytes
            payload['evidence_manifest_sha256'] = hashlib.sha256(manifest_bytes).hexdigest()
        elif '--identity' in args:
            payload = {'status': 'pass'}
        elif 'hashlib' in command:
            probe = '/home/user1/w1-remote-runtime-dependency-repair-v1/runtime_preflight.py'
            identity = '/home/user1/w1-remote-runtime-dependency-repair-v1/runtime-identity-v2.json'
            payload = {probe: launcher._sha256(launcher.ROOT / 'remote_runtime_preflight.py'),
                       identity: self.checked['contract']['runtime_identity_sha256']}
        elif 'attempt_path_exists' in command:
            payload = {'attempt_path_exists': False}
        else:
            payload = {p: {'exists': False, 'symlink': False}
                       for p in (request['remote_execution_root'], request['evidence_directory'])}
        return Stream(), Stream(json.dumps(payload).encode()), Stream()


class ControllerOrchestrationTests(unittest.TestCase):
    def run_fixture(self, directory, *, upload_seconds=2, fail_upload=False, connect_error=False):
        request = request_fixture()
        checked = {'request': request, 'request_sha256': '1' * 64,
                   'identity': {'manifest_sha256': '2' * 64, 'hashes_sha256': '3' * 64,
                                'manifest': {'files': {}}},
                   'contract': {'native_python': request['runtime']['python_executable'],
                                'runtime_identity_sha256': request['runtime']['runtime_identity_sha256']}}
        token = 'NONSECRET_SYNTHETIC_TOKEN'
        auth = Path(directory) / 'fixture-authorization.json'
        auth.write_text(json.dumps(authorization_fixture(request, token)), encoding='utf-8')
        clock = Clock()
        client = SSH(checked, clock, upload_seconds=upload_seconds, fail_upload=fail_upload)
        def connect(_contract, _password, *, wall_budget):
            if connect_error: raise OSError('synthetic connection failure')
            return wall_budget.bind(client)
        def budget(limit, **kwargs): return WallBudget(limit, **kwargs, clock=clock)
        with (patch.object(launcher.time, 'monotonic', clock),
              patch.object(launcher, 'WallBudget', side_effect=budget),
              patch.object(launcher.controller, 'connect', side_effect=connect),
              patch.object(launcher.controller_runtime, 'verify_controller_runtime'),
              patch.object(launcher, '_local_evidence_path', return_value=Path(directory) / 'download'),
              patch.object(launcher.sys, 'stdin', io.StringIO('NONSECRET_PASSWORD\n' + token + '\n'))):
            result = launcher.run_formal(checked, auth, 'hzy', password_stdin=True)
        return result, client, json.loads((Path(directory) / 'controller-wall-phases.json').read_text())

    def test_real_controller_routes_cap_and_verified_download(self):
        with tempfile.TemporaryDirectory() as directory:
            result, client, evidence = self.run_fixture(directory)
            self.assertEqual(result.get('remote_exit_code'), 0, result)
            self.assertTrue(result['uploaded_authorization_verification']['json_verified'])
            self.assertEqual(result['local_evidence']['files'], 1)
            self.assertTrue(result['engineering_job_consumed'])
            commands = [shlex.split(cmd) for cmd, _ in client.commands]
            formal = next(cmd for cmd in commands if '--remaining-wall-seconds' in cmd)
            cap = float(formal[formal.index('--remaining-wall-seconds') + 1])
            self.assertGreater(cap, 5)
            self.assertLessEqual(cap, 150)
            self.assertEqual([row['phase'] for row in evidence['phases']],
                             ['connect', 'remote_preflight', 'upload', 'bootstrap', 'remote_acceptance', 'download'])
            self.assertTrue(client.closed)

    def test_expired_upload_prevents_remote_worker_and_preserves_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            result, client, evidence = self.run_fixture(directory, upload_seconds=176)
            self.assertEqual(result['error_type'], 'WallDeadlineExceeded')
            self.assertIsNone(result['engineering_job_consumed'])
            self.assertFalse(result['automatic_retry'])
            self.assertFalse(any('--bootstrap-only' in cmd or '--evidence-dir' in cmd for cmd, _ in client.commands))
            self.assertEqual(evidence['phases'][-1]['outcome'], 'error')
            self.assertTrue(client.closed)

    def test_upload_inspection_error_does_not_replace_first_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self.run_fixture(directory, fail_upload=True)
            self.assertEqual(result['error_type'], 'RuntimeError')
            self.assertIn('original upload failure', result['exception_chain'])
            self.assertEqual(result['consumption_inspection_error_type'], 'OSError')
            self.assertIsNone(result['engineering_job_consumed'])

    def test_connect_failure_is_recorded_before_consumption(self):
        with tempfile.TemporaryDirectory() as directory:
            result, client, evidence = self.run_fixture(directory, connect_error=True)
            self.assertEqual(result['error_type'], 'OSError')
            self.assertFalse(result['token_consumed'])
            self.assertFalse(result['engineering_job_consumed'])
            self.assertEqual(client.commands, [])
            self.assertEqual(evidence['phases'][0]['outcome'], 'error')

    def test_main_reports_smoke_success_as_exit_zero(self):
        with (patch.object(launcher, 'verify_structure', return_value={}),
              patch.object(launcher, 'run_formal', return_value={'status': 'gpu_smoke_test_pass'}),
              patch('sys.stdout', io.StringIO())):
            self.assertEqual(launcher.main(['--authorization-file', 'fixture.json', '--real-name', 'hzy']), 0)


if __name__ == '__main__': unittest.main()
