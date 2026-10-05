"""Exercise archive validation/extraction itself; no model, environment or network."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from bulk_payload_transport import EXTRACT_SCRIPT, download_evidence
from manifest_contract import write_identity_files, verify_package

class PayloadTests(unittest.TestCase):
    def fixture(self, base, *, extra=None, corrupt=False):
        source = base / 'source'
        source.mkdir()
        shutil.copy2(Path(__file__).parent / 'manifest_contract.py', source)
        (source / 'fixture.txt').write_text('NONSECRET frozen synthetic fixture', encoding='utf-8')
        write_identity_files(source, attempt='synthetic-bulk-payload')
        staged = base / 'staged'
        staged.mkdir()
        archive = staged / '.frozen-payload.zip'
        with zipfile.ZipFile(archive, 'x') as z:
            for path in source.iterdir():
                z.writestr(path.name, b'tampered' if corrupt and path.name == 'fixture.txt' else path.read_bytes())
            if extra: z.writestr(extra, 'unexpected')
        return source, staged, hashlib.sha256(archive.read_bytes()).hexdigest()

    def execute(self, staged, digest):
        return subprocess.run([sys.executable, '-I', '-B', '-c', EXTRACT_SCRIPT, str(staged), digest],
                              capture_output=True, text=True)

    def test_extract_production_validator_accepts_identical_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, staged, digest = self.fixture(Path(tmp))
            result = self.execute(staged, digest)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(verify_package(staged)['manifest_sha256'], verify_package(source)['manifest_sha256'])
            self.assertFalse((staged / '.frozen-payload.zip').exists())

    def test_digest_mismatch_rejected_before_extract(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, staged, _ = self.fixture(Path(tmp))
            result = self.execute(staged, '0' * 64)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('PAYLOAD_ARCHIVE_DIGEST_MISMATCH', result.stderr)
            self.assertFalse((staged / 'fixture.txt').exists())

    def test_unlisted_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, staged, digest = self.fixture(Path(tmp), extra='../escape')
            result = self.execute(staged, digest)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('PAYLOAD_ALLOWLIST_MISMATCH', result.stderr)
            self.assertFalse((Path(tmp) / 'escape').exists())

    def test_corrupt_member_and_duplicate_destination_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, staged, digest = self.fixture(Path(tmp), corrupt=True)
            result = self.execute(staged, digest)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('PAYLOAD_MEMBER_DIGEST_MISMATCH', result.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            _, staged, digest = self.fixture(Path(tmp))
            (staged / 'fixture.txt').write_text('never overwrite')
            result = self.execute(staged, digest)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((staged / 'fixture.txt').read_text(), 'never overwrite')


class EvidenceTests(unittest.TestCase):
    def run_download(self, base, *, corrupt=False):
        evidence = base / 'remote-evidence'
        evidence.mkdir()
        body = b'{"synthetic":true}\n'
        (evidence / 'result.json').write_bytes(b'tampered' if corrupt else body)
        manifest = {'verified': True, 'engineering_job_id': 'synthetic-download',
                    'formal_research_attempt_created': False,
                    'files': {'result.json': {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}}}
        data = json.dumps(manifest).encode()
        (evidence / 'acceptance-evidence-manifest.json').write_bytes(data)
        request = {'engineering_job_id': 'synthetic-download', 'remote_execution_root': str(base / 'remote-job'),
                   'evidence_directory': str(evidence), 'runtime': {'python_executable': sys.executable},
                   'limits': {'aggregate_storage_bytes': 100000}}
        class SFTP:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def get(self, remote, local): shutil.copy2(remote, local)
        class Client:
            def open_sftp(self): return SFTP()
        def execute(_client, argv, **kwargs):
            result = subprocess.run(argv, capture_output=True, text=True, timeout=kwargs['timeout'])
            return {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
        return download_evidence(Client(), request, hashlib.sha256(data).hexdigest(),
                                 destination=base / 'download', execute=execute)

    def test_actual_sealed_manifest_download_extract_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_download(Path(tmp))
            self.assertEqual(result['files'], 1)
            self.assertTrue(result['temporary_remote_transport_removed'])
            self.assertEqual((Path(tmp) / 'download/result.json').read_bytes(), b'{"synthetic":true}\n')

    def test_corrupt_evidence_stops_before_local_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'EVIDENCE_FILE_IDENTITY_MISMATCH'):
                self.run_download(Path(tmp), corrupt=True)
            self.assertFalse((Path(tmp) / 'download').exists())

if __name__ == '__main__': unittest.main()
