"""Batch the identical frozen allowlist; retain exclusive root/auth creation and hashes."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

def gpu_preflight(client, execute):
    gpu = execute(client, ['nvidia-smi', '--id=1', '--query-gpu=index,uuid,memory.free,name',
                           '--format=csv,noheader,nounits'], timeout=10)
    compute = execute(client, ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                               '--format=csv,noheader,nounits'], timeout=10)
    if gpu['exit_code'] or compute['exit_code']: raise RuntimeError('GPU1_PREFLIGHT_QUERY_FAILED')
    parts = [part.strip() for part in gpu['stdout'].strip().split(',')]
    if len(parts) != 4 or parts[0] != '1' or int(parts[2]) < 9216:
        raise RuntimeError('GPU1_FREE_MEMORY_REQUIREMENT_FAILED:' + gpu['stdout'])
    occupied = [line for line in compute['stdout'].splitlines() if line.split(',')[0].strip() == parts[1]]
    if occupied: raise RuntimeError('GPU1_OTHER_COMPUTE_TASK_PRESENT:' + '\n'.join(occupied))
    return {'gpu_physical_device': 1, 'gpu_uuid': parts[1], 'free_memory_mib': int(parts[2]),
            'name': parts[3], 'other_compute_tasks': [], 'allocation_attestation_required': False}

EXTRACT_SCRIPT = '''
import hashlib,json,pathlib,runpy,sys,zipfile
root=pathlib.Path(sys.argv[1]); digest=sys.argv[2]
archive=root/'.frozen-payload.zip'
if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest: raise RuntimeError('PAYLOAD_ARCHIVE_DIGEST_MISMATCH')
with zipfile.ZipFile(archive) as z:
    names=z.namelist()
    if len(names)!=len(set(names)): raise RuntimeError('PAYLOAD_DUPLICATE_MEMBER')
    manifest=json.loads(z.read('execution-manifest.json'))
    hashes=json.loads(z.read('hashes.json'))
    expected={**manifest['files'], 'execution-manifest.json':hashes['execution_manifest_sha256']}
    if set(names)!=set(expected)|{'hashes.json'}: raise RuntimeError('PAYLOAD_ALLOWLIST_MISMATCH')
    for name in names:
        rel=pathlib.PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or '\\\\' in name or ':' in name: raise RuntimeError('PAYLOAD_PATH_INVALID')
        data=z.read(name)
        if name in expected and hashlib.sha256(data).hexdigest()!=expected[name]: raise RuntimeError('PAYLOAD_MEMBER_DIGEST_MISMATCH:'+name)
        path=root.joinpath(*rel.parts)
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f: f.write(data)
verify=runpy.run_path(str(root/'manifest_contract.py'))['verify_package']
identity=verify(root)
archive.unlink()
print(json.dumps({'status':'payload_verified','manifest_sha256':identity['manifest_sha256'],'hashes_sha256':identity['hashes_sha256'],'files':len(names)}))
'''


def upload_once(client, checked, authorization_path, *, source_root, execute, mkdir_parents):
    contract = checked['contract']
    remote = contract['native_execution_root']
    identities = {**checked['identity']['manifest']['files'],
                  'execution-manifest.json': checked['identity']['manifest_sha256'],
                  'hashes.json': checked['identity']['hashes_sha256']}
    with tempfile.TemporaryDirectory(prefix='w1-frozen-transfer-') as temporary:
        archive = Path(temporary) / 'payload.zip'
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for relative, digest in sorted(identities.items()):
                path = Path(source_root) / relative
                data = path.read_bytes()
                if path.is_symlink() or hashlib.sha256(data).hexdigest() != digest:
                    raise RuntimeError('LOCAL_PAYLOAD_IDENTITY_MISMATCH:' + relative)
                z.writestr(relative, data)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        with client.open_sftp() as sftp:
            sftp.mkdir(remote, mode=0o700)
            sftp.put(str(archive), remote + '/.frozen-payload.zip')
        result = execute(client, [contract['native_python'], '-I', '-B', '-c', EXTRACT_SCRIPT, remote, digest], timeout=30)
        if result['exit_code']:
            raise RuntimeError('FROZEN_PAYLOAD_EXTRACT_FAILED:' + result['stderr'] + result['stdout'])
        receipt = json.loads(result['stdout'])
        if (receipt.get('status') != 'payload_verified'
                or receipt.get('manifest_sha256') != checked['identity']['manifest_sha256']
                or receipt.get('hashes_sha256') != checked['identity']['hashes_sha256']):
            raise RuntimeError('FROZEN_PAYLOAD_RECEIPT_MISMATCH')
    auth_root = '/home/user1/.w1-external-authorizations'
    auth_path = auth_root + '/' + contract['attempt'] + '.json'
    with client.open_sftp() as sftp:
        mkdir_parents(sftp, auth_root)
        with sftp.open(auth_path, 'wx') as handle:
            handle.write(Path(authorization_path).read_bytes())
        sftp.chmod(auth_path, 0o600)
    return remote, auth_path


DOWNLOAD_SCRIPT = '''
import hashlib,json,pathlib,sys,zipfile
root=pathlib.Path(sys.argv[1]); expected=sys.argv[2]; target=pathlib.Path(sys.argv[3])
manifest_path=root/'acceptance-evidence-manifest.json'
data=manifest_path.read_bytes()
if hashlib.sha256(data).hexdigest()!=expected: raise RuntimeError('EVIDENCE_MANIFEST_DIGEST_MISMATCH')
manifest=json.loads(data)
with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for name,identity in sorted(manifest['files'].items()):
        rel=pathlib.PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or '\\\\' in name or ':' in name: raise RuntimeError('EVIDENCE_PATH_INVALID')
        path=root.joinpath(*rel.parts)
        if path.is_symlink(): raise RuntimeError('EVIDENCE_SYMLINK')
        body=path.read_bytes()
        if len(body)!=identity['bytes'] or hashlib.sha256(body).hexdigest()!=identity['sha256']: raise RuntimeError('EVIDENCE_FILE_IDENTITY_MISMATCH:'+name)
        z.writestr(name,body)
    z.writestr('acceptance-evidence-manifest.json',data)
print(json.dumps({'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size}))
'''


def download_evidence(client, request, expected_manifest_sha256, *, destination, execute):
    destination = Path(destination)
    if destination.exists(): raise RuntimeError('LOCAL_ACCEPTANCE_EVIDENCE_PATH_ALREADY_EXISTS_NO_RETRY')
    remote_zip = request['remote_execution_root'] + '-verified-transport.zip'
    result = execute(client, [request['runtime']['python_executable'], '-I', '-B', '-c', DOWNLOAD_SCRIPT,
                             request['evidence_directory'], expected_manifest_sha256, remote_zip], timeout=30)
    if result['exit_code']: raise RuntimeError('EVIDENCE_ARCHIVE_FAILED:' + result['stderr'])
    receipt = json.loads(result['stdout'])
    with tempfile.TemporaryDirectory(prefix='w1-verified-download-') as temporary:
        archive = Path(temporary) / 'evidence.zip'
        with client.open_sftp() as sftp: sftp.get(remote_zip, str(archive))
        if archive.stat().st_size != receipt['bytes'] or hashlib.sha256(archive.read_bytes()).hexdigest() != receipt['sha256']:
            raise RuntimeError('DOWNLOADED_ARCHIVE_DIGEST_MISMATCH')
        with zipfile.ZipFile(archive) as z:
            manifest_data = z.read('acceptance-evidence-manifest.json')
            if hashlib.sha256(manifest_data).hexdigest() != expected_manifest_sha256:
                raise RuntimeError('SERVER_ACCEPTANCE_EVIDENCE_MANIFEST_DIGEST_MISMATCH')
            manifest = json.loads(manifest_data)
            if (manifest.get('verified') is not True or manifest.get('engineering_job_id') != request['engineering_job_id']
                    or manifest.get('formal_research_attempt_created') is not False):
                raise RuntimeError('SERVER_ACCEPTANCE_EVIDENCE_MANIFEST_INVALID')
            if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(manifest['files']) | {'acceptance-evidence-manifest.json'}:
                raise RuntimeError('EVIDENCE_ARCHIVE_ALLOWLIST_MISMATCH')
            total = sum(identity['bytes'] for identity in manifest['files'].values()) + len(manifest_data)
            # Charge frozen staging as well as native/download/evidence copies.
            staged = sum(p.stat().st_size for p in Path(__file__).resolve().parent.rglob('*')
                         if p.is_file() and '__pycache__' not in p.parts)
            if 3 * total + receipt['bytes'] + staged > request['limits']['aggregate_storage_bytes']:
                raise RuntimeError('ACCEPTANCE_THREE_COPY_AGGREGATE_STORAGE_LIMIT')
            destination.mkdir(parents=True, exist_ok=False)
            for name, identity in manifest['files'].items():
                rel = PurePosixPath(name)
                if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name: raise RuntimeError('EVIDENCE_PATH_INVALID')
                data = z.read(name)
                if len(data) != identity['bytes'] or hashlib.sha256(data).hexdigest() != identity['sha256']:
                    raise RuntimeError('SERVER_ACCEPTANCE_EVIDENCE_IDENTITY_MISMATCH:' + name)
                path = destination.joinpath(*rel.parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as f: f.write(data)
            with (destination / 'acceptance-evidence-manifest.json').open('xb') as f: f.write(manifest_data)
    # Only this identity's temporary transport archive is removed, after receipt verification.
    cleanup = execute(client, [request['runtime']['python_executable'], '-I', '-B', '-c',
        'import pathlib,sys; pathlib.Path(sys.argv[1]).unlink()', remote_zip], timeout=10)
    return {'path': str(destination), 'files': len(manifest['files']), 'verified_bytes': total,
            'manifest_sha256': expected_manifest_sha256, 'transport_sha256': receipt['sha256'],
            'temporary_remote_transport_removed': cleanup['exit_code'] == 0,
            'aggregate_storage_check_includes_three_payload_copies_and_transport': True}
