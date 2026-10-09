"""Bind the private relocated runtime; existing server environments are rejected."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

def verify(root):
    contract=json.loads((Path(root)/'server-evidence/runtime-binding.json').read_text())
    if sys.executable!=contract['interpreter']:raise ValueError('SERVER_RUNTIME_INTERPRETER_CHANGED')
    identity=Path(contract['remote_identity_file'])
    if hashlib.sha256(identity.read_bytes()).hexdigest()!=contract['runtime_identity_sha256']:
        raise ValueError('SERVER_RUNTIME_IDENTITY_CHANGED')
    observed=json.loads(identity.read_text())
    if observed['status']!='pass' or observed['differences'] or not observed['all_original_files_match']:
        raise ValueError('SERVER_RUNTIME_NOT_ADMITTED')
    versions={d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions()}
    if versions!=observed['observed']['versions'] or sys.version!=observed['observed']['python']:
        raise ValueError('SERVER_RUNTIME_VERSION_CHANGED')
    for path,digest in contract['critical_runtime_files'].items():
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        if h.hexdigest()!=digest:raise ValueError('SERVER_CRITICAL_RUNTIME_FILE_CHANGED:'+path)
    return contract
