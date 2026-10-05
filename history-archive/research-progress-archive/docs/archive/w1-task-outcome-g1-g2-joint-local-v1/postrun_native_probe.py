"""Read-only process and frozen native package inspection; no torch import."""
import hashlib
import json
import os
from pathlib import Path
import time

attempt = 'w1-task-outcome-g1-g2-joint-local-v1-once'
root = Path('/home/asus') / attempt / 'package'
identity = json.loads((root / 'hashes.json').read_text(encoding='utf-8'))
mismatches = []
for name, expected in identity['files'].items():
    p = root / name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
        mismatches.append(name)
manifest_matches = hashlib.sha256((root / 'execution-manifest.json').read_bytes()).hexdigest() == identity['execution_manifest_sha256']
matches = []
for p in Path('/proc').iterdir():
    if not p.name.isdecimal() or int(p.name) == os.getpid():
        continue
    try:
        command = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if str(root) in command:
        matches.append({'pid': int(p.name), 'command': command})
print(json.dumps({'timestamp_unix': time.time(), 'native_content_files': len(identity['files']),
    'native_content_mismatches': mismatches, 'native_manifest_matches': manifest_matches,
    'task_processes': matches, 'task_processes_remaining': len(matches),
    'no_model_environment_calls': True, 'probe_is_read_only': True}))
assert not mismatches and manifest_matches and not matches
