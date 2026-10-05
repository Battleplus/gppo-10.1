import hashlib
import json
from pathlib import Path
import shutil
import sys

root = Path(__file__).resolve().parent
package = root / 'package'
source = root.parent / 'w1-gpu-smoke-consolidated-acceptance-v3/package'
for name in ('local_launcher.py', 'local_gpu_worker.py', 'local_runtime.py'):
    shutil.copy2(root / name, package / name)
unchanged = {}
for p in source.rglob('*'):
    if p.is_file() and p.name not in ('execution-manifest.json', 'hashes.json'):
        relative = p.relative_to(source)
        assert p.read_bytes() == (package / relative).read_bytes(), relative
        unchanged[relative.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0, str(package))
from manifest_contract import write_identity_files, verify_package
result = write_identity_files(package, attempt='w1-gpu-smoke-local-v1-once', entrypoint='local_launcher.py')
result['local_request_sha256'] = hashlib.sha256((package / 'LOCAL_ACCEPTANCE_REQUEST.json').read_bytes()).hexdigest()
result['runtime_identity_sha256'] = hashlib.sha256((package / 'local-runtime-identity.json').read_bytes()).hexdigest()
result['source_files_preserved'] = len(unchanged)
(root / 'local-frozen-identity.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
(root / 'preserved-source-hashes.json').write_text(json.dumps(unchanged, indent=2), encoding='utf-8')
print(json.dumps(result))
