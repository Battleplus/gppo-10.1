import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
package = root / 'package'
native = '/home/asus/w1-gpu-smoke-local-v1'
python = '/home/asus/.venvs/w1-light-repaired-fair-rerun-py31116/bin/python'
parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['sync', 'preflight', 'system', 'gpu', 'download'])
args = parser.parse_args()

def identities():
    return {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in package.rglob('*') if p.is_file()}

before = identities()
if args.mode == 'sync':
    command = ['wsl', '-d', 'Ubuntu-24.04', '--', 'python3', '-B', '-c',
        "import shutil,pathlib; source=pathlib.Path('/mnt/e/Z博士/diagnostic-work/w1-gpu-smoke-local-v1/package'); target=pathlib.Path('"+native+"/package'); "
        "assert not (target.parent/'package-supervision').exists(); shutil.copytree(source,target,dirs_exist_ok=True)"]
elif args.mode == 'download':
    command = ['wsl', '-d', 'Ubuntu-24.04', '--', 'python3', '-B', '-c',
        "import pathlib,shutil,hashlib,json; s=pathlib.Path('"+native+"'); d=pathlib.Path('/mnt/e/Z博士/diagnostic-work/w1-gpu-smoke-local-v1/download'); d.mkdir(exist_ok=False); "
        "names=['gpu-evidence','package-supervision','local-supervisor-result.json','evidence']; "
        "[(shutil.copytree(s/n,d/n) if (s/n).is_dir() else shutil.copy2(s/n,d/n)) for n in names if (s/n).exists()]; "
        "files={str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest() for p in d.rglob('*') if p.is_file()}; "
        "assert all(hashlib.sha256((s/n).read_bytes()).hexdigest()==h for n,h in files.items()); "
        "(d/'download-hashes.json').write_text(json.dumps(files,sort_keys=True,indent=2)); print(json.dumps({'files':len(files),'hashes_verified':True}))"]
else:
    command = ['wsl', '-d', 'Ubuntu-24.04', '--', 'env', '-u', 'LD_LIBRARY_PATH', '-u', 'LD_PRELOAD',
        '-u', 'PYTHONPATH', '-u', 'PYTHONHOME', 'PYTHONNOUSERSITE=1', 'CUDA_VISIBLE_DEVICES=0',
        'OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
        python, '-I', '-B', '-X', 'pycache_prefix='+str(Path(python).parent.parent).replace('\\','/')+'/.w1-no-bytecode-cache',
        native+'/package/local_launcher.py']
    if args.mode != 'gpu':
        command.append('--preflight-only' if args.mode == 'preflight' else '--system-only')
started, cpu = time.monotonic(), time.process_time()
try:
    result = subprocess.run(command, text=True, capture_output=True, encoding='utf-8', errors='replace',
                            timeout=180 if args.mode == 'gpu' else 120)
    stdout, stderr, code = result.stdout, result.stderr, result.returncode
except subprocess.TimeoutExpired as error:
    stdout = (error.stdout or b'').decode('utf-8', 'replace') if isinstance(error.stdout, bytes) else error.stdout or ''
    stderr = str(error)
    code = 124
after = identities()
evidence = {'classification': 'GPU_SMOKE_TEST' if args.mode == 'gpu' else 'LOCAL_SMOKE_PREPARATION',
    'mode': args.mode, 'command': command, 'exit_code': code, 'wall_seconds': time.monotonic()-started,
    'windows_controller_cpu_seconds': time.process_time()-cpu, 'full_cpu_measurement': False,
    'cpu_gap': 'Windows WSL bridge process CPU not included in controller process time; Linux SELF+waited separately recorded',
    'frozen_contents_unchanged': before == after, 'before': before, 'after': after}
(root / (args.mode+'.stdout.txt')).write_text(stdout, encoding='utf-8')
(root / (args.mode+'.stderr.txt')).write_text(stderr, encoding='utf-8')
(root / (args.mode+'-evidence.json')).write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in evidence.items() if k not in ('before','after')}, ensure_ascii=False))
print(stdout[-12000:])
print(stderr[-2000:])
if before != after:
    raise SystemExit('FROZEN_PACKAGE_CHANGED')
raise SystemExit(code)
