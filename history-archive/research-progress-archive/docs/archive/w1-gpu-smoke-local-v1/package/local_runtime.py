import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def gpu_snapshot():
    command = ['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total,memory.free', '--format=csv,noheader,nounits']
    result = subprocess.run(command, text=True, capture_output=True, timeout=10, check=True)
    rows = [row.split(', ') for row in result.stdout.strip().splitlines()]
    if len(rows) != 1 or rows[0][0] != '0':
        raise RuntimeError('LOCAL_GPU_INVENTORY_MISMATCH')
    row = rows[0]
    return {'gpu_physical_device': 0, 'gpu_uuid': row[1], 'name': row[2], 'driver': row[3],
            'total_bytes': int(row[4])*1024**2, 'free_bytes': int(row[5])*1024**2,
            'owned_process_memory_bytes': None, 'exclusive_allocation_proven': False,
            'scope': 'shared Windows desktop GPU; per-process WDDM/WSL accounting unavailable'}

def snapshot():
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode or not sys.flags.no_user_site:
        raise RuntimeError('LOCAL_ISOLATED_INTERPRETER_FLAGS_REQUIRED')
    cache = Path(sys.prefix) / '.w1-no-bytecode-cache'
    if os.environ.get('PYTHONNOUSERSITE') != '1' or sys.pycache_prefix != str(cache) or cache.exists():
        raise RuntimeError('LOCAL_BYTECODE_ISOLATION_REQUIRED')
    version = f'python{sys.version_info.major}.{sys.version_info.minor}'
    allowed = {str(Path(sys.base_prefix) / 'lib' / part) for part in
        (version, f'python{sys.version_info.major}{sys.version_info.minor}.zip', version+'/lib-dynload')}
    allowed.update((str(Path(sys.prefix)/'lib'/version/'site-packages'), str(ROOT), str(ROOT/'native')))
    if set(sys.path) - allowed:
        raise RuntimeError('LOCAL_IMPORT_SEARCH_PATH_OUTSIDE_RUNTIME_AND_PACKAGE')
    if any(os.environ.get(key) for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH', 'PYTHONHOME')):
        raise RuntimeError('LOCAL_IMPLICIT_RUNTIME_PATH_FORBIDDEN')
    import torch
    distributions = {}
    for d in importlib.metadata.distributions():
        record = Path(d._path) / 'RECORD'
        distributions[d.metadata['Name']] = {'version': d.version,
            'record_sha256': digest(record) if record.is_file() else None}
    if torch.cuda.is_initialized():
        raise RuntimeError('LOCAL_PREFLIGHT_MUST_NOT_INITIALIZE_CUDA')
    return {'python_executable': sys.executable, 'python_prefix': sys.prefix,
        'python': platform.python_version(), 'python_binary_sha256': digest(Path(sys.executable).resolve()),
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'torch_module': str(Path(torch.__file__).resolve()),
        'torch_module_sha256': digest(torch.__file__), 'installed_distributions': distributions,
        'cuda_initialized': False}

def verify_runtime(request):
    actual = snapshot()
    for key in ('python_executable', 'python', 'torch', 'cuda'):
        if actual[key] != request['runtime'][key]:
            raise RuntimeError('LOCAL_RUNTIME_MISMATCH:' + key)
    expected = json.loads((ROOT / 'local-runtime-identity.json').read_text(encoding='utf-8'))
    if actual != expected:
        raise RuntimeError('LOCAL_RUNTIME_IDENTITY_CHANGED')
    if not Path(actual['torch_module']).is_relative_to(Path(sys.prefix)):
        raise RuntimeError('LOCAL_TORCH_OUTSIDE_EXPLICIT_RUNTIME')
    return actual
