import hashlib,json,os,resource
from pathlib import Path
source=Path('/mnt/e/Z博士/research-plans/w1-g1-gppo-phase-accounting-repair-v1')
native=Path('/home/asus/w1-g1-gppo-phase-accounting-repair-v1-once')
manifest=json.loads((source/'package/execution-manifest.json').read_text())
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
result={'native_frozen_mismatches':[n for n,v in manifest['files'].items() if h(native/'package'/n)!=v],
        'archive_frozen_mismatches':[n for n,v in manifest['files'].items() if h(source/'frozen-archive/package'/n)!=v],
        'authorization_consumed':(native/'authorization-consumed.json').is_file(),
        'remaining_owned_processes':[]}
needle=str(native/'package')+'/'
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid():continue
    try:
        argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
        if any(a.startswith(needle) for a in argv):result['remaining_owned_processes'].append(int(p.name))
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
r=resource.getrusage(resource.RUSAGE_SELF)
result['readonly_self_cpu_seconds']=r.ru_utime+r.ru_stime
print(json.dumps(result))
