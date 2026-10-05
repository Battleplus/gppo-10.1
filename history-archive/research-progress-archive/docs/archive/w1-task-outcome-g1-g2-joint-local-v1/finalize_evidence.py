"""Artifact-only final checks. Does not launch a research entry or import torch."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
def read(p):
    return json.loads(p.read_text(encoding='utf-8'))
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name, value):
    (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

native = subprocess.run(['wsl','-d','Ubuntu-24.04','--exec',
    '/home/asus/.venvs/w1-light-repaired-fair-rerun-py31116/bin/python','-I','-B',
    '/mnt/e/Z博士/research-plans/w1-task-outcome-g1-g2-joint-local-v1/postrun_native_probe.py'],
    stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
(ROOT/'postrun-native-probe.stdout.txt').write_bytes(native.stdout)
(ROOT/'postrun-native-probe.stderr.txt').write_bytes(native.stderr)
assert native.returncode == 0, 'NATIVE_POSTRUN_PROBE_FAILED'
probe = json.loads(native.stdout.decode('utf-8'))
gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.free,utilization.gpu',
    '--format=csv,noheader'],capture_output=True,text=True,timeout=15)
write('postrun-cleanup-evidence.json', {'timestamp_unix':time.time(),'native_probe_exit_code':native.returncode,
    'native_probe':probe,'gpu_inventory_exit_code':gpu.returncode,'gpu_inventory':gpu.stdout.strip(),
    'gpu_memory_scope':'whole desktop GPU inventory, not per-process or exclusive-memory proof',
    'processes_killed':0,'environment_calls':0,'model_calls':0})

package=ROOT/'package'
frozen=read(ROOT/'final-frozen-identity.json')
outer={name:digest(package/name) for name in ('execution-manifest.json','hashes.json','RESOURCE_REQUEST.json','local-runtime-identity.json')}
assert all(outer[n]==frozen[n] for n in outer)
hashes=read(package/'hashes.json')
content={n:digest(package/n) for n in hashes['files']}
assert content==hashes['files']
assert outer['execution-manifest.json']==hashes['execution_manifest_sha256']
exports=read(ROOT/'verified-export/export-hashes.json')
assert all(digest(ROOT/'verified-export'/n)==h for n,h in exports.items())
training=read(ROOT/'verified-export/run-once/world-model-training-summary.json')
assert len(training['routes'])==6
checkpoints={r['checkpoint_relative_path']:digest(ROOT/'verified-export/run-once'/r['checkpoint_relative_path']) for r in training['routes']}
assert all(checkpoints[r['checkpoint_relative_path']]==r['checkpoint_sha256'] for r in training['routes'])
write('final-artifact-verification.json', {'status':'pass','timestamp_unix':time.time(),
    'frozen_identity':outer,'content_files_checked':len(content),'windows_frozen_contents_unchanged':True,
    'native_frozen_contents_unchanged':True,'export_files_checked':len(exports),'export_hashes_match':True,
    'checkpoint_files_checked':6,'checkpoint_sha256':checkpoints,'checkpoint_loads_for_audit':0,
    'environment_calls':0,'model_calls':0})
controller=read(ROOT/'native-controller-settlement.json')
launcher=read(ROOT/'launch-evidence.json')
supervisor=read(ROOT/'verified-export/supervisor-status.json')
worker=read(ROOT/'verified-export/run-once/resource-settlement.json')
request=read(package/'RESOURCE_REQUEST.json')
actual=worker['ledger']['totals']
counts={n:{'actual':v,'limit':request['totals'][n],'pass':v<=request['totals'][n]} for n,v in actual.items()}
assert all(r['pass'] for r in counts.values())
write('FINAL_RESOURCE_SUMMARY.json', {'status':'measured_limits_pass_with_disclosed_gaps',
    'full_resource_acceptance':False,'attempt':controller['attempt'],'counts':counts,
    'failed_calls':worker['ledger']['failed_calls'],'pending_calls':worker['ledger']['pending_calls'],
    'gppo_updates':0,'task_comparison_calls':0,
    'wall_seconds':{'native_controller':controller['wall_seconds'],'windows_launcher':launcher['wall_seconds'],
        'conservative_max':max(controller['wall_seconds'],launcher['wall_seconds']),
        'limit':request['totals']['wall_seconds'],'cross_clock_difference_seconds':controller['wall_seconds']-launcher['wall_seconds'],
        'cross_clock_difference_source':'来源未确定；保留两个时钟域原值，不校正或归为噪声'},
    'cpu_seconds':{'native_self':controller['self_cpu_seconds'],'native_waited_children':controller['waited_children_cpu_seconds'],
        'native_self_plus_waited':controller['self_plus_waited_cpu_seconds'],
        'windows_controller':launcher['windows_controller_cpu_seconds'],
        'measured_disjoint_sum':controller['self_plus_waited_cpu_seconds']+launcher['windows_controller_cpu_seconds'],
        'supervisor_nested_do_not_add':supervisor['cpu_seconds'],'limit':request['totals']['complete_process_cpu_seconds'],
        'counting':'native SELF + kernel waited CHILDREN; supervisor/worker nested totals are breakdown only'},
    'stage_boundaries':supervisor['phase_rows'],'controller_phases':controller['controller_phases'],
    'rss_upper_bound_peak_bytes':supervisor['peak_rss_upper_bound_bytes'],
    'native_active_bytes':controller['active_native_bytes'],'verified_export_bytes':controller['verified_export_bytes'],
    'aggregate_storage_bytes':controller['active_native_bytes']+controller['verified_export_bytes'],
    'resource_gaps':controller['resource_gaps']+['Auxiliary operator audit/monitor processes outside the research process tree are not included'],
    'practical_decision_cost':'未评价；无GPPO任务决策，不能判10ms CPU/50ms wall门通过'})
print(json.dumps({'status':'pass','frozen_files':len(content),'export_files':len(exports),
    'checkpoints':6,'task_processes_remaining':probe['task_processes_remaining']},ensure_ascii=False))
