"""Audit stopped attempt read-only; never import research modules or torch."""
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'package'
EXPORT = ROOT / 'verified-export'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix(): digest(p)
            for p in root.rglob('*') if p.is_file()}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    expected = read(ROOT / 'final-delivery-evidence.json')['after']
    source = inventory(PACKAGE)
    archive = inventory(ROOT / 'frozen-archive' / 'package')
    exported = read(EXPORT / 'export-hashes.json')
    export_before = inventory(EXPORT)
    export_mismatches = [n for n, h in exported.items()
                         if not (EXPORT / n).is_file() or digest(EXPORT / n) != h]
    export_inventory_mismatches = sorted(set(export_before) ^ (set(exported) | {'export-hashes.json'}))
    db = EXPORT / 'run-once' / 'budget.sqlite3'
    connection = sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)
    connection.execute('PRAGMA query_only=ON')
    rows = connection.execute('SELECT stage,name,amounts,status,error FROM calls').fetchall()
    integrity = connection.execute('PRAGMA integrity_check').fetchall()
    connection.close()
    totals = {}
    statuses = {}
    for stage, name, amounts, status, error in rows:
        statuses[status] = statuses.get(status, 0) + 1
        for key, amount in json.loads(amounts).items():
            totals[key] = totals.get(key, 0) + amount
    native_code = '''import hashlib,json,pathlib,sys
payload=json.load(sys.stdin)
r=pathlib.Path('/home/asus/w1-g1-gppo-task-validation-local-v1-once/package')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
input_bad=[n for n,h in payload['inputs'].items() if not (r/n).is_file() or sha(r/n)!=h]
export_bad=[]
for n,h in payload['export'].items():
 p=(r.parent/n) if n in ['native-launch.stdout.txt','controller-first-error.txt','authorization-consumed.json'] else r/n
 if not p.is_file() or sha(p)!=h: export_bad.append(n)
live=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit(): continue
 try: argv=(p/'cmdline').read_bytes().split(b'\\0')
 except (OSError,PermissionError): continue
 if any(a.startswith(str(r).encode()+b'/') and a.endswith(b'.py') for a in argv):
  live.append({'pid':int(p.name),'executable':argv[0].decode(errors='replace')})
marker=json.loads((r.parent/'authorization-consumed.json').read_text())
print(json.dumps({'native_frozen_file_count':len(payload['inputs']),
 'native_frozen_identity_mismatches':input_bad,'native_export_identity_mismatches':export_bad,
 'live_attempt_processes':live,'authorization_consumed':True,
 'consumed_attempt':marker['attempt'],'supervisor_started_marker':marker['supervisor_started']}))
'''
    command = ['wsl', '-d', 'Ubuntu-24.04', '--',
               '/home/asus/.venvs/w1-light-repaired-fair-rerun-py31116/bin/python',
               '-I', '-B', '-c', native_code]
    started = time.monotonic()
    result = subprocess.run(command, input=json.dumps({'inputs': expected, 'export': exported}).encode(),
                            capture_output=True, timeout=60)
    (ROOT / 'formal-stop-native-audit.stdout.raw').write_bytes(result.stdout)
    (ROOT / 'formal-stop-native-audit.stderr.raw').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError('NATIVE_READ_ONLY_AUDIT_FAILED:' + result.stderr.decode('utf-8', 'replace'))
    native = json.loads(result.stdout.decode('utf-8'))
    controller = read(ROOT / 'native-controller-settlement.json')
    windows = read(ROOT / 'launch-evidence.json')
    supervisor = read(EXPORT / 'supervisor-status.json')
    evidence = {
        'schema': 'w1-g1-gppo-formal-technical-stop-audit/1.0.0',
        'attempt': windows['attempt'], 'formal_exit_code': windows['exit_code'],
        'stop_reason': supervisor['stop_reason'],
        'package_file_count': len(source), 'source_matches_approved_final_delivery': source == expected,
        'full_frozen_archive_matches_source': archive == source,
        'export_hash_count': len(exported), 'export_hash_manifest_sha256': digest(EXPORT / 'export-hashes.json'),
        'export_identity_mismatches': export_mismatches,
        'export_inventory_mismatches': export_inventory_mismatches,
        'export_unchanged_by_read_only_audit': export_before == inventory(EXPORT),
        'sqlite_open_mode': 'ro', 'sqlite_integrity': integrity,
        'ledger_call_count': len(rows), 'ledger_call_status_counts': statuses,
        'ledger_resource_totals': totals, 'pending_calls': statuses.get('pending', 0),
        'failed_calls': statuses.get('failed', 0),
        'environment_steps': totals.get('environment_steps', 0),
        'policy_optimizer_updates': totals.get('policy_optimizer_updates', 0),
        'world_batch_forwards': totals.get('world_batch_forwards', 0),
        'model_initializations_or_loads': totals.get('model_initializations_or_loads', 0),
        'checkpoint_loads': totals.get('checkpoint_loads', 0),
        'checkpoint_writes': totals.get('checkpoint_writes', 0),
        'task_episodes': totals.get('task_episodes', 0),
        'worker_status_retained': read(EXPORT / 'run-once' / 'status.json')['status'],
        'worker_final_resource_settlement_present': (EXPORT / 'run-once' / 'resource-settlement.json').exists(),
        'settlement_scope': 'supervisor and outer controller records plus read-only ledger reconciliation; worker final settlement missing',
        'controller_native_cpu_seconds': controller['self_plus_waited_cpu_seconds'],
        'windows_controller_cpu_seconds': windows['windows_controller_cpu_seconds'],
        'known_cpu_sum_without_nested_double_count_seconds': controller['self_plus_waited_cpu_seconds'] + windows['windows_controller_cpu_seconds'],
        'known_cpu_sum_is_complete_lifecycle_cpu': False,
        'supervisor_nested_cpu_seconds_not_added': supervisor['cpu_seconds'],
        'launch_wall_seconds': windows['wall_seconds'],
        'supervisor_peak_rss_upper_bound_bytes': supervisor['peak_rss_upper_bound_bytes'],
        'active_native_bytes': controller['active_native_bytes'],
        'verified_export_bytes': controller['verified_export_bytes'],
        'measured_controller_limits_pass': controller['measured_controller_limits_pass'],
        'full_resource_acceptance': False,
        'resource_gaps': controller['resource_gaps'] + [
            'Windows launcher timing starts after its package/authorization verification; PowerShell startup and prior checks unmeasured'],
        'formal_runner_readiness_disposition': 'not_ready_after_observed_phase_handshake_failure',
        'runner_ready': False,
        'all_task_and_cost_effects': 'not_evaluated',
        'research_code_modified': False, 'automatic_retry': False,
        'offline_audit_model_or_environment_calls': 0,
        'native_audit_exit_code': result.returncode, 'native_audit_wall_seconds': time.monotonic() - started,
        **native,
    }
    (ROOT / 'formal-stop-audit.json').write_text(json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=False))
    if (source != expected or archive != source or export_mismatches or export_inventory_mismatches
            or native['native_frozen_identity_mismatches'] or native['native_export_identity_mismatches']
            or native['live_attempt_processes']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
