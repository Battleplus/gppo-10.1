"""Aggregate existing engineering receipts and immutable evidence, never execute models."""
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent
DIAG = ROOT.parent.parent/'research-diagnostics/w1-drone-consequence-v3-candidate-contract-repair-20261008'


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
def save(path, value): path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n', encoding='utf-8')


def main():
    cases = []
    total_calls = {}
    for name in ('first-controlled-stop', 'second-controlled-stop', 'boundary-before-type-checks',
                 'before-keyed-lookup', 'final-controlled-evidence'):
        folder = DIAG/name
        dbpath = folder/'controlled-workspace/controlled-output/ledger.sqlite'
        db = sqlite3.connect(dbpath.as_uri()+'?mode=ro&immutable=1', uri=True)
        totals = {}; statuses = dict(db.execute('SELECT status,COUNT(*) FROM calls GROUP BY status'))
        counters = {}
        start = None; finish = None
        for stage, amounts, began, ended in db.execute('SELECT stage,amounts,started,finished FROM calls'):
            start = began if start is None else min(start, began)
            if ended is not None: finish = ended if finish is None else max(finish, ended)
            for k, v in json.loads(amounts).items():
                totals[k] = totals.get(k, 0)+v
                counters[(stage, k)] = counters.get((stage, k), 0)+v
        reservations = {(s, r): n for s, r, n in db.execute('SELECT stage,resource,units FROM reservations')}
        assert counters == reservations
        integrity = db.execute('PRAGMA integrity_check').fetchone()[0]
        assert integrity == 'ok'
        db.close()
        for k, v in totals.items(): total_calls[k] = total_calls.get(k, 0)+v
        receipt = folder/'controlled-resource-receipt.json'
        failure = folder/'acceptance-first-error.json'
        spec = read(folder/'controlled-workspace/RESOURCE_REQUEST.json')
        assert spec['approval_status'] == 'ENGINEERING_ONLY'
        assert all(totals[k] <= spec['totals'][k] for k in totals)
        assert totals.get('world_model_updates', 0) == totals.get('policy_updates', 0) == totals.get('world_model_batch_forwards', 0) == 0
        cases.append({'case': name, 'ledger_sha256': sha(dbpath), 'sqlite_integrity': integrity,
                      'status_counts': statuses, 'totals': totals, 'reservation_projection_matches': True,
                      'caught_first_error': read(failure) if failure.exists() else None,
                      'final_engineering_receipt': read(receipt) if receipt.exists() else None,
                      'CPU_scope_complete': receipt.exists(),
                      'observed_ledger_span_wall_seconds_lower_bound': finish-start,
                      'measured_CPU_if_missing_receipt': 'unknown; inclusive operation timings must not be summed',
                      'formal_research': False})
    assert total_calls['environment_steps'] == 3316
    inventory = read(DIAG/'controlled-input-inventory.json')
    assert all(sha(ROOT/name) == expected for name, expected in inventory['code_files'].items())
    receipts = [r['final_engineering_receipt'] for r in cases if r['final_engineering_receipt']]
    save(DIAG/'engineering-resource-accounting.json', {
        'cases': cases, 'total_ledger_reservations': total_calls,
        'synthetic_environment_steps': total_calls['environment_steps'],
        'neural_initializations_forwards_training_GPPO': 0, 'formal_environment_steps': 0,
        'known_completed_process_CPU_seconds_at_receipt_lower_bound': sum(r['process_CPU_seconds_including_import'] for r in receipts),
        'failed_process_CPU': 'unknown; not zero', 'whole_preparation_CPU_complete': False,
        'uncounted': ['failed-run final CPU', 'receipt/output/exit tails', 'PowerShell/WSL bridge and copy CPU'],
        'final_production_code_matches_acceptance_input_hashes': True,
        'controlled_cache_reuse': 'source-inventory/window SHA and full labels reverified; no formal results read in controlled acceptance',
        'end_to_end_scope': 'live production collection plus separately completed no-forward identity boundaries and authentic stage finalization',
        'scientific_metrics': '未评价'})
    files = {p.relative_to(DIAG).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(DIAG.rglob('*')) if p.is_file() and p.name != 'controlled-evidence-hashes.json'}
    save(DIAG/'controlled-evidence-hashes.json', {'files': files, 'excludes': ['controlled-evidence-hashes.json']})
    print(json.dumps({'cases': len(cases), 'synthetic_steps': total_calls['environment_steps'],
                      'code_matches_acceptance': True, 'neural_forwards': 0, 'GPPO_updates': 0}))


if __name__ == '__main__': main()
