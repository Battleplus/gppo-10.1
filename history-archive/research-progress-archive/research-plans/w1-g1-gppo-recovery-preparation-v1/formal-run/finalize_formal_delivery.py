"""Finalize post-run review/archive metadata; stdlib only, never launch a worker."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
P = R / 'package'
A = R / 'formal-result-audit'
E = R / 'verified-export'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                            allow_nan=False) + '\n', encoding='utf-8')

def inventory(root):
    return {p.relative_to(root).as_posix(): sha(p)
            for p in root.rglob('*') if p.is_file()}

expected = read(R / 'audit/final-frozen-input-hashes.json')
assert inventory(P) == expected
assert inventory(R / 'frozen-archive/package') == expected
identity = read(R / 'frozen-identity.json')
for name, field in (('execution-manifest.json', 'execution_manifest_sha256'),
                    ('hashes.json', 'hashes_sha256'),
                    ('RESOURCE_REQUEST.json', 'resource_request_sha256')):
    assert sha(P / name) == identity[field]
assert read(P / 'RESOURCE_REQUEST.json')['status'] == 'NOT_APPROVED'
payload = read(E / 'export-hashes.json')
assert len(payload) == 35
assert all(sha(E / rel) == digest for rel, digest in payload.items())
assert set(inventory(E)) - {'export-hashes.json'} == set(payload)
old = read(R / 'audit/recovery-audit.json')
oldroot = Path(old['source_root'])
oldpaths = {
    'execution_manifest_sha256': 'package/execution-manifest.json',
    'experiment_matrix_sha256': 'package/experiment-matrix.json',
    'policy_routes_sha256': 'verified-export/run-once/policy-training-routes.jsonl',
    'sqlite_sha256': 'verified-export/run-once/budget.sqlite3',
    'resource_settlement_sha256': 'verified-export/run-once/resource-settlement.json',
    'export_hashes_sha256': 'verified-export/export-hashes.json',
}
assert all(sha(oldroot / rel) == old['inputs'][key] for key, rel in oldpaths.items())
assert not (R / 'external-token.txt').exists()
assert read(R / 'launch-evidence.json')['exit_code'] == 0
review = read(A / 'luna-final-review.json')
budget = review['ledger_and_cumulative_resources']
contract = read(P / 'cumulative-resource-contract.json')
assert budget['new_attempt_wall_cap_seconds'] == contract['new_wall_cap'] == 4670
assert budget['new_launch_wall_seconds'] < contract['new_wall_cap']
assert budget['authorized_cumulative_measured_old_wall_plus_new_cap_seconds'] == contract['cumulative_measured_old_wall_plus_new_cap']
assert budget['cumulative_wall_within_authorized_envelope']
assert review['conclusion']['task_gate_pass'] is False
assert review['conclusion']['cost_gate_pass'] is False

statuspath = R / 'formal-git-archive-status.json'
status = read(statuspath)
D = Path(status['local_archive'])
assert D.is_dir() and not status['remote_archived']
prior_manifest = read(D / 'archive-hashes.json')
preserved = {}
for rel in ('formal-result-audit/luna-final-review.json',
            'formal-result-audit/luna-final-review.md'):
    destination = D / rel
    assert sha(destination) == prior_manifest[rel]
    if sha(destination) != sha(R / rel):
        prior = D / 'superseded-budget-scope-review' / Path(rel).name
        assert not prior.exists()
        prior.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(destination, prior)
        preserved[prior.relative_to(D).as_posix()] = sha(prior)
        shutil.copyfile(R / rel, destination)
    assert sha(destination) == sha(R / rel)
erratum = 'formal-result-audit/luna-final-review-budget-scope-erratum.md'
assert not (D / erratum).exists()
shutil.copyfile(R / erratum, D / erratum)
status['budget_scope_erratum_synced'] = True
status['superseded_review_preserved'] = preserved
status['files'].update({rel: sha(R / rel) for rel in (
    'formal-result-audit/luna-final-review.json',
    'formal-result-audit/luna-final-review.md', erratum)})
status['files'].update(preserved)
status['no_additional_git_retry'] = True
write(statuspath, status)

report = R / 'formal-run-report.md'
addition = '''
## 最终独立复核及归档状态

Luna已完成九路模型、240 episode、收益门、成本与账本的制品复核。其初稿曾误将旧新累计wall与本次4670秒上限比较；已按冻结累计合同勘误：本次1175.926秒通过，新旧累计4805.325秒低于旧实测加新批准上限8299.399秒。full_resource_acceptance=false仍来自已披露计量缺口与旧pending，不是本次wall超限。初稿与勘误均保存在本地归档。

本地小型归档位于 research-progress-archive/research-plans/w1-g1-gppo-recovery-preparation-v1/formal-run。Git add因既有index.lock退出128；没有删除锁、关闭签名或触碰907项其他暂存内容。远端ls-remote超时，未创建新提交、未核验远端归档；状态明确为“远端未归档”。本地材料不包含token、授权秘密、checkpoint、SQLite或大型原始日志。
'''
assert '## 最终独立复核及归档状态' not in report.read_text(encoding='utf-8')
report.write_text(report.read_text(encoding='utf-8') + addition, encoding='utf-8')
shutil.copyfile(report, D / report.name)
status['files'][report.name] = sha(report)
write(statuspath, status)
shutil.copyfile(statuspath, D / statuspath.name)

result = {
    'attempt': 'w1-g1-gppo-recovery-v1-once',
    'frozen_227_files_and_preparation_archive_unchanged': len(expected) == 227,
    'export_35_payload_hashes_unchanged': True,
    'old_failure_evidence_and_settlement_unchanged': True,
    'luna_review_completed_and_budget_scope_corrected': True,
    'plaintext_token_absent': True,
    'formal_exit_code': 0,
    'task_episodes': 240,
    'research_status': 'task_gain_gate_stop',
    'task_gate_pass': False,
    'G1_cost_gate_pass': False,
    'new_measured_wall_cap_pass': True,
    'full_resource_acceptance': False,
    'remote_archived': False,
    'no_experiment_restart': True,
    'this_finalizer_environment_or_model_calls': 0,
    'torch_imported': 'torch' in sys.modules,
}
write(A / 'final-post-review-verification.json', result)
shutil.copyfile(A / 'final-post-review-verification.json',
                D / 'formal-result-audit/final-post-review-verification.json')
files = {p.relative_to(R).as_posix(): sha(p) for p in A.rglob('*') if p.is_file()}
files.update({name: sha(R / name) for name in (
    'formal-run-report.md', 'launch-evidence.json', 'native-controller-settlement.json',
    'formal-git-archive-status.json', 'token-cleanup.json', 'finalize_formal_delivery.py',
    'verified-export/export-hashes.json')})
write(R / 'formal-delivery-hashes.json', files)
shutil.copyfile(R / 'formal-delivery-hashes.json', D / 'formal-delivery-hashes.json')
shutil.copyfile(Path(__file__), D / Path(__file__).name)
archivefiles = inventory(D)
archivefiles.pop('archive-hashes.json', None)
write(D / 'archive-hashes.json', archivefiles)
assert all(sha(D / rel) == digest for rel, digest in archivefiles.items())
assert all(sha(R / rel) == digest for rel, digest in files.items())
assert inventory(P) == expected
print(json.dumps(result, ensure_ascii=False))
