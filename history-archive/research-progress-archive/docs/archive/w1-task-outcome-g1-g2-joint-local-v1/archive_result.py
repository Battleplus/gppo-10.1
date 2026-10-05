import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
REPO=Path('E:/Z博士/research-progress-archive')
DEST=REPO/'docs/archive/w1-task-outcome-g1-g2-joint-local-v1'
assert not DEST.exists(),'ARCHIVE_ALREADY_EXISTS'
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args])
def index():return {row.split(b'\t',1)[1]:row for row in git('ls-files','--stage','-z').split(b'\0') if row}
before=index();copies={}
staged_before=git('diff','--cached','--name-only','-z').split(b'\0')
names=['RESEARCH_REPORT.md','RESEARCH_RESULT.json','final-frozen-identity.json','migration-audit.json',
    'offline-preparation-evidence.json','entry-adaptation-no-model-evidence.json','windows-preflight-evidence.json',
    'native-preflight-evidence.json','native-preflight.stdout.txt','native-preflight.stderr.txt',
    'launch-evidence.json','launch.stdout.txt','launch.stderr.txt','native-controller-settlement.json',
    'independent-metric-recalculation.json','postrun-cleanup-evidence.json',
    'postrun-independent-review.md','final-artifact-verification.json','FINAL_RESOURCE_SUMMARY.json',
    'postrun-native-probe.stdout.txt','postrun-native-probe.stderr.txt','independent_analysis.py',
    'assemble_final_report.py','finalize_evidence.py','postrun_native_probe.py',
    'verified-export/export-hashes.json','verified-export/supervisor-status.json',
    'verified-export/run-once/status.json','verified-export/run-once/data-admission.json',
    'verified-export/run-once/prediction-metrics.json','verified-export/run-once/world-model-training-summary.json',
    'verified-export/run-once/world-model-training-progress.json','verified-export/run-once/resource-settlement.json',
    'verified-export/run-once/local-runtime-execution.json',
    'verified-export/run-once/environment.json','verified-export/run-once/runtime-input-check.json',
    'verified-export/run-once/label-coverage-summary.json',
    'package/local_research_entry.py','package/local_research_worker.py','package/local_pipeline_controller.py',
    'package/run_local_research.py','package/local_runtime.py','package/supervise.py','package/runner.py',
    'package/joint_pipeline.py','package/production_data.py','package/production_world.py',
    'package/LOCAL_REAL_RESEARCH.md','package/LOCAL_ACCOUNTING_CONTRACT.json','package/RESOURCE_REQUEST.json',
    'package/experiment-matrix.json','package/parent-split.json','package/launch-contract.json',
    'package/data-gate.json','package/prediction-gates.json','package/execution-manifest.json','package/hashes.json',
    'package/unique-launch-command.md','summarize_result.py','offline_prepare_and_test.py','archive_result.py']
for name in names:
    source=ROOT/name
    if not source.exists():continue
    if source.stat().st_size>500000:continue
    assert source.suffix not in ('.pt','.sqlite3')
    target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    assert source.read_bytes()==target.read_bytes()
    copies[name]=hashlib.sha256(target.read_bytes()).hexdigest()
review=ROOT/'local-independent-review.md'
if not review.exists():review=Path('E:/Z博士/local-independent-review.md')
if review.exists():
    shutil.copy2(review,DEST/'local-independent-review.md');copies['local-independent-review.md']=hashlib.sha256(review.read_bytes()).hexdigest()
(DEST/'archive-copy-hashes.json').write_text(json.dumps(copies,indent=2)+'\n',encoding='utf-8')
git('add','--',DEST.relative_to(REPO).as_posix());after=index()
assert all(after.get(path)==row for path,row in before.items()),'EXISTING_INDEX_ENTRY_CHANGED'
status={'local_archive':str(DEST),'files':len(copies),'all_copy_hashes_verified':True,'existing_staged_entries_preserved':len(before),
    'existing_staged_paths_preserved':len([p for p in staged_before if p]),
    'signed_commit_created':False,'remote_archived':False,
    'blocker':'ssh-agent Stopped/Disabled; SSH signing required (commit.gpgsign=true, gpg.format=ssh); requirements unchanged',
    'exclusions':'authorization/token/credentials/checkpoints/SQLite/raw tapes/large traces',
    'archive_kind':'small code/protocol/result snapshot; full executable local package and research artifacts remain at original local paths'}
(ROOT/'archive-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(status,ensure_ascii=False))
