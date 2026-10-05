"""Run no-model Linux checks and preserve real invocation/output."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT/'package'
NATIVE_PYTHON='/home/asus/.venvs/w1-light-repaired-fair-rerun-py31116/bin/python'
MOUNT='/mnt/e/Z博士/research-plans/'+ROOT.name


def inventory():return {p.relative_to(PACKAGE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in PACKAGE.rglob('*') if p.is_file()}
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')


def run(name,args,timeout):
    command=['wsl','-d','Ubuntu-24.04','--','env','-u','LD_LIBRARY_PATH','-u','LD_PRELOAD',
             '-u','PYTHONPATH','-u','PYTHONHOME','PYTHONNOUSERSITE=1','CUDA_VISIBLE_DEVICES=',
             'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',NATIVE_PYTHON,
             '-I','-B','-X','pycache_prefix=/home/asus/.venvs/w1-light-repaired-fair-rerun-py31116/.w1-no-bytecode-cache',*args]
    start=time.monotonic()
    failure=None
    try:
        result=subprocess.run(command,capture_output=True,timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        failure=traceback.format_exc()
        result=subprocess.CompletedProcess(command,124,stdout=exc.stdout or b'',stderr=(exc.stderr or b'')+failure.encode())
    (ROOT/(name+'.stdout.raw')).write_bytes(result.stdout)
    (ROOT/(name+'.stderr.raw')).write_bytes(result.stderr)
    (ROOT/(name+'.stdout.txt')).write_text(result.stdout.decode('utf-8','replace'),encoding='utf-8')
    (ROOT/(name+'.stderr.txt')).write_text(result.stderr.decode('utf-8','replace'),encoding='utf-8')
    evidence={'command':command,'exit_code':result.returncode,'wall_seconds':time.monotonic()-start,'error':failure,
              'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(result.stderr).hexdigest()}
    write(ROOT/(name+'.invocation.json'),evidence)
    return evidence


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['unit','production'])
    parser.add_argument('--tag',default='development');args=parser.parse_args()
    before=inventory()
    if args.mode=='unit':
        script="""import ast,json,pathlib,sys,unittest,resource,os
root=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(root))
evidence=pathlib.Path(sys.argv[2]);evidence.mkdir(parents=True,exist_ok=True)
os.environ['W1_TEST_EVIDENCE_DIR']=str(evidence)
for p in root.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
names=['test_phase_handshake','test_worker_phase_accounting','test_supervise_settlement','test_linux_process_scope','test_supervised_phase_integration','test_runner_bootstrap']
suite=unittest.defaultTestLoader.loadTestsFromNames(names)
result=unittest.TextTestRunner(verbosity=2).run(suite)
a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN)
value={'success':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'model_or_environment_calls':0,'self_cpu':a.ru_utime+a.ru_stime,'waited_children_cpu':b.ru_utime+b.ru_stime}
print(json.dumps(value));raise SystemExit(0 if result.wasSuccessful() else 1)
"""
        evidence=run('accounting-regressions-'+args.tag,['-c',script,MOUNT+'/package',MOUNT+'/system-test-evidence-'+args.tag],240)
        raw=(ROOT/('accounting-regressions-'+args.tag+'.stdout.txt')).read_text(encoding='utf-8')
        try:summary=json.loads(raw.strip().splitlines()[-1])
        except (IndexError,json.JSONDecodeError):summary={'success':False,'summary_unavailable':True}
        summary.update(invocation=evidence,before=before,after=inventory(),bytes_unchanged=before==inventory())
        write(ROOT/'accounting-regressions-evidence.json',summary)
        print(json.dumps({k:v for k,v in summary.items() if k not in ('before','after')},ensure_ascii=False))
        if not summary['success']:raise SystemExit(1)
    else:
        native='/home/asus/w1-phase-cpu-repair-engineering-'+args.tag
        evidence=run('production-handshake-'+args.tag,[MOUNT+'/run_production_handshake_acceptance.py',MOUNT+'/package',native],240)
        copy=run('copy-handshake-'+args.tag,['-c',
            'import pathlib,shutil,sys; d=pathlib.Path(sys.argv[2]); assert not d.exists(); shutil.copytree(sys.argv[1],d)',
            native,MOUNT+'/engineering-evidence-'+args.tag],120)
        if copy['exit_code']:raise SystemExit(1)
        if evidence['exit_code']:
            print(json.dumps(evidence));raise SystemExit(1)
        witness=json.loads((ROOT/('engineering-evidence-'+args.tag)/'production-handshake-acceptance.json').read_text(encoding='utf-8'))
        write(ROOT/'production-handshake-acceptance.json',witness)
        print(json.dumps({'production_invocation':evidence,'native_evidence':native,'all_cases_pass':witness['all_cases_pass'],'source_unchanged':before==inventory()},ensure_ascii=False))


if __name__=='__main__':main()
