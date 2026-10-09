"""Bounded local export; records incomplete copies instead of overwriting them."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time


def export(source,destination,wall_cap,cpu_cap,storage_cap):
    started=time.monotonic();cpu=time.process_time();size=0;files={}
    if wall_cap<=0 or cpu_cap<=0:raise ValueError('EXPORT_BUDGET_EXHAUSTED')
    resource.setrlimit(resource.RLIMIT_CPU,(max(1,int(cpu_cap)+1),max(2,int(cpu_cap)+2)))
    destination.parent.mkdir(parents=True,exist_ok=True);destination.mkdir(exist_ok=False)
    def guard():
        if time.monotonic()-started>wall_cap or time.process_time()-cpu>cpu_cap:
            raise TimeoutError('EXPORT_RESOURCE_CAP_EXCEEDED')
        if size>storage_cap:raise ValueError('EXPORT_STORAGE_CAP_EXCEEDED')
    try:
        for src in sorted(source.rglob('*')):
            if src.is_symlink():raise ValueError('EXPORT_SYMLINK_REJECTED')
            if not src.is_file() or '__pycache__' in src.parts:continue
            rel=str(src.relative_to(source));dst=destination/rel
            dst.parent.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256();count=0
            with src.open('rb') as inp,dst.open('xb') as out:
                while chunk:=inp.read(1024*1024):
                    guard();out.write(chunk);digest.update(chunk);size+=len(chunk);count+=len(chunk)
                out.flush();os.fsync(out.fileno())
            with dst.open('rb') as inp:
                verify=hashlib.sha256()
                while chunk:=inp.read(1024*1024):guard();verify.update(chunk)
            if verify.hexdigest()!=digest.hexdigest():raise ValueError('EXPORT_HASH_MISMATCH:'+rel)
            files[rel]={'bytes':count,'sha256':digest.hexdigest()};guard()
        evidence={'status':'verified_export','files':files,'bytes':size,
                  'wall_seconds':time.monotonic()-started,'CPU_seconds':time.process_time()-cpu,
                  'scope':'export child SELF; controller adds waited-child once','digest_excludes_self':True}
        (destination/'export-verification.json').write_text(json.dumps(evidence,sort_keys=True,indent=2)+'\n',encoding='utf-8')
        guard();return evidence
    except BaseException:
        import traceback
        (destination/'export-first-error.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
    p.add_argument('--wall',type=float,required=True);p.add_argument('--cpu',type=float,required=True)
    p.add_argument('--storage',type=int,required=True);a=p.parse_args()
    result=export(a.source,a.destination,a.wall,a.cpu,a.storage)
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))
