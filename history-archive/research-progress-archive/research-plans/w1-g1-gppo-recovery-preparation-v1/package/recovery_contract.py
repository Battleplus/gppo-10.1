"""Stdlib-only recovery contract. Reuse is identity based, never score based."""
from pathlib import Path
import hashlib
import json

REUSE_KEYS = {('G0',s) for s in (8301,8302,8303)} | {('T',s) for s in (8301,8302,8303)} | {('G1',8301)}
TRAIN_KEYS = {('G1',8302),('G1',8303)}

class RecoveryContractError(RuntimeError): pass

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def training_route_keys(matrix):
    if 'recovery' not in matrix:
        return { (m,s) for m in ('G0','T','G1') for s in (8301,8302,8303) }
    config=matrix['recovery']
    if config != {'enabled':True,'contract_file':'recovery-contract.json','mode':'reuse_final_seven_restart_missing_two'}:
        raise RecoveryContractError('RECOVERY_MODE_INVALID')
    return set(TRAIN_KEYS)

def verify_recovery(root, matrix, request):
    root=Path(root).resolve()
    if 'recovery' not in matrix:
        return None
    training_route_keys(matrix)
    c=read(root/'recovery-contract.json')
    if c.get('schema')!='w1-final-policy-reuse/1.0.0' or c.get('attempt')!=matrix.get('attempt') or c.get('attempt')!=request.get('attempt'):
        raise RecoveryContractError('RECOVERY_IDENTITY_MISMATCH')
    if c.get('train_routes')!=[{'method':'G1','seed':8302},{'method':'G1','seed':8303}] or c.get('partial_8302_resume')!='not_exactly_recoverable_restart_from_seed':
        raise RecoveryContractError('RECOVERY_TRAIN_SELECTION_MISMATCH')
    scientific={k:v for k,v in matrix.items() if k not in ('attempt','recovery')}
    old=read(root/c['source_matrix_file'])
    old={k:v for k,v in old.items() if k!='attempt'}
    if scientific!=old or canonical(scientific)!=c['scientific_matrix_sha256']:
        raise RecoveryContractError('RECOVERY_SCIENCE_CHANGED')
    for name,digest in c['unchanged_inputs'].items():
        if sha(root/name)!=digest:
            raise RecoveryContractError('RECOVERY_INPUT_CHANGED:'+name)
    routes=c.get('reuse_routes',[])
    keys=[(x['method'],x['seed']) for x in routes]
    if len(keys)!=7 or set(keys)!=REUSE_KEYS:
        raise RecoveryContractError('RECOVERY_REUSE_ROUTE_SET_INVALID')
    for row in routes:
        key=(row['method'],row['seed'])
        path=(root/row['checkpoint_relative_path']).resolve()
        if not path.is_relative_to(root/'frozen-models'/'reused-policy') or not path.is_file() or sha(path)!=row['checkpoint_sha256']:
            raise RecoveryContractError('RECOVERY_CHECKPOINT_IDENTITY_INVALID:'+str(key))
        if row['training_steps']!=2048 or row['optimizer_updates']!=128 or row['ppo_update_count']!=128:
            raise RecoveryContractError('RECOVERY_CHECKPOINT_NOT_FINAL:'+str(key))
        if row['ledger_completed_steps']!=2048 or row['ledger_completed_updates']!=128 or row['ledger_completed_saves']!=1 or row['ledger_pending_calls']!=0:
            raise RecoveryContractError('RECOVERY_SAVE_NOT_CONFIRMED:'+str(key))
        if row['method']=='G1':
            binding=read(root/'g1-model-binding.json')['routes'][0]
            if row['world_model_seed']!=8201 or row['world_model_variant']!='G1' or row['world_checkpoint_sha256']!=binding['checkpoint_sha256']:
                raise RecoveryContractError('RECOVERY_WORLD_BINDING_MISMATCH')
        elif row['world_model_variant'] is not None or row['world_model_seed'] is not None:
            raise RecoveryContractError('RECOVERY_UNEXPECTED_WORLD_MODEL')
    caps=request['stages']['conditional_policy_training']
    for name,expected in {'environment_steps':4096,'policy_optimizer_updates':256,'policy_backward_calls':256,'checkpoint_writes':2,
                         'checkpoint_loads':3,'model_initializations_or_loads':5}.items():
        if caps.get(name)!=expected:
            raise RecoveryContractError('RECOVERY_REMAINING_BUDGET_MISMATCH:'+name)
    original=read(root/c['source_request_file'])['stages']['conditional_task_confirmation']
    if request['stages']['conditional_task_confirmation']!=original:
        raise RecoveryContractError('RECOVERY_TASK_BUDGET_CHANGED')
    if request['status']!='NOT_APPROVED':
        raise RecoveryContractError('RECOVERY_REQUEST_STATUS_CHANGED')
    return c

def reused_route_records(root, matrix, request):
    c=verify_recovery(root,matrix,request)
    if c is None:return {}
    output={}
    for row in c['reuse_routes']:
        record={k:row[k] for k in ('method','seed','training_steps','optimizer_updates','checkpoint_sha256',
                                 'checkpoint_state_sha256','world_model_variant','world_model_seed','ppo_update_count')}
        record.update({'checkpoint':str((Path(root)/row['checkpoint_relative_path']).resolve()),
                       'rollout_steps':64,'updates_per_rollout':4,'source_attempt':c['source_attempt'],
                       'reuse':True,'summary':row['summary']})
        output[(row['method'],row['seed'])]=record
    return output

def validate_checkpoint_location(root, output, route):
    resolved=Path(route['checkpoint']).resolve()
    allowed=resolved.is_relative_to(Path(output).resolve())
    if route.get('reuse') is True:
        allowed=resolved.is_relative_to(Path(root).resolve()/'frozen-models'/'reused-policy')
    if not allowed or not resolved.is_file() or sha(resolved)!=route['checkpoint_sha256']:
        raise RecoveryContractError('RECOVERY_FINAL_CHECKPOINT_PATH_OR_HASH_INVALID')
    return resolved
