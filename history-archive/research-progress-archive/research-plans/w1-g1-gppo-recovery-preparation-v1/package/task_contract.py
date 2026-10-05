"""Pure identity validation for the four-arm task-only proposal."""
import hashlib
import json
from pathlib import Path
class TaskContractError(RuntimeError):pass
def read(p):
    value=json.loads(p.read_text(encoding='utf-8'))
    if not isinstance(value,dict):raise TaskContractError('CONTRACT_OBJECT_REQUIRED:'+p.name)
    return value
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def _verify_task_inputs(root):
    root=Path(root).resolve()
    from verify_runtime_inputs import verify_runtime_inputs
    source_identity=verify_runtime_inputs(root)
    matrix=read(root/'experiment-matrix.json');request=read(root/'RESOURCE_REQUEST.json')
    split=read(root/'parent-split.json');binding=read(root/'g1-model-binding.json')
    if len({matrix.get('attempt'),request.get('attempt'),split.get('attempt')})!=1:raise TaskContractError('TASK_ATTEMPT_MISMATCH')
    if matrix.get('methods')!=['G0','T','G1','H'] or matrix.get('policy_seeds')!=[8301,8302,8303]:raise TaskContractError('TASK_ROUTE_SET_MISMATCH')
    if matrix.get('world_model_device')!='cpu' or request.get('gpu') is not None or request['totals']['gpu_devices']!=0:raise TaskContractError('FIXED_CPU_DEVICE_REQUIRED')
    if matrix.get('e2e_fixture',{}).get('enabled'):raise TaskContractError('FORMAL_REJECTS_TEST_PROFILE')
    if matrix.get('cpu_execution',{}).get('affinity')!=[0] or matrix['cpu_execution'].get('torch_threads')!=1:raise TaskContractError('CPU_CONDITION_MISMATCH')
    tape=root/'native/source-evidence/tapes-train.json'
    if sha(tape)!=split.get('source_tape_sha256'):raise TaskContractError('TASK_TAPE_DIGEST_MISMATCH')
    rows=json.loads(tape.read_text(encoding='utf-8'));identities=set()
    for role,indices in [('train',range(64,88)),('task_confirmation',range(104,112))]:
        frozen=split.get(role)
        if not isinstance(frozen,list) or [r.get('parent') for r in frozen]!=[f'train-{i:04d}' for i in indices]:raise TaskContractError('FIXED_TASK_PARENT_SET_MISMATCH:'+role)
        if matrix['splits'].get(role)!=frozen:raise TaskContractError('MATRIX_PARENT_REFERENCE_MISMATCH:'+role)
        for r in frozen:
            matching=[x for x in rows if x['parent']==r['parent']]
            if len(matching)!=1:raise TaskContractError('TASK_PARENT_NOT_UNIQUE')
            source=matching[0]
            if canonical(source['scenario'])!=r['scenario_sha256'] or any(source.get(k)!=v for k,v in r.items()):raise TaskContractError('TASK_PARENT_CONTENT_MISMATCH:'+r['parent'])
            if r['scenario_sha256'] in identities or r['structural_sha256'] in identities:raise TaskContractError('TASK_PARENT_IDENTITY_COLLISION')
            identities.update((r['scenario_sha256'],r['structural_sha256']))
    if binding.get('schema')!='w1-g1-model-binding/1.0.0':raise TaskContractError('G1_BINDING_SCHEMA_INVALID')
    routes=binding.get('routes',[])
    if [(r.get('policy_seed'),r.get('world_seed')) for r in routes]!=[(8301,8201),(8302,8202),(8303,8203)]:raise TaskContractError('G1_MODEL_PAIRING_MISMATCH')
    for r in routes:
        path=(root/r['checkpoint_relative_path']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path)!=r['checkpoint_sha256']:raise TaskContractError('G1_CHECKPOINT_IDENTITY_MISMATCH')
        metadata=r['checkpoint_metadata']
        if metadata.get('continuation_id')!='hungarian-v1-fixed' or metadata.get('variant')!='G1' or metadata.get('seed')!=r['world_seed'] or metadata.get('architecture')!=matrix['world_model_architecture']:raise TaskContractError('G1_METADATA_CONTRACT_MISMATCH')
    from recovery_contract import training_route_keys, verify_recovery
    verify_recovery(root,matrix,request)
    if request['stages']['conditional_policy_training']['policy_optimizer_updates']!=128*len(training_route_keys(matrix)) or request['stages']['conditional_task_confirmation']['task_episodes']!=240:raise TaskContractError('TASK_BUDGET_BINDING_MISMATCH')
    config=read(root/'environment-config-contract.json')
    if canonical(config['config'])!=config['config_sha256'] or (config['config']['task_completion_mode'],config['config']['deadline_basis'])!=('arrival_to_region','physical_arrival'):raise TaskContractError('TASK_SEMANTICS_MISMATCH')
    q=read(root/'g1-prediction-qualification.json')
    if not q.get('G1_gate_pass') or q.get('G2_gate_pass') is not False:raise TaskContractError('G1_HISTORICAL_QUALIFICATION_MISSING')
    return {'status':'task_inputs_pass','source_identity':source_identity,'parent_roles':{'train':24,'task_confirmation':8},'models':3,'model_loads':0,'gppo_training_authorized':False}

def verify_task_inputs(root):
    try:
        return _verify_task_inputs(root)
    except TaskContractError:
        raise
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise TaskContractError('TASK_CONTRACT_INVALID:'+type(exc).__name__+':'+str(exc)) from exc
