"""Offline preparation assertions only; no model, simulator, or CUDA use."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
PKG=ROOT/'package'
SOURCE=Path('E:/Z博士/diagnostic-work/w1-gpu-smoke-local-v1/package')
def read(root,name):return json.loads((root/name).read_text(encoding='utf-8'))
def same(name):assert (PKG/name).read_bytes()==(SOURCE/name).read_bytes(),name
for name in ('production_data.py','production_world.py','joint_pipeline.py','runner.py','budget_ledger.py',
    'w1_graph_jepa.py','w1_training.py','event_jepa_losses.py','sequence_data_contract.py',
    'task_outcome_contract.py','environment-config-contract.json','data-gate.json','prediction-gates.json'):
    same(name)
matrix,old=read(PKG,'experiment-matrix.json'),read(SOURCE,'experiment-matrix.json')
assert {k:v for k,v in matrix.items() if k!='attempt'}=={k:v for k,v in old.items() if k!='attempt'}
split,oldsplit=read(PKG,'parent-split.json'),read(SOURCE,'parent-split.json')
assert {k:v for k,v in split.items() if k!='attempt'}=={k:v for k,v in oldsplit.items() if k!='attempt'}
request,prior=read(PKG,'RESOURCE_REQUEST.json'),read(SOURCE,'RESOURCE_REQUEST.json')
assert request['stages']==prior['stages']
assert {k:v for k,v in request['totals'].items() if k!='gpu_memory_bytes'}=={k:v for k,v in prior['totals'].items() if k!='gpu_memory_bytes'}
assert request['totals']['gpu_memory_bytes']<prior['totals']['gpu_memory_bytes']
assert request['status']=='NOT_APPROVED' and request['totals']['gppo_updates']==request['totals']['task_comparison_calls']==0
assert not matrix.get('synthetic_profile') and not read(PKG,'launch-contract.json')['integration_test']
assert matrix['world_model_seeds']==[8201,8202,8203]
for file in PKG.rglob('*.py'):ast.parse(file.read_text(encoding='utf-8-sig'),filename=str(file))
sys.path.insert(0,str(PKG))
from joint_inputs import verify_joint_inputs
inputs=verify_joint_inputs(PKG)
assert inputs['parent_roles']=={'train':24,'model_selection':8,'prediction_confirmation':8}
assert not any(name.startswith(('torch','gppo_world')) for name in sys.modules), 'PREPARATION_IMPORTED_MODEL_OR_SIMULATOR'
evidence={'status':'pass','production_research_code_byte_identical':True,'matrix_identical_except_attempt':True,
    'parents_identical_except_attempt':True,'budgets_unchanged_except_tighter_gpu':True,'source_inputs':inputs,
    'all_python_ast_parsed':True,'model_simulator_cuda_calls':0,'gppo_calls':0}
(ROOT/'offline-preparation-evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(evidence,ensure_ascii=False))
