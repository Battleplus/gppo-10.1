"""Read persisted artifacts only. No model loading or prediction."""
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT=Path(__file__).resolve().parent
EXPORT=ROOT/'verified-export'
RUN=EXPORT/'run-once'
def read(path):return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
status=read(RUN/'status.json')
admission=read(RUN/'data-admission.json')
metrics=read(RUN/'prediction-metrics.json')
training=read(RUN/'world-model-training-summary.json') or read(RUN/'world-model-training-progress.json')
supervisor=read(EXPORT/'supervisor-status.json')
runtime=read(RUN/'local-runtime-execution.json')
resource=read(RUN/'resource-settlement.json')
windows=[]
path=RUN/'world-model-windows.jsonl'
if path.is_file():windows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
labels={role:{'windows':0,'complete':0,'candidates':0,'heads':{head:{'valid':0,'positive':0,'negative':0,'unknown':0,'unknown_reasons':{}} for head in ('physical_on_time_completion','task_expired','host_confirmation')}} for role in ('train','model_selection','prediction_confirmation')}
for row in windows:
    group=labels[row['split']];group['windows']+=1;group['complete']+=row.get('status')=='complete'
    records=row.get('task_outcome_target',[]);group['candidates']+=len(records)
    for record in records:
        for head in group['heads']:
            label=record[head];c=group['heads'][head]
            if label['valid']:
                c['valid']+=1;c['positive' if label['value'] else 'negative']+=1
            else:
                c['unknown']+=1;reason=str(label.get('reason',label.get('source','unspecified')))
                c['unknown_reasons'][reason]=c['unknown_reasons'].get(reason,0)+1
checkpoint_files=[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in RUN.rglob('*.pt')]
ledger=None
if (RUN/'budget.sqlite3').exists():
    connection=sqlite3.connect('file:'+str((RUN/'budget.sqlite3').resolve()).replace('\\','/')+'?mode=ro',uri=True)
    ledger={'statuses':dict(connection.execute('SELECT status,COUNT(*) FROM calls GROUP BY status')),
        'operations':[{'stage':stage,'name':name,'calls':n} for stage,name,n in connection.execute('SELECT stage,name,COUNT(*) FROM calls GROUP BY stage,name ORDER BY stage,name')]}
    connection.close()
complete_routes={variant:sum(route.get('variant')==variant for route in (training or {}).get('routes',[])) for variant in ('G1','G2')}
summary={'classification':'REAL_W1_LOCAL_JOINT_RESEARCH','status':status,'data_admission':admission,
    'label_distribution':labels,'completed_routes':complete_routes,'training':training,'metrics':metrics,
    'checkpoints':checkpoint_files,'runtime':runtime,'worker_settlement':resource,'supervisor':supervisor,
    'native_controller':read(ROOT/'native-controller-settlement.json'),
    'windows_launcher':{k:v for k,v in (read(ROOT/'launch-evidence.json') or {}).items() if k not in ('before','after')},
    'ledger':ledger,'gppo':'未评价；预算为零','practical_cost':'未评价；没有GPPO任务决策',
    'synthetic_data_mixed':False,'research_success_not_inferred_from_completion':True}
(ROOT/'RESEARCH_RESULT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text=['# 本机真实 W1 联合运行结果','',f"程序状态：{(status or {}).get('status','没有worker状态')}；G1 {complete_routes['G1']}/3，G2 {complete_routes['G2']}/3。",'']
if admission:
    text += [f"数据门：{admission['status']}；完整窗口 {admission['complete_windows']}/{admission['expected_window_count']}。",'']
    if not admission['coverage_pass']:
        text += ['冻结数据门未通过，世界模型训练与预测未评价；不能据此断言G1/G2机制有效或失败。','', '失败依据：', '```json',json.dumps(admission['failures'],ensure_ascii=False,indent=2),'```','']
if metrics:
    text += ['| 方法 | 父场景宏平均regret | Top-1 |','|---|---:|---:|']
    for method in ('transparent','G1','G2'):
        values=metrics['parent_macro_regret'][method];tops=metrics['parent_macro_top1'][method]
        text.append(f"| {method} | {sum(values.values())/len(values):.8f} | {sum(tops.values())/len(tops):.4f} |")
    text += ['', '冻结推进判断：','```json',json.dumps(metrics['worth_gppo_suggestions'],ensure_ascii=False,indent=2),'```','']
text += ['GPPO收益、相对Hungarian实用收益及10ms/50ms任务成本标准：未评价，本轮预算为零。',
    '', '资源计量是已测范围：native controller SELF＋waited CHILDREN，Windows controller另报；WSL桥接CPU、最终退出尾段、WDDM进程GPU内存/独占证明没有完整测量，不宣称完整资源验收通过。',
    '', '数据、checkpoint、逐候选trace和账本位于 verified-export/run-once；逐文件hash位于 verified-export/export-hashes.json。历史开发使用排除尚未证明，确认集仅作本研究内部隔离评价。']
(ROOT/'RESEARCH_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
print(json.dumps({'status':(status or {}).get('status'),'routes':complete_routes,'complete_windows':None if admission is None else admission['complete_windows'],'metrics_available':bool(metrics),'checkpoint_files':len(checkpoint_files)},ensure_ascii=False))
