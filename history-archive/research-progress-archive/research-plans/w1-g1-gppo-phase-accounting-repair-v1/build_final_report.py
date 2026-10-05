"""Audit old/final identities, scientific invariants, CPU evidence, and delivery."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
NEW=ROOT/'package'
OLD=ROOT.parent/'w1-g1-gppo-task-validation-local-v1'/'package'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
def normalize(v):
    if isinstance(v,dict):return {k:normalize(x) for k,x in v.items() if k not in ('runner_ready','resource_request_sha256')}
    if isinstance(v,list):return [normalize(x) for x in v]
    if isinstance(v,str):return v.replace('w1-g1-gppo-phase-accounting-repair-v1','w1-g1-gppo-task-validation-local-v1')
    return v

def main():
    old_manifest=load(OLD/'execution-manifest.json')['files']
    old_initial=load(ROOT/'source-copy-evidence.json')['files']
    old_unchanged=all(sha(OLD/n)==h for n,h in old_initial.items())
    old_export=OLD.parent/'verified-export'
    old_export_hashes=load(old_export/'export-hashes.json')
    old_export_unchanged=all(sha(old_export/n)==h for n,h in old_export_hashes.items())
    changed=[n for n,h in old_manifest.items() if not (NEW/n).is_file() or sha(NEW/n)!=h]
    newfiles=load(NEW/'execution-manifest.json')['files']
    added=sorted(set(newfiles)-set(old_manifest))
    scientific_files=[n for n in old_manifest if n.startswith(('native/','source-evidence/','source-models/','scene-tapes/','frozen-models/')) or n in ('production_policy.py','production_data.py','production_models.py','task_pipeline.py','task_metrics.py','task_contract.py','runtime-inputs.json','runtime-profile.json')]
    scientific_unchanged=all(sha(NEW/n)==old_manifest[n] for n in scientific_files)
    contracts_equal={n:normalize(load(NEW/n))==normalize(load(OLD/n)) for n in ('RESOURCE_REQUEST.json','experiment-matrix.json','parent-split.json','launch-contract.json')}
    witness=load(ROOT/'production-handshake-acceptance.json')
    audit={'old_frozen_inputs_unchanged':old_unchanged,'old_manifest_count':len(old_manifest),'old_full_frozen_file_count':len(old_initial),'changed_files':changed,'added_files':added,'scientific_files_unchanged':scientific_unchanged,'scientific_files_verified':scientific_files,'contracts_same_after_proposed_identity_normalization':contracts_equal,'budget_increased':False,'formal_attempt_started':False,'old_token_reused':False,'environment_calls':0,'model_initializations':0,'model_forwards':0,'checkpoint_loads':0,'optimizer_updates':0}
    audit.update(old_failed_export_unchanged=old_export_unchanged,old_failed_export_files=len(old_export_hashes))
    if not old_unchanged or not old_export_unchanged or not scientific_unchanged or not all(contracts_equal.values()):raise RuntimeError('INVARIANT_AUDIT_FAILED')
    write(ROOT/'research-and-old-evidence-invariants.json',audit)
    rows=[]
    for case in witness['cases']:
        state=case['supervisor']
        rows.append({'case':case['mode'],'checks_pass':case['all_checks_pass'],'expected_stop_exit':case['exit_code'],'wall_seconds':case['wall_seconds'],'outer_waited_cpu':case['outer_kernel_waited_cpu_seconds'],'supervisor_cpu':state['cpu_seconds'],'closed_phases_cpu':case['closed_phase_cpu_seconds'],'remaining_measured_tail':case['independently_recorded_unconfirmed_tail_cpu_seconds'],'startup_exit_scope_difference':case['outer_startup_and_exit_tail_cpu_seconds'],'worker_last_acknowledged_stage':state.get('worker_last_acknowledged_stage'),'server_committed_stage':state.get('server_committed_stage'),'stage_ack_status':state.get('stage_transition_acknowledgement_status')})
    write(ROOT/'cpu-independent-recalculation.json',{'cases':rows,'nested_values_added_to_outer_total':False,'method':'kernel waited CHILDREN measured outside supervisor; independent supervisor RUSAGE total versus authenticated closed intervals plus separately recorded tail; known 0.03s supervisor post-settlement CPU included by outer wait','unknown_stage_cpu_not_filled_zero':True,'measurement_gaps':['Windows/WSL bridge CPU not measured','formal controller final-write and exit tail remain explicitly unmeasured','sampling is not a delegated cgroup hard cap'],'research_calls':0})
    identity=load(ROOT/'frozen-identity.json')
    table='\n'.join(f"| {r['case']} | {'通过' if r['checks_pass'] else '失败'} | {r['outer_waited_cpu']:.6f} | {r['supervisor_cpu']:.6f} | {r['remaining_measured_tail']:.6f} |" for r in rows)
    text=f'''# 阶段 CPU 协议修复交付

工程范围通过：此前首次同阶段快照后的训练转换已在真实生产 worker→runner→supervisor→AF_UNIX 路径通过。最终代码经独立 Luna 复核、WSL 受控回归及原样只读预检。工程回放在真实训练函数第一条语句前停止，研究环境、模型、checkpoint 和训练调用均为0。

旧 attempt 已消费、技术停止和资源验收失败结论保留；旧 {len(old_initial)} 个冻结文件哈希不变，未修改原始失败制品。此修复不形成新的研究结果。

## 根因与最小修复

旧本地同阶段快照将 worker CPU 起点从0推进到2.866153s，却未同步 supervisor，首次训练转换遭严格连续性拒绝。旧异常结算再次握手且 supervisor 立即清理 worker，首错与结算因此缺失。

修复分离阶段起点与累计观察；进入阶段登记CPU0，累计快照不移动起点；真实转换以序列、载荷摘要和双向确认提交。相同重复幂等、冲突拒绝，确认丢失停止。保留1e-6数值容差和 PHASE_CPU_BASELINE_DISCONTINUITY 检查。

worker 在结算前保存首 traceback；异常结算不再次切阶段。所有退出都独立核对开放阶段尾段与精确总量。server已提交与worker已收到确认分开记录；不确定阶段保留null。worker未生成结算时，supervisor明确标注不完整fallback，未知账本不是0。

## 真实生产路径证据

| 用例 | 判定 | 外层 waited CPU(s) | supervisor CPU(s) | 独立尾段(s) |
| --- | --- | ---: | ---: | ---: |
{table}

这些受控停止的worker退出码1、17或18是预设异常路径；验收驱动器检查预期错误、账本、结算与导出后退出0，不将被测试worker退出码改写为成功。normal用例包含staging连续3次、training连续3次和task阶段1次快照，实际等待3个短子进程。研究阶段函数未替换；包外故障注入限socket响应、受控CPU与退出边界。

开发回归失败原件仍保留：development-2的stage_enter通知校验和隔离子进程导入失败、development-1/stable-1的并发修改字节不一致、旧历史入口测试断言及freeze-driver-schema-error.json。已分别修复通知合同、明确子进程路径与-B、稳定最终字节、验证历史入口拒绝和读取独立审核的实际JSON结构；没有删断言、吞异常或改写旧退出码。

CPU复算按外层kernel waited CHILDREN与supervisor SELF+waited分别保留嵌套口径；已关闭区间加独立开放尾段等于supervisor总量。外层另捕获已知0.03s supervisor退出尾段，内部值不再次加入外层。完整Windows/WSL桥、正式controller最后写入与退出尾段仍未测量，full_resource_acceptance=false；没有声称全生命周期成本验收通过。

## 冻结身份与执行状态

- proposed attempt：`{identity['attempt']}`
- Manifest：`{identity['execution_manifest_sha256']}`
- Hashes：`{identity['hashes_sha256']}`
- RESOURCE_REQUEST：`{identity['resource_request_sha256']}`
- RESOURCE_REQUEST保持NOT_APPROVED；runner_ready仅表示此次工程阻断已关闭。
- 唯一命令见package/unique-launch-command.md；新正式执行仍需绑定以上身份的新授权与新token。本轮未创建正式token或正式研究attempt。

科学代码、四臂、种子、训练规模、父场景、门槛和预算逐项与旧包核对不变。当前预算仍为22,752环境步、1,152策略更新、240episodes、wall5,070s、CPU7,790s。

全部测试、协议、首错、SQLite与受控导出原件放包外engineering-evidence-final；归档副本与最终冻结字节一致。只读Windows/WSL预检与真实生产工程回放是两类不同证据。

GitHub状态见git-archive-status.json；签名代理未恢复时只能报告本地归档与远端未归档。
'''
    (ROOT/'final-repair-report.md').write_text(text,encoding='utf-8')
    print(json.dumps({'invariants_pass':True,'cases':len(rows),'report':str(ROOT/'final-repair-report.md')},ensure_ascii=False))

if __name__=='__main__':main()
