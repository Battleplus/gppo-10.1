import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
download = root / 'download'
evidence = download / 'gpu-evidence'
run = evidence / 'joint-controlled-export/run-once'
def read(p):
    return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
result = read(evidence / 'local-gpu-result.json')
progress = read(run / 'world-model-training-progress.json')
metrics = read(run / 'prediction-metrics.json')
summary = read(run / 'world-model-training-summary.json')
settlement = read(run / 'resource-settlement.json')
supervisor = read(download / 'local-supervisor-result.json')
assert result['status'] == 'complete' and result['completed_routes'] == {'G1': 3, 'G2': 3}
assert settlement['ledger']['pending_calls'] == 0 and settlement['ledger']['failed_calls'] == 0
assert settlement['ledger_settlement_error'] is None
routes = []
for route in progress['routes']:
    variant, seed = route['variant'], route['seed']
    model_route = next(row for row in summary['routes'] if row['variant'] == variant and row['seed'] == seed)
    checkpoint = run / model_route['checkpoint_relative_path']
    assert sha(checkpoint) == model_route['checkpoint_sha256']
    assert len(route['checkpoint_restores']) == 1 and route['checkpoint_restores'][0]['device'] == 'cuda:0'
    loss = route['loss_by_epoch'][0]['loss']
    assert math.isfinite(loss)
    key = f'{variant}:{seed}'
    routes.append({'variant': variant, 'seed': seed, 'epoch': 1, 'loss': loss,
        'updates': model_route['updates'], 'training_candidate_rows': model_route['train_candidate_rows'],
        'event_brier': metrics['event_brier_parent_macro'][key],
        'physical_on_time_brier': metrics['horizon_task_outcome_brier_by_head_parent_macro'][key]['physical_on_time_completion'],
        'macro_regret': statistics.mean(metrics['seed_parent_macro_regret'][f'{variant}_seed_{seed}'].values()),
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint), 'restored': True})

trace = [json.loads(line) for line in (run / 'prediction-trace.jsonl').read_text(encoding='utf-8').splitlines()]
regrets, events = collections.defaultdict(list), collections.defaultdict(list)
candidate_rows = 0
for window in trace:
    rows = window['candidate_rows']
    candidate_rows += len(rows)
    assert len(rows) == window['candidate_count']
    assert len({c['candidate_id'] for c in rows}) == len(rows)
    assert window['continuation_id'] == 'hungarian-v1-fixed'
    assert all(math.isfinite(c['true_utility']) for c in rows)
    oracle = max(c['true_utility'] for c in rows)
    for method, decision in window['decisions'].items():
        selected = next(c for c in rows if c['action_id'] == decision['selected_action'])
        regret = oracle - selected['true_utility']
        assert abs(regret - decision['regret']) < 1e-12
        regrets[method, window['parent']].append(regret)
    for c in rows:
        for variant in ('G1', 'G2'):
            for seed, prediction in c['predictions'][variant].items():
                for h, (label, valid) in enumerate(zip(c['event_labels'], c['event_valid'])):
                    if valid:
                        value = prediction['event_probability'][h]
                        assert label is not None and math.isfinite(value)
                        events[f'{variant}:{seed}', window['parent']].append((value-label)**2)
macro_regret = {method: statistics.mean(statistics.mean(regrets[method,parent]) for m,parent in regrets if m==method)
                for method in sorted({key[0] for key in regrets})}
event_brier = {method: statistics.mean(statistics.mean(events[method,parent]) for m,parent in events if m==method)
               for method in sorted({key[0] for key in events})}
assert all(abs(event_brier[k]-metrics['event_brier_parent_macro'][k]) < 1e-12 for k in event_brier)
assert all(abs(macro_regret[k]-statistics.mean(metrics['parent_macro_regret'][k].values())) < 1e-12 for k in macro_regret)
recalculation = {'model_calls': 0, 'trace_windows': len(trace), 'candidate_rows': candidate_rows,
    'parent_count': len({w['parent'] for w in trace}), 'macro_regret': macro_regret,
    'event_brier_parent_macro': event_brier, 'persisted_metrics_match': True}
(root / 'independent-metric-recalculation.json').write_text(json.dumps(recalculation, indent=2), encoding='utf-8')
for relative, digest in read(download / 'download-hashes.json').items():
    assert sha(download / relative) == digest
for stage in ('preflight', 'system', 'gpu', 'download'):
    assert read(root / (stage+'-evidence.json'))['frozen_contents_unchanged']
profile = {'train_parents': 2, 'selection_parents': 2, 'confirmation_parents': 8,
    'seeds': [8201,8202,8203], 'epochs': 1, 'actual_training_candidates': 10, 'maximum_training_candidates': 50}
resource = {'launch_wall_seconds': read(root/'gpu-evidence.json')['wall_seconds'],
    'download_and_hash_verify_wall_seconds': read(root/'download-evidence.json')['wall_seconds'],
    'native_supervisor_cpu_seconds': supervisor['supervisor']['cpu_seconds'],
    'worker_cpu_seconds_nested_not_added': result['resources']['complete_process_cpu_seconds'],
    'gpu_peak_allocated_bytes': result['gpu_resources']['framework_peak_allocated_bytes'],
    'gpu_peak_reserved_bytes': result['gpu_resources']['framework_peak_reserved_bytes'],
    'process_tree_peak_rss_bytes': supervisor['supervisor']['peak_rss_upper_bound_bytes'],
    'worker_active_storage_peak_bytes': result['resources']['active_storage_peak_bytes'],
    'downloaded_bytes': sum(p.stat().st_size for p in download.rglob('*') if p.is_file()),
    'measured_limits_pass': True, 'full_resource_acceptance': False,
    'measurement_gaps': result['resource_measurement_gaps'] + ['Windows WSL bridge and copy subprocess CPU not measured'],
    'no_automatic_retry': True}
delivery = {'classification': 'GPU_SMOKE_TEST', 'status': 'complete', 'exit_code': 0,
    'training_completed': True, 'completed_routes': {'G1':3, 'G2':3}, 'profile': profile,
    'routes': routes, 'independent_metrics': recalculation, 'resources': resource,
    'gpu_call_accounting': settlement['model_calls'],
    'gpu_plus_cpu_regression_call_accounting': result['model_call_totals'],
    'research_environment_calls': 0, 'gppo_calls': 0, 'research_effect_established': False,
    'checkpoint_restore_validated_routes': 6, 'downloaded_artifact_hashes_verified': 242,
    'source_identity': read(root/'local-frozen-identity.json')}
(root/'GPU_SMOKE_TEST_RESULT.json').write_text(json.dumps(delivery, ensure_ascii=False, indent=2), encoding='utf-8')
report = ['# 本机 GPU_SMOKE_TEST：实际训练完成', '',
    '退出码0，G1/G2各三个固定种子全部完成1 epoch、checkpoint保存/恢复、逐候选预测、生产指标与离线复算、账本结算及受控导出。仅合成环境夹具；没有真实研究环境或GPPO调用。', '',
    '| 方案 | 种子 | 实际loss | 事件Brier | 物理按时完成Brier | regret | 恢复 |',
    '| --- | --- | --- | --- | --- | --- | --- |']
for row in routes:
    report.append(f"| {row['variant']} | {row['seed']} | {row['loss']:.6f} | {row['event_brier']:.6f} | {row['physical_on_time_brier']:.6f} | {row['macro_regret']:.6f} | cuda:0已恢复 |")
report += ['', '训练输入为生产collector在合成环境上生成的2个父场景、10条候选（上限50）；选择2个父场景、确认8个父场景/40条候选。完整主体结构未缩小，每路2次优化。', '',
    f"G1/G2集成事件Brier={metrics['event_brier_parent_macro']['G1:ensemble']:.6f}/{metrics['event_brier_parent_macro']['G2:ensemble']:.6f}，这里仅公开续行确认事件头有有效标签，其他即时事件头保持unknown。独立horizon物理按时完成Brier={metrics['horizon_task_outcome_brier_by_head_parent_macro']['G1:ensemble']['physical_on_time_completion']:.6f}/{metrics['horizon_task_outcome_brier_by_head_parent_macro']['G2:ensemble']['physical_on_time_completion']:.6f}。这是不同时间语义的两组头，不能互相代替。",
    '', '透明方法、G1/G2所有种子和集成的regret均为0；G1相对透明、G2相对G1的冻结改善门均未通过。本次仅证明工程链跑通，不能据合成指标判断研究效果或事件监督独立有效。GPPO和真实调度收益/成本未评价。', '',
    f"运行时：本机WSL Python3.11.16 / torch2.7.0+cu128 / CUDA12.8，RTX3060 Laptop GPU0，驱动571.96。启动wall={resource['launch_wall_seconds']:.3f}s；复制及hash验收={resource['download_and_hash_verify_wall_seconds']:.3f}s；native supervisor SELF+waited CPU={resource['native_supervisor_cpu_seconds']:.3f}s，不再加worker嵌套CPU。显存allocated/reserved峰值={resource['gpu_peak_allocated_bytes']/1024**2:.3f}/{resource['gpu_peak_reserved_bytes']/1024**2:.3f}MiB；进程树RSS峰值={resource['process_tree_peak_rss_bytes']/1024**3:.3f}GiB。",
    '', 'GPU主链：12次初始化/加载、84次forward、12次反向/更新、6次checkpoint写与加载、420样本评价。附带CPU sequence回归另计12次初始化/加载、66次forward、6次反向/更新、6次checkpoint写与加载、330样本；合计恰好原工程上限24/150/18/18/12/12/750，CUDA context为1。真实环境步0；288步/12reset/60branch均为合成替身操作。', '',
    '资源实测范围内通过；full_resource_acceptance=false。Windows WSL桥接及copy子进程CPU未测量，桌面GPU非独占、自有进程显存不可可靠读取；4GiB上限指本作业PyTorch allocator，不能解释为全卡占用上限。CPU为采样和退出结算，不宣称cgroup硬限制。历史0.274925s差值仍来源未确定。', '',
    '最终包182个历史内容/新增本地入口和runtime身份经清单校验。历史source-evidence随包做完整性复制和摘要校验；训练collector实际runtime-inputs重新指向本次生成的synthetic-tapes，未用历史研究标签训练。179个历史源文件字节不变。前后包内容身份一致，242个导出文件与原生副本hash一致。', '',
    '数据/模型/trace：download/gpu-evidence/joint-controlled-export/run-once/；模型目录world-model-checkpoints/G1和G2；逐候选prediction-trace.jsonl；prediction-metrics.json；resource-settlement.json；完整原始日志download/package-supervision/console.log。CPU额外回归制品另存gpu-evidence。', '',
    '唯一实际运行命令及逐阶段原样参数保存在gpu-evidence.json；单次身份w1-gpu-smoke-local-v1-once已消费，不能作为可再次执行的命令。没有创建正式研究attempt；旧服务器身份未消费。', '',
    'Luna静态入口复核无确定性阻断；结果复核独立记录。签名要求保持，ssh-agent停用，尚无新签名提交/远端提交；远端未归档。']
(root/'GPU_SMOKE_TEST_REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
print(json.dumps({'status':'complete','routes':routes,'resources':resource,'trace_recomputed':True}, ensure_ascii=False))
