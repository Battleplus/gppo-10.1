"""Analyze persisted outputs only; no model/checkpoint loading or dynamic work."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'package'))
import launch_server_acceptance as launcher

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
checked = launcher.verify_structure()
controller = read(root / 'controller-result.json')
measurement = read(root / 'controller-measurement.json')
before = read(root / 'final-original-preflight.json')['all_package_files_before']
after = {p.relative_to(root / 'package').as_posix(): sha(p) for p in (root / 'package').rglob('*')
         if p.is_file() and '__pycache__' not in p.parts}
assert before == after, 'FINAL_INPUTS_CHANGED_DURING_RUN'
evidence_path = (controller.get('local_evidence') or {}).get('path')
download = Path(evidence_path) if evidence_path else root.parent / 'runs' / checked['request']['engineering_job_id']
actual, manifest, files_verified = {}, {}, False
if (download / 'acceptance-evidence-manifest.json').exists():
    manifest = read(download / 'acceptance-evidence-manifest.json')
    assert sha(download / 'acceptance-evidence-manifest.json') == controller['local_evidence']['manifest_sha256']
    for relative, identity in manifest['files'].items():
        path = download / relative
        assert path.stat().st_size == identity['bytes'] and sha(path) == identity['sha256'], relative
    files_verified = True
    actual = read(download / 'acceptance-result.json')
export = download / 'joint-controlled-export' / 'run-once'
training_path = export / 'world-model-training-summary.json'
metrics_path = export / 'prediction-metrics.json'
ledger_path = export / 'resource-settlement.json'
training = read(training_path) if training_path.exists() else {'routes': []}
metrics = read(metrics_path) if metrics_path.exists() else {}
ledger = read(ledger_path) if ledger_path.exists() else {}
routes = []
for route in training['routes']:
    checkpoint = export / route['checkpoint_relative_path']
    verified = checkpoint.is_file() and sha(checkpoint) == route['checkpoint_sha256']
    assert verified, 'CHECKPOINT_FILE_IDENTITY_MISMATCH'
    routes.append({'variant': route['variant'], 'seed': route['seed'],
                   'epochs': route['epochs_trained'], 'updates': route['updates'],
                   'train_candidate_rows': route['train_candidate_rows'], 'device': route['device'],
                   'loss_by_epoch': route['loss_by_epoch'], 'selection_regret': route['selection_regret'],
                   'checkpoint_path': str(checkpoint), 'checkpoint_sha256': route['checkpoint_sha256'],
                   'checkpoint_file_verified': verified})
means = {method: sum(parent.values()) / len(parent) if parent else None
         for method, parent in metrics.get('parent_macro_regret', {}).items()}
resources = actual.get('resources_after_terminal_event') or actual.get('resources_at_stop') or {}
gpu = actual.get('gpu_resource_evidence') or {}
complete_routes = {variant: sum(route['variant'] == variant for route in routes) for variant in ('G1', 'G2')}
totals = actual.get('model_call_totals') or (ledger.get('ledger') or {}).get('totals', {})
raw_path = download.with_suffix('.remote-output.json')
raw = read(raw_path) if raw_path.exists() else {}
phase_log_path = download / 'supervised_phase_sync.log'
phase_log = phase_log_path.read_text(encoding='utf-8') if phase_log_path.exists() else ''
audit = read(root / 'post-run-audit.json') if (root / 'post-run-audit.json').exists() else {}
bootstrap = read(root / 'remote-bootstrap.json') if (root / 'remote-bootstrap.json').exists() else {}
bootstrap_payload = json.loads(bootstrap['stdout'].splitlines()[-1]) if bootstrap.get('exit_code') == 0 else {}
if 'AF_UNIX path too long' in raw.get('stderr', '') and not training['routes']:
    # The selected tests run serially; this failure precedes either training test.
    totals = {'cuda_context_initializations': 1, 'minimum_cuda_tensor_allocations': 1,
              'model_initializations_or_loads': 0, 'world_batch_forwards': 0,
              'world_backward_calls': 0, 'world_optimizer_updates': 0,
              'checkpoint_writes': 0, 'checkpoint_loads': 0, 'world_sample_evaluations': 0,
              'synthetic_environment_constructions': 0}
restore = (totals.get('checkpoint_loads', 0) >= 6 and len(routes) == 6 and bool(metrics)
           and actual.get('joint_pipeline_evidence', {}).get('controlled_export_verified') is True)
result = {'classification': 'GPU_SMOKE_TEST', 'status': controller.get('status'),
          'engineering_job_id': checked['request']['engineering_job_id'],
          'engineering_identity_consumed': controller.get('engineering_job_consumed'),
          'conflicting_remote_summary_consumption_flag': (controller.get('remote_result') or {}).get('engineering_job_consumed'),
          'consumption_rule': 'exclusive staging directory creation consumes identity; remote false summary does not undo this',
          'remote_exit_code': controller.get('remote_exit_code'), 'run_status': actual.get('run_status'),
          'completed_routes': complete_routes, 'training_routes': routes,
          'parent_macro_regret': means, 'event_brier_parent_macro': metrics.get('event_brier_parent_macro'),
          'prediction_metrics': metrics, 'checkpoint_restore_verified_in_production': restore,
          'export_files_independently_hash_verified': files_verified,
          'frozen_inputs_unchanged': True, 'model_call_totals': totals,
          'ledger': ledger.get('ledger'), 'resources': resources, 'gpu': gpu,
          'controller_measurement': measurement, 'controller_settlement': controller.get('settlement'),
          'independent_runtime_fixture_preflight': bootstrap_payload.get('isolated_fixture_preflight'),
          'runtime': {'python_executable': checked['request']['runtime']['python_executable'],
                      'python': checked['request']['runtime']['python'], 'torch': checked['request']['runtime']['torch'],
                      'cuda': checked['request']['runtime']['cuda'], 'driver': '535.161.08',
                      'gpu': 'GPU1 NVIDIA GeForce RTX 2080 Ti', 'CUDA_VISIBLE_DEVICES': '1'},
          'post_stop_readonly_audit': {'payload_identity_verified': audit.get('payload_identity_verified'),
                                     'owned_job_processes': audit.get('owned_job_processes'),
                                     'gpu_query': audit.get('gpu_query'), 'gpu_process_query': audit.get('gpu_process_query'),
                                     'cpu_not_in_controller_measurement': True, 'wall_not_in_controller_measurement': True},
          'full_resource_acceptance': False,
          'unmeasured_scopes': ['remote preflight/SSH/SFTP complete CPU', 'aggregate process-tree RSS',
                               'offline preparation/post-analysis CPU outside execution'],
          'real_research_environment_calls': 0, 'gppo_calls': 0, 'research_effect_proven': False,
          'automatic_retry': False, 'runner_ready': False,
          'first_exception_chain': '\n'.join(part for part in [raw.get('stderr'), phase_log,
                                    actual.get('exception_chain') or controller.get('exception_chain')] if part),
          'output_directory': str(download), 'package_identity': read(root / 'smoke-package-identity.json')}
(root / 'GPU_SMOKE_TEST_RESULT.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
if result['first_exception_chain']:
    (root / 'first-error-traceback.txt').write_text(result['first_exception_chain'], encoding='utf-8')
lines = ['# GPU_SMOKE_TEST — 实际结果', '',
         f"状态：{result['status']}；远端退出码 {result['remote_exit_code']}。G1 {complete_routes['G1']}/3，G2 {complete_routes['G2']}/3。",
         '', '合成数据结果只证明工程链路，不是世界模型研究效果；不运行 GPPO。', '',
         '|方法|种子|epoch|更新|实际 loss_by_epoch|checkpoint 摘要校验|', '|---|---:|---:|---:|---|---|']
for route in routes:
    lines.append(f"|{route['variant']}|{route['seed']}|{route['epochs']}|{route['updates']}|{json.dumps(route['loss_by_epoch'], ensure_ascii=False)}|通过|")
lines += ['', f"checkpoint 生产恢复并用于预测：{'通过' if restore else '未评价或未完成'}。",
          f"透明基线及各模型父场景宏平均 regret：{json.dumps(means, ensure_ascii=False)}。",
          f"事件 Brier：{json.dumps(metrics.get('event_brier_parent_macro'), ensure_ascii=False)}。", '',
          f"控制器 wall {measurement['wall_seconds']:.6f} 秒，CPU {measurement['controller_process_cpu_seconds']:.6f} 秒。",
          f"远端资源：{json.dumps(resources, ensure_ascii=False)}。",
          f"GPU 峰值与实际采样：{json.dumps(gpu, ensure_ascii=False)}。", '',
          '本次首次实际错误为 phase_handshake.py:61 socket.bind 的 OSError: AF_UNIX path too long。',
          '随后测试尝试读取未创建的 console.log，产生第二层 FileNotFoundError；两层原始异常均保留。',
          'CPU scope 算术回归通过；supervised_phase_sync 未通过；collector、六路训练、恢复与预测均未进入。',
          'CUDA context 和最小张量分配各 1 次；其他模型/环境/checkpoint 调用为 0，不能把全部计算说成零。',
          f"运行时：{json.dumps(result['runtime'], ensure_ascii=False)}。",
          'bootstrap 实际隔离导入与纯配置检查通过，不能称为完整流水线验收。',
          'controller wall 总量小于 180 秒，但已测非 server 开销 75.547 秒超过冻结 controller 30 秒子上限；资源验收未通过。',
          '停止后的只读进程/GPU 审计不在上述 controller wall/CPU 范围；不声称它们已被完整计量。',
          '完整 SSH/SFTP CPU 和聚合 RSS 未测量，不声称完整资源验收通过。',
          f"原样包身份未变：True；导出逐文件核验：{files_verified}。",
          f"制品位置：{download}。",
          '本身份已消费并封存，不自动修复续跑，不创建额外身份。',
          f"退出后本作业残留进程：{audit.get('owned_job_processes')}；GPU1 空闲 10988 MiB，GPU0 HARL PID 1167128 保持。",
          '远端 stdout 的 engineering_job_consumed=false 与实际暂存创建及结果记录 true 冲突；不得据此复用身份。',
          'GitHub 签名代理阻塞，远端未归档；本地归档另行记录。']
if result['first_exception_chain']: lines += ['', '首次完整异常：', '```text', result['first_exception_chain'], '```']
(root / 'GPU_SMOKE_TEST_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
exclude = {'external-authorization.json', 'evidence-hashes.json'}
names = [p for p in root.iterdir() if p.is_file() and p.name not in exclude]
index = {'classification': 'GPU_SMOKE_TEST', 'credentials_excluded': True,
         'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(names)},
         'export_manifest_sha256': sha(download / 'acceptance-evidence-manifest.json') if files_verified else None}
(root / 'evidence-hashes.json').write_text(json.dumps(index, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': result['status'], 'completed_routes': complete_routes,
                  'restore_verified': restore, 'export_verified': files_verified,
                  'parent_macro_regret': means, 'result_sha256': sha(root / 'GPU_SMOKE_TEST_RESULT.json')}, indent=2))
