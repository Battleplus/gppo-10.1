# GPU_SMOKE_TEST — 实际结果

状态：technical_stop；远端退出码 1。G1 0/3，G2 0/3。

合成数据结果只证明工程链路，不是世界模型研究效果；不运行 GPPO。

|方法|种子|epoch|更新|实际 loss_by_epoch|checkpoint 摘要校验|
|---|---:|---:|---:|---|---|

checkpoint 生产恢复并用于预测：未评价或未完成。
透明基线及各模型父场景宏平均 regret：{}。
事件 Brier：null。

控制器 wall 91.269552 秒，CPU 0.500000 秒。
远端资源：{"active_storage_peak_bytes": 0, "aggregate_storage_peak_bytes": 1485, "complete_process_cpu_scope": "RUSAGE_SELF plus RUSAGE_CHILDREN, including waited/reaped descendants", "complete_process_cpu_seconds": 11.106864, "current_active_storage_bytes": 0, "current_aggregate_storage_bytes": 1485, "root_process_rss_high_water_bytes": 508653568, "rss_measurement_scope": "root process /proc/self/status VmRSS and VmHWM only; sampled checks; excludes descendants", "wall_seconds": 15.721878582146019}。
GPU 峰值与实际采样：{"ceiling_bytes": 8589934592, "framework_peak_allocated_bytes": 512, "framework_peak_reserved_bytes": 2097152, "framework_samples": "torch.cuda.max_memory_allocated/reserved at production call boundaries; no extra model forwards", "observed_owned_process_memory_peak_bytes": 163577856, "process_samples": [{"boundary": "post_cuda_context", "gpu_exclusive_allocation_observed": true, "gpu_owned_compute_process_pids": [1376809], "gpu_owned_process_memory_bytes": 163577856, "gpu_physical_device": 1, "gpu_uuid": "GPU-7b708ead-e044-6ce7-a530-6b3093ac905e", "phase": "initialization"}, {"boundary": "start", "gpu_exclusive_allocation_observed": true, "gpu_owned_compute_process_pids": [1376809], "gpu_owned_process_memory_bytes": 163577856, "gpu_physical_device": 1, "gpu_uuid": "GPU-7b708ead-e044-6ce7-a530-6b3093ac905e", "phase": "cpu_scope_regression"}, {"boundary": "end", "gpu_exclusive_allocation_observed": true, "gpu_owned_compute_process_pids": [1376809], "gpu_owned_process_memory_bytes": 163577856, "gpu_physical_device": 1, "gpu_uuid": "GPU-7b708ead-e044-6ce7-a530-6b3093ac905e", "phase": "cpu_scope_regression"}, {"boundary": "start", "gpu_exclusive_allocation_observed": true, "gpu_owned_compute_process_pids": [1376809], "gpu_owned_process_memory_bytes": 163577856, "gpu_physical_device": 1, "gpu_uuid": "GPU-7b708ead-e044-6ce7-a530-6b3093ac905e", "phase": "supervised_phase_sync"}, {"boundary": "end", "gpu_exclusive_allocation_observed": true, "gpu_owned_compute_process_pids": [1376809], "gpu_owned_process_memory_bytes": 163577856, "gpu_physical_device": 1, "gpu_uuid": "GPU-7b708ead-e044-6ce7-a530-6b3093ac905e", "phase": "supervised_phase_sync"}]}。

本次首次实际错误为 phase_handshake.py:61 socket.bind 的 OSError: AF_UNIX path too long。
随后测试尝试读取未创建的 console.log，产生第二层 FileNotFoundError；两层原始异常均保留。
CPU scope 算术回归通过；supervised_phase_sync 未通过；collector、六路训练、恢复与预测均未进入。
CUDA context 和最小张量分配各 1 次；其他模型/环境/checkpoint 调用为 0，不能把全部计算说成零。
运行时：{"python_executable": "/home/user1/.venvs/w1-runtime-v1/bin/python", "python": "3.10.12", "torch": "2.5.1+cu121", "cuda": "12.1", "driver": "535.161.08", "gpu": "GPU1 NVIDIA GeForce RTX 2080 Ti", "CUDA_VISIBLE_DEVICES": "1"}。
bootstrap 实际隔离导入与纯配置检查通过，不能称为完整流水线验收。
controller wall 总量小于 180 秒，但已测非 server 开销 75.547 秒超过冻结 controller 30 秒子上限；资源验收未通过。
停止后的只读进程/GPU 审计不在上述 controller wall/CPU 范围；不声称它们已被完整计量。
完整 SSH/SFTP CPU 和聚合 RSS 未测量，不声称完整资源验收通过。
原样包身份未变：True；导出逐文件核验：True。
制品位置：E:\Z博士\diagnostic-work\runs\w1-gpu-smoke-test-fixture-runtime-repair-v2-once。
本身份已消费并封存，不自动修复续跑，不创建额外身份。
退出后本作业残留进程：[]；GPU1 空闲 10988 MiB，GPU0 HARL PID 1167128 保持。
远端 stdout 的 engineering_job_consumed=false 与实际暂存创建及结果记录 true 冲突；不得据此复用身份。
GitHub 签名代理阻塞，远端未归档；本地归档另行记录。

首次完整异常：
```text
Traceback (most recent call last):
  File "/home/user1/w1-gpu-smoke-test-fixture-runtime-repair-v2-once/supervise.py", line 176, in main
    phase_server=StageServer(root/'.phase-accounting.sock',limits,tree_reader=_proc_tree,
  File "/home/user1/w1-gpu-smoke-test-fixture-runtime-repair-v2-once/phase_handshake.py", line 61, in __init__
    self.socket.bind(str(self.path))
OSError: AF_UNIX path too long

test_short_subprocesses_and_stage_boundaries_through_real_supervisor (test_supervised_phase_integration.SupervisedPhaseIntegrationTests) ... ERROR

======================================================================
ERROR: test_short_subprocesses_and_stage_boundaries_through_real_supervisor (test_supervised_phase_integration.SupervisedPhaseIntegrationTests)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/user1/w1-gpu-smoke-test-fixture-runtime-repair-v2-once/test_supervised_phase_integration.py", line 60, in test_short_subprocesses_and_stage_boundaries_through_real_supervisor
    self.fail(str(state) + "\n" + (root / "console.log").read_text())
  File "/usr/lib/python3.10/pathlib.py", line 1134, in read_text
    with self.open(mode='r', encoding=encoding, errors=errors) as f:
  File "/usr/lib/python3.10/pathlib.py", line 1119, in open
    return self._accessor.open(self, mode, buffering, encoding, errors,
FileNotFoundError: [Errno 2] No such file or directory: '/home/user1/w1-gpu-smoke-test-fixture-runtime-repair-v2-evidence/work/w1-supervised-cpu-o62199jf/console.log'

----------------------------------------------------------------------
Ran 1 test in 0.009s

FAILED (errors=1)

Traceback (most recent call last):
  File "/home/user1/w1-gpu-smoke-test-fixture-runtime-repair-v2-once/acceptance_entry.py", line 964, in _run_server_acceptance
    job, result = run_registered_once(
  File "/home/user1/w1-gpu-smoke-test-fixture-runtime-repair-v2-once/acceptance_entry.py", line 706, in run_registered_once
    result = run_workload(job)
  File "/home/user1/w1-gpu-smoke-test-fixture-runtime-repair-v2-once/acceptance_entry.py", line 911, in run_workload
    results = _run_bounded_tests(
  File "/home/user1/w1-gpu-smoke-test-fixture-runtime-repair-v2-once/acceptance_entry.py", line 663, in _run_bounded_tests
    raise AcceptanceError("SERVER_ACCEPTANCE_TEST_FAILED:" + phase)
AcceptanceError: SERVER_ACCEPTANCE_TEST_FAILED:supervised_phase_sync

```
