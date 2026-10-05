# GPU_SMOKE_TEST — bootstrap 修复版实际执行结果

本次未跑通训练链路，状态 technical_stop；G1/G2 各三个种子均未开始（0/6）。
已消费身份 w1-gpu-smoke-test-bootstrap-repair-v1-once，不重跑、不创建替代身份。
本次没有世界模型或 GPPO 研究结果，runner_ready=false。

## 实际执行

核验并接续原控制器 PID 40424 / 子进程 19428，没有重复启动。密码仅通过
已核验本机控制台输入，未保存。外部授权 868 字节，SFTP wx 回读和摘要绑定通过。
远端 bootstrap-only 退出 0，六个关键模块来自冻结暂存目录，Python 运行时匹配。
GPU1 启动时没有其他计算任务，空闲 10988 MiB，满足 9 GiB 要求。
CUDA_VISIBLE_DEVICES=1，程序使用 cuda:0，实际 CUDA 初始化成功。

## 第一处失败

正式远端入口退出 1。_run_bounded_tests 在任何测试执行前导入 test_sequence_world，
它经 test_sequence_data 导入 test_action_conditioned_task_outcomes。
该文件第 16 行 runpy.run_path 调用指向不存在的包外历史路径：

`/home/runs/w1-light-repaired-fair-rerun-v2-nativefs-once/native/gppo_world/task_lifecycle.py`

完整 FileNotFoundError 异常链保存于 first-error-traceback.txt。
该夹具的 SOURCE_ROOT 由 PACKAGE.parents[1]/runs/... 推导；在服务器上变成
/home/runs/...。冻结包内 native/gppo_world/task_lifecycle.py 实际存在且摘要正确，
因此是夹具对工作区路径的不必要依赖，不能解释为 GPU 或世界模型失败。
本轮没有修复失败包或自动重跑。

|要求|实际结果|
|---|---|
|G1 三种子|0/3，未开始|
|G2 三种子|0/3，未开始|
|collector 合成候选|未执行|
|loss、regret、Brier、逐候选预测|未评价|
|checkpoint 保存/恢复|0/0，恢复校验未评价|
|模型初始化/forward/backward/update|均为 0|
|CUDA context/最小张量分配|1/1，不报告全部计算为零|
|真实环境/GPPO|0/0|
|控制器/远端入口退出码|1/1|

## 运行时与资源

解释器 /home/user1/.venvs/w1-runtime-v1/bin/python；冻结身份 Python 3.10.12，
实际 torch/CUDA 检查通过（PyTorch 2.5.1+cu121，CUDA 12.1），驱动 535.161.08。
GPU1 是 RTX 2080 Ti。框架峰值 allocated=512 字节、reserved=2097152 字节；
实际进程显存观测峰值 163577856 字节（156 MiB）。

控制器从认证后启动到结果落盘实测 wall 219.222977 秒，CPU 0.734375 秒。
远端 acceptance 进程实测 wall 8.782984 秒、SELF+CHILDREN CPU 9.077643 秒。
这些 wall 范围嵌套，不能相加；CPU 不重复但仍缺少远端预检与 SSH/SFTP 范围。
控制器 wall 已超过 180 秒全局上限约 39.223 秒，不能称为在原预算内完成，
也不能因取消精确传输 CPU 前置要求而忽略这个已测得的超限。
当前控制器未在预检/暂存/下载全生命周期硬限制 180 秒，需保留这项事实。
完整 CPU 预算验收未证明；RSS 只覆盖 acceptance 根进程。停止记录存储为 0
是在失败文件写入前的采样，不能当作最终制品存储验收。

最终只读核查：无残留本作业进程，GPU1 恢复 10988 MiB 空闲；GPU0 HARL
PID 1167128 未受影响。无需再终止任何任务。

## 制品与验证

下载目录：E:\Z博士\diagnostic-work\runs\w1-gpu-smoke-test-bootstrap-repair-v1-once。
这是通过哈希核验的失败证据，不是完整训练结果导出。
远端证据清单 SHA-256：9c6c957a6adba0e20281a230b3d1356b76882d32d598a49a574a6eb0e3264d2e。
acceptance-result.json 共 5784 字节，SHA-256：
03fea7594594741acc7d730d20a9b3f42b7e5d03d173927edd3e323caabe785a。

冻结包 Manifest：69b3d14fdc98f93afb67fe6e28166b0dd75ddf8d05c240ff743c0f6e872b9fea；
Hashes：7dd180d999a32d1273986e1709c2e7f5af5452d78da3f9b3b5d85a0b9c0c2c53。
结束后对暂存 payload 身份再次核验，通过；没有改写冻结包。
本地结果哈希见 evidence-hashes.json，授权/凭据文件不进入该归档清单。
GitHub 远端归档未核验，不宣称已完成。

## 尚未完成的目标

六路训练、checkpoint 恢复、预测与指标复算均未达成。
下一项具体工程缺口是消除该夹具的包外历史路径依赖，并在隔离暂存目录
真实导入所有将使用的夹具；此外需约束全生命周期 wall。
这两项并非模型设计变化。本次停止规则禁止自动重跑，未申请或启动新身份。
