# 本机 GPU_SMOKE_TEST：实际训练完成

退出码0，G1/G2各三个固定种子全部完成1 epoch、checkpoint保存/恢复、逐候选预测、生产指标与离线复算、账本结算及受控导出。仅合成环境夹具；没有真实研究环境或GPPO调用。

| 方案 | 种子 | 实际loss | 事件Brier | 物理按时完成Brier | regret | 恢复 |
| --- | --- | --- | --- | --- | --- | --- |
| G1 | 8201 | 1.477375 | 0.250145 | 0.288117 | 0.000000 | cuda:0已恢复 |
| G1 | 8202 | 1.631045 | 0.289005 | 0.290296 | 0.000000 | cuda:0已恢复 |
| G1 | 8203 | 1.542348 | 0.260183 | 0.250230 | 0.000000 | cuda:0已恢复 |
| G2 | 8201 | 2.273652 | 0.284468 | 0.272696 | 0.000000 | cuda:0已恢复 |
| G2 | 8202 | 2.498233 | 0.338501 | 0.289922 | 0.000000 | cuda:0已恢复 |
| G2 | 8203 | 2.273731 | 0.292684 | 0.213582 | 0.000000 | cuda:0已恢复 |

训练输入为生产collector在合成环境上生成的2个父场景、10条候选（上限50）；选择2个父场景、确认8个父场景/40条候选。完整主体结构未缩小，每路2次优化。

G1/G2集成事件Brier=0.265983/0.304435，这里仅公开续行确认事件头有有效标签，其他即时事件头保持unknown。独立horizon物理按时完成Brier=0.275596/0.256667。这是不同时间语义的两组头，不能互相代替。

透明方法、G1/G2所有种子和集成的regret均为0；G1相对透明、G2相对G1的冻结改善门均未通过。本次仅证明工程链跑通，不能据合成指标判断研究效果或事件监督独立有效。GPPO和真实调度收益/成本未评价。

运行时：本机WSL Python3.11.16 / torch2.7.0+cu128 / CUDA12.8，RTX3060 Laptop GPU0，驱动571.96。启动wall=36.378s；复制及hash验收=13.934s；native supervisor SELF+waited CPU=25.461s，不再加worker嵌套CPU。显存allocated/reserved峰值=45.993/50.000MiB；进程树RSS峰值=2.106GiB。

GPU主链：12次初始化/加载、84次forward、12次反向/更新、6次checkpoint写与加载、420样本评价。附带CPU sequence回归另计12次初始化/加载、66次forward、6次反向/更新、6次checkpoint写与加载、330样本；合计恰好原工程上限24/150/18/18/12/12/750，CUDA context为1。真实环境步0；288步/12reset/60branch均为合成替身操作。

资源实测范围内通过；full_resource_acceptance=false。Windows WSL桥接及copy子进程CPU未测量，桌面GPU非独占、自有进程显存不可可靠读取；4GiB上限指本作业PyTorch allocator，不能解释为全卡占用上限。CPU为采样和退出结算，不宣称cgroup硬限制。历史0.274925s差值仍来源未确定。

最终包182个历史内容/新增本地入口和runtime身份经清单校验。历史source-evidence随包做完整性复制和摘要校验；训练collector实际runtime-inputs重新指向本次生成的synthetic-tapes，未用历史研究标签训练。179个历史源文件字节不变。前后包内容身份一致，242个导出文件与原生副本hash一致。

数据/模型/trace：download/gpu-evidence/joint-controlled-export/run-once/；模型目录world-model-checkpoints/G1和G2；逐候选prediction-trace.jsonl；prediction-metrics.json；resource-settlement.json；完整原始日志download/package-supervision/console.log。CPU额外回归制品另存gpu-evidence。

唯一实际运行命令及逐阶段原样参数保存在gpu-evidence.json；单次身份w1-gpu-smoke-local-v1-once已消费，不能作为可再次执行的命令。没有创建正式研究attempt；旧服务器身份未消费。

Luna静态入口复核无确定性阻断；结果复核独立记录。签名要求保持，ssh-agent停用，尚无新签名提交/远端提交；远端未归档。
