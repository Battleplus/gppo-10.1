# GPU_SMOKE_TEST 集中工程收尾 v3

新身份：w1-gpu-smoke-test-consolidated-acceptance-v3-once。
所有旧包、失败记录、消费证据原样保留。RESOURCE_REQUEST及工程申请
保持NOT_APPROVED。新外部授权按本轮明确授权绑定，旧token不迁移。

沿用原完整生产模型、标签、损失、固定种子和研究门槛；GPU合成profile
2/2/8父场景、G1/G2各3seed、1epoch50trainrows，附带原CPU sequence
1/1/8、25rows。真实研究环境/数据/GPPO禁止。全部合成模型计算如实计费。

集中修复安全短私有Unix socket、构造失败清理、早期首异常和缺失日志、
子进程依赖/退出/超时、消费状态、部分训练制品和结算/导出。原预算
180wall/360CPU，服务器150/300、控制器30/30、显存8GiB、空闲9GiB不变。
已取消用户自设登记挂载、独占分配证明和完整传输CPU精确计量前置要求。
完整runtime hash和隔离graph验证合并到metered worker中、先于CUDA；
去除重复扫描和已验证ZIP后逐文件覆盖。GPU前检查控制器阶段剩余额度。

实际入口：控制器→SSH/SFTP→smoke_supervisor_entry→生产supervise
→PDEATHSIG worker_bootstrap→smoke_worker exec隔离acceptance_entry
→生产collector/训练/选择/checkpoint/预测/复算/账本/受控导出→hash下载。
系统验收使用同一wrapper/worker、同一解释器/目录布局/环境参数，
--system-only只执行真实文件、认证socket、process、退出/异常/超时、
缺失日志、导出，不创建CUDA或模型。证据在包外，与最终hashes绑定。

SELF+waited CHILDREN只计一次；外层与worker计量嵌套核对不相加。
训练是否完成与完整资源验收分开报告。完整SSH/SFTP CPU、退出尾段
分项和未覆盖RSS写未测量。原关闭余量和0.05s同范围门不放宽。
历史0.274925s差值来源未确定。GPU首失败封存停止，不追加测试。
合成指标仅证明工程链路，不支持研究效用或腾讯会议路线成功。

静态全链检查另发现并修复GPU设备字符串cuda与生产合同cuda:0不一致；
统一为cuda:0，不改模型/训练预算。系统only导出独立schema，不能错误
标记工程identity已消费。系统证据下载拒绝Windowsdrive/UNC/逃逸/符号链接。
