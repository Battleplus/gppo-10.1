# 唯一入口

Attempt：`w1-drone-action-consequence-world-model-v5-recovery-v1-linux-release-v3-output-writes-contract-repair-once`。

当前资源申请为 `NOT_APPROVED`。新批准需绑定同一 `frozen-identity.json` 中的 manifest、hashes、request 摘要，并覆盖所有 stage、total 和外部传输 limits。不得使用 v5 或更早版本的授权。

服务器部署后先执行只读预检：

```sh
/home/user1/w1-runtimes/w1-py31116-torch270cu128-v2/w1-light-repaired-fair-rerun-py31116/bin/python -B /home/user1/w1-pilot/packages/w1-drone-action-consequence-world-model-v5-recovery-v1-linux-release-v3-output-writes/launch_pilot.py --preflight
```

仅在身份和部署文件核验通过、取得绑定本 attempt 的一次新授权后，执行唯一正式入口一次：

```sh
/home/user1/w1-runtimes/w1-py31116-torch270cu128-v2/w1-light-repaired-fair-rerun-py31116/bin/python -B /home/user1/w1-pilot/packages/w1-drone-action-consequence-world-model-v5-recovery-v1-linux-release-v3-output-writes/launch_pilot.py --authorization /home/user1/w1-pilot/authorizations/w1-drone-action-consequence-world-model-v5-recovery-v1-linux-release-v3-output-writes-contract-repair-once.json
```

`persistent_handoff` 只表示提交。之后只读监控同一 systemd user service，不重复调用入口。CPU0/单线程/GPU 关闭及既有 Restart=no、KillMode=control-group 继续使用。失败保留首错、pending 和已测资源，不自动重试。
