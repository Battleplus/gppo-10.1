# Unique commands

Read-only final package validation, no token consumption and no attempt:

```powershell
python E:\Z博士\research-plans\w1-action-relative-auxiliary-runtime-dependency-repair-v1\launch_once.py --authorization-file E:\Z博士\.codex-tmp\w1-action-relative-auxiliary-runtime-dependency-repair-v1.authorization.json --preflight-only
```

The authorized zero-step dependency validation performs native staging,
dependency probing, settlement, and controlled export without starting the
worker, environment, model, checkpoint loader, or trainer:

```powershell
python E:\Z博士\research-plans\w1-action-relative-auxiliary-runtime-dependency-repair-v1\launch_once.py --attempt-token-file E:\Z博士\.codex-tmp\w1-action-relative-auxiliary-runtime-dependency-repair-v1.token --authorization-file E:\Z博士\.codex-tmp\w1-action-relative-auxiliary-runtime-dependency-repair-v1.authorization.json --dependency-only
```

Any later full experiment would require separate authorization and would omit
`--dependency-only`. It is not authorized by this repair task:

```powershell
python E:\Z博士\research-plans\w1-action-relative-auxiliary-runtime-dependency-repair-v1\launch_once.py --attempt-token-file E:\Z博士\.codex-tmp\w1-action-relative-auxiliary-runtime-dependency-repair-v1.token --authorization-file E:\Z博士\.codex-tmp\w1-action-relative-auxiliary-runtime-dependency-repair-v1.authorization.json
```

The request remains `NOT_APPROVED`; neither command edits frozen inputs.
