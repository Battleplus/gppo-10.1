# Unique commands

Read-only final package validation, no token consumption and no attempt:

```powershell
python E:\Z博士\research-plans\w1-action-relative-auxiliary-production-backend-completion-v1\launch_once.py --authorization-file E:\Z博士\.codex-tmp\w1-action-relative-auxiliary-production-backend-completion-v1.authorization.json --preflight-only
```

The eventual one-shot execution requires the external authorization file and
the separate token file outside the package:

```powershell
python E:\Z博士\research-plans\w1-action-relative-auxiliary-production-backend-completion-v1\launch_once.py --attempt-token-file E:\Z博士\.codex-tmp\w1-action-relative-auxiliary-production-backend-completion-v1.token --authorization-file E:\Z博士\.codex-tmp\w1-action-relative-auxiliary-production-backend-completion-v1.authorization.json
```

The request remains `NOT_APPROVED`; neither command edits frozen inputs.
