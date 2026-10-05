# Unique launch commands

Read-only final-package preflight; it does not consume the token or create an attempt:

```powershell
python E:\Z博士\research-plans\w1-eawm-jepa-world-model-production-repair-v3\launch_once.py --authorization-file E:\Z博士\.codex-private\w1-eawm-jepa-world-model-production-repair-v3\authorization.json --preflight-only
```

The full command below is a template only; external approval must bind the final package and budget before execution:

```powershell
python E:\Z博士\research-plans\w1-eawm-jepa-world-model-production-repair-v3\launch_once.py --attempt-token-file <approved-external-token-file> --authorization-file <approved-external-authorization-file>
```
