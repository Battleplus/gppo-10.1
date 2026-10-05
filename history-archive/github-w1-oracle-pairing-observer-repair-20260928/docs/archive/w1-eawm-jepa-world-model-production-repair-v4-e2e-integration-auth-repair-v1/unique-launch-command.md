# Unique launch commands

Read-only structure and identity preflight. This accepts the frozen `NOT_APPROVED` status, does not grant approval, reads no token, and does not create an attempt:

```powershell
python "E:\Z博士\research-plans\w1-eawm-jepa-world-model-production-repair-v4-e2e-integration-auth-repair-v1\launch_once.py" --authorization-file "E:\Z博士\.codex-private\w1-eawm-jepa-world-model-production-repair-v4-e2e-integration-auth-repair-v1\authorization.json" --preflight-only
```

Formal start template only. Obtain a new external authorization and one-time token bound to this final identity; the resource request remains `NOT_APPROVED`:

```powershell
python "E:\Z博士\research-plans\w1-eawm-jepa-world-model-production-repair-v4-e2e-integration-auth-repair-v1\launch_once.py" --attempt-token-file "<new-external-token-file>" --authorization-file "<new-approved-authorization-file>"
```
