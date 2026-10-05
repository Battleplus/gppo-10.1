# Final package read-only preflight

Package: `E:\Z博士\research-plans\w1-action-relative-auxiliary-production-backend-completion-v1`

Command:

```powershell
python E:\Z博士\research-plans\w1-action-relative-auxiliary-production-backend-completion-v1\launch_once.py --authorization-file E:\Z博士\.codex-tmp\w1-action-relative-auxiliary-production-backend-completion-v1.authorization.json --preflight-only
```

Windows launcher exit code: `0`.

WSL Ubuntu-24.04 returned `0` and emitted:

```json
{"attempt":"w1-action-relative-auxiliary-production-backend-completion-v1-once","schema":"w1-linux-preflight/2.0.0","staging_started":false,"status":"preflight_pass","worker_started":false}
{"attempt":"w1-action-relative-auxiliary-production-backend-completion-v1-once","elapsed_wall_seconds":0.5595163000016328,"staging_started":false,"status":"preflight_pass","worker_started":false,"wsl_returncode":0}
```

The WSL wrapper also emitted a localized informational warning before the JSON
line; it did not alter the return code. No native attempt directory, lock,
worker, environment, model, checkpoint, SQLite ledger, or export was created.

Final identity:

- execution-manifest.json SHA-256: `77a83b74c06326953517728f38b2532f58b2f1774579901396a740d06cf83a41`
- hashes.json SHA-256: `0b26832fbd52c51868b48b2693bdcdc38bda230a9e4526b89a43ff156aeb9e17`
- RESOURCE_REQUEST status: `NOT_APPROVED`
