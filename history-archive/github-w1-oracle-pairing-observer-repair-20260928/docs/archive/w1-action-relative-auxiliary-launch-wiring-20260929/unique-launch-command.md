# Unique launch command

Preparation check:

```powershell
python E:\Z博士\research-plans\w1-action-relative-auxiliary-launch-wiring-v1\launch_once.py --attempt-token-file E:\Z博士\.codex-tmp\w1-action-relative-auxiliary-launch-wiring-v1.token
```

The token file must be outside the package and is never written to package
state. The request remains `NOT_APPROVED`; the external token authorizes only
the exact frozen attempt and does not modify the request or any input hash.
