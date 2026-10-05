# Unique launch command

Preparation identity and preflight:

```powershell
python E:\Z博士\research-plans\w1-action-relative-auxiliary-v1\launch_once.py
```

The command must print `NOT_APPROVED` while `RESOURCE_REQUEST.json` is
`NOT_APPROVED`. A future approved run must use the exact frozen package,
`--execute`, and the externally supplied one-shot value in
`W1_ACTION_RELATIVE_AUXILIARY_TOKEN`; the token is never stored in this
package. The preparation entrypoint intentionally refuses to attach a dynamic
runner in this round.
