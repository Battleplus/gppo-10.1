$package = Split-Path -Parent $MyInvocation.MyCommand.Path
$windows = 'E:\Z博士\.codex-exports\w1-eawm-jepa-world-model-production-repair-v4-e2e-integration-auth-repair-v1-once\run-once\world-model-windows.jsonl'
$source = 'E:\Z博士\research-plans\w1-eawm-jepa-world-model-production-repair-v4-e2e-integration-auth-repair-v1\production_data.py'
python -B (Join-Path $package 'audit_saved_windows.py') --windows $windows --source $source --out (Join-Path $package 'data-availability-audit.json')
