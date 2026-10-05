# Server audit update, 2026-10-03

## Current status

SSH to user1@172.17.27.173 works. This dated update supplements the earlier connection-failure report; it does not overwrite that report or any frozen package.

runner_ready=false. No engineering workload or research attempt has started. No registration, staging, token creation/consumption, real environment call, torch import, CUDA context, model initialization/forward/update, or checkpoint operation occurred in this probe.

## Runtime identity discrepancy resolved

Actual command on Windows:

    & 'E:\\Z博士\\.runtime-tools\\w1-ssh-controller-v1\\Scripts\\python.exe' -B 'E:\\Z博士\\diagnostic-work\\w1-joint-production-acceptance-v3-20261002\\runtime_record_identity_probe.py'

The controller verified its pinned runtime and host key, then invoked the frozen remote interpreter with -I -B. Password was entered through getpass and retained only in memory. Exit code: 0. Probe wall: 2.73760089999996 seconds, including connection and client close. This is a read-only diagnostic, not a production resource settlement. Complete SSH lifetime CPU was not measured.

Remote interpreter: /home/user1/.venvs/w1-runtime-v1/bin/python, Python 3.10.12. All 25 dependency versions and raw RECORD SHA-256 values match the source frozen runtime identity; dependency set and Python binary SHA-256 also match.

The earlier metadata probe used distribution.read_text('RECORD').encode(), normalizing CRLF to LF. Raw byte hashes match the frozen record, while text hashes reproduce the prior discrepancies. CRLF counts: torch 12774, NumPy 1407, filelock 48. No runtime repair or dependency installation is justified by that discrepancy.

Scope: this probe checked metadata and the interpreter binary. It did not rehash all dependency payloads or execute the full production runtime validator, import torch, or test GPU calculations. It is not server production-chain acceptance.

Evidence:
E:\\Z博士\\diagnostic-work\\w1-joint-production-acceptance-v3-20261002\\runtime-record-identity-20261003-104304.json
SHA-256: 032c2cc230e1f74ad0f93c090a59fa21476b727c76daf2b509c6f34781c1b6ad

## Remaining blocking conditions

1. Mandatory registry /media/abc_disk/admin123/lbh/dengji.txt and /media/abc_disk parents do not exist. /media is not writable by user1. No record was appended. Administrator must restore the correct mount/path and history with lock-append access; creating a substitute file would not meet the user's registration rule.
2. GPU1 has 10988 MiB free at this snapshot, but exclusive allocation is not proven. GPU0/HARL was not modified. The frozen GPU allocation requirement remains unchanged.
3. Current SSH session scope has CPUAccounting=yes and Delegate=no, and its cgroup is not writable by user1. The user systemd manager responds, but that does not demonstrate counting of SSH authentication, separate SFTP channels or final exit. No host accounting permission or final lifetime record is available. Full CPU acceptance remains incomplete; the historical 0.274925s discrepancy remains unreconstructible from preserved records, not classified as noise.
4. Development-only receipt validation and production release wiring are still incomplete; no successful server acceptance receipt exists. The v3-metered-remote directory remains unfrozen and unlaunchable. Copied source identities are not new approved execution identities.

External prerequisites were requested together. No new scientific design, seed, threshold, dataset, budget or fallback registration path was introduced.

## Local executor and archive

Windows briefly returned CreateProcess error 1455 (pagefile too small), and one process returned -1073741502. The two yielded read-only sessions no longer existed when checked. A minimal cmd /c ver and subsequent commands succeeded using login=false. Recovery does not establish the underlying Windows memory cause is fixed.

Git signing remains gpg.format=ssh, commit.gpgsign=true; ssh-agent is Stopped/Disabled. No unsigned commit or push was attempted. Remote archive is NOT COMPLETE. Existing 535 staged files and index are preserved. Index SHA-256: 401ea4494d8a47cada77d4687bb3119f2a111d2b771433f69a96c5b2464ff421. No index.lock exists or was removed.

This report's archive copy is a small local summary, not a runnable package or a remote commit. All data, G1/G2 training and prediction outcomes remain unevaluated. GPPO calls remain zero.

