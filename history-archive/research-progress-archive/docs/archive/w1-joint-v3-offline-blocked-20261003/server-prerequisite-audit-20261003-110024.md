# Retained prerequisite audit, 2026-10-03 11:00 Asia/Shanghai

The three startup prerequisites remain mandatory. This read-only probe did not create an attempt, registration record, staging directory, lock, SQLite database, model call, environment call, or GPU context.

Evidence: server-prerequisite-access-20261003-110024.json

Observed facts:

- The required registration path /media/abc_disk/admin123/lbh/dengji.txt and its intermediate mount/path do not exist. /media is not writable by user1. Non-interactive sudo is unavailable; no substitute path was created.
- GPU0 is occupied by the existing HARL process PID 1167128. GPU1 is visible with 10,988 MiB free and compute mode Default, but no scheduler or administrator evidence proves exclusive allocation. No process was modified.
- The current SSH session scope reports CPUAccounting=yes and Delegate=no. It is not writable by user1. This does not establish accounting for SSH setup, independent SFTP channels, authentication, or the post-exit tail.
- The frozen descendant meter explicitly reports remote_transport_scope_status=incomplete_requires_host_scope; it cannot be upgraded by a local flag.

Conclusion: runner_ready remains false. The research entry must continue to reject formal start until an administrator restores the specified registration mount/history, provides GPU1 exclusive allocation evidence, and provides an authoritative per-connection lifetime CPU scope or final record covering SSH/SFTP and exit tail. Research budgets remain unused; G1/G2 and all model stages are 未评价.

