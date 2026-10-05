# Remote Transport CPU Meter Diagnostics

These were bounded synthetic CPU tests run with the installed `Ubuntu-24.04`
WSL distribution. No SSH connection, remote command, real SFTP server, model,
training run, or formal attempt was made by this work. The parent reported a
separate read-only connection timeout and instructed that it not be retried.

## Capture limits

For runs 1 through 5, `functions.exec_command` exposed one combined `output`
field. It did not expose stdout and stderr separately. The combined capture
included a WSL localhost translation warning decoded with NULs and replacement
characters, followed by the unittest output transcribed below. Those runs
therefore record stdout/stderr as unavailable separately instead of assigning
the combined text to one stream. Run 6 used redirected .NET process streams;
its stdout and stderr are recorded separately below.

The tests assert CPU thresholds but do not print their internal JSON scope
reports. Exact measured process-tree CPU seconds and the exact SFTP stub
`process_time` value were not saved. No measured total is inferred here.

## Runs

### 1. Initial focused run, invalid SFTP evidence

Command:

```text
wsl.exe -d Ubuntu-24.04 --cd '/mnt/e/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote' python3 -m unittest -v test_transport_cpu_scope
```

stdout/stderr: not separated by the command tool; combined output contained
the WSL warning and:

```text
test_sftp_server_child_is_owned_by_the_same_root (test_transport_cpu_scope.TransportCpuScopeTests.test_sftp_server_child_is_owned_by_the_same_root) ... ok
test_waited_short_lived_grandchild_and_root_tail_are_inclusive_once (test_transport_cpu_scope.TransportCpuScopeTests.test_waited_short_lived_grandchild_and_root_tail_are_inclusive_once) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.873s

OK
```

Exit code: `0`. This apparent pass is invalid as SFTP CPU evidence: the SFTP
stub's one-line `while` syntax raised `SyntaxError`, and the test ignored the
server return code. The short-lived grandchild requested 0.36 s CPU and the
root tail requested 0.08 s. The SFTP stub did not execute its 0.07 s loop;
only unspecified interpreter/error handling CPU occurred.

### 2. Full related run, assertion failure and invalid SFTP evidence

Command:

```text
wsl.exe -d Ubuntu-24.04 --cd '/mnt/e/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote' python3 -m unittest -v test_transport_cpu_scope test_linux_process_scope
```

stdout/stderr: not separately exposed; combined output contained the WSL
warning and:

```text
test_sftp_server_child_is_owned_by_the_same_root (test_transport_cpu_scope.TransportCpuScopeTests.test_sftp_server_child_is_owned_by_the_same_root) ... ok
test_waited_short_lived_grandchild_and_root_tail_are_inclusive_once (test_transport_cpu_scope.TransportCpuScopeTests.test_waited_short_lived_grandchild_and_root_tail_are_inclusive_once) ... FAIL
test_nonmonotonic_snapshot_delta_fails_closed (test_linux_process_scope.LinuxProcessScopeTests.test_nonmonotonic_snapshot_delta_fails_closed) ... ok
test_reaper_kills_only_post_baseline_adopted_children (test_linux_process_scope.LinuxProcessScopeTests.test_reaper_kills_only_post_baseline_adopted_children) ... ok
test_waited_grandchild_ticks_survive_identity_disappearance_and_error_exit (test_linux_process_scope.LinuxProcessScopeTests.test_waited_grandchild_ticks_survive_identity_disappearance_and_error_exit) ... ok

======================================================================
FAIL: test_waited_short_lived_grandchild_and_root_tail_are_inclusive_once (test_transport_cpu_scope.TransportCpuScopeTests.test_waited_short_lived_grandchild_and_root_tail_are_inclusive_once)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/mnt/e/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote/test_transport_cpu_scope.py", line 60, in test_waited_short_lived_grandchild_and_root_tail_are_inclusive_once
    self.assertIn("sshd session/channel CPU", " ".join(report["unmeasured_remote_transport_components"]))
AssertionError: 'sshd session/channel CPU' not found in "sshd session/channel process that starts this root, including setup before root start and cleanup after root exit sshd exec or SFTP subsystem server processes used by independent channels this root's own final report serialization and process-exit tail after the snapshot"

----------------------------------------------------------------------
Ran 5 tests in 1.278s

FAILED (failures=1)
```

Exit code: `1`. This run also had the invalid SFTP stub described in run 1.
Synthetic CPU requests were 0.36 s grandchild plus 0.08 s root tail; the
existing Linux process-scope test additionally requested 0.22 s grandchild
CPU. The other existing tests used sleeps or validation only.

### 3. Full related run, apparent pass still invalid for SFTP

Command:

```text
wsl.exe -d Ubuntu-24.04 --cd '/mnt/e/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote' python3 -m unittest -v test_transport_cpu_scope test_linux_process_scope
```

stdout/stderr: not separately exposed; combined output contained the WSL
warning and:

```text
test_sftp_server_child_is_owned_by_the_same_root (test_transport_cpu_scope.TransportCpuScopeTests.test_sftp_server_child_is_owned_by_the_same_root) ... ok
test_waited_short_lived_grandchild_and_root_tail_are_inclusive_once (test_transport_cpu_scope.TransportCpuScopeTests.test_waited_short_lived_grandchild_and_root_tail_are_inclusive_once) ... ok
test_nonmonotonic_snapshot_delta_fails_closed (test_linux_process_scope.LinuxProcessScopeTests.test_nonmonotonic_snapshot_delta_fails_closed) ... ok
test_reaper_kills_only_post_baseline_adopted_children (test_linux_process_scope.LinuxProcessScopeTests.test_reaper_kills_only_post_baseline_adopted_children) ... ok
test_waited_grandchild_ticks_survive_identity_disappearance_and_error_exit (test_linux_process_scope.LinuxProcessScopeTests.test_waited_grandchild_ticks_survive_identity_disappearance_and_error_exit) ... ok

----------------------------------------------------------------------
Ran 5 tests in 1.363s

OK
```

Exit code: `0`. The SFTP stub was still syntactically invalid and its return
code was not checked, so this apparent pass is invalid as SFTP CPU evidence.
Synthetic CPU requests were 0.36 s grandchild plus 0.08 s root tail; the
existing Linux process-scope test additionally requested 0.22 s grandchild
CPU. No exact process-tree total was printed or saved.

### 4. Corrected focused run

Command:

```text
wsl.exe -d Ubuntu-24.04 --cd '/mnt/e/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote' python3 -m unittest -v test_transport_cpu_scope.TransportCpuScopeTests.test_sftp_server_child_is_owned_by_the_same_root test_transport_cpu_scope.TransportCpuScopeTests.test_waited_short_lived_grandchild_and_root_tail_are_inclusive_once
```

stdout/stderr: not separately exposed; combined output contained the WSL
warning and:

```text
test_sftp_server_child_is_owned_by_the_same_root (test_transport_cpu_scope.TransportCpuScopeTests.test_sftp_server_child_is_owned_by_the_same_root) ... ok
test_waited_short_lived_grandchild_and_root_tail_are_inclusive_once (test_transport_cpu_scope.TransportCpuScopeTests.test_waited_short_lived_grandchild_and_root_tail_are_inclusive_once) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.739s

OK
```

Exit code: `0`. The SFTP CPU stub now used valid multiline Python, emitted a
completion marker and its measured `process_time` to captured stderr, and had
its exit code checked. Its requested CPU loop was 0.07 s. The grandchild and
root-tail requests were 0.36 s and 0.08 s. The test compared root waited-child
ticks against the stub's emitted CPU value and required a 0.06 s minimum
between tail snapshots. Exact values were consumed by assertions but not
printed or saved.

### 5. Corrected full related run

Command:

```text
wsl.exe -d Ubuntu-24.04 --cd '/mnt/e/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote' python3 -m unittest -v test_transport_cpu_scope test_linux_process_scope
```

stdout/stderr: not separately exposed; combined output contained the WSL
warning and:

```text
test_sftp_server_child_is_owned_by_the_same_root (test_transport_cpu_scope.TransportCpuScopeTests.test_sftp_server_child_is_owned_by_the_same_root) ... ok
test_waited_short_lived_grandchild_and_root_tail_are_inclusive_once (test_transport_cpu_scope.TransportCpuScopeTests.test_waited_short_lived_grandchild_and_root_tail_are_inclusive_once) ... ok
test_nonmonotonic_snapshot_delta_fails_closed (test_linux_process_scope.LinuxProcessScopeTests.test_nonmonotonic_snapshot_delta_fails_closed) ... ok
test_reaper_kills_only_post_baseline_adopted_children (test_linux_process_scope.LinuxProcessScopeTests.test_reaper_kills_only_post_baseline_adopted_children) ... ok
test_waited_grandchild_ticks_survive_identity_disappearance_and_error_exit (test_linux_process_scope.LinuxProcessScopeTests.test_waited_grandchild_ticks_survive_identity_disappearance_and_error_exit) ... ok

----------------------------------------------------------------------
Ran 5 tests in 1.249s

OK
```

Exit code: `0`. Synthetic CPU requests were 0.36 s grandchild, 0.08 s root
tail, and 0.07 s SFTP stub loop. The existing Linux process-scope test
additionally requested 0.22 s grandchild CPU. These are requested synthetic
workloads, not measured aggregate CPU totals.

### 6. Latest focused run after child-exit propagation fix

The grandchild wrapper was changed to propagate its child's exit status with
`sys.exit(child.wait())`.

Command:

```text
wsl.exe -d Ubuntu-24.04 --cd /mnt/e/Z博士/research-plans/w1-task-outcome-g1-g2-joint-production-acceptance-v3-metered-remote python3 -m unittest -v test_transport_cpu_scope.TransportCpuScopeTests.test_waited_short_lived_grandchild_and_root_tail_are_inclusive_once
```

This run used redirected .NET process streams. `stdout` was empty. The captured
`stderr` value (JSON-escaped by the command wrapper) was:

```text
w\u0000s\u0000l\u0000:\u0000 \u0000�hKm0R \u0000l\u0000o\u0000c\u0000a\u0000l\u0000h\u0000o\u0000s\u0000t\u0000 \u0000�N\u0006tM�n\f�FO*g\\��P0R \u0000W\u0000S\u0000L\u0000\u00020N\u0000A\u0000T\u0000 \u0000!j\u000f_\u000bN�v \u0000W\u0000S\u0000L\u0000 \u0000\rN/e\u0001c \u0000l\u0000o\u0000c\u0000a\u0000l\u0000h\u0000o\u0000s\u0000t \u0000�N\u0006t\u00020\r\u0000\n\u0000test_waited_short_lived_grandchild_and_root_tail_are_inclusive_once (test_transport_cpu_scope.TransportCpuScopeTests.test_waited_short_lived_grandchild_and_root_tail_are_inclusive_once) ... ok\n\n----------------------------------------------------------------------\nRan 1 test in 0.543s\n\nOK\n```

Exit code: `0`. Synthetic CPU requests were 0.36 s grandchild plus 0.08 s root
tail. No exact process-tree total was printed or saved.

## Interpretation

Only runs 4 through 6 validate the corrected synthetic behavior. The SFTP
case uses a Python CPU stub launched through the SFTP child API; it does not
exercise OpenSSH `sftp-server` or the SFTP wire protocol. A verified descendant
tree is not a complete remote SSH/SFTP total: sshd setup, independent channel
servers, and the post-snapshot root-exit tail remain outside the user-owned
meter. A host-admin configured, validated per-connection accounting scope is
still required before any full remote transport CPU status can pass.
