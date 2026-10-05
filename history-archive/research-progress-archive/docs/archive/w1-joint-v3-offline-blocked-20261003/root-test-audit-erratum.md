# CPU stub test evidence correction

The first reported pass for
`TransportCpuScopeTests.test_sftp_server_child_is_owned_by_the_same_root`
is not valid evidence of its intended synthetic CPU workload.

Root source review found that the child command combined a compound `while`
statement after a semicolon. The child therefore had invalid Python syntax,
and the test did not assert its exit status. Interpreter startup CPU could
satisfy its weak waited-tick assertions. This says nothing about a real SFTP
server or a real server run.

The worker was instructed to use valid multiline Python, require exit zero,
check completion output and child process CPU, and compare a separate root-tail
CPU delta. Only the corrected rerun may be used as finite synthetic CPU-work
evidence. Preserve the first pass and this correction; do not rewrite it into
a successful server acceptance claim.

No model, environment, checkpoint or server command was used by these tests.
