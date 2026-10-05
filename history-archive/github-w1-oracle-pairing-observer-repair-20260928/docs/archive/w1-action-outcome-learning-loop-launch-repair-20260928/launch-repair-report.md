# Windows to WSL launch repair

Status: `NOT_APPROVED`

The prior attempt stopped before dynamic execution because its Windows process
constructed `Path('/home/asus/...')` and passed that value to Windows copy and
directory-fsync operations. Windows interpreted it on the current `E:` drive,
creating a one-file partial directory before `PermissionError`.

The repaired chain has two explicit ownership boundaries:

1. `launch_once.py` runs on Windows. It verifies the frozen package and external
   token, asks the named WSL distribution to convert the source path, and starts
   the Linux entry with a subprocess argument array. The token travels only on
   stdin. A Windows preflight failure emits a structured technical-stop record
   with stage, wall/CPU time, and explicit no-staging/no-worker flags.
2. `wsl_stage_and_launch.py` runs under the frozen Linux interpreter. It checks
   Linux mount metadata, creates and fsyncs native state, invokes the supervisor,
   settles, and performs the single verified export from Linux.

`launcher-status.json` is durable before the first package file copy. A staging
failure remains on ext4 and blocks reuse. File-operation retries retain the
three-attempt/two-second limit; worker, environment, model, and training calls
are never retried.

The repair does not change the parent matrix, opportunity rule, fixed
Hungarian continuation, labels, model architecture, seeds, training settings,
prediction gate, task arms, materiality thresholds, or any dynamic-call maximum.
The existing 600-second/1,200-CPU-second staging and export ceilings already
include process startup and file copying, so no resource ceiling is expanded.

The Windows preflight wall/CPU is included in the staging gate. The total wall
check combines Windows preflight with the Linux entry lifetime. The total CPU
check adds three non-overlapping measurements: Windows preflight process CPU,
Linux launcher process CPU, and the supervisor's complete-process CPU for its
worker subtree. The resulting settlement is appended to the same verified
export manifest as `export-status.json`. The small Windows interval after
preflight is not visible to Linux and is explicitly marked unavailable; the
Windows wrapper separately checks and prints its full outer wall interval.
