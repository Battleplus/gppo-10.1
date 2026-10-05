# Arrival Reward Boundary Erratum

The native M10 completion record retains its historical inclusive deadline
flag: `physical_arrival_before_deadline` is computed with `arrival_time <=
deadline` in both the ordinary and bounded-retry notice paths. Native reward
accounting consumes that flag without modification. The task audit added in
`runtime_hooks.task_outcome_labels` deliberately labels physical on-time
completion with the strict predicate `physical_arrival_time < deadline`,
matching the task-lifecycle contract that expires a task at exact deadline
equality. Host confirmation remains independently inclusive (`<= deadline`)
in both native recording and the audit.

Consequently, the native physical-arrival reward flag and the strict audited
task label are not interchangeable at exact equality. The task audit is the
terminal-label authority for this validation package; it does not rewrite
native reward values or completion records. No native simulator or reward
source was changed for this correction.

Pure component regression `test_arrival_deadline_boundary.py` calls the native
ServiceClock and TaskLifecycle with arrivals before, at and after the deadline.
At exact equality, `arrive` first expires the task and then raises
`ValueError: Arrival requires the assigned UAV`; it does not create a completed
task or arrival log. Thus an inclusive completion-record flag at equality is
not reachable through that moving-arrival path. This is also a retained native
technical-stop edge case, not evidence that equality is harmless. The protocol
stops and preserves evidence if it occurs. No real environment was constructed
for this regression, and no historic reward or result was changed.
