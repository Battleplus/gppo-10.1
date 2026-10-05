# Zero-Call Verification

`python -B -m unittest discover -s research-plans/w1-action-conditioned-task-outcome-source-check-v1 -p test_*.py -v`
returned exit code 0: three tests passed. The independent source-check command
returned exit code 0 and identified the default mode as
`continuous_service_until_deadline`, `physical_completed_at_in_lifecycle=true`,
`physical_completed_at_in_step_info=false`, and
`completion_notice_enabled_in_default_mode=false`. It also records that the
collector and runtime hooks instantiate default configuration while the named
source run's `environment.json` records `arrival_to_region`.

Synthetic counts: one lifecycle task completion, one lifecycle task expiry;
W1 environment constructions/resets/steps: 0; model initializations/loads,
forwards, optimizer updates, and checkpoint writes: 0. No dynamic attempt was
created. These tests validate a bottom boundary and source contract, not the
production collector or any research effect.
