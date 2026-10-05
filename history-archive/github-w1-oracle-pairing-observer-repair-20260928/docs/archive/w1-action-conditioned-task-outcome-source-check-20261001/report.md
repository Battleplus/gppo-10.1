# Action-Conditioned Task Outcome Source Check

## Finding

Under the unchanged W1-light default `continuous_service_until_deadline` mode,
physical service completion is an internal, timestamped event, but the
production one-step label export does not record it. A distinct host completion
confirmation is structurally unavailable: `_emit_completion_notices` returns
before emitting any completion message in this mode. Ordinary task telemetry
reports `pending` and `remaining_service`; it does not certify a completion
message receipt. A delayed `pending=0` observation could also mean an assigned
or expired task, so it cannot be relabeled as host confirmation.

This supports a data-contract and observation-horizon diagnosis. It does not
show whether a model trained on valid task outcomes could distinguish actions.
The sealed `prediction_gate_stop` remains unchanged.

## Evidence

The checked native files are `runs/w1-light-repaired-fair-rerun-v2-nativefs-once/native/gppo_world/m10_environment.py`
(SHA-256 `0f4615d11604a7cd1cc74c94f0258dfc6cb59748f64a075276ec90941db094b1`)
and `task_lifecycle.py`
(SHA-256 `50248afa519fd18dbc5995ba0bbf7e58adc0833ae501760ea032ab91fca5ed89`).

There are two distinct configuration identities. The named source run's
`environment.json` records `arrival_to_region` / `physical_arrival` (SHA-256
`95b8b6e293fd7f500c3e89114424ff850717914cd3950a01e5cfbfb426f625ae`).
The world-model collector actually instantiates bare `M10Config()` at
`production_data.py:247`, and the policy and task hooks do likewise in
`runtime_hooks.py:243,391-392`. Those calls use the native defaults
`continuous_service_until_deadline` / `physical_service`. The collector and
hooks do not load the source run's `environment.json`. This check identifies
the implementation that produced the sealed labels; it does not establish
which configuration was originally intended. A future protocol must name
the intended task contract explicitly before requesting dynamic collection.

- `m10_environment.py:55` fixes the default mode. `_emit_completion_notices`
  returns for every non-`arrival_to_region` mode at lines 454-457. The
  completion-record and message creation follows that return at lines 458-484.
- In the default mode, `task_lifecycle.py:107-112` sets `completed_at` when
  sufficient service finishes by the deadline. `m10_environment.py:1004-1021` exports terminal
  task states and arrival-mode `completion_records`, but no service-completion
  timestamp or command-linked service record.
- `m10_environment.py:787-798` builds ordinary task telemetry without a
  completion field. `m10_environment.py:755-756` raises a shared
  completion-or-invalidation trigger for both completed and expired tasks.
- `task_execution.py:78-105` retains command IDs and fenced task tokens. A
  future label collector would have to persist the first command identity and
  verify the completion belongs to that command rather than a later reassignment.
- The prior sealed-data audit counted 56 complete windows: train 24,
  model-selection 8, prediction confirmation 24. The 24 confirmation windows
  are 8 parents x 3 repeats and contain 330 candidates. Across all 771 saved
  candidates, both task event heads have zero valid labels and all branches
  stop before termination or horizon. This check does not consume the sealed
  task comparison set.

The earlier v1 world-model collector used the aggregate `counts.completed`
as a one-step physical-completion indicator, which would not itself identify
the selected task or subtract prior completions. The sealed v4 collector
instead sets physical completion and host confirmation to unknown in
`production_data.py:182`. Neither implementation supplies the desired
first-action-specific terminal task label.

`source_contract_check.py` parses the frozen source and asserts the mode,
completion guard, task telemetry keys, lifecycle timestamp write, step-info
fields, and actual collector/runtime configuration calls. The three tests pass. The lifecycle-only fixture records physical
completion at t=6 and expiry at t=9 without constructing an environment.
The test does not demonstrate that the current production collector emits a
command-attributed physical label; that collector still needs a future
post-action trajectory and explicit audit record.

## Consequence

An unchanged-mode continuation could make physical completion and expiry
observable to a new audited collector, subject to command attribution. It
cannot make the host-confirmation head valid through the current notice path.
No missing value should be imputed to failure. The earlier provisional
eight-parent label-only request requires revision before any authorization:
its nonzero host-confirmation coverage gate is impossible under this frozen
mode. Keep that head unknown or explicitly change the research target in a
separate protocol; do not silently switch to `arrival_to_region`.

The next bounded engineering action is a synthetic production-collector
fixture for command-linked physical completion and expiry, with no environment
or model calls. A real collection matrix or training request is premature
until that label path, parent identities, and full resource accounting are
frozen. No GPPO, world-model loss, prediction gate, or old result changes here.
