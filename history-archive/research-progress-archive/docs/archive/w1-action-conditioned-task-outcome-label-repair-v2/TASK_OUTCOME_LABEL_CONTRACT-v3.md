# W1 Action-Conditioned Task Outcome Label Contract v3

Status: frozen for label-pipeline qualification; no real collection is approved.

The executable validator is `task_outcome_contract.py`, contract version
`w1-action-conditioned-task-outcome/3.0.0`. Each row is the outcome of one
legal first action followed by `hungarian-v1-fixed` until native termination or
the configured observation horizon. It is a sequence-outcome label, not a
claim that the first command alone caused a later completion.

## Identity and boundary

Each candidate is joined by the enclosing `window_id`, `parent`, and `repeat`,
plus `candidate_id`, integer `action_id`, public target task and UAV IDs,
`public_input_hash`, environment-config SHA-256, and `continuation_id`. The
candidate must be in the frozen legal-action set. Every candidate in a window
shares the same pre-branch public input hash. The input contains the public
observation, received telemetry, legal mask, and already confirmed public
continuation only. Lifecycle state, command execution identity, and all
post-decision observations are label-side audit data and never policy input.

The decision boundary is `decision_time`. The observation interval is
`(decision_time, horizon_time]`; the post-action trajectory starts at step 0,
has contiguous step numbers and strictly increasing finite times, and may not
extend past the frozen horizon. Its final row must show native termination or
truncation at the horizon for a terminal outcome to be valid. A branch that
ends earlier without a terminal event stays unresolved.

## Targets and masks

| Field | Positive/negative evidence | Unknown or masked cases |
|---|---|---|
| `physical_on_time_completion` | `true` only with `TaskLifecycle.completed_at <= deadline`; `false` only after explicit task expiry | Missing completion timestamp, unresolved episode end, or malformed lifecycle evidence is never imputed as failure |
| `task_expired` | `true` only when final `TaskLifecycle.state == expired` and final time is at or after the task deadline; `false` when an explicit on-time completion is observed | No explicit expiry by observation end remains unknown |
| `host_confirmation` | Not supervised in `continuous_service_until_deadline` | Always `value=null, valid=false`; physical completion and ordinary telemetry cannot stand in for a host-received completion message |
| `completion_by_first_command` | Audit-only identity comparison; not a model target | Rejected/lost first commands are explicitly false; unknown acceptance or a pre-existing command reused as the first ID stays unknown; later Hungarian execution is not credited to the first action |

Completion source is `TaskLifecycle.completed_at`; expiry source is the explicit
`TaskLifecycle.state` transition at the configured deadline. The label retains
the absolute deadline, decision time, observation cutoff, first-command ID and
status, initial and completion execution identities where available, raw
post-action feedback, candidate identity, and continuation identity. Command
rejection or loss does not erase a later valid sequence outcome, but the later
outcome is not renamed as a direct first-command effect.

`NOOP` has no target task; task completion and expiry masks are false. A
no-opportunity window is not converted into a candidate row or zero label.
Malformed identity, non-finite time, non-contiguous trajectory, conflicting
completion/termination evidence, or a continuation mismatch raises a contract
error and stops collection.

## Runtime configuration

The only supported task semantics in this version are
`continuous_service_until_deadline / physical_service`, with host completion
notice unavailable. Before constructing the environment, the collector
instantiates the exact frozen `M10Config` values, validates them against
`environment-config-contract.json`, and writes `environment.json` from that
same runtime object. A mismatch aborts before the first reset. The label also
stores the canonical configuration SHA-256.

This contract supersedes neither old schemas nor old results. Historical
arrival-to-region labels remain separately identified and are not pooled.
