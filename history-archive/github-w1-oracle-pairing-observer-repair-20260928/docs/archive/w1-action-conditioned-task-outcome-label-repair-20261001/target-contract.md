# Action-Conditioned Task Outcome Contract

This package defines labels for a future, separately approved confirmation
collection. It does not alter the sealed experiment or infer missing labels.

For every non-NOOP candidate, the record must preserve the candidate action,
target task and UAV identity, first command ID, decision time, fixed
continuation identity, first command feedback, and a post-action trajectory
through native termination or the frozen 18-second horizon. In the frozen
`continuous_service_until_deadline` environment, physical completion needs an
explicit `TaskLifecycle.completed_at` audit record with matching task, UAV and
command identity. The separate `arrival_to_region` completion record is not a
source for this experiment. Host confirmation needs a matching completion
message received by the host; a physical completion does not imply it. Expiry
needs an explicit task terminal state. On-time flags compare the corresponding
event time to that task's deadline, without changing the native reward basis.

Missing records, a window ending before the required observation, a dropped or
unknown message, a completion by a later reassigned command, and an unavailable
host confirmation remain `unknown` for the first-action-specific head; they
are never converted to zero or failure. A NOOP has no target task, so task
completion, host confirmation, expiry, and completion-time labels are masked.

All post-action records must have time at or after the decision boundary. The
candidate action and continuation identity must match the frozen decision
input. The public-input snapshot remains pre-branch; future observations are
label-side evidence only.

The production label set should retain separate fields for:

- command acceptance (`accepted`, `rejected`, `lost`, `corrupt`, `timeout`, or `unknown`);
- physical service completion time and on-time physical completion;
- host confirmation receipt time and on-time host confirmation;
- explicit task expiry or completion-notice expiry;
- aggregate utility and energy, with the existing masks unchanged.
