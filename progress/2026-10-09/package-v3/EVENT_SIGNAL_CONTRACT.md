# Event Signal Contract

`event_signal` is a decision-prefix field. Its semantics are frozen as `pre_action_public_observation`: the value is computed from the producer's trigger flags at the moment the public state is exposed, before a candidate action is executed.

`0.0` is a valid false value when `event_signal_valid=true`; it is not missing. A missing field, non-boolean validity flag, non-finite value, or `event_signal_valid=false` is explicit unknown and is rejected before graph encoding or model forward. Unknown is never converted to zero.

Restoring a candidate branch must preserve the pre-action event value. The restored simulator's internal trigger flags may represent a later or different lifecycle state, so they cannot redefine the candidate prefix. The branch probe records the expected value and validity and projects the public observation back to the frozen pre-action semantics before the full prefix comparison.

The value and validity are carried through collector, branch snapshot, label/target construction, graph/action encoding, world-model input, prediction export, replay and GPPO verification. Any missing field, type conversion, time-point substitution, or digest mismatch stops before model forward.
