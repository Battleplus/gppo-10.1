# Fixed-continuation learning target

Date: 2026-09-28. Status: implementation complete, execution `NOT_APPROVED`.

## Primary target

For each legal candidate at a frozen public decision state, the input is the
candidate first action plus information legally available before that action.
The label is discounted remaining utility from that decision through native
termination after forcing the candidate once and using
`hungarian-v1-fixed` thereafter. The utility uses the original reward vectors,
preference `(0.8, 0.2)`, discount `0.99`, task scale `0.5`, six-task capacity,
and 36-unit initial fleet energy. No reward or preference is changed.

Energy, first-command fate, physical completion, reassignment, expiration, and
host confirmation remain explanation labels. They are not additional primary
training heads in this first learner. Host-confirmation unknown remains unknown
and does not replace physical completion or reward-derived utility.

## Input boundary

The 827-dimensional learner input is the complete native public `flat[770]`
vector, a fixed `history[32]` summary of up to four earlier public observations,
and a 25-way one-hot candidate identity. Each history lag contributes its age,
legal-candidate count, public UAV and task known/valid counts, continuation
count, and public event signal.

The history is truncated at the decision. Hidden faults, environment state,
undelivered messages, branch results, and feedback from the candidate action
cannot enter the input.

## Provenance boundary

The repaired fair rerun contains 774 real factual-action labels and 3,321
unknown unexecuted candidates. Those labels are valid only for their saved
controller continuations. The sealed oracle contains 144 real branches under
`hungarian-v1-fixed`, but they are development data from the closed first-window
analysis. `training_eligible=false` means these records are not admitted to the
new split; it does not mean their labels are absent or fabricated.

New learning records must be true branches, contain every legal candidate for
one public state, use `hungarian-v1-fixed`, and belong to exactly one frozen
parent group. Factual-policy and fixed-Hungarian labels cannot be pooled.

## Task use

The task experiment follows Hungarian until the first predeclared qualifying
state at or after decision step 4. The transparent or learned selector may
change exactly that one action. Hungarian resumes on the next observation and
continues to termination. The learned score is the arithmetic mean of all
three frozen training seeds; no seed is selected on task results. This is not
GPPO fusion and does not estimate GPPO continuation value.

No-op remains a candidate only when the shared execution filter allows it. Its
utility label is valid; target-task fields remain masked.
