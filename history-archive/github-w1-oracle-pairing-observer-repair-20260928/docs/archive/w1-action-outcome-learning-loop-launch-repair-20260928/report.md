# Action-outcome learning-loop preparation report

## What now runs

The package contains a real lazily constructed PyTorch MLP, a strict training
schema, parent-grouped loading, three-seed supervised training, checkpoint
identity checks, independent prediction metrics, a fail-closed prediction gate,
and a paired one-shot task controller. The runtime bridge reuses the repaired
environment, public adapter, Hungarian selector, packet observer, labeler,
native-filesystem supervisor, and verified export.

The old 7,344-step label-feasibility request is withdrawn. The new
`RESOURCE_REQUEST.json` covers one attempt: fixed-continuation label collection,
limited training and model selection, independent prediction evaluation, and
task comparison only when prediction gates pass.

## Data facts

The repaired-run export contains 774 real factual labels and 3,321 unknown
unexecuted candidates. The completed oracle has 144 real fixed-Hungarian branch
labels. They remain development evidence with different continuation provenance
and are not pooled into the new training split. `training_eligible=false` is an
admission decision, not a claim that existing labels are invalid.

The new matrix freezes 48 outcome-unselected parents from the existing training
tape into new-learner-disjoint roles: 24 train, 8 selection, 8 prediction
evaluation, and 8 task comparison.
No heldout/test result is read. Every new supervised decision requires complete
true branches for its legal candidate set.

## Budget and gates

The matrix-derived upper bound is 42,480 environment steps, 160 resets, 2,200
branches, 5,700 optimizer updates, 7,902 model batch forwards, and 483,600 model
sample evaluations. The 1,296 task steps are conditional and remain unused if
prediction fails. Stage limits cannot borrow and historical use receives no
credit. Time and storage values are conservative technical-stop ceilings, with
their measured-rate and artifact-size basis recorded in `experiment-design.md`
and `RESOURCE_REQUEST.json`; they are not predicted consumption.

Prediction must strictly beat current-public and transparent-history
parent-macro MAE and improve transparent selected-action regret. Only then do
the eight untouched task parents run. Task criteria remain `0.01`, `10 ms` CPU
mean, and `50 ms` wall p95. The measured decision interval includes feature and
tensor construction, all seed forwards, ensemble averaging, and action choice;
any nonfinite task prediction stops the attempt.

Preparation constructed no environment or neural network, loaded no checkpoint,
performed no model forward, and ran no optimizer update. Tests use contracts,
synthetic records, fake predictors, and a fake pipeline backend. Passing them
shows wiring, not prediction accuracy or task benefit.

Once approved with the matching external token, this attempt can produce an
actual prediction comparison and, after a successful prediction gate, a paired
task comparison. No label-feasibility experiment remains. The unavoidable
empirical conditions are sufficient qualifying states in each frozen data group
and a learner that beats the transparent baseline before task episodes.
