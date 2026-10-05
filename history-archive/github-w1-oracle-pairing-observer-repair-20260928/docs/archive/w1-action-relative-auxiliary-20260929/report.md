# Preparation report

Status: `NOT_APPROVED`. No dynamic attempt was created.

The package implements two controlled objective variants. A is absolute
remaining-utility SmoothL1. B adds a frozen coefficient `0.25` times the
SmoothL1 loss after subtracting the complete window candidate mean from both
prediction and target vectors. The model architecture, public input, fixed
Hungarian continuation, parent split, three seeds, optimizer, early stopping,
and equal per-window weighting are shared.

The sealed source audit found 219 reusable train records over 24 parents and
68 reusable model-selection records over 8 parents. The 111 old prediction
records are excluded from fitting and selection. Eight new confirmation
parents are frozen by tape index after excluding all 48 parents in the prior
matrix. The available manifests do not establish absence of historical use by
other strategy runs, so no stronger independence claim is made.

The trace schema requires three seed predictions, their ensemble mean, true
label validity, both transparent baselines, continuation identity, public
input hash, and complete legal candidate sets. `metrics.py` independently
reconstructs MAE, RMSE, parent-macro regret, Top-1, tie selection, and selected
actions from that trace.

The prediction gate is fixed before new confirmation: B minus A parent-macro
regret must be at most `-0.005`, with B no worse in at least 4 of 8 parents.
The effect size is half of the existing `0.01` task-materiality threshold.
The report must also show B versus transparent history, per-parent and
per-repeat values, centered and pairwise errors, NOOP use, and descriptive
parent-bootstrap uncertainty. A passed prediction gate is required before any
conditional task episode; task utility and CPU/wall costs remain separate.

The unique entrypoint performs identity checks and refuses dynamic execution
while the request is `NOT_APPROVED`. The resource request is derived for one
new 24-unit confirmation pass, two three-seed variants, complete trace output,
and a conditional four-arm task stage. The data-reuse audit is separately
metered with zero environment/model calls. It does not transfer historical
credit or authorize automatic retry, extension, tuning, or scene replacement.

The corrected training adapter uses one complete window per update. Therefore
the frozen upper bounds are 28,800 optimizer updates and 38,688 counted model
batch forwards, including model-selection forwards. The conditional task
stage is bound to the eight sealed task parents from the prior matrix and
allows 144 learned seed forwards; it does not reuse the eight new prediction
parents.

Preparation validation: 15 unit tests passed; no environment, reset/step,
checkpoint, model-forward, optimizer, or training call was made.
