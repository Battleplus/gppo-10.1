# Offline prediction-accuracy and action-ranking diagnosis

## Decision

The frozen conclusion remains **prediction gate not passed**. No task
comparison was run or reconstructed.

The saved evidence does not support a definitive answer to whether the learned
MAE gain came mainly from fitting each window's overall utility level. That
question requires the learned score for every candidate. Those scores were not
persisted, and this analysis was prohibited from loading checkpoints or adding
model forwards.

The evidence does establish a concrete objective mismatch: the model was
trained with candidate-row Smooth L1 on absolute remaining utility, while the
failed gate component depends on within-window action ordering.

## Numerical evidence

The frozen learned ensemble has parent-macro MAE `0.046080960`, compared with
`0.202324152` for current-public and `0.209299444` for transparent-history.
However, learned regret is `0.007620270`, above both baselines' `0.006897738`,
and learned Top-1 is `0.375`, below both baselines' `0.500`.

Since every parent has three windows, the aggregate Top-1 values imply 9 exact
best selections for the learned ensemble and 12 for each baseline. Aggregate
regret sums are `0.182886482` learned and `0.165545702` baseline, a learned
increase of `0.017340781`. The missing predictions prevent locating that
increase to particular windows, actions, NOOP choices, ties, acceptance states,
or individual seeds.

For the two saved public baselines, parent-macro window-level absolute MAE drops
sharply after removing each window's candidate mean:

| Measure | Current public | Transparent history |
|---|---:|---:|
| Window absolute MAE | 0.202021970 | 0.207077383 |
| Window-centered MAE | 0.029063686 | 0.030124831 |
| Pairwise difference MAE | 0.043821208 | 0.045804765 |
| Pairwise direction accuracy | 0.649206 | 0.641270 |
| Predicted score span | 0.071361439 | 0.071955224 |

The true utility span averages `0.053612763`; the best-versus-second-best gap
averages `0.007394222`. One window has an exact best tie. Exploratorily, without
changing frozen Top-1, 9/24 gaps are at most `0.001` and 21/24 are at most
`0.005`. Thus many windows require fine relative discrimination. The baseline
span comparison does not indicate score compression for those baselines.
Learned score span, centered MAE, pairwise difference error, and pairwise
direction accuracy are unavailable.

## Where saved regret occurs

Current-public and transparent-history selected the same action in all 24
windows, so their window regret and acceptance rows are identical. Neither
selected NOOP, although NOOP was legal in all windows and truly optimal in four.

The largest baseline loss is
`prediction_evaluation:train-0227:r1:s4`: action 0 was accepted and incurred
regret `0.127821079`, which is 77.21% of total baseline regret. Accepted actions
account for `0.163606324` of the `0.165545702` total. One selected action had
unknown acceptance and contributes `0.001939378`; one transport-lost selected
action has zero regret. Baseline loss therefore is not primarily explained by
NOOP selection or failed command transport. Learned loss concentration and all
learned-versus-transparent different-action windows cannot be recovered from
the saved aggregates.

`window-comparison.csv` gives the true range, best/second gap, baseline spans,
centered errors, pairwise diagnostics, selections, regret, and acceptance for
every window. `parent-repeat-regret.csv` preserves the requested parent/repeat
layout and marks every learned field `NOT_SAVED`. `parent-summary.csv` provides
parent-level averages. `candidate-evidence.csv` contains all 111 candidate
labels and baseline scores, with missing learned scores explicitly marked.

## Training-target review

The feature contract correctly includes a 25-way candidate-action one-hot in
addition to the frozen 770-value public observation and 32 history features.
Train, model-selection, and prediction parents are disjoint. All evaluation
records use the same fixed Hungarian continuation.

The loss contains no window-centering or ranking term. Each candidate is one
training row, so windows with more candidates receive more training weight: the
48 training windows contributed 48 rows from 3-candidate windows, 52 from
4-candidate windows, 40 from 5-candidate windows, 70 from 7-candidate windows,
and 9 from one 9-candidate window. This weighting follows from the frozen
implementation; this diagnostic does not claim it caused the failed ranking.

## Testable next hypothesis

A specific next hypothesis is supported for discussion: under the same public
features, architecture, fixed Hungarian continuation, and parent-group split,
an objective that explicitly represents within-window relative utility, such
as a centered-utility or pairwise-ranking auxiliary term, may reduce selected
action regret even when absolute MAE is already low.

This is a hypothesis, not an approved change. A valid test would need to persist
per-seed and ensemble candidate predictions, predefine tie handling and loss
weights, and evaluate once on newly frozen parents that have not been used for
this diagnosis. It must compare absolute MAE, centered error, pairwise ranking,
regret, and eventual task outcomes without tuning on these eight parents.

The current data cannot distinguish that hypothesis from alternatives such as
seed instability or a few learned action changes. No expansion, training, or
new sampling follows automatically from this report.
