# Research contract

## Controlled comparison

Variant A is a new controlled baseline: candidate remaining utility trained
with SmoothL1. Variant B uses the identical model, input, windows, optimizer,
three seeds, batch organisation, early stopping and model-selection parents,
plus one frozen window-centered SmoothL1 term. A is not described as identical
to the old candidate-row-weighted training result; that result remains
historical evidence.

For a complete legal candidate window, let `p_i` be the prediction and `y_i`
the fixed-Hungarian remaining-utility label. A uses
`mean_i SmoothL1(p_i, y_i)`. B uses that term plus `0.25 *`
`mean_i SmoothL1(p_i-mean(p), y_i-mean(y))`. Both variants average their loss
equally over complete decision windows, so candidate count cannot change the
relative weighting of A and B.

The coefficient `0.25` and SmoothL1 beta `1.0` are frozen before any new
prediction confirmation output is read. No other ranking term, architecture
change, reward shaping, GPPO fusion, or coefficient search is allowed.

## Data boundary

The sealed run supplies 219 train records over 24 parents and 68
model-selection records over 8 parents. These records are reused only after
the generated audit verifies fixed-Hungarian continuation and parent
disjointness. The 111 old prediction-evaluation records and all task-stage
records are excluded from training and model selection.

Eight new prediction-confirmation parents are selected by tape index after
excluding all 48 parents in the prior matrix. Selection occurs before outcomes
and is not based on model or rule performance. The available manifests do not
prove that no historical strategy ever used those scenarios; the claim is
limited to disjointness from this package's training, selection, and prior
diagnostic sets. The eight old prediction parents become development evidence
and are not a fresh confirmation set.

Each new parent has three fixed external repeats. Each repeat chooses at most
one public-data-qualified window, preserves no-op and no-op absence, and
retains missing opportunities. Labels use only the fixed Hungarian continuation
and are not mixed with factual-controller or old-oracle labels.

## Prediction gate

Primary metric: B minus A parent-macro selected-action regret. The gate is
declared passed only when the difference is at most `-0.005` and B is no worse
in at least 4 of 8 parents. The `0.005` effect size is one half of the frozen
`0.01` task-materiality threshold and is fixed before confirmation. The report
also gives B versus transparent-history regret, Top-1, ties, selected actions,
centered error, pairwise difference error, NOOP use, and per-parent/per-repeat
values. An eight-parent percentile bootstrap is descriptive uncertainty only;
it is not a generalisation claim.

Only a passed prediction gate may enter the conditional task comparison. Task
utility and the existing 10 ms complete-CPU mean and 50 ms wall p95 standards
remain separate decisions.

Task-stage cost is attributed by method. B acceptance requires B relative to
Hungarian utility of at least `0.01`, B CPU mean at most 10 ms, and B wall p95
at most 50 ms. A cost is reported separately and cannot enter B's cost checks.
A method with no learned decision has cost status `not_evaluated`; zero samples
cannot satisfy a cost threshold. The timing boundary includes feature
construction, three seed-model forwards, ensemble scoring, and action choice.

## Traceability

The already-counted prediction forwards must persist every candidate's three
seed predictions and ensemble mean, true-label validity, continuation identity,
input hash, both transparent baseline scores, selected action, tie rule, and
regret. The trace schema rejects incomplete candidate sets, nonfinite values,
imputed unknown targets, and seed/ensemble inconsistency.
