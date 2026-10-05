# Evaluation-contract review

## Frozen metric definitions

The independent prediction split contains 111 candidate records grouped into
24 decision windows from 8 parents and 3 repeats per parent. Every record has
continuation identity `hungarian-v1-fixed`. Within each decision, the records
contain the complete saved legal candidate set and share the same frozen public
state, history, parent, repeat, and true-branch label source.

For each candidate, absolute error is `abs(score - remaining_utility)`. MAE is
computed by pooling candidate errors within each parent, averaging within that
parent, and then averaging the eight parent values. Windows with more candidates
therefore have more weight inside a parent's MAE.

For regret and Top-1, selection occurs once per window. The selected candidate
maximizes `(score, -action)`, so an exact score tie selects the lower action
number. Regret is `max(true utility) - true utility of selected action`. Top-1
is one exactly when regret is at most `1e-12`. Window regrets/Top-1 indicators
are averaged within parent and then across the eight parents. Because each
parent has three repeats here, these macro averages also equal the simple mean
over 24 windows.

NOOP is action 24. It remains in the legal candidate set, scoring, selection,
and true-label comparison whenever legal. NOOP was legal in all 24 evaluation
windows. It was truly optimal in four windows but neither public baseline
selected it. This diagnostic does not remove it or recalculate the frozen gate.

## Reproduction

The saved current-public and transparent-history metrics were independently
recalculated from `learning-records.jsonl`. MAE, RMSE, regret, and Top-1 match
the frozen aggregate values within `1e-15`. Both baselines use exactly the same
111 records, 24 windows, legal candidates, and true remaining-utility labels.
They selected the same action in every window.

The learned aggregate was not reproducible at candidate level because neither
the three seed predictions nor the ensemble predictions were persisted. The
only saved learned evidence is the aggregate in
`prediction-evaluation.json` and three checkpoints. This analysis did not load
those checkpoints and performed zero model forwards.

## Unknown fields

Every candidate has a finite `remaining_utility` reconstructed during the
authorized run from its complete reward sequence through native termination.
Unknown host confirmations were not filled with zero and do not invalidate
that scalar target. They do prevent host-confirmation-specific conclusions.
Likewise, unknown first-command acceptance prevents a definitive command-state
interpretation for that branch but does not erase its saved utility.

The 111 evaluation candidates contain explanation records with explicit unknown
task and host-confirmation fields. Those fields remain categorical/missing in
`candidate-evidence.csv`; no imputation was performed.

## Evidence status

The eight prediction-evaluation parents have now been used for post-run
diagnosis and method design. They are development evidence from this point
forward and cannot be described as a fresh confirmation set for a modified
method. The conditional task-comparison parents remain unexecuted and were not
read or consumed in this analysis.
