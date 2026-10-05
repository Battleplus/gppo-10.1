# Complete finite learning-loop design

Status: `NOT_APPROVED`. This package does not run during preparation.

## Frozen parent groups

`experiment-matrix.json` takes the first 48 unique rows of the already frozen,
randomized 2,048-row W1-light training tape before observing outcomes. The
groups are disjoint: 24 training parents, 8 model-selection parents, 8
independent prediction-evaluation parents, and 8 separate task-comparison
parents. Existing validation/test tapes are not read or reused.

Training and model-selection parents use two external repeats. Independent
prediction and task parents use three. Every repeat has its own frozen external
key. The four groups share the repaired runtime and scenario generator contract
but no parent identity.

## Label mechanism

Each data parent-repeat follows Hungarian from reset. It selects at most one
state: the first state at decision step 4 or later with at least two legal
non-noop candidates and at least two distinct transparent-history scores.
This task condition depends only on public pre-action data. It does not select
on rule loss, learned output, oracle gain, hidden fault, or outcome. Missing
opportunities remain missing.

At that state, every legal candidate, including legal noop, is copied from one
isolated snapshot, forced once, and followed by Hungarian to native termination.
Packet identities and common primitive fates are checked. All candidate labels
for a decision must be present. Minimum parent coverage is 18/24 training, 6/8
selection, and 6/8 independent prediction parents. These are data conditions,
not task-benefit thresholds.

## Learner and selection

The model is a real `827 -> 128 -> 64 -> 1` tanh MLP in `learning.py`. It trains
only remaining frozen utility with Smooth L1 loss, Adam `3e-4`, batch size 64,
at most 100 epochs, and seeds 7101/7102/7103. Each seed selects its epoch only
by model-selection-parent MAE, with patience 12. Checkpoints carry architecture,
continuation, seed, split identities, and update counts.

Independent prediction evaluation uses the arithmetic mean of the three seed
predictions. It reports parent-macro MAE/RMSE, top-1 accuracy, and selected
regret against current-public and transparent-history scores. Task comparison
runs only if learned MAE is strictly lower than both baselines and learned
selected regret is strictly lower than the transparent baseline.

## Conditional task comparison

On eight untouched task parents and three repeats, compare paired Hungarian,
transparent one-shot plus Hungarian continuation, and learned three-seed
one-shot plus Hungarian continuation. The trigger is identical across arms and
no arm is replaced when no opportunity appears.

Acceptance requires learned-minus-Hungarian parent-macro utility at least
`0.01`, learned decision CPU mean at most `10 ms`, and wall p95 at most `50 ms`.
Decision cost covers public-history feature construction, tensor construction,
all three seed forwards, ensemble averaging, and candidate selection as one
continuous interval. A nonfinite seed or ensemble score is a technical stop.
A prediction gate pass is not task success. A task failure does not authorize
more parents, windows, epochs, seeds, or parameter search.

Identity, public-boundary, snapshot, packet, label, nonfinite, checkpoint,
ledger, or resource failure stops the attempt. Insufficient data stops before
training. A failed prediction gate stops before task episodes. There is no
retry, budget borrowing, follow-up training, or GPPO fusion.

## Resource-limit basis

Dynamic-call ceilings are exact matrix maxima. Time and storage values are
conservative technical-stop ceilings rather than forecasts. The completed
repaired fair rerun used 3,036.79 wall seconds and 3,658.30 complete CPU seconds
for 6,918 environment steps. Label collection therefore reserves 24,000 wall
and 48,000 complete CPU seconds for 41,184 steps plus snapshot and branch work;
the conditional 1,296-step task stage reserves 1,200/2,400 seconds. Training
reserves 3,600/7,200 seconds for at most 5,700 updates and 483,600 sample
evaluations.

The completed 144-branch oracle archive occupies 465,837,556 bytes, about
3.24 MB per branch. Projecting that rate to 2,200 branches gives about 7.12 GB;
the 16 GiB label-stage ceiling leaves more than a twofold allowance for
snapshots, logs, and ledger state. The 24 GiB active ceiling includes three
checkpoints and settlement artifacts; 48 GiB permits one native copy and one
verified export.
