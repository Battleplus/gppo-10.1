---
title: "W1 current research progress snapshot"
date: 2026-09-29
experiment_id: W1-B-260929-001
status: archived_locally_pending_signed_commit
research_status: prediction_gate_not_passed
active_attempt: none
branch: research/w1-action-outcome-prior-prototype-20260928
remote_base_commit: d2c885c9940f1428a404385587bef64fe9eeacfd
---

# W1 Current Research Progress

This snapshot records the project state after the action-relative auxiliary-objective run on 2026-09-29. It separates completed evidence from the proposed next architecture. It does not authorize another experiment.

## Tencent-meeting objective

The target remains GPPO with a world model that predicts candidate-action consequences. GPPO, not the world model, should make the final action decision. The predicted consequences may be supplied to GPPO as action features or a learned prior. The final claim must come from task-level performance under the original W1-light setting and fair baselines.

## Completed evidence

1. The repaired W1-light runtime, public-information boundary, counterfactual branch pairing, Windows-to-WSL launch path, and pre-decision history freeze have passed their respective technical checks.
2. A fixed-continuation action-outcome learner was trained with three seeds. It reduced absolute utility MAE from approximately `0.20` for the transparent baselines to `0.046081`, but its selected-action regret was `0.007620`, worse than the baseline value `0.006898`. The frozen prediction gate therefore failed and no task comparison was run.
3. The follow-up controlled experiment compared:
   - A: absolute remaining-utility SmoothL1;
   - B: the same absolute loss plus a window-centered auxiliary loss.
4. The follow-up experiment completed technically on 8 frozen parents, 3 repeats per parent, and 24 decision windows. It used 113 unique candidate labels and trained both A and B with seeds `7101`, `7102`, and `7103`.
5. A and B selected the same action in all 24 windows. Their parent-macro selected-action regret was identical at `0.024674480730902813`; the required A-minus-B improvement was `0.005`, so the auxiliary-objective effect gate failed.
6. B was non-worse than A in 8/8 parents and better than the transparent-history regret baseline (`0.03541961257778265`), but these subconditions did not override the failed primary effect gate.
7. The four-arm task comparison and decision-cost stage were not entered. Task utility relative to Hungarian or transparent selection, the 10 ms CPU gate, and the 50 ms wall gate remain not evaluated for this method.

The latest completed run is documented in [the one-shot run report](../w1-action-relative-auxiliary-task-cost-attribution-repair-v1-once-run-20260929/report.md). Its [result summary](../w1-action-relative-auxiliary-task-cost-attribution-repair-v1-once-run-20260929/result-summary.json), [prediction gate](../w1-action-relative-auxiliary-task-cost-attribution-repair-v1-once-run-20260929/prediction-gate.json), and [independent review](../w1-action-relative-auxiliary-task-cost-attribution-repair-v1-once-run-20260929/luna-final-review.md) are the controlling evidence for the latest result.

## Current interpretation

The evidence shows that the learned predictor can estimate absolute continuation utility more accurately than simple baselines, but neither the original absolute objective nor the centered auxiliary objective has demonstrated a reliable improvement in action selection. This is a failure of the tested prediction-to-selection mechanism. It is not evidence that every world-model design is useless.

The result does not establish a GPPO plus world-model advantage because the learned consequence signal has not yet been integrated as an input to a newly trained GPPO policy and tested at task level. It also does not establish task utility or production decision cost for that integration.

## Proposed next research step

The next hypothesis, if separately prepared and approved, is a controlled GPPO fusion experiment:

- `GPPO_noWM`: GPPO using the original public graph state;
- `GPPO+WM`: GPPO receives frozen per-candidate consequence predictions as additional action features or through a learnable gate;
- `GPPO+shuffled-WM`: the same architecture and budget with candidate predictions shuffled within each decision window.

GPPO must remain the final decision-maker, legal masks must remain authoritative, and the world model must not restore illegal actions or supply private/future information. The world model's MAE, ranking accuracy, and regret remain mechanism diagnostics; the research decision must use task utility, deadline completion, cross-parent and cross-seed consistency, and complete decision cost.

This proposed experiment has not been implemented, frozen, approved, or run as of this snapshot.

## Operational state

- Active environment run: none.
- Active model training: none.
- Active model forward evaluation: none.
- Current research decision: `PREDICTION_GATE_NOT_PASSED` for the centered auxiliary-objective route.
- Historical negative results, technical stops, resource ledgers, and unknown labels remain preserved.
- The latest completed run closed its ledger with zero pending and zero failed calls and verified an 87-file controlled export.

## Git archive state

- Local branch: `research/w1-action-outcome-prior-prototype-20260928`.
- Remote base at snapshot time: `d2c885c9940f1428a404385587bef64fe9eeacfd`.
- The existing research archive files and this progress snapshot are staged locally.
- No new signed commit or remote archive is claimed.
- Repository policy requires SSH-signed commits. Windows `ssh-agent` is currently `Stopped/Disabled`, so the staged archive remains pending administrator restoration of the signing agent.
- Credentials, authorization tokens, SQLite ledgers, checkpoints, and large raw logs are excluded from the Git archive.
