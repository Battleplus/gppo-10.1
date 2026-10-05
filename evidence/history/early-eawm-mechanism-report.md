# GPPO World-Model Mechanism Diagnostic

This is a read-only replay of the sealed prediction-confirmation export. It did not construct an environment, load a checkpoint, initialize a model, run a forward pass, train, inspect the sealed task set, or request a new attempt.

## Evidence and replay

- 24 prediction-confirmation windows from 8 parent scenarios, 3 repeats per parent.
- 330 candidate rows; candidate counts are 10, 13, or 17.
- `candidate-audit.jsonl` contains every candidate's saved true utility, transparent score, target masks, event labels, and saved seed/ensemble score reconstruction.
- `window-audit.jsonl` and `window-audit.csv` contain the per-window oracle, best/second-best gap, selected actions, regret, seed actions, and event variation.
- `parent-summary.json` contains the parent-level macro recomputation.
- `diagnostic-summary.json` records the independent counts and frozen metric values.

The replay uses the production formula: transparent score plus the saved outcome residual with task weight `0.5 * 0.8` and energy weight `0.2`. It does not recompute predictions.

The final replay uses the production strict direction rule: a pair is correct only when `predicted_delta * true_delta > 0`; a predicted tie is not counted as correct. The earlier `replay-final-20261001` output used a non-strict tie comparison and is superseded by `replay-final-strict-20261001`. The strict rerun reproduces the frozen production pairwise values (`transparent=0.1006021756`, `G1=0.1734459984`, `G2=0.8645493395`).

## What the event labels contain

The five event dimensions are `public_field_change`, `new_measurement_received`, `continuation_publicly_confirmed`, `physical_completion_observed`, and `host_confirmation_observed`.

Across all 330 candidates, valid counts are respectively `98, 330, 330, 0, 0`. Candidate-level event variation occurs only in `continuation_publicly_confirmed`, in all 24 windows. The physical-completion and host-confirmation dimensions are entirely unknown in this export. Acceptance is valid for only 98 candidates and has no candidate variation in these windows; state-change is valid for all candidates but also has no candidate variation.

Therefore the G2 event Brier improvement (`-0.219393`) is evidence about the saved event targets, especially public continuation confirmation. It is not evidence that the model predicted physical completion, expiry, or host confirmation. Those mechanisms cannot be separated from this export.

## Candidate sorting and selected actions

The frozen aggregate values are:

| Method | Parent-macro regret | Pairwise direction accuracy | Top-1 |
|---|---:|---:|---:|
| Transparent | 0.001476692 | 0.100602 | 0.041667 |
| G1 ensemble | 0.001407247 | 0.173446 | 0.083333 |
| G2 ensemble | 0.000347222 | 0.864549 | 0.666667 |

Relative to transparent, G1 selects a different action in 22/24 windows but usually selects another candidate with the same saved true utility. Its regret improves in only a small subset and is worse in `train-0046`; its seed behavior is unstable. G2 selects a different action in all 24 windows and selects NOOP (`action_id=24`) in 19/24 windows. G2 matches the oracle in 16/24 windows. All three G2 seeds are not equivalent: seed 8202 selects NOOP zero times and has zero oracle matches, while seed 8203 selects NOOP 22 times and matches the oracle 16 times. The ensemble therefore hides a material seed difference.

The G2 improvement is concentrated in windows where NOOP is the saved utility optimum. For an active-action to NOOP change, the saved task residual difference is approximately `+0.066667` and the transparent-score difference is about `-0.063` to `-0.064`; the resulting true utility gap is about `+0.001667`. This is a task/energy residual and candidate-type distinction, not a measured completion-time effect. In windows where the two methods choose different active actions, the true utility is often exactly tied.

No two windows share the same `input_hash`, so this export contains no exact repeated public input on which to test opposite action directions. With completion-time and host-confirmation labels unknown, it cannot distinguish a repeatable public-information mechanism from an implementation or later-execution difference. The supported answer is **无法判断**.

## Parent-level regret

| Parent | Transparent | G1 | G1 - T | G2 | G2 - G1 |
|---|---:|---:|---:|---:|---:|
| train-0040 | 0.001667 | 0.001667 | 0.000000 | 0.000556 | -0.001111 |
| train-0041 | 0.000702 | 0.000702 | 0.000000 | 0.000000 | -0.000702 |
| train-0042 | 0.001667 | 0.001111 | -0.000556 | 0.001111 | 0.000000 |
| train-0043 | 0.001667 | 0.001111 | -0.000556 | 0.000000 | -0.001111 |
| train-0044 | 0.001667 | 0.001667 | 0.000000 | 0.001111 | -0.000556 |
| train-0045 | 0.001667 | 0.001667 | 0.000000 | 0.000000 | -0.001667 |
| train-0046 | 0.001111 | 0.001667 | +0.000556 | 0.000000 | -0.001667 |
| train-0047 | 0.001667 | 0.001667 | 0.000000 | 0.000000 | -0.001667 |

The original G1 prediction gate remains failed: G1 versus transparent improvement is only `0.0000694444`, below `0.005`. G2 versus G1 improvement is `0.0010600253`, below its frozen `0.0025` threshold. No task calls were made, so task benefit and decision cost remain **未评价**.

## Mechanism conclusion

The data support a specific diagnostic observation: event supervision improved prediction of a public continuation-related label and the G2 score then separated NOOP from active candidates more effectively on these windows. The data do not support the stronger claim that G2 learned physical completion, deadline/expiry, or host-confirmation consequences, nor that the gain would transfer to GPPO task utility.

The next defensible step is a new, separately frozen confirmation design that adds valid action-conditioned labels for completion/expiry and host confirmation while preserving the public-input boundary. Do not tune on these 8 parents again, do not lower the existing gates, and do not start a task experiment from this diagnostic. If those labels cannot be obtained under the causal contract, stop this mechanism line rather than infer them from the current Brier result.
