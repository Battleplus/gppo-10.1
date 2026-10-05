# Evidence cases

Date: 2026-09-28

All examples below are post-hoc development evidence from the repaired W1-light evaluation. Source lines refer to `run-once/evaluation/steps.jsonl`. The estimator only reads public records available at or before the decision; a later public record is used solely when its original measurement time exactly matches the target decision time.

## Position information gain

At source line 60 (`validation-0000`, PPO, repeat 0, decision time 8), the public position was two seconds old. The exact-time telemetry label first became public at source line 61, one second later. Current observation and existing search both predicted `(0.995136, 0.098508)` and had distance error `2.000000`. Integrating the continuously public ACK-known continuation predicted `(2.985409, 0.295523)`, with error `5.1e-8` against the actual telemetry measurement. This is a valid example of legal history containing state information absent from the stale snapshot.

The estimate is not universally exact. At source line 767 (`validation-0007`, FULL, repeat 0, time 10), the stale public position was already close to the later exact-time label. The history projection had error `0.980118`, while the current observation had error `0.019882`. The aggregate conclusion therefore rests on all common rows, not selected examples.

## Energy information gain

At source line 159 (`validation-0001`, PPO, repeat 0, time 9), the exact-time energy label first appeared at line 160. Current observation error was `0.350000`; the existing idle-power correction error was `0.300000`; using the frozen travel-power rate during the continuously confirmed continuation reduced error to `3.8e-7`.

At source line 73 (`validation-0000`, GPPO_noWM, repeat 0, time 8), the transparent history estimate over-deducted travel energy: its error was `0.300000`, while the existing search estimate error was `2.9e-7`. This preserves the counterexample and shows why action/continuation transitions still matter.

## Remaining travel information and execution uncertainty

At source line 60, the same verified continuation had an eventual same-UAV physical arrival at time `8.130815`. Current public geometry overestimated remaining travel by `2.000000` seconds, existing search by `4.000000`, and history integration missed by `1.3e-7` seconds.

At source line 156 (`validation-0001`, PPO, repeat 0, time 6), the actual same-UAV arrival occurred `6.340831` seconds later. History projection predicted `1.894235` seconds and had error `4.446595`, worse than both snapshot estimates. The record supports an execution outcome label, but the excess duration may include later events unavailable to the simple kinematic estimator; it is not evidence that a particular hidden cause was predictable.

## Acceptance and completion label boundary

The 367 saved non-NOOP choices contain 350 `accepted`, 8 `command_lost`, and 9 `resource_unavailable` feedback labels. Of the accepted choices, 302 have a saved physical completion by the selected UAV, 25 complete with a different UAV, and 23 have no saved completion. These labels establish availability only. No pre-frozen acceptance estimator or same-state alternative-action truth exists here, so the audit does not compute counterfactual benefit or candidate-ranking accuracy.

## Coverage and selection

- Position: 2,316 exact-time labels from 3,096 UAV-decision candidates; history estimate emitted on 584 labeled rows across all eight parents and seven methods.
- Energy: 2,512 exact-time labels from 3,096 candidates; history estimate emitted on 660 labeled rows across all eight parents and seven methods.
- Remaining travel: 501 usable same-UAV arrival labels from 833 public continuation candidates; history estimate emitted on 455 labeled rows.
- Main position abstentions were 2,269 rows without a continuously confirmed continuation, 81 asynchronous x/y rows, and one unknown-target row.

Exact-time telemetry labels preferentially exist when a packet eventually arrives. The common comparison further selects states with a continuously visible ACK-known continuation. These results do not describe permanently lost telemetry, disconnected UAVs, or all decision states.
