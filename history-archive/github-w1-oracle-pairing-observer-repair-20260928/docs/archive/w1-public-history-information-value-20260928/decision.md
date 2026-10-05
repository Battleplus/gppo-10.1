# Decision

Date: 2026-09-28

## Primary judgment

**There is a verifiable public-history information increment.** On the exact same labeled rows where the predeclared transparent history estimator can emit, it materially improves position, energy, and remaining-travel estimates over the current public snapshot and the frozen search correction. The effect appears in all eight development parents and all seven evaluated methods.

This is an information-value result, not a learning or task-benefit result. The increment is already exposed by a deterministic estimator using a public ACK-known continuation, stale telemetry time, frozen speed, and frozen power. The audit therefore provides no present need to train a learned history model for these three quantities. A future study would first have to show that the transparent estimate changes legal candidate ranking or improves task utility under the existing `0.01` utility and cost criteria.

## Common-row results

| Target | Common rows | Current MAE | Existing search MAE | Public-history MAE | Parents |
|---|---:|---:|---:|---:|---:|
| Position | 584 | 0.812104 | 0.812104 | 0.026703 | 8/8 |
| Energy | 660 | 0.287371 | 0.237718 | 0.062629 | 8/8 |
| Remaining travel | 455 | 1.044045 s | 2.002287 s | 0.098086 s | 8/8 |

The frozen search remaining-time expression is conservative and adds position age, so its larger MAE does not by itself mean it is defective for planning. The public-history estimator is a point estimate evaluated against realized same-UAV arrival.

## Limits

- This is a post-hoc analysis of eight already observed development parents. Repeated rows within a parent are not independent scenes.
- Exact-time labels require later successful public delivery, creating selection toward observable states.
- The position estimate emits on 25.2% of valid position labels (`584/2316`) and the energy estimate on 26.3% (`660/2512`). It abstains when continuation or motion assumptions are not publicly supported.
- Remaining-travel labels exclude missing arrivals and different-UAV completions; later interruption effects produce real counterexamples.
- Controller slices reflect different visited states and cannot be read as causal method comparisons.
- State-error reduction does not establish better action ranking, counterfactual value, utility gain, deployability, or world-model advantage.

## Research implication

Do not train or modify FULL on the strength of this audit. Preserve the old repaired-runtime FULL result, the failed first-window oracle criterion, all technical stops, historical costs, and the 10 ms cost failure. Any later proposal should compare this transparent estimator against the unchanged execution filter and demonstrate a decision-level contribution before introducing a learned predictor.

Environment calls, model initialization/loading, model forward calls, and training updates in this audit were all zero.
