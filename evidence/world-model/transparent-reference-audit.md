# Transparent Baseline Independent Review

## Finding

The transparent baseline chose NOOP (action 24) in all eight completed prediction-confirmation windows because the implemented ideal-continuation formula equalizes projected task completion across NOOP and every active action. The active actions then incur a negative energy component while NOOP incurs zero energy cost. NOOP therefore wins by a strict score margin in every parent. The saved scores reproduce the documented formula exactly: I found no scale/sign arithmetic, validity-mask, or NOOP tie-breaking error.

This is an implementation-consistent consequence of the explicitly optimistic surrogate, but it makes the frozen comparator a do-nothing policy on these confirmation windows. The official G1 gate still passes against that comparator. Interpret that as a relative result against this NOOP-selecting ideal-continuation reference, not as evidence that G1 outperforms a generally competent transparent policy.

## Formula And Reconstruction

The two normalization helpers, [transparent_utility.py](/E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-local-v1/package/transparent_utility.py:12) and [transparent_utility_base.py](/E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-local-v1/package/native/source-evidence/transparent_utility_base.py:12), use

```text
task_component   = (public_score + 0.1 * energy_cost) / (100 * task_capacity)
energy_component = -energy_cost / initial_total_energy
horizon_task     = task_component + ideal_continuation_completions / task_capacity
policy_score     = 0.5 * 0.8 * horizon_task + 0.2 * energy_component
```

[production_data.py](/E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-local-v1/package/production_data.py:264) counts decision-time visible pending tasks. For a non-NOOP action, it removes the addressed task from the ideal count when the public scorer marks that action on-time; the scorer reads `own_on_time_public` at line 706. It adds all remaining counted tasks as ideal future completions at zero future energy cost. The recorded weights are task scale 0.5, task preference 0.8, and energy preference 0.2. [production_world.py](/E:/Z博士/research-plans/w1-task-outcome-g1-g2-joint-local-v1/package/production_world.py:531) selects the highest score, breaking exact ties by the lowest action ID and then candidate ID.

I independently reconstructed every candidate's final score from the saved `transparent_horizon_components` and weights in `world-model-windows.jsonl`. Across all 101 confirmation candidates, the maximum absolute difference from the saved score is 0. The prediction trace agrees on the selected NOOPs and scores. The saved public metadata marks the continuation assumption `uses_private_truth: false` and records zero ideal-continuation energy cost.

The per-parent component pattern explains the choice. NOOP's base task component is zero and its ideal count is 2, 3, or 4 tasks. Every active action's base task component is 1/6, while its ideal count is one lower. Thus both get the same horizon task component: 2/6, 3/6, or 4/6. Every active action has a strictly negative energy component; NOOP's is zero. The score advantage is exactly the energy penalty, not a tie or a mask outcome. Utility-validity masks are checked after selection for regret calculation and do not select the action; all 101 candidate rows also have valid task and energy residual labels.

| Parent | Candidates | NOOP score | Best active action | Best active score | NOOP margin | NOOP regret | First-step-only regret | G1 regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train-0096 | 10 | 0.200000 | 0 | 0.197209 | 0.002791 | 0.264123 | 0.265602 | 0.132155 |
| train-0097 | 9 | 0.133333 | 0 | 0.130215 | 0.003119 | 0.270245 | 0.136591 | 0.000000 |
| train-0098 | 13 | 0.200000 | 0 | 0.196951 | 0.003049 | 0.264830 | 0.263617 | 0.000000 |
| train-0099 | 13 | 0.200000 | 0 | 0.196180 | 0.003820 | 0.132663 | 0.000000 | 0.000000 |
| train-0100 | 13 | 0.200000 | 0 | 0.196904 | 0.003096 | 0.398692 | 0.136173 | 0.000000 |
| train-0101 | 17 | 0.266667 | 0 | 0.263274 | 0.003393 | 0.267507 | 0.132694 | 0.001459 |
| train-0102 | 17 | 0.266667 | 0 | 0.262952 | 0.003715 | 0.135909 | 0.006159 | 0.139240 |
| train-0103 | 9 | 0.133333 | 0 | 0.130591 | 0.002742 | 0.138049 | 0.134871 | 0.134883 |

Action 0 is the best active action under the saved score and tie rule in each parent; several active actions can tie with one another. The best-active score remains below NOOP even before ties matter. NOOP's score margin ranges from 0.002742 to 0.003820, with mean 0.003216.

## Effect On G1 Interpretation

The saved metrics report mean parent regret of 0.234002 for transparent and 0.050967 for G1, a G1-minus-transparent difference of -0.183035. G1 is no worse on 7/8 parents; its official frozen gate records 3/3 seed improvements and passes. Transparent has a 100% NOOP selection rate and zero top-1 selections. The contrast is therefore substantially a comparison against an ineffective realized NOOP policy, even though NOOP was selected by the documented, optimistically augmented score.

As a sensitivity check, I ranked the same saved candidates using `first_step_transparent_scores`, before adding ideal continuations. This simple post-hoc scorer selects action 0 in all eight parents and has mean realized regret 0.134463. Keeping the already-evaluated G1 choices fixed, G1's regret is lower by 0.083496 on average and no worse on 6/8 parents. This is not the registered gate and does not substitute for it; it shows that the G1 advantage is smaller against an action-taking public-score comparator, while remaining positive on this eight-parent sample.

## Scope And Limitations

I read the named package sources and exported JSONL/JSON artifacts only. I did not construct an environment, execute the public scorer, load models or checkpoints, or run inference. The raw unnormalized mapping returned by the native public scorer (`score` and `energy_cost`) is not exported per candidate; reconstruction therefore starts from the saved normalized public components and independently verifies the final candidate scores, but does not re-execute the underlying graph scorer. No package or evidence files were modified. The counterfactual first-step comparison is post-hoc and uses only eight within-study held-out parents.
