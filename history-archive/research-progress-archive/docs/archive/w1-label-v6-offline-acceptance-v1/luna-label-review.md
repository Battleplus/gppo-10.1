# Luna independent label review

Reviewer: `/root/v6_label_independent_review`, luna_worker (gpt-6-luna, max).
Review independence: fresh same-family agent; acceptance is provisional.
The reviewer read source and the saved JSONL independently. No files were
modified, no sealed task results were read, and no dynamic calls were made.

Primary physical, expiry and host labels are separately represented and their
counts agree: physical 82/18/8, expiry 18/82/8, host 42/50/16 (true/false/unknown).
Among the 82 physical completions, 42 host receipts were timely, 32 were late,
and eight were not observed. NOOP accounts for eight further host unknowns.

Blocking interpretation defect: `production_data.py:201` obtains the final
execution command from a retained task token without requiring completion.
`task_outcome_contract.py:214` compares that command to the first command and
sets completion attribution/identifiability without checking `completed_at`.
Nine expired candidates are therefore incorrectly marked true:
train-0057/action-13; train-0058/action-1; train-0059/action-1;
train-0060/actions-1,7,13,19; train-0061/action-1; train-0063/action-19.
The old 83 count is not valid as a physical-completion attribution count.
Only 74 true flags also have physical-completion evidence. These remain
identity-associated completions, not proof of a counterfactual causal effect.
The defect concerns audit-only fields and does not invalidate the three
primary terminal labels. A separate erratum is required; do not edit old data.

The reviewer independently recomputed 13/25 same-task groups with different
physical outcomes, spanning 7/8 parents. train-0060 has only cross-task
differences. Candidate pairs are within-window branches, not independent
replicates. Decisions were at t=4, measurement times no later than t=3 and
receipt times no later than t=4. Production source captures the public input
before branch copies and checks parent identity after branch execution.

The saved `true_utility` is first-step utility: `production_data.py:483`
computes the first reward vector, line 535 discards continuation reward, and
lines 541-550 omit global reward/counts/energy from continuation records.
The final target-task label cannot substitute for whole-schedule utility.

Decision: current evidence supports pipeline/development use and the
feasibility of physical task labels. It lacks parent-disjoint study roles and
whole-continuation multi-objective supervision for a formal world-model/GPPO
evaluation. Limited additional data collection is justified after contract and
accounting repairs. This is not a new sample-size pass threshold, and does not
show either model success or model failure. Resource settlement must be
reported separately from the internal coverage pass.
