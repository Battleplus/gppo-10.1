# Luna read-only field review

Date: 2026-09-29

- The 770-value `flat` layout is decoded correctly: four 32-wide UAV rows, three region rows, four target rows, then six 32-wide task rows. The task offset is `11 * 32 = 352`. Public fields use `(value, known, valid, age)` quartets; `flat[768]` is decision time divided by the horizon 18.
- Candidate actions below 24 map to UAV `action // 6` and task slot `action % 6`; action 24 is NOOP. Distances are valid only when the relevant public coordinate fields are known and valid.
- Exact current confirmed-continuation identities were not persisted. `continuation_id` identifies the fixed Hungarian policy and is not the current continuation handle list. The history vector preserves only earlier continuation counts. Branch-step continuation records are post-decision audit data and cannot be used to reconstruct decision input.
- Candidate labels persist stepwise vector rewards, discounted increments, final counts, energy use, and terminal task records. `first_feedback` and `first_acceptance_class` reliably describe the first command response, but do not establish eventual task completion.
- Host confirmation remains separate from physical completion and is not used as a substitute for the physical task label.

No correction to the analysis field mapping was required.
