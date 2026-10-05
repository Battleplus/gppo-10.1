# Task cost attribution repair

## Scope

This independent package preserves the frozen A/B research data, objectives,
model, seeds, prediction gates, task utility threshold, and resource ceilings.
It does not reuse or modify the preceding package or any consumed attempt. No
real environment, model, checkpoint, forward, or training call was made.

## Repair

Each learned one-shot task decision now records `method`, `parent`, `repeat`,
`decision_step`, `candidate_count`, `cpu_seconds`, and `wall_seconds`. The
timer still begins before history and feature construction and ends after the
three model forwards, ensemble construction, and action selection.

`A_one_shot` and `B_one_shot` are summarized independently. Each summary
contains its sample count, CPU mean, wall p95, and threshold results. The
frozen task gate uses only B minus Hungarian utility, B's CPU mean, and B's
wall p95. A's cost is reported but cannot change B's acceptance result.

When a method makes no learned decision, its cost summary is explicitly
`not_evaluated`, has sample count zero, and has null threshold results. The B
gate cannot pass without an evaluated B cost sample.

Raw records are saved in `task-decision-costs.jsonl` and embedded in
`task-comparison.json`, allowing independent recomputation by method, parent,
repeat, and decision step.

## Regression evidence

Controlled production-task tests use only fake clocks, bottom environment and
model-compute substitutes. They verify fast-A/slow-B, slow-A/fast-B, grouped
recomputation, and no-opportunity cases. Actions and utilities are invariant to
the clock schedule. Exactly three model forwards remain charged for each A or
B learned decision; the repair performs no extra forward.

The complete selected suite passed: 13 tests and 2 subtests. The prior nine
production repair tests remain passing.
