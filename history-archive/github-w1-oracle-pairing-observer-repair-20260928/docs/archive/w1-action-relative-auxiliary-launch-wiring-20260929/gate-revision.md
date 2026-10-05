# Gate and coverage revision

The confirmation stage has eight preselected prediction parents and three
repeats, for 24 units. Each unit may produce at most one qualifying decision
window and at most 25 total legal branches (24 non-NOOP plus an optional
NOOP). `NO_OPPORTUNITY` is persisted as a unit record and is never converted
to a zero regret or replaced with another parent.

The confirmation coverage gate requires all eight parents to have at least one
valid window before prediction evaluation. The parent macro denominator is
always the eight frozen parents; a missing parent contributes no regret value
and makes the coverage gate fail. The four-parent non-worse count is computed
only over parents with valid A/B windows and cannot count a missing parent.

The first gate is `B - A <= -0.005` in parent-macro selected regret and at
least four valid parents where B is no worse than A. The task stage has an
additional, predeclared condition: B's parent-macro regret must be strictly
lower than the transparent-history baseline. If the first gate passes and this
second condition fails, the run settles with zero task calls.
