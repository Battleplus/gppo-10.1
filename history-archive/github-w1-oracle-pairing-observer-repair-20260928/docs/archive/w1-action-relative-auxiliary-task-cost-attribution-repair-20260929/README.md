# W1 A/B production backend completion

This is an independent, unapproved preparation package for the frozen
absolute-vs-relative-auxiliary objective comparison. It preserves the study
contract, parent split, seeds, thresholds and resource ceilings.

The formal path is `launch_once.py` -> WSL staging -> `native_launch.py` ->
supervisor -> `runner.py` -> `ABRuntimeBackend` -> `runtime_adapter.py`.
No fake backend is selected by the formal runner. Integration tests inject
only bottom-level environment/model/optimizer/checkpoint substitutes.

Dynamic execution is not part of this delivery.

This revision attributes task decision cost separately to `A_one_shot` and
`B_one_shot`. B task acceptance uses B's own CPU mean and wall p95; an absent
B decision is reported as not evaluated. See `cost-attribution-repair-report.md`.
