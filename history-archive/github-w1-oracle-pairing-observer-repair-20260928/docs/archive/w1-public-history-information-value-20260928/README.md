# W1 public-history information-value audit

This directory contains the 2026-09-28 post-hoc development audit of repaired W1-light public-history state information. Read `decision.md` for the judgment, `analysis-plan.md` for the frozen method, and `evidence-cases.md` for sourced examples and counterexamples.

Generated tables:

- `input-label-availability.csv`: label source, time alignment, censoring, and evaluation validity.
- `estimation-results.csv`: overall, parent, method, age, and continuation slices using common-row comparisons.
- `analysis-summary.json`: coverage, refusal reasons, action-label counts, and headline results.
- `input-index.json`: immutable input identities and exclusions.

Code and verification:

- `analyze_history_value.py`
- `test_history_value.py`
- `test-output.txt`

The audit made zero environment and model calls and performed no training.
