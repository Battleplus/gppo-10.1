# W1 relative-utility auxiliary objective preparation

This package prepares one bounded, one-shot comparison of two learned
remaining-utility objectives under the repaired W1-light contract. It is
`NOT_APPROVED` and contains no dynamic attempt.

The frozen research decision remains `PREDICTION_GATE_NOT_PASSED`. The package
does not claim that the earlier failure had a unique cause. It tests one
specific mechanism: a window-centered SmoothL1 auxiliary term may improve
within-window action ordering while preserving the same absolute target.

Preparation actions are offline only. `launch_once.py` verifies the manifest
and refuses dynamic execution while the request is not approved. No environment
runtime, checkpoint, model forward, optimizer update, or training call is
made by the preparation scripts or tests.
