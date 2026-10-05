# W1 Task Outcome Source Check

Zero-environment-call follow-up to `w1-action-conditioned-task-outcome-label-repair-v1`.
The earlier preparation package and sealed experiment artifacts are unchanged.

Run `python -B source_contract_check.py` and
`python -B -m unittest discover -s . -p test_*.py -v` from this directory.
The source check parses frozen Python files without importing the environment.
The tests import only `task_lifecycle.py` and exercise its deterministic task
object; they do not construct the W1 environment or call reset/step.

This is a static and bottom-boundary check, not production collection coverage
or evidence of prediction accuracy. There is no approved dynamic attempt.
