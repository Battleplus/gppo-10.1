# Zero-Call Verification

Command:

```text
python -B -m unittest discover -s E:\Z博士\research-plans\w1-action-conditioned-task-outcome-label-repair-v1 -p test_*.py -v
```

Result: 9 tests passed, exit 0. The tests use synthetic records and temporary
JSONL files only. They do not instantiate the W1 environment, initialize a
model, load a checkpoint, run a forward pass, train, or create an attempt.

Read-only audit input:

`E:\Z博士\.codex-exports\w1-eawm-jepa-world-model-production-repair-v4-e2e-integration-auth-repair-v1-once\run-once\world-model-windows.jsonl`

SHA-256: `932b9f6638dbe332359301b26b098db411d5f75cee93ec239114fb0ac421bb51`

The audit counted 56 complete windows and 771 candidate branches; 0 branches
have nonempty completion records or completion messages. All candidate
branches have `terminated=false` and `truncated=false`. The two task event
heads have 0 valid labels. Details are in `data-availability-audit.json`.

Frozen native source identities:

- `m10_environment.py`: `0f4615d11604a7cd1cc74c94f0258dfc6cb59748f64a075276ec90941db094b1`
- `task_lifecycle.py`: `50248afa519fd18dbc5995ba0bbf7e58adc0833ae501760ea032ab91fca5ed89`

These are interface and data-availability checks, not evidence of model
accuracy, task benefit, or a successful new confirmation collection.
