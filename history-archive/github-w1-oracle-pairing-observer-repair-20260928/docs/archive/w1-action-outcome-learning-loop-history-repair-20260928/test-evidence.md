# History-boundary repair test evidence

## Results

- Combined package and regression suite: 49 tests passed in 41.702 seconds.
- Current WSL-native infrastructure suite: 12 tests passed in 2.354 seconds.
- Production collector real-entry integration: 1 test passed in 19.700 seconds.
- Python bytecode compilation: performed again during final freeze validation.

The generated logs are `test-output.txt`,
`native-current-test-output.txt`, and `history-entry-test-output.txt`.

## Production-chain coverage

The production collector integration starts at the actual Windows
`launch_once.py` entry. It invokes the frozen `Ubuntu-24.04` WSL distribution,
uses native Linux staging, starts the real supervisor, and runs the production
`AuthorizedRuntimeBackend` with a controlled fake environment. The chain
freezes decision input before branch continuation, constructs a label after
later observations are generated, persists the label to JSONL, reloads and
validates it, settles the ledger, and performs the controlled hash-verified
Windows export.

This is not a fake-worker-only check. It uses the production collector, public
adapter, feature builder, labeler, validator, persistence path, budget ledger,
supervisor, and export path. It does not construct the real W1 environment,
initialize or load a model, perform a model forward, or execute training.

## Boundary and isolation coverage

The targeted regression cases verify:

- decision input is frozen before continuation observations are produced;
- multiple candidates with different termination times remain isolated;
- exchanging candidate execution order preserves candidate-keyed inputs and
  labels;
- parent history and frozen input hashes do not change around branches;
- an intentionally injected future public observation still raises
  `LearningContractError`;
- an old measurement received only after the decision does not enter input;
- nested lists, dictionaries, and arrays do not share mutable references;
- the first repaired label persists, reloads, and passes contract validation;
- normal, rejected, unknown, and no-opportunity paths retain their contracts.

## Infrastructure coverage

The native suite exercises atomic concurrent readers, bounded fsync failure,
worker exit and interruption, stale-state classification, unresolved ledger
preservation, native staging, verified export, and the additive final
settlement record. Tracebacks in the two injected-failure tests are expected
evidence that the failures remain visible; both tests pass by confirming the
supervisor stops and settles correctly.

Passing these checks establishes the repaired input boundary and launch wiring.
It does not authorize or execute the learning experiment and does not establish
prediction accuracy or task benefit.
