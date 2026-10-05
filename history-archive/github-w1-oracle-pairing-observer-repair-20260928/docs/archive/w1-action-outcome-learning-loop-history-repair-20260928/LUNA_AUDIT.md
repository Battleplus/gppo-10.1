# Independent luna_worker audit

Status: **PASS**. The review was read-only. It did not modify files, run the
research experiment, or submit Git changes.

## Findings

- Decision-time public input is frozen before candidate branching. The
  existing future-observation guard remains active, and continuation
  observations do not feed candidate input.
- Every candidate deep-copies the same parent capture. Parent and frozen-input
  digests are checked around each branch. Targeted regressions cover nested
  mutable isolation, candidate order invariance, and unequal termination
  times.
- The production collector integration begins at the real Windows entry,
  traverses WSL-native staging and the supervisor, runs the production
  `AuthorizedRuntimeBackend` with a controlled fake environment, persists and
  reloads the first label, validates it, settles the ledger, and verifies the
  controlled export.
- Eleven unrelated research implementation files, `experiment-matrix.json`,
  and `derived-budget.json` are byte-identical to the reference package. The
  declared production changes are limited to `runtime_backend.py` and
  `wsl_stage_and_launch.py`.
- `RESOURCE_REQUEST.json` remains `NOT_APPROVED`. All 58 frozen files match the
  execution manifest. The Windows export path, WSL-native formal attempt path,
  and erroneous Windows path mapping are unused.

Final package identities reviewed:

- `execution-manifest.json` SHA-256:
  `3ece85737fe484b3ca8312b77bc9c8bed7f4183bd1ecacb666b547ff514bf71c`
- `hashes.json` SHA-256:
  `1bd02c6af66f4571bac1574b3afee3af2e700aac1dbb55fa8a708616a99b8e8e`

## Residual limits

The WSL production-collector end-to-end integration covers the successful
label path. Rejected, unknown, and no-opportunity behavior is covered by the
separate production-component regression harness rather than by three more
Windows-to-WSL end-to-end runs. WSL also emitted a garbled localhost/NAT warning
while returning an unambiguous successful path-existence result; this is kept
as environment noise and did not affect the audit conclusion.
