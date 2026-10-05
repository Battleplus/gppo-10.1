# W1 Action-Conditioned Task Outcome Label Repair v2

This is a frozen label-contract and production-collector qualification package.
It does not start a formal attempt and does not run the real environment, a
model, a checkpoint, a forward pass, or training. The only proposed dynamic
activity is the separate label-only request in `RESOURCE_REQUEST.json`, which
remains `NOT_APPROVED`.

The contract binds physical on-time completion and explicit expiry to a legal
first action plus fixed Hungarian continuation. Host confirmation stays
unlabeled in `continuous_service_until_deadline`. See
`TASK_OUTCOME_LABEL_CONTRACT-v3.md`, `configuration-identity-audit.md`, and
`data-reuse-audit.json` for semantics, provenance, and excluded historical
records.

Run the production collector lifecycle tests on Ubuntu-24.04 with Python 3:

```text
python3 -B -m unittest discover -s . -p 'test_*.py' -v
```

The tests use a deterministic fake environment and native Linux temporary
storage. They are not real label coverage or model-effect evidence.

Freeze the preparation payload only after all source files and reports are
final:

```text
python3 -B freeze_contract.py
python3 -B freeze_contract.py --verify
```

`hashes.json` covers every payload file except itself and
`contract-manifest.json`; the manifest records the exact hash-list digest.
The manifest SHA-256 is the external package identity. This package has no
formal launch entry or consumed authorization.
