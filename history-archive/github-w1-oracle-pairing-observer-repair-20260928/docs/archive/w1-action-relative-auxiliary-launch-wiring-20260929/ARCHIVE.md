# A/B launch wiring archive

This archive contains the new preparation package
`w1-action-relative-auxiliary-launch-wiring-v1` after fixing the two launch
gaps in the prior A/B package. The frozen request remains `NOT_APPROVED`.

- Attempt: `w1-action-relative-auxiliary-launch-wiring-v1-once`
- `execution-manifest.json` SHA-256: `a6a0c7597221f2d6e8cba983f76b0c8bd767c7f27389a41cc9e5feae5094d12e`
- `hashes.json` SHA-256: `11113e2af6f2a13b4ad885ce93381faa7f82c7c77e61a2c6aac9f366200ecce1`
- Dynamic execution: not run; no real environment, model forward, checkpoint, training, or attempt was created.
- Validation: 20 pure-logic tests and 5 real Windows-to-Ubuntu-24.04-to-native-WSL integration tests passed.

The package reuses the validated native staging, supervisor, settlement, and
verified-export chain. The production entry rejects integration-test contracts.
The test-only fake backend is enabled only by an explicit temporary clone and
is not part of the formal native runner.

The archive excludes the external one-shot token file, credentials, raw logs,
SQLite databases, checkpoints, and temporary integration output directories.
