# Contract repair report

The stopped package used two incompatible identity formats. Its generator wrote
`hashes.json.files` as a list and omitted `execution_manifest_sha256`, while the
three launch layers expected a mapping plus that field. The resulting
`KeyError` occurred before staging.

This package uses one shared `manifest_contract.py` implementation. The
acyclic dependency is:

```text
hashes.json --execution_manifest_sha256--> execution-manifest.json
execution-manifest.json --files--> content file SHA-256 values
external authorization.json --binds--> attempt, both package digests, request digest, token digest
```

The manifest does not hash `hashes.json`, so the package has no circular
digest. Windows, WSL staging, and Linux native entry all call
`manifest_contract.verify_package`. Missing fields, bad schemas, altered files,
attempt mismatches, request mismatches, and digest mismatches raise named
`PackageContractError`/contract errors.

The freeze script now delegates identity creation to the same
`freeze_wiring.freeze_identity()`/`manifest_contract.write_identity_files()` path
used by integration clones; it cannot regenerate the removed list-shaped
schema. The final `test-output.txt` is part of the sealed content set and was
written before the final identity was regenerated.

`--preflight-only` performs the same Windows and WSL/Linux validation without
reading the one-shot token, creating native or export directories, creating a
lock, staging files, or starting a worker. The integration clone uses the same
`freeze_identity()` generator as the formal package; it does not rewrite a
second hashes format.

Research configuration, A/B objectives, parent splits, seeds, gates, and
resource limits are copied unchanged. `RESOURCE_REQUEST.json` remains
`NOT_APPROVED`.

## Identity drift and resolution

An attempted final preflight was stopped before staging with
`PACKAGE_FILE_DIGEST_MISMATCH: manifest_contract.py`. The current file hash was
`f9371fa4327c319f2e6eb818738d0382e41cb1765116f365d5bedf1d22faab68`, while the
sealed manifest and archive recorded
`a179ad1fa65787c5e4c0ba9b1dfd922cffc186a86f8a2f1843c3d790b594ba5e`.
The byte-level comparison found exactly one difference: the current copy had
one fewer trailing newline. No executable statement, schema, or contract
behavior differed. The source of the drift was the later formatting edit that
removed that final blank line; it was not an environment or research change.

The exact archived/previously verified bytes were restored before this
re-freeze. The failed preflight record is kept outside the package, and the
new package identity is regenerated only after the restoration and regression
tests.
