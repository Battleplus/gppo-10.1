"""Fail-closed unique entrypoint for the future approved dynamic run.

Preparation never starts the experiment.  The entrypoint only performs cheap
identity checks and refuses while RESOURCE_REQUEST.json is NOT_APPROVED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="reserved for a separately approved runner")
    parser.add_argument("--token", default=None, help="one-shot token; never written to disk")
    args = parser.parse_args()
    request = json.loads((ROOT / "RESOURCE_REQUEST.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "execution-manifest.json").read_text(encoding="utf-8"))
    if sha256(ROOT / "hashes.json") != manifest["hashes_sha256"]:
        raise SystemExit("IDENTITY_STOP: hashes.json does not match execution manifest")
    if request.get("status") != "APPROVED":
        print("NOT_APPROVED: preparation entrypoint refuses dynamic execution")
        return 2 if args.execute else 0
    if not args.execute:
        print("READY_CHECK_ONLY: use --execute only after an external approval updates the frozen request")
        return 0
    expected = os.environ.get("W1_ACTION_RELATIVE_AUXILIARY_TOKEN")
    if not expected or args.token != expected:
        raise SystemExit("IDENTITY_STOP: external token mismatch")
    raise SystemExit("RUNNER_NOT_ATTACHED: dynamic runner is outside this preparation scope")


if __name__ == "__main__":
    raise SystemExit(main())
