"""Freeze preparation files only; this does not create or authorize an attempt."""
import argparse
import hashlib
import json
from pathlib import Path


def identity(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()
            and path.name != "delivery-hashes.json" and "__pycache__" not in path.parts
            and path.suffix != ".pyc"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    path = root / "delivery-hashes.json"
    files = identity(root)
    if args.verify:
        expected = json.loads(path.read_text())
        if expected.get("schema") != "w1-runtime-preparation-files/1.0.0" or expected["files"] != files:
            raise SystemExit("PREPARATION_FILE_IDENTITY_MISMATCH")
    else:
        if path.exists():
            raise SystemExit("DELIVERY_ALREADY_FROZEN")
        path.write_text(json.dumps({"schema": "w1-runtime-preparation-files/1.0.0", "files": files,
                                   "dynamic_authorization": False, "training_authorization": False,
                                   "formal_attempt_created": False}, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"verified": args.verify, "files": len(files),
                      "delivery_hashes_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "formal_attempt_created": False}))


if __name__ == "__main__":
    main()
