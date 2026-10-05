"""Explicit Windows transport interpreter; not a model execution environment."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
from manifest_contract import sha256_file

PACKAGES = ("paramiko", "bcrypt", "cryptography", "invoke", "PyNaCl", "cffi", "pycparser")
ROOT = Path(__file__).resolve().parent


def describe():
    return {"schema": "w1-windows-ssh-controller-runtime/1.0.0",
        "python_executable": sys.executable, "python_prefix": sys.prefix,
        "python_version": sys.version, "python_binary_sha256": sha256_file(Path(sys.executable)),
        "distributions": {name: {"version": importlib.metadata.version(name),
            "record_sha256": sha256_file(Path(importlib.metadata.distribution(name)._path) / "RECORD")}
            for name in PACKAGES}, "model_execution": False,
        "credentials_saved": False, "installation_during_experiment": False}


def verify_controller_runtime():
    spec = json.loads((ROOT / "controller-runtime-contract.json").read_text(encoding="utf-8"))
    if describe() != spec:
        raise RuntimeError("WINDOWS_CONTROLLER_RUNTIME_IDENTITY_MISMATCH")
    return {"status": "pass", "model_execution": False, "python_executable": sys.executable}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        (ROOT / "controller-runtime-contract.json").write_text(json.dumps(describe(), sort_keys=True,
            indent=2)+"\n", encoding="utf-8")
    else:
        print(json.dumps(verify_controller_runtime()))
