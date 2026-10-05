"""Prepare, freeze, and read-only verify; never invoke a formal launch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

EVIDENCE = Path(__file__).resolve().parent
WORKSPACE = EVIDENCE.parents[1]
PACKAGE = EVIDENCE.parent / "w1-action-conditioned-task-outcome-label-repair-v6-dependency-cpu-settlement"
ATTEMPT = "w1-action-conditioned-task-outcome-label-qualification-v6-dependency-cpu-settlement-once"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def snapshot(root):
    return {path.relative_to(root).as_posix(): {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in sorted(root.rglob("*")) if path.is_file()}


def invoke(name, argv, cwd=PACKAGE):
    process = subprocess.run(argv, cwd=cwd, capture_output=True, timeout=180, check=False)
    (EVIDENCE / (name + ".stdout")).write_bytes(process.stdout)
    (EVIDENCE / (name + ".stderr")).write_bytes(process.stderr)
    record(EVIDENCE / (name + "-execution.json"), {"argv": argv, "cwd": str(cwd), "exit_code": process.returncode})
    if process.returncode:
        raise RuntimeError(name + " failed; original outputs retained")
    return process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-only", action="store_true")
    parser.add_argument("--freeze-and-preflight", action="store_true")
    args = parser.parse_args()
    if args.tests_only == args.freeze_and_preflight:
        raise SystemExit("CHOOSE_ONE_PREPARATION_MODE")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if args.tests_only:
        invoke("full-regression", ["wsl.exe", "--distribution", "Ubuntu-24.04", "--exec",
                                  "/home/asus/w1-action-relative-auxiliary-runtime-dependency-repair-v1-runtime/bin/python",
                                  "-B", "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py", "-v"])
        print(json.dumps({"regression_exit_code": 0, "formal_attempt_started": False}))
        return
    if not (EVIDENCE / "full-regression-execution.json").is_file():
        raise SystemExit("FINAL_REGRESSION_EVIDENCE_REQUIRED")
    if json.loads((EVIDENCE / "full-regression-execution.json").read_text())["exit_code"] != 0:
        raise SystemExit("FINAL_REGRESSION_NOT_PASSED")
    invoke("freeze", [sys.executable, "-B", str(PACKAGE / "freeze_contract.py")])
    invoke("freeze-verify", [sys.executable, "-B", str(PACKAGE / "freeze_contract.py"), "--verify"])
    authorization = WORKSPACE / ".codex-private" / ATTEMPT / "authorization.json"
    if authorization.exists():
        raise SystemExit("EXTERNAL_AUTHORIZATION_ALREADY_EXISTS_DO_NOT_OVERWRITE")
    authorization.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "w1-external-launch-authorization/2.0.0", "attempt": ATTEMPT,
               "execution_manifest_sha256": digest(PACKAGE / "execution-manifest.json"),
               "hashes_sha256": digest(PACKAGE / "hashes.json"),
               "resource_request_sha256": digest(PACKAGE / "RESOURCE_REQUEST.json"),
               "token_sha256": None, "status": "NOT_APPROVED"}
    record(authorization, payload)
    if json.loads(authorization.read_text(encoding="utf-8")) != payload:
        raise SystemExit("EXTERNAL_DRAFT_BINDING_ROUNDTRIP_FAILED")
    before = snapshot(PACKAGE)
    record(EVIDENCE / "final-package-before.json", before)
    process = invoke("final-original-preflight", [sys.executable, "-B", str(PACKAGE / "launch_once.py"),
                                                 "--authorization-file", str(authorization), "--preflight-only"])
    after = snapshot(PACKAGE)
    record(EVIDENCE / "final-package-after.json", after)
    output = process.stdout.decode("utf-8", errors="replace")
    reports = []
    for line in output.splitlines():
        try:
            reports.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    windows = next((item for item in reversed(reports) if item.get("status") == "preflight_pass"), None)
    if before != after or not windows or windows.get("staging_started") is not False or windows.get("worker_started") is not False:
        raise RuntimeError("FINAL_ORIGINAL_PREFLIGHT_CONTRACT_FAILED")
    result = {"schema": "w1-label-v6-final-preflight-evidence/1.0.0", **payload,
              "exit_code": process.returncode, "before_after_equal": before == after,
              "all_file_count": len(before), "staging_started": windows["staging_started"],
              "worker_started": windows["worker_started"], "official_token_created": False,
              "authorization_is_unapproved_draft": True, "windows_output": windows,
              "full_windows_wsl_output_file": "final-original-preflight.stdout",
              "stderr_file": "final-original-preflight.stderr"}
    record(EVIDENCE / "final-preflight-summary.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
