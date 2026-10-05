"""Reconcile returned remote file identities with the frozen local delivery."""
import hashlib
import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent / "remote-runtime"
PACKAGE = EVIDENCE.parents[1] / "w1-remote-runtime-dependency-repair-v1"
REMOTE_ROOT = "/home/user1/w1-remote-runtime-dependency-repair-v1/"


def read_snapshot(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, absolute = line.split("  ", 1)
        if not absolute.startswith(REMOTE_ROOT) or len(digest) != 64:
            raise RuntimeError("REMOTE_IDENTITY_LINE_INVALID")
        result[absolute[len(REMOTE_ROOT):]] = digest
    return result


before = read_snapshot(EVIDENCE / "final-original-files-before.sha256")
after = read_snapshot(EVIDENCE / "final-original-files-after.sha256")
local = {path.relative_to(PACKAGE).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
         for path in PACKAGE.rglob("*") if path.is_file()}
output = json.loads((EVIDENCE / "final-original-preflight.stdout").read_text())
exit_code = int((EVIDENCE / "final-original-preflight.exit").read_text())
if exit_code != 0 or before != after or after != local:
    raise RuntimeError("FINAL_REMOTE_ORIGINAL_PREFLIGHT_IDENTITY_FAILED")
if not (output["status"] == "pass" and output["tensor_calls"] == 0
        and output["model_calls"] == 0 and output["optimizer_updates"] == 0
        and output["staging_started"] is False and output["worker_started"] is False):
    raise RuntimeError("REMOTE_PREFLIGHT_SCOPE_FAILED")
command = next(line for line in (PACKAGE / "README.md").read_text(encoding="utf-8").splitlines()
               if line.startswith("env -u LD_LIBRARY_PATH"))
summary = {"actual_command": command, "exit_code": exit_code,
           "file_count_including_outer_manifest": len(local),
           "local_remote_equal": True, "before_after_equal": True,
           "remote_output": output, "formal_attempt_created": False,
           "stdout_file": "final-original-preflight.stdout",
           "stderr_file": "final-original-preflight.stderr",
           "stderr_observation": "empty; SCP omits progress for zero-byte file",
           "training_authorization": False}
(EVIDENCE / "final-remote-preflight-summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in summary.items() if key != "actual_command"}))
