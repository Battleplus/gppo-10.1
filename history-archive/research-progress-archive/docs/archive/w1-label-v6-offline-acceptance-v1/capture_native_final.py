"""Read original terminal WSL artifacts; never execute a research worker."""
import json
from pathlib import Path
import subprocess

from analyze_v6 import ATTEMPT, read_json, require, sha, snapshot, write_json


def main():
    output = Path(__file__).resolve().parent
    native = Path("//wsl$/Ubuntu-24.04/home/asus") / ATTEMPT
    before = snapshot(native)
    final = read_json(native / "launcher-status.json")
    export = read_json(native / "export-status.json")
    require(final["status"] == "stopped" and final["stage"] == "export_failed", "native_not_terminal")
    require(export["status"] == "failed", "native_export_not_failed")
    native_sample = read_json(native / "supervisor-status.json")
    pids = [native_sample["supervisor_pid"], native_sample["pid"]]
    probe = "import json,pathlib; print(json.dumps({str(p):pathlib.Path('/proc/'+str(p)).exists() for p in " + repr(pids) + "}))"
    command = ["wsl.exe", "--distribution", "Ubuntu-24.04", "--exec", "/usr/bin/python3", "-I", "-B", "-c", probe]
    completed = subprocess.run(command, capture_output=True, timeout=20, check=False)
    require(completed.returncode == 0, "native_process_readonly_probe_failed")
    stdout = completed.stdout.decode("utf-8", errors="strict")
    stderr = completed.stderr.decode("utf-8", errors="backslashreplace")
    existing = json.loads(stdout.strip())
    require(not any(existing.values()), "old_worker_or_supervisor_pid_present")
    after = snapshot(native)
    require(before == after, "native_changed_during_audit")
    local = output.parents[1] / "runs" / ATTEMPT
    payload = read_json(local / "export-manifest.json")
    native_payload_mismatches = [name for name, record in payload["files"].items() if before.get(name) != record["sha256"]]
    evidence = {"native_source": str(native), "capture_scope": "readonly terminal artifacts and /proc PID presence",
                "native_launcher_final": final, "native_export_final": export,
                "native_status_sha256": sha((native / "launcher-status.json").read_bytes()),
                "native_export_sha256": sha((native / "export-status.json").read_bytes()),
                "native_vs_copied_payload_mismatches": native_payload_mismatches,
                "native_file_count": len(before), "native_bytes": sum(p.stat().st_size for p in native.rglob("*") if p.is_file()),
                "process_probe_command": command, "process_probe_exit_code": completed.returncode,
                "process_probe_stdout": stdout, "process_probe_stderr": stderr,
                "process_probe_stdout_hex": completed.stdout.hex(), "process_probe_stderr_hex": completed.stderr.hex(),
                "original_pids_present": existing, "native_files_unchanged": True,
                "research_environment_calls": 0, "model_calls": 0, "formal_attempt_created": False}
    write_json(output / "native-final-status-evidence-v2.json", evidence)
    print(json.dumps({"native_status": final["status"], "native_stage": final["stage"],
                      "export_status": export["status"], "original_pids_present": existing,
                      "native_files_unchanged": True, "native_vs_copied_payload_mismatches": native_payload_mismatches}))


if __name__ == "__main__":
    main()
