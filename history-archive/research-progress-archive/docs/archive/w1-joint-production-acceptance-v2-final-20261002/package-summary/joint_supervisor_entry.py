"""Isolated child wrapper around the reused native supervisor."""
from pathlib import Path
import json
import os
import resource
import sys
import time

root = Path(__file__).resolve().parent
PROCESS_MODULE_ENTRY_MONOTONIC = time.monotonic()
stage = "supervisor_module_entry"
result_status = "supervisor_exception"
error = None
code = 1
sys.path.insert(0, str(root))
from supervise import main
from infra_io import durable_atomic_json

try:
    stage = "supervisor_and_worker"
    code = main(root, "runner.py")
    result_status = "supervisor_returned"
except BaseException as exc:
    error = type(exc).__name__ + ": " + str(exc)
    raise
finally:
    own = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    payload = {
        "schema": "w1-native-launcher-cpu-accounting/3.0.0",
        "status": result_status,
        "stage": stage,
        "error": error,
        "pid": os.getpid(),
        "process_module_entry_monotonic": PROCESS_MODULE_ENTRY_MONOTONIC,
        "sampled_monotonic": time.monotonic(),
        "native_launcher_process_cpu_seconds": own.ru_utime + own.ru_stime,
        "native_launcher_reaped_children_cpu_seconds": child.ru_utime + child.ru_stime,
        "native_launcher_complete_process_cpu_snapshot_seconds": own.ru_utime + own.ru_stime + child.ru_utime + child.ru_stime,
        "scope": "joint supervisor self plus reaped workers; sampled after supervise.main return or exception, before this write and process exit",
    }
    try:
        durable_atomic_json(root / "runtime-output" / "native-launcher-accounting.json", payload)
    except BaseException as accounting_error:
        print(
            "native launcher accounting write failed: "
            + type(accounting_error).__name__ + ": " + str(accounting_error),
            file=sys.stderr,
        )
raise SystemExit(code)
