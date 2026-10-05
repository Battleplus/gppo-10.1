"""Exec the isolated acceptance interpreter; secrets remain solely on stdin."""
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
request = json.loads((root / "SERVER_ACCEPTANCE_REQUEST.json").read_text(encoding="utf-8"))
if request["gpu"]["physical_device"] != 1 or request.get("smoke_adapter") is not True:
    raise RuntimeError("SMOKE_WORKER_GPU_SCOPE_INVALID")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.pop("W1_PHASE_ACCOUNTING_SOCKET", None)
arguments = json.loads(os.environ.pop("W1_SMOKE_WORKER_ARGUMENTS_JSON"))
if not isinstance(arguments, list) or any(not isinstance(arg, str) for arg in arguments):
    raise RuntimeError("SMOKE_WORKER_ARGUMENTS_INVALID")
os.execv(sys.executable, [sys.executable, "-I", "-B", "-X",
    "pycache_prefix=/home/user1/.venvs/w1-runtime-v1/.w1-no-bytecode-cache",
    str(root / "acceptance_entry.py"), *arguments])
