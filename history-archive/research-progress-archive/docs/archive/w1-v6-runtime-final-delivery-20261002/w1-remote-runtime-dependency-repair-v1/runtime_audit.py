"""Read-only host/environment audit, apart from writing its own evidence file."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


COMMANDS = {
    "host": ["uname", "-a"],
    "os": ["cat", "/etc/os-release"],
    "gpu": ["nvidia-smi"],
    "gpu_identity": ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader"],
    "gpu_processes": ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
    "load": ["uptime"], "cpus": ["nproc"], "memory": ["free", "-b"],
    "disk": ["df", "-B1", "/", "/home", "/tmp"],
    "cuda_loader": ["ldconfig", "-p"],
    "cuda_compiler": ["nvcc", "--version"],
    "default_python": ["/usr/bin/python3", "-B", "-c", "import sys,json; print(json.dumps(dict(executable=sys.executable,version=sys.version,prefix=sys.prefix,path=sys.path)))"],
    "default_torch_import": ["/usr/bin/python3", "-B", "-c", "import torch; print(torch.__version__); print(torch.version.cuda)"],
    "default_dependencies": ["/usr/bin/python3", "-m", "pip", "list", "--format=json"],
    "conda_environments": ["/home/user1/anaconda3/bin/conda", "env", "list", "--json"],
    "pytorch_environment": ["/home/user1/anaconda3/envs/pytorch/bin/python3.9", "-B", "-c", "import sys,importlib.util,json; print(json.dumps(dict(executable=sys.executable,version=sys.version,prefix=sys.prefix,torch_spec=str(importlib.util.find_spec('torch')))))"],
    "pytorch_dependencies": ["/home/user1/anaconda3/envs/pytorch/bin/python3.9", "-m", "pip", "list", "--format=json"],
    "process_snapshot": ["ps", "-eo", "pid,ppid,user,stat,pcpu,pmem,etime,comm", "--sort=-pcpu"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise SystemExit("AUDIT_EVIDENCE_ALREADY_EXISTS")
    result = {"schema": "w1-remote-runtime-readonly-audit/1.0.0", "commands": {},
              "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "formal_attempt_created": False, "research_data_read": False,
              "model_calls": 0, "optimizer_updates": 0,
              "loader_environment": {key: os.environ.get(key) for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "CUDA_VISIBLE_DEVICES")}}
    for name, command in COMMANDS.items():
        try:
            process = subprocess.run(command, text=True, capture_output=True, timeout=30)
            result["commands"][name] = {"argv": command, "returncode": process.returncode,
                                       "stdout": process.stdout, "stderr": process.stderr}
        except BaseException as exc:
            result["commands"][name] = {"argv": command, "error": f"{type(exc).__name__}: {exc}"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_path": str(path), "commands": len(result["commands"]), "formal_attempt_created": False}))


if __name__ == "__main__":
    main()
