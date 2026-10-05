"""Read-only server audit for preparation; no attempt or tensor construction."""
import argparse
import getpass
import json
from pathlib import Path
import sys
from launch_joint_once import ROOT, connect, execute, remote_readonly_preflight

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads((ROOT / "launch-contract.json").read_text(encoding="utf-8"))
    password = getpass.getpass("Remote SSH password: ")
    client = connect(contract, password)
    password = None
    try:
        result = remote_readonly_preflight(client, contract)
        result["load"] = execute(client, ["cat", "/proc/loadavg"], timeout=10)
        result["gpu"] = execute(client, ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], timeout=10)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))
    finally:
        client.close()
