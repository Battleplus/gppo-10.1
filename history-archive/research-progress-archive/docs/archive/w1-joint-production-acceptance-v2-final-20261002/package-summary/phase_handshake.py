"""Synchronous worker stage boundaries; authenticated by Linux peer credentials."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import resource
import socket
import struct
import time


def request_boundary(row):
    path = os.environ.get("W1_PHASE_ACCOUNTING_SOCKET")
    if not path:
        return {"scope": "local_test_without_supervisor_handshake"}
    payload = json.dumps(row, allow_nan=False).encode() + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(10)
        channel.connect(path)
        channel.sendall(payload)
        response = b""
        while not response.endswith(b"\n"):
            block = channel.recv(4096)
            if not block:
                raise RuntimeError("PHASE_ACCOUNTING_ACK_MISSING")
            response += block
            if len(response) > 8192:
                raise RuntimeError("PHASE_ACCOUNTING_ACK_TOO_LARGE")
    value = json.loads(response)
    if value.get("status") != "acknowledged":
        raise RuntimeError("PHASE_ACCOUNTING_REJECTED:" + str(value.get("error")))
    return value


class StageServer:
    def __init__(self, path, request, *, tree_reader, cpu_origin_seconds=0.0,
                 wall_origin_monotonic=None):
        if (type(cpu_origin_seconds) not in (int, float)
                or not math.isfinite(cpu_origin_seconds) or cpu_origin_seconds < 0):
            raise ValueError("PHASE_CPU_ORIGIN_INVALID")
        self.path = Path(path)
        self.request = request
        self.tree_reader = tree_reader
        self.rows = []
        self.stage = "staging_and_zero_step_gate"
        self.cpu_origin_seconds = float(cpu_origin_seconds)
        self.cpu_start = float(cpu_origin_seconds)
        self.wall_origin_monotonic = (time.monotonic() if wall_origin_monotonic is None
                                      else wall_origin_monotonic)
        if (type(self.wall_origin_monotonic) not in (int, float)
                or not math.isfinite(self.wall_origin_monotonic)
                or self.wall_origin_monotonic < 0):
            raise ValueError("PHASE_WALL_ORIGIN_INVALID")
        self.wall_start = float(self.wall_origin_monotonic)
        self.last_worker_cpu_end = 0.0
        self.last_worker_wall_end = None
        self.last_cpu_cumulative = 0.0
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.bind(str(self.path))
        os.chmod(self.path, 0o600)
        self.socket.listen(1)
        self.socket.setblocking(False)

    def process_pending(self, worker_pid):
        try:
            channel, _address = self.socket.accept()
        except BlockingIOError:
            return []
        with channel:
            channel.settimeout(5)
            try:
                pid, uid, _gid = struct.unpack("3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                if pid != worker_pid or uid != os.getuid():
                    raise RuntimeError("PHASE_PEER_IDENTITY_MISMATCH")
                data = b""
                while not data.endswith(b"\n"):
                    block = channel.recv(4096)
                    if not block:
                        raise RuntimeError("PHASE_REQUEST_INCOMPLETE")
                    data += block
                    if len(data) > 8192:
                        raise RuntimeError("PHASE_REQUEST_TOO_LARGE")
                value = json.loads(data)
                if not isinstance(value, dict):
                    raise RuntimeError("PHASE_REQUEST_NOT_OBJECT")
                request_pid = value.get("pid")
                if isinstance(request_pid, bool) or not isinstance(request_pid, int) or request_pid != pid:
                    raise RuntimeError("PHASE_TRANSITION_IDENTITY_MISMATCH")
                if value.get("stage") != self.stage:
                    raise RuntimeError("PHASE_TRANSITION_IDENTITY_MISMATCH")
                next_stage = value.get("next_stage")
                if not isinstance(next_stage, str) or next_stage not in self.request["stages"]:
                    raise RuntimeError("PHASE_NEXT_STAGE_UNKNOWN")
                numeric = {}
                for name in ("cpu_start_seconds", "cpu_end_seconds", "cpu_seconds",
                             "wall_start_monotonic", "wall_end_monotonic", "wall_seconds"):
                    number = value.get(name)
                    if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                        raise RuntimeError("PHASE_VALUE_INVALID:" + name)
                    numeric[name] = float(number)
                if numeric["cpu_end_seconds"] < numeric["cpu_start_seconds"]:
                    raise RuntimeError("PHASE_CPU_CLOCK_REVERSED")
                if numeric["wall_end_monotonic"] < numeric["wall_start_monotonic"]:
                    raise RuntimeError("PHASE_WALL_CLOCK_REVERSED")
                if abs(numeric["cpu_seconds"] - (numeric["cpu_end_seconds"] - numeric["cpu_start_seconds"])) > 1e-6:
                    raise RuntimeError("PHASE_CPU_DELTA_MISMATCH")
                if abs(numeric["wall_seconds"] - (numeric["wall_end_monotonic"] - numeric["wall_start_monotonic"])) > 1e-6:
                    raise RuntimeError("PHASE_WALL_DELTA_MISMATCH")
                if abs(numeric["cpu_start_seconds"] - self.last_worker_cpu_end) > 1e-6:
                    raise RuntimeError("PHASE_CPU_BASELINE_DISCONTINUITY")
                if (self.last_worker_wall_end is not None
                        and numeric["wall_start_monotonic"] < self.last_worker_wall_end - 1e-6):
                    raise RuntimeError("PHASE_WALL_BASELINE_REVERSED")
                if numeric["wall_end_monotonic"] > time.monotonic() + 0.05:
                    raise RuntimeError("PHASE_WALL_SNAPSHOT_IN_FUTURE")
                tree = self.tree_reader(os.getpid())
                if set(tree) != {os.getpid(), worker_pid}:
                    raise RuntimeError("PHASE_BOUNDARY_HAS_ACTIVE_DESCENDANTS")
                own = resource.getrusage(resource.RUSAGE_SELF)
                waited = resource.getrusage(resource.RUSAGE_CHILDREN)
                cpu = own.ru_utime + own.ru_stime + waited.ru_utime + waited.ru_stime + numeric["cpu_end_seconds"]
                cumulative = cpu - self.cpu_origin_seconds
                if cumulative < self.last_cpu_cumulative - 1e-6:
                    raise RuntimeError("PHASE_CPU_CUMULATIVE_REVERSED")
                wall = time.monotonic()
                row = {"stage": self.stage, "next_stage": next_stage,
                       "cpu_seconds": cpu - self.cpu_start, "wall_seconds": wall - self.wall_start,
                       "cpu_seconds_cumulative": cumulative, "sampled_monotonic": wall,
                       "worker_reported_cpu_seconds": numeric["cpu_seconds"],
                       "worker_reported_wall_seconds": numeric["wall_seconds"],
                       "worker_cpu_end_seconds": numeric["cpu_end_seconds"],
                       "worker_wall_end_monotonic": numeric["wall_end_monotonic"],
                       "scope": "supervisor self and waited helpers plus quiescent worker self and waited descendants"}
                cap = self.request["stages"][self.stage]
                if row["cpu_seconds"] < 0 or row["wall_seconds"] < 0:
                    raise RuntimeError("PHASE_CLOCK_REVERSED")
                prior_cpu = sum(item["cpu_seconds"] for item in self.rows if item["stage"] == self.stage)
                prior_wall = sum(item["wall_seconds"] for item in self.rows if item["stage"] == self.stage)
                if row["cpu_seconds"] + prior_cpu > cap["complete_process_cpu_seconds"] or row["wall_seconds"] + prior_wall > cap["wall_seconds"]:
                    raise RuntimeError("PHASE_RESOURCE_CAP_EXCEEDED:" + self.stage)
                self.rows.append(row)
                self.stage = next_stage
                self.cpu_start, self.wall_start = cpu, wall
                self.last_worker_cpu_end = numeric["cpu_end_seconds"]
                self.last_worker_wall_end = numeric["wall_end_monotonic"]
                self.last_cpu_cumulative = cumulative
                channel.sendall(json.dumps({"status": "acknowledged", **row}).encode() + b"\n")
                return [row]
            except BaseException as exc:
                channel.sendall(json.dumps({"status": "rejected", "error": str(exc)}).encode() + b"\n")
                raise

    def close(self):
        self.socket.close()
        self.path.unlink(missing_ok=True)
