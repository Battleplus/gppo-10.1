"""Meter one persistent Linux transport root and all processes it owns.

This module verifies only the root process and its descendants. It cannot
account for the sshd process that starts this root, or observe this root after
it exits; scope_report() keeps those transport boundaries explicit.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from typing import IO

from linux_process_scope import (
    ProcessScopeError,
    enable_subreaper,
    proc_tree,
    reap_owned_children,
    tree_cpu_ticks,
)


TRANSPORT_CPU_SCOPE_SCHEMA = "w1-remote-transport-cpu-scope/1.0.0"
_ADMIN_CONDITION = (
    "A host-admin configured and validated per-connection accounting scope "
    "must include the sshd session process, each exec/SFTP server process, "
    "their descendants, and usage through scope removal; concurrent SSH "
    "connections must not share the charged scope."
)


class TransportCpuScopeError(RuntimeError):
    """Raised when this process cannot verify its Linux descendant scope."""


class TransportCpuScope:
    """A persistent remote root that starts and accounts for transport work.

    Start this object in the one long-lived remote SSH exec command. Launch all
    remote exec payloads and any SFTP server as its children. Separate SSH
    channels opened directly by the SSH client are outside this process tree.
    """

    def __init__(self) -> None:
        if not sys.platform.startswith("linux"):
            raise TransportCpuScopeError("LINUX_PROC_AND_SUBREAPER_REQUIRED")
        self.pid = os.getpid()
        try:
            self.clock_ticks = int(os.sysconf("SC_CLK_TCK"))
            if self.clock_ticks <= 0:
                raise ValueError
            enable_subreaper()
            rows = proc_tree(self.pid)
        except (OSError, ValueError, ProcessScopeError) as exc:
            raise TransportCpuScopeError("TRANSPORT_SCOPE_START_FAILED") from exc
        if self.pid not in rows:
            raise TransportCpuScopeError("TRANSPORT_SCOPE_ROOT_MISSING")
        if len(rows) != 1:
            raise TransportCpuScopeError("TRANSPORT_SCOPE_PREEXISTING_DESCENDANTS")
        self.root_start_ticks = rows[self.pid]["start"]
        self._last_ticks = 0
        self._children: dict[int, tuple[subprocess.Popen, str]] = {}
        self._closed = False
        self._final_report: dict[str, object] | None = None

    def spawn_remote_process(
        self,
        argv: Sequence[str],
        *,
        purpose: str,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        stdin: int | IO[bytes] | None = None,
        stdout: int | IO[bytes] | None = None,
        stderr: int | IO[bytes] | None = None,
    ) -> subprocess.Popen:
        """Start one remote exec payload or an SFTP server below this root."""
        if self._closed:
            raise TransportCpuScopeError("TRANSPORT_SCOPE_ALREADY_CLOSED")
        if (isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence)
                or not argv or any(not isinstance(part, str) or not part for part in argv)):
            raise ValueError("TRANSPORT_PROCESS_ARGV_INVALID")
        if not isinstance(purpose, str) or purpose not in {"ssh_exec", "sftp_server"}:
            raise ValueError("TRANSPORT_PROCESS_PURPOSE_INVALID")
        try:
            process = subprocess.Popen(
                list(argv), cwd=cwd, env=None if env is None else dict(env),
                stdin=stdin, stdout=stdout, stderr=stderr,
                close_fds=True, start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            raise TransportCpuScopeError("TRANSPORT_PROCESS_START_FAILED") from exc
        self._children[process.pid] = (process, purpose)
        return process

    def spawn_sftp_server(
        self,
        argv: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        stderr: int | IO[bytes] | None = None,
    ) -> subprocess.Popen:
        """Start a stdio SFTP server as a measured child of the persistent root."""
        return self.spawn_remote_process(
            argv, purpose="sftp_server", cwd=cwd, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
        )

    def wait(self, process: subprocess.Popen, *, timeout: float | None = None) -> int:
        """Wait for a process started by this scope; its CPU stays in root totals."""
        entry = self._children.get(getattr(process, "pid", None))
        if entry is None or entry[0] is not process:
            raise TransportCpuScopeError("TRANSPORT_PROCESS_NOT_OWNED")
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise
        except (ChildProcessError, OSError) as exc:
            raise TransportCpuScopeError("TRANSPORT_PROCESS_WAIT_FAILED") from exc

    def scope_report(self) -> dict[str, object]:
        """Return a measured subtree snapshot and explicit transport gaps.

        Kernel cumulative counters include this root from process creation and
        include short-lived children after their parent waits for them. A stable
        proc-tree scan counts each live or waited process once.
        """
        if self._final_report is not None:
            return dict(self._final_report)
        try:
            rows = proc_tree(self.pid, expected_start=self.root_start_ticks)
            ticks = tree_cpu_ticks(rows)
            if ticks < self._last_ticks:
                raise ProcessScopeError("CPU_SCOPE_PROC_SNAPSHOT_NONMONOTONIC")
            self._last_ticks = ticks
            root = rows[self.pid]
            descendants = len(rows) - 1
            process_tree_status = (
                "verified_complete_at_snapshot" if descendants == 0
                else "verified_live_descendants_at_snapshot"
            )
            errors: list[str] = []
            measured_seconds: float | None = ticks / self.clock_ticks
            root_waited_ticks: int | None = root["waited_ticks"]
            row_count: int | None = len(rows)
            root_start: int | None = self.root_start_ticks
        except (KeyError, ProcessScopeError, OSError) as exc:
            process_tree_status = "unavailable"
            errors = ["TRANSPORT_PROCESS_TREE_SNAPSHOT_FAILED:" + str(exc)]
            measured_seconds = None
            root_waited_ticks = None
            row_count = None
            root_start = self.root_start_ticks

        return {
            "schema": TRANSPORT_CPU_SCOPE_SCHEMA,
            "status": "partial_unverified" if not errors else "measurement_unavailable",
            "process_tree_scope_status": process_tree_status,
            "process_tree_cpu_seconds_measured": measured_seconds,
            "process_tree_cpu_seconds_charged_once": measured_seconds,
            "process_tree_root_pid": self.pid,
            "process_tree_root_start_ticks": root_start,
            "process_tree_row_count": row_count,
            "root_reaped_descendant_cpu_ticks": root_waited_ticks,
            "clock_ticks_per_second": self.clock_ticks,
            "snapshot_monotonic": time.monotonic(),
            "root_cpu_boundary": "root process creation through this snapshot, including interpreter startup at kernel tick resolution",
            "nested_worker_cpu_rule": "all descendants are already included in the one process-tree total; nested snapshots must remain diagnostic and must not be added",
            "remote_transport_scope_status": "incomplete_requires_host_scope",
            "authoritative_remote_transport_cpu_seconds": None,
            "scope_boundary_reason": (
                "This is a same-UID descendant-tree meter. It does not read or charge ancestor sshd work; "
                "server privilege separation varies, and a child cannot sample its ancestor after root exit."
            ),
            "unmeasured_remote_transport_components": [
                "sshd session/channel process that starts this root, including setup before root start and cleanup after root exit",
                "sshd exec or SFTP subsystem server processes used by independent channels",
                "this root's own final report serialization and process-exit tail after the snapshot",
            ],
            "independent_ssh_channel_scope_status": "excluded_by_process_tree_root",
            "required_host_admin_condition": _ADMIN_CONDITION,
            "cleanup_status": "not_finalized",
            "errors": errors,
        }

    def finalize(self, *, reap_timeout_seconds: float = 2.0) -> dict[str, object]:
        """Stop and reap owned descendants, then capture their inclusive CPU."""
        if self._final_report is not None:
            return dict(self._final_report)
        if (isinstance(reap_timeout_seconds, bool)
                or not isinstance(reap_timeout_seconds, (int, float))
                or not math.isfinite(reap_timeout_seconds)
                or reap_timeout_seconds < 0):
            raise ValueError("TRANSPORT_SCOPE_REAP_TIMEOUT_INVALID")
        cleanup: dict[str, object]
        try:
            cleanup = reap_owned_children(float(reap_timeout_seconds))
            cleanup_status = str(cleanup.get("status", "unknown"))
        except (OSError, ProcessScopeError) as exc:
            cleanup = {"status": "failed", "error": str(exc)}
            cleanup_status = "failed"
        for process, _purpose in self._children.values():
            for stream_name in ("stdin", "stdout", "stderr"):
                stream = getattr(process, stream_name, None)
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except OSError:
                        pass
        report = self.scope_report()
        report["cleanup_status"] = cleanup_status
        report["cleanup"] = cleanup
        self._closed = True
        self._final_report = report
        return dict(report)


def scope_report(scope: TransportCpuScope) -> dict[str, object]:
    """Small functional entry point for callers that keep the scope object."""
    if not isinstance(scope, TransportCpuScope):
        raise TypeError("TRANSPORT_SCOPE_OBJECT_REQUIRED")
    return scope.scope_report()
