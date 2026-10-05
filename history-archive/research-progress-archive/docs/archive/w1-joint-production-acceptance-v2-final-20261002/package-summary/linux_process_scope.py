"""Owner-scoped Linux process-tree accounting and subreaper cleanup."""
from __future__ import annotations

import ctypes
import errno
import math
import os
from pathlib import Path
import signal
import time


PR_SET_CHILD_SUBREAPER = 36
PR_GET_CHILD_SUBREAPER = 37
_PROC = Path("/proc")
_BASELINE_CHILDREN: set[tuple[int, int]] | None = None
_SUBREAPER_ENABLED = False


class ProcessScopeError(RuntimeError):
    """Raised when the owned Linux process scope cannot be verified."""


class _SnapshotRace(Exception):
    pass


def _prctl(option: int, argument: int) -> int:
    if not sys_platform_linux():
        raise ProcessScopeError("LINUX_PRCTL_REQUIRED")
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.prctl
    call.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                     ctypes.c_ulong, ctypes.c_ulong]
    call.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = call(option, argument, 0, 0, 0)
    if result == -1:
        number = ctypes.get_errno() or errno.EPERM
        raise OSError(number, os.strerror(number))
    return result


def sys_platform_linux() -> bool:
    return os.name == "posix" and _PROC.is_dir()


def _read_proc_record(pid: int, *, owner_uid: int | None = None) -> dict:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ProcessScopeError("PROC_PID_INVALID")
    path = _PROC / str(pid)
    expected_uid = os.getuid() if owner_uid is None else owner_uid
    try:
        before = path.stat()
        if before.st_uid != expected_uid:
            raise ProcessScopeError("CPU_SCOPE_DESCENDANT_OWNER_MISMATCH")
        data = (path / "stat").read_text(encoding="ascii")
        after = path.stat()
    except (FileNotFoundError, ProcessLookupError) as exc:
        raise _SnapshotRace from exc
    except PermissionError as exc:
        raise ProcessScopeError("CPU_SCOPE_PROC_PERMISSION_DENIED") from exc
    if after.st_uid != expected_uid:
        raise ProcessScopeError("CPU_SCOPE_DESCENDANT_OWNER_MISMATCH")
    close = data.rfind(")")
    if close < 0:
        raise ProcessScopeError("CPU_SCOPE_PROC_STAT_MALFORMED")
    fields = data[close + 2:].split()
    if len(fields) <= 19:
        raise ProcessScopeError("CPU_SCOPE_PROC_STAT_MALFORMED")
    try:
        record = {
            "pid": pid,
            "ppid": int(fields[1]),
            "state": fields[0],
            "self_ticks": int(fields[11]) + int(fields[12]),
            "waited_ticks": int(fields[13]) + int(fields[14]),
            "start": int(fields[19]),
            "owner_uid": after.st_uid,
        }
    except (ValueError, IndexError) as exc:
        raise ProcessScopeError("CPU_SCOPE_PROC_STAT_MALFORMED") from exc
    if record["ppid"] < 0 or record["self_ticks"] < 0 or record["waited_ticks"] < 0 or record["start"] < 0:
        raise ProcessScopeError("CPU_SCOPE_PROC_STAT_NEGATIVE_FIELD")
    record["ticks"] = record["self_ticks"]
    record["total_ticks"] = record["self_ticks"] + record["waited_ticks"]
    return record


def _children_of(pid: int) -> set[int]:
    task_root = _PROC / str(pid) / "task"
    children: set[int] = set()
    try:
        tasks = tuple(task_root.iterdir())
    except (FileNotFoundError, ProcessLookupError) as exc:
        raise _SnapshotRace from exc
    except PermissionError as exc:
        raise ProcessScopeError("CPU_SCOPE_PROC_PERMISSION_DENIED") from exc
    for task in tasks:
        try:
            raw = (task / "children").read_text(encoding="ascii").strip()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise ProcessScopeError("CPU_SCOPE_PROC_PERMISSION_DENIED") from exc
        for value in raw.split():
            try:
                child = int(value)
            except ValueError as exc:
                raise ProcessScopeError("CPU_SCOPE_PROC_CHILDREN_MALFORMED") from exc
            if child <= 0:
                raise ProcessScopeError("CPU_SCOPE_PROC_CHILDREN_MALFORMED")
            children.add(child)
    return children


def _collect_once(root_pid: int, *, expected_start: int | None = None,
                  owner_uid: int | None = None) -> dict[int, dict]:
    expected_uid = os.getuid() if owner_uid is None else owner_uid
    root = _read_proc_record(root_pid, owner_uid=expected_uid)
    if expected_start is not None and root["start"] != expected_start:
        raise ProcessScopeError("CPU_SCOPE_ROOT_IDENTITY_CHANGED")
    rows = {root_pid: root}
    child_sets: dict[int, set[int]] = {}
    pending = [root_pid]
    while pending:
        parent_pid = pending.pop()
        children = _children_of(parent_pid)
        child_sets[parent_pid] = children
        parent = rows[parent_pid]
        for child_pid in children:
            if child_pid in rows:
                existing = rows[child_pid]
                if existing["ppid"] != parent_pid:
                    raise _SnapshotRace
                continue
            child = _read_proc_record(child_pid, owner_uid=expected_uid)
            if child["ppid"] != parent_pid:
                raise _SnapshotRace
            rows[child_pid] = child
            pending.append(child_pid)

    # Reject a scan where a parent reaped or gained children mid-traversal.
    for parent_pid, children_before in child_sets.items():
        parent_before = rows[parent_pid]
        parent_after = _read_proc_record(parent_pid, owner_uid=expected_uid)
        if (parent_after["start"] != parent_before["start"]
                or parent_after["ppid"] != parent_before["ppid"]
                or parent_after["waited_ticks"] != parent_before["waited_ticks"]
                or _children_of(parent_pid) != children_before):
            raise _SnapshotRace
    for pid, before in rows.items():
        after = _read_proc_record(pid, owner_uid=expected_uid)
        if after["start"] != before["start"] or after["ppid"] != before["ppid"]:
            raise _SnapshotRace
    return rows


def proc_tree(root_pid: int, *, expected_start: int | None = None,
              owner_uid: int | None = None, attempts: int = 4) -> dict[int, dict]:
    """Return a stable owner-filtered descendant snapshot rooted at ``root_pid``.

    Each live process contributes its own ticks plus kernel-maintained waited
    child ticks. A process appears either as a live row or in its living
    parent's ``cutime+cstime`` after wait; stable snapshots do not add both.
    """
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("CPU_SCOPE_SNAPSHOT_ATTEMPTS_INVALID")
    last_race = None
    for _ in range(attempts):
        try:
            return _collect_once(root_pid, expected_start=expected_start, owner_uid=owner_uid)
        except _SnapshotRace as exc:
            last_race = exc
    raise ProcessScopeError("CPU_SCOPE_PROC_SNAPSHOT_UNSTABLE") from last_race


def tree_cpu_ticks(rows: dict[int, dict]) -> int:
    if not isinstance(rows, dict) or not rows:
        raise ProcessScopeError("CPU_SCOPE_TREE_EMPTY")
    total = 0
    for pid, row in rows.items():
        if (isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
                or not isinstance(row, dict) or row.get("pid") != pid):
            raise ProcessScopeError("CPU_SCOPE_TREE_ROW_INVALID")
        self_ticks = row.get("self_ticks")
        waited_ticks = row.get("waited_ticks")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (self_ticks, waited_ticks)):
            raise ProcessScopeError("CPU_SCOPE_TREE_TICKS_INVALID")
        total += self_ticks + waited_ticks
    return total


def monotonic_cpu_delta(previous_ticks: int, current_ticks: int, clock_ticks: int) -> float:
    values = (previous_ticks, current_ticks, clock_ticks)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("CPU_SCOPE_TICK_VALUE_INVALID")
    if clock_ticks == 0:
        raise ValueError("CPU_SCOPE_CLOCK_TICKS_INVALID")
    if current_ticks < previous_ticks:
        raise ProcessScopeError("CPU_SCOPE_PROC_SNAPSHOT_NONMONOTONIC")
    return (current_ticks - previous_ticks) / clock_ticks


def _direct_child_identities(parent_pid: int) -> set[tuple[int, int]]:
    result = set()
    for pid in _children_of(parent_pid):
        row = _read_proc_record(pid)
        if row["ppid"] != parent_pid:
            raise _SnapshotRace
        result.add((pid, row["start"]))
    return result


def _verify_subreaper() -> bool:
    if not sys_platform_linux():
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.prctl
    call.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                     ctypes.c_ulong, ctypes.c_ulong]
    call.restype = ctypes.c_int
    value = ctypes.c_int()
    ctypes.set_errno(0)
    pointer = ctypes.cast(ctypes.byref(value), ctypes.c_void_p).value
    result = call(PR_GET_CHILD_SUBREAPER, pointer, 0, 0, 0)
    if result == -1:
        number = ctypes.get_errno() or errno.EPERM
        raise OSError(number, os.strerror(number))
    return value.value == 1


def enable_subreaper() -> None:
    """Enable and verify Linux child-subreaper behavior before spawning work."""
    global _BASELINE_CHILDREN, _SUBREAPER_ENABLED
    if not sys_platform_linux():
        raise ProcessScopeError("LINUX_SUBREAPER_REQUIRED")
    _prctl(PR_SET_CHILD_SUBREAPER, 1)
    if not _verify_subreaper():
        raise ProcessScopeError("CHILD_SUBREAPER_ENABLE_NOT_CONFIRMED")
    if not _SUBREAPER_ENABLED:
        self_pid = os.getpid()
        last_race = None
        for _ in range(4):
            try:
                _BASELINE_CHILDREN = _direct_child_identities(self_pid)
                break
            except _SnapshotRace as exc:
                last_race = exc
        else:
            raise ProcessScopeError("SUBREAPER_BASELINE_UNSTABLE") from last_race
        _SUBREAPER_ENABLED = True


def _owned_descendant_rows() -> dict[int, dict]:
    if not _SUBREAPER_ENABLED or _BASELINE_CHILDREN is None or not _verify_subreaper():
        raise ProcessScopeError("SUBREAPER_OWNERSHIP_BASELINE_MISSING")
    self_pid = os.getpid()
    last_race = None
    for _ in range(4):
        try:
            direct = _direct_child_identities(self_pid)
            roots = [identity for identity in direct if identity not in _BASELINE_CHILDREN]
            result = {}
            for pid, start in roots:
                subtree = proc_tree(pid, expected_start=start)
                result.update(subtree)
            return result
        except _SnapshotRace as exc:
            last_race = exc
    raise ProcessScopeError("OWNED_DESCENDANT_SCAN_UNSTABLE") from last_race


def _signal_identity(pid: int, start: int, sig: int) -> bool:
    try:
        row = _read_proc_record(pid)
    except _SnapshotRace:
        return False
    if row["start"] != start:
        return False
    if row["state"] == "Z":
        return True
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return False
    return True


def _reap_direct_owned_children() -> None:
    if _BASELINE_CHILDREN is None:
        return
    try:
        identities = _direct_child_identities(os.getpid())
    except _SnapshotRace:
        return
    for pid, start in identities:
        if (pid, start) in _BASELINE_CHILDREN:
            continue
        try:
            row = _read_proc_record(pid)
            if row["start"] == start and row["state"] == "Z":
                os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, ProcessLookupError, _SnapshotRace):
            pass


def reap_owned_children(timeout_seconds: float = 2.0) -> dict:
    """Stop/reap only post-enable direct children adopted by this subreaper.

    Descendant PIDs are individually checked for the current UID and starttime;
    no process-group or global child signal is used. Existing direct children
    present when subreaping was enabled are excluded.
    """
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds < 0):
        raise ValueError("OWNED_CHILD_TIMEOUT_INVALID")
    rows = _owned_descendant_rows()
    signaled = set()
    for row in rows.values():
        key = (row["pid"], row["start"])
        if key not in signaled:
            _signal_identity(*key, signal.SIGTERM)
            signaled.add(key)
    started = time.monotonic()
    deadline = started + float(timeout_seconds)
    kill_at = started + float(timeout_seconds) / 2.0
    killed = set()
    while True:
        _reap_direct_owned_children()
        rows = _owned_descendant_rows()
        if not rows:
            return {"status": "reaped", "signaled_pids": sorted(pid for pid, _ in signaled),
                    "killed_pids": sorted(pid for pid, _ in killed), "unresolved": []}
        now = time.monotonic()
        force = now >= kill_at
        for row in rows.values():
            key = (row["pid"], row["start"])
            if force and key not in killed:
                _signal_identity(*key, signal.SIGKILL)
                killed.add(key)
        if now >= deadline:
            unresolved = sorted({row["pid"] for row in rows.values() if row["state"] != "Z"})
            for row in rows.values():
                if row["state"] == "Z" and row["ppid"] == os.getpid():
                    try:
                        os.waitpid(row["pid"], os.WNOHANG)
                    except (ChildProcessError, ProcessLookupError):
                        pass
            if unresolved:
                raise ProcessScopeError("OWNED_DESCENDANTS_REMAIN:" + ",".join(map(str, unresolved)))
            return {"status": "reaped", "signaled_pids": sorted(pid for pid, _ in signaled),
                    "killed_pids": sorted(pid for pid, _ in killed), "unresolved": []}
        time.sleep(min(0.02, max(0.0, deadline - now)))
