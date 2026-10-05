"""Append-only server task registration for the native Linux launcher."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys
import time
import unicodedata
from typing import Any, Callable, Iterable, Mapping
import uuid

from registration_identity import derive_name_id


REGISTRATION_PATH = Path("/media/abc_disk/admin123/lbh/dengji.txt")
SCHEMA = "w1-server-task-registration/1.0.0"
TERMINAL_EVENTS = frozenset({"SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED"})
SHARED_NAMES = frozenset({"admin", "admin123", "root", "user", "user1", "ubuntu", "test", "example"})
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
NAME_ID_RE = re.compile(r"[a-z]+\Z")
SENSITIVE_KEY_RE = re.compile(
    r"(?i)(?:password|passwd|secret|token|api[_-]?key|private[_-]?key|credential)"
)
CREDENTIAL_VALUE_RE = re.compile(
    r"(?i)(?:--?(?:password|passwd|secret|token|api[_-]?key|private[_-]?key|credential)"
    r"\s*(?:=|\s)\s*|(?:password|passwd|secret|token|api[_-]?key|private[_-]?key|credential)"
    r"\s*[:=]\s*)(?P<value>[^\s,;]+)"
)
REDACTION_MARKERS = frozenset({"<redacted>", "[redacted]", "***", "redacted"})
DETAIL_FIELDS = frozenset({
    "ai", "task_name", "purpose", "workdir", "training_files", "dataset", "model",
    "command_redacted", "gpu_request", "cpu_cores_estimate", "ram_estimate_gb",
    "disk_growth_estimate_gb", "duration_estimate", "estimate_basis", "log_path",
    "checkpoint_path", "resource_snapshot",
})
RESERVED_FIELDS = frozenset({
    "schema", "job_id", "event", "registered_at", "real_name", "name_id",
    "linux_user", "linux_uid", "host", "status", "binding", "started_at",
    "updated_at", "ended_at", "pid", "pgid", "worker_pids", "actual_gpu_ids",
    "worker_pgids", "process_starttimes", "exit_code", "reason_redacted", "result_path",
})


class RegistrationError(RuntimeError):
    """Raised when a registration event cannot be safely recorded or verified."""


def _require_linux() -> None:
    if not sys.platform.startswith("linux"):
        raise RegistrationError("server registration is supported only on native Linux")


def _current_identity() -> tuple[str, int]:
    _require_linux()
    import pwd

    uid = os.getuid()
    try:
        username = pwd.getpwuid(uid).pw_name
    except KeyError as exc:
        raise RegistrationError("current Linux UID has no account name") from exc
    if not username:
        raise RegistrationError("current Linux username is unavailable")
    return username, uid


def _validate_real_name(real_name: str, name_id: str, linux_user: str) -> None:
    if not isinstance(real_name, str) or not real_name.strip():
        raise RegistrationError("real_name is required; registration cannot use an example default")
    if real_name != real_name.strip() or any(ord(ch) < 32 for ch in real_name):
        raise RegistrationError("real_name must be a clean, nonempty user-provided value")
    english_name = re.fullmatch(r"[a-z]+", real_name) is not None
    chinese_name = all(
        ("CJK UNIFIED IDEOGRAPH" in unicodedata.name(ch, "")
         or ch in {"〇", "·"})
        for ch in real_name
    ) and any("CJK UNIFIED IDEOGRAPH" in unicodedata.name(ch, "") for ch in real_name)
    if not (english_name or chinese_name):
        raise RegistrationError("real_name must be a Chinese name or lowercase English name")
    if not isinstance(name_id, str) or NAME_ID_RE.fullmatch(name_id) is None:
        raise RegistrationError("name_id must contain lowercase ASCII letters only")
    try:
        expected_name_id = derive_name_id(real_name)
    except ValueError as exc:
        raise RegistrationError(str(exc)) from exc
    if name_id != expected_name_id:
        raise RegistrationError("name_id does not match the derived real-name identity")
    shared = {value.casefold() for value in SHARED_NAMES}
    shared.add(linux_user.casefold())
    if real_name.casefold() in shared or name_id.casefold() in shared:
        raise RegistrationError("shared Linux account names cannot stand in for a real name")


def _validate_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise RegistrationError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_attempt(attempt: Any) -> str:
    if not isinstance(attempt, str) or not attempt.strip() or any(ord(ch) < 32 for ch in attempt):
        raise RegistrationError("attempt must be a nonempty string")
    return attempt


def _validate_positive_pid(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RegistrationError(f"{field} must be a positive integer")
    return value


def _validate_pid_array(value: Any, field: str) -> list[int]:
    if not isinstance(value, list):
        raise RegistrationError(f"{field} must be a JSON array")
    result = [_validate_positive_pid(item, field) for item in value]
    if len(result) != len(set(result)):
        raise RegistrationError(f"{field} must not contain duplicate PIDs")
    return result


def _validate_gpu_ids(value: Any, *, allow_empty: bool = False) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise RegistrationError("GPU IDs must be an integer array")
    ids: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise RegistrationError("GPU IDs must be nonnegative integers")
        ids.append(item)
    if not allow_empty and not ids:
        raise RegistrationError("actual_gpu_ids must identify at least one GPU")
    if len(ids) != len(set(ids)):
        raise RegistrationError("GPU IDs must not contain duplicates")
    return sorted(ids)


def _validate_json_value(value: Any, label: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RegistrationError(f"{label} must contain finite JSON values only") from exc


def _validate_budget(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise RegistrationError("budget must be a nonempty object")

    def visit(item: Any) -> None:
        if isinstance(item, bool) or item is None or isinstance(item, str):
            return
        if isinstance(item, (int, float)):
            if item < 0:
                raise RegistrationError("budget values cannot be negative")
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise RegistrationError("budget object keys must be strings")
                visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        raise RegistrationError("budget must contain JSON-compatible values")

    result = dict(value)
    visit(result)
    _validate_json_value(result, "budget")
    return result


def _reject_credentials(value: Any, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise RegistrationError(f"{path} contains a non-string field name")
            if SENSITIVE_KEY_RE.search(key):
                raise RegistrationError(f"credential-like field is forbidden: {path}.{key}")
            if key == "command":
                raise RegistrationError("only command_redacted may be registered")
            _reject_credentials(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_credentials(child, f"{path}[{index}]")
    elif isinstance(value, str):
        for match in CREDENTIAL_VALUE_RE.finditer(value):
            if match.group("value").strip("\"'").casefold() not in REDACTION_MARKERS:
                raise RegistrationError(f"possible credential value in {path}; redact it before registration")


def _lock_file(handle: Any, exclusive: bool) -> None:
    _require_linux()
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)


def _unlock_file(fd: int) -> None:
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def _state_transition_lock(path: Path):
    """Serialize each registration state transition end to end."""
    _require_linux()
    lock_path = Path(str(path) + ".lock")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise RegistrationError("registration transition lock must be a regular file")
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _append_line(path: Path, record: Mapping[str, Any]) -> None:
    """Private path-injectable primitive; public APIs always use REGISTRATION_PATH."""
    _require_linux()
    _reject_credentials(record)
    try:
        line = (json.dumps(dict(record), ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RegistrationError("registration event is not valid JSON") from exc
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise RegistrationError(f"cannot open registration ledger for append: {exc.strerror}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise RegistrationError("registration ledger must be a regular file")
        with os.fdopen(fd, "ab", closefd=False) as handle:
            _lock_file(handle, exclusive=True)
            try:
                offset = 0
                while offset < len(line):
                    offset += os.write(fd, line[offset:])
                os.fsync(fd)
            finally:
                _unlock_file(fd)
    finally:
        os.close(fd)


def _read_events(path: Path) -> list[dict[str, Any]]:
    _require_linux()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RegistrationError(f"cannot read registration ledger: {exc.strerror}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise RegistrationError("registration ledger must be a regular file")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            _lock_file(handle, exclusive=False)
            try:
                raw = handle.read()
            finally:
                _unlock_file(fd)
    finally:
        os.close(fd)
    events: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        try:
            event = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError, RegistrationError) as exc:
            raise RegistrationError(f"registration ledger line {number} is invalid") from exc
        if not isinstance(event, dict):
            raise RegistrationError(f"registration ledger line {number} is not an object")
        events.append(event)
    return events


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RegistrationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _events_for_job(path: Path, job_id: str) -> list[dict[str, Any]]:
    events = [event for event in _read_events(path) if event.get("job_id") == job_id]
    if not events:
        raise RegistrationError("job_id is not present in the registration ledger")
    if events[0].get("event") != "REGISTERED" or events[0].get("status") != "REGISTERED":
        raise RegistrationError("job does not begin with a valid REGISTERED event")
    for event in events:
        if event.get("binding") != events[0].get("binding"):
            raise RegistrationError("event binding differs from REGISTERED identity")
    return events


def _verify_registered(path: Path, job_id: str, real_name: str, name_id: str) -> dict[str, Any]:
    events = _events_for_job(path, job_id)
    event = events[0]
    if (event.get("event") != "REGISTERED" or event.get("status") != "REGISTERED"
            or event.get("real_name") != real_name or event.get("name_id") != name_id):
        raise RegistrationError("REGISTERED readback did not match the requested identity")
    if event.get("schema") != SCHEMA:
        raise RegistrationError("REGISTERED schema is invalid")
    return event


def verify_registered(job_id: str, *, real_name: str, name_id: str) -> dict[str, Any]:
    """Read and validate a REGISTERED event from the fixed server ledger."""
    return _verify_registered(REGISTRATION_PATH, job_id, real_name, name_id)


def _make_binding(attempt: str, manifest_sha256: str, hashes_sha256: str,
                  resource_request_sha256: str | None, budget: Mapping[str, Any]) -> dict[str, Any]:
    request_digest = None
    if resource_request_sha256 is not None:
        request_digest = _validate_digest(resource_request_sha256, "resource_request_sha256")
    canonical_budget = json.dumps(dict(budget), ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":"), allow_nan=False)
    budget_sha256 = hashlib.sha256(canonical_budget.encode("utf-8")).hexdigest()
    return {
        "attempt": _validate_attempt(attempt),
        "manifest_sha256": _validate_digest(manifest_sha256, "manifest_sha256"),
        "hashes_sha256": _validate_digest(hashes_sha256, "hashes_sha256"),
        "resource_request_sha256": request_digest,
        "budget": dict(budget),
        "budget_sha256": budget_sha256,
    }


def _register_before_launch(path: Path, *, real_name: str, name_id: str,
                            details: Mapping[str, Any], attempt: str,
                            manifest_sha256: str, hashes_sha256: str,
                            budget: Mapping[str, Any],
                            resource_request_sha256: str | None = None) -> dict[str, Any]:
    if not isinstance(details, Mapping):
        raise RegistrationError("details must be an object")
    missing = sorted(DETAIL_FIELDS - details.keys())
    unexpected = sorted(details.keys() & RESERVED_FIELDS)
    if missing:
        raise RegistrationError("task details missing fields: " + ",".join(missing))
    if unexpected:
        raise RegistrationError("task details cannot override: " + ",".join(unexpected))
    linux_user, linux_uid = _current_identity()
    _validate_real_name(real_name, name_id, linux_user)
    if not isinstance(details.get("task_name"), str) or not details["task_name"].strip():
        raise RegistrationError("task_name is required")
    if not isinstance(details.get("command_redacted"), str) or not details["command_redacted"].strip():
        raise RegistrationError("command_redacted is required")
    gpu_request = details.get("gpu_request")
    if not isinstance(gpu_request, Mapping):
        raise RegistrationError("gpu_request must be an object")
    requested_ids = _validate_gpu_ids(gpu_request.get("ids"), allow_empty=True)
    requested_count = gpu_request.get("count")
    if (isinstance(requested_count, bool) or not isinstance(requested_count, int)
            or requested_count != len(requested_ids)):
        raise RegistrationError("gpu_request.count must match gpu_request.ids")
    canonical_budget = _validate_budget(budget)
    binding = _make_binding(
        attempt, manifest_sha256, hashes_sha256, resource_request_sha256, canonical_budget,
    )
    now = datetime.now().astimezone()
    job_id = f"{now:%Y%m%d-%H%M%S}-{name_id}-{uuid.uuid4().hex[:8]}"
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "job_id": job_id,
        "event": "REGISTERED",
        "registered_at": now.isoformat(timespec="seconds"),
        "real_name": real_name,
        "name_id": name_id,
        "linux_user": linux_user,
        "linux_uid": linux_uid,
        "host": socket.gethostname(),
        **dict(details),
        "binding": binding,
        "status": "REGISTERED",
    }
    _reject_credentials(record)
    _validate_json_value(record, "REGISTERED event")
    with _state_transition_lock(path):
        existing = _read_events(path) if path.exists() else []
        if any(event.get("job_id") == job_id for event in existing):
            raise RegistrationError("generated job_id already exists")
        _append_line(path, record)
        return _verify_registered(path, job_id, real_name, name_id)


def register_before_launch(*, real_name: str, name_id: str,
                           details: Mapping[str, Any], attempt: str,
                           manifest_sha256: str, hashes_sha256: str,
                           budget: Mapping[str, Any],
                           resource_request_sha256: str | None = None) -> dict[str, Any]:
    """Append and read back REGISTERED before the native launcher starts work.

    No path override is exposed here. Tests may call the private helper with a
    temporary path; production calls always use REGISTRATION_PATH.
    """
    return _register_before_launch(
        REGISTRATION_PATH, real_name=real_name, name_id=name_id, details=details,
        attempt=attempt, manifest_sha256=manifest_sha256, hashes_sha256=hashes_sha256,
        budget=budget, resource_request_sha256=resource_request_sha256,
    )


def _read_proc_identity(proc_root: Path, pid: int) -> dict[str, int | str]:
    process_dir = proc_root / str(pid)
    try:
        stat_before = (process_dir / "stat").read_text(encoding="ascii")
        status_text = (process_dir / "status").read_text(encoding="ascii")
        stat_after = (process_dir / "stat").read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise RegistrationError(f"/proc/{pid} is not readable as a live process") from exc

    def parse_stat(raw: str) -> tuple[str, int, int, int]:
        right = raw.rfind(")")
        if right < 0:
            raise RegistrationError(f"/proc/{pid}/stat is malformed")
        fields = raw[right + 1:].split()
        if len(fields) <= 19:
            raise RegistrationError(f"/proc/{pid}/stat is truncated")
        try:
            return fields[0], int(fields[1]), int(fields[2]), int(fields[19])
        except ValueError as exc:
            raise RegistrationError(f"/proc/{pid}/stat has invalid numeric fields") from exc

    before = parse_stat(stat_before)
    after = parse_stat(stat_after)
    if before != after or before[0] in {"Z", "X", "x"}:
        raise RegistrationError(f"/proc/{pid} exited or changed during verification")
    uid_line = next((line for line in status_text.splitlines() if line.startswith("Uid:")), None)
    if uid_line is None:
        raise RegistrationError(f"/proc/{pid}/status has no UID")
    try:
        owner_uid = int(uid_line.split()[1])
    except (IndexError, ValueError) as exc:
        raise RegistrationError(f"/proc/{pid}/status has an invalid UID") from exc
    return {"state": before[0], "ppid": before[1], "pgid": before[2],
            "starttime": before[3], "uid": owner_uid}


def _verify_process_binding(event: Mapping[str, Any], *,
                            proc_root: Path = Path("/proc")) -> dict[str, dict[str, int]]:
    pid = _validate_positive_pid(event.get("pid"), "pid")
    pgid = _validate_positive_pid(event.get("pgid"), "pgid")
    workers = _validate_pid_array(event.get("worker_pids"), "worker_pids")
    if pid in workers:
        raise RegistrationError("worker_pids must not repeat the root pid")
    expected_uid = event.get("linux_uid")
    if isinstance(expected_uid, bool) or not isinstance(expected_uid, int) or expected_uid < 0:
        raise RegistrationError("REGISTERED linux_uid is invalid")
    root = _read_proc_identity(proc_root, pid)
    if root["uid"] != expected_uid or root["pgid"] != pgid:
        raise RegistrationError("root process owner or process group does not match registration")
    identities: dict[int, dict[str, int | str]] = {pid: root}
    for worker_pid in workers:
        worker = _read_proc_identity(proc_root, worker_pid)
        if worker["uid"] != expected_uid:
            raise RegistrationError(f"worker {worker_pid} owner does not match registration")
        identities[worker_pid] = worker
        ancestor_pid = int(worker["ppid"])
        seen: set[int] = set()
        while ancestor_pid != pid:
            if ancestor_pid <= 1 or ancestor_pid in seen:
                raise RegistrationError(f"worker {worker_pid} is not a child of the registered root")
            seen.add(ancestor_pid)
            ancestor = identities.get(ancestor_pid)
            if ancestor is None:
                ancestor = _read_proc_identity(proc_root, ancestor_pid)
                identities[ancestor_pid] = ancestor
            if ancestor["uid"] != expected_uid:
                raise RegistrationError(f"worker {worker_pid} ancestor owner differs")
            ancestor_pid = int(ancestor["ppid"])
    worker_pgids = {str(worker_pid): int(identities[worker_pid]["pgid"])
                    for worker_pid in workers}
    process_starttimes = {str(process_pid): int(identities[process_pid]["starttime"])
                          for process_pid in (pid, *workers)}
    recorded_worker_pgids = event.get("worker_pgids")
    if recorded_worker_pgids is not None and recorded_worker_pgids != worker_pgids:
        raise RegistrationError("worker_pgids do not match the live /proc process groups")
    recorded_starttimes = event.get("process_starttimes")
    if recorded_starttimes is not None and recorded_starttimes != process_starttimes:
        raise RegistrationError("process_starttimes do not match live /proc processes")
    # Re-read each registered PID to detect exit or PID reuse during traversal.
    for checked_pid in (pid, *workers):
        if _read_proc_identity(proc_root, checked_pid) != identities[checked_pid]:
            raise RegistrationError(f"process {checked_pid} changed during binding verification")
    return {"worker_pgids": worker_pgids, "process_starttimes": process_starttimes}


def _query_and_verify_gpu_ids(actual_gpu_ids: Any,
                              readonly_gpu_query: Callable[[tuple[int, ...]], Iterable[int]],
                              process_pids: tuple[int, ...]) -> list[int]:
    expected = _validate_gpu_ids(actual_gpu_ids, allow_empty=True)
    if not callable(readonly_gpu_query):
        raise RegistrationError("a read-only GPU query callable is required")
    try:
        observed = _validate_gpu_ids(
            list(readonly_gpu_query(process_pids)), allow_empty=True,
        )
    except RegistrationError:
        raise
    except Exception as exc:
        raise RegistrationError(f"read-only GPU query failed: {type(exc).__name__}") from exc
    if observed != expected:
        raise RegistrationError("actual_gpu_ids do not match the read-only GPU query")
    return expected


def wait_for_gpu_binding(process_pids: Iterable[int], expected_gpu_ids: Iterable[int],
                         timeout_seconds: float, *,
                         readonly_gpu_query: Callable[[tuple[int, ...]], Iterable[int]],
                         poll_seconds: float = 0.1) -> list[int]:
    """Wait until a process set owns exactly the requested GPUs."""
    expected = sorted(set(expected_gpu_ids))
    if not expected:
        return []
    pids = tuple(process_pids)
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        observed = sorted(set(readonly_gpu_query(pids)))
        if observed == expected:
            return observed
        if time.monotonic() >= deadline:
            raise RegistrationError("GPU_REGISTRATION_BINDING_TIMEOUT")
        time.sleep(max(0.0, poll_seconds))


def _verify_running(path: Path, job_id: str, *,
                    readonly_gpu_query: Callable[[tuple[int, ...]], Iterable[int]],
                    proc_root: Path = Path("/proc")) -> dict[str, Any]:
    events = _events_for_job(path, job_id)
    event = events[-1]
    if event.get("event") != "RUNNING" or event.get("status") != "RUNNING":
        raise RegistrationError("latest job event is not RUNNING")
    if not isinstance(event.get("worker_pgids"), dict) or not isinstance(
            event.get("process_starttimes"), dict):
        raise RegistrationError("RUNNING process group and start-time snapshots are required")
    registered = events[0]
    _verify_process_binding({**registered, **event}, proc_root=proc_root)
    process_pids = (event["pid"], *event["worker_pids"])
    observed = _query_and_verify_gpu_ids(
        event.get("actual_gpu_ids"), readonly_gpu_query, process_pids,
    )
    requested = _validate_gpu_ids(registered["gpu_request"].get("ids"), allow_empty=True)
    if not set(observed).issubset(requested):
        raise RegistrationError("actual GPU IDs are outside the registered GPU request")
    return event


def verify_running(job_id: str, *,
                   readonly_gpu_query: Callable[[tuple[int, ...]], Iterable[int]]) -> dict[str, Any]:
    """Validate /proc and query GPU use attributable to the supplied process IDs.

    The injected query must be read-only and return only GPUs reported for the
    supplied root/worker PIDs by the server's GPU process accounting source.
    """
    return _verify_running(REGISTRATION_PATH, job_id,
                           readonly_gpu_query=readonly_gpu_query)


def _append_running_unlocked(path: Path, job_id: str, *, pid: int, pgid: int,
                    worker_pids: list[int], actual_gpu_ids: list[int],
                    readonly_gpu_query: Callable[[tuple[int, ...]], Iterable[int]],
                    proc_root: Path = Path("/proc"),
                    log_path: str | None = None,
                    timestamp_field: str = "started_at") -> dict[str, Any]:
    events = _events_for_job(path, job_id)
    previous = events[-1]
    if previous.get("event") not in {"REGISTERED", "RUNNING"}:
        raise RegistrationError("cannot append RUNNING after a terminal event")
    registered = events[0]
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "job_id": job_id,
        "event": "RUNNING",
        timestamp_field: datetime.now().astimezone().isoformat(timespec="seconds"),
        "pid": _validate_positive_pid(pid, "pid"),
        "pgid": _validate_positive_pid(pgid, "pgid"),
        "worker_pids": _validate_pid_array(worker_pids, "worker_pids"),
        "actual_gpu_ids": _validate_gpu_ids(actual_gpu_ids, allow_empty=True),
        "linux_uid": registered["linux_uid"],
        "binding": registered["binding"],
        "status": "RUNNING",
    }
    process_snapshot = _verify_process_binding(
        {**registered, **record}, proc_root=proc_root,
    )
    record.update(process_snapshot)
    process_pids = (record["pid"], *record["worker_pids"])
    record["actual_gpu_ids"] = _query_and_verify_gpu_ids(
        record["actual_gpu_ids"], readonly_gpu_query, process_pids,
    )
    requested_gpu_ids = _validate_gpu_ids(
        registered["gpu_request"].get("ids"), allow_empty=True,
    )
    if requested_gpu_ids and record["worker_pids"] and not record["actual_gpu_ids"]:
        raise RegistrationError("GPU request requires a nonempty actual GPU binding")
    if not set(record["actual_gpu_ids"]).issubset(requested_gpu_ids):
        raise RegistrationError("actual GPU IDs are outside the registered GPU request")
    if log_path is None:
        log_path = registered.get("log_path")
    if log_path is not None:
        record["log_path"] = log_path
    _reject_credentials(record)
    _append_line(path, record)
    return _verify_running(path, job_id, readonly_gpu_query=readonly_gpu_query,
                           proc_root=proc_root)


def _append_running(path: Path, job_id: str, *, pid: int, pgid: int,
                    worker_pids: list[int], actual_gpu_ids: list[int],
                    readonly_gpu_query: Callable[[tuple[int, ...]], Iterable[int]],
                    proc_root: Path = Path("/proc"),
                    log_path: str | None = None,
                    timestamp_field: str = "started_at") -> dict[str, Any]:
    with _state_transition_lock(path):
        return _append_running_unlocked(
            path, job_id, pid=pid, pgid=pgid, worker_pids=worker_pids,
            actual_gpu_ids=actual_gpu_ids, readonly_gpu_query=readonly_gpu_query,
            proc_root=proc_root, log_path=log_path, timestamp_field=timestamp_field,
        )


def append_running(job_id: str, *, pid: int, pgid: int,
                   worker_pids: list[int], actual_gpu_ids: list[int],
                   readonly_gpu_query: Callable[[tuple[int, ...]], Iterable[int]],
                   log_path: str | None = None,
                   restarted: bool = False) -> dict[str, Any]:
    """Append a verified RUNNING event to the fixed server ledger."""
    if not isinstance(restarted, bool):
        raise RegistrationError("restarted must be a boolean")
    return _append_running(
        REGISTRATION_PATH, job_id, pid=pid, pgid=pgid, worker_pids=worker_pids,
        actual_gpu_ids=actual_gpu_ids, readonly_gpu_query=readonly_gpu_query,
        log_path=log_path, timestamp_field="updated_at" if restarted else "started_at",
    )


def _append_terminal_unlocked(path: Path, job_id: str, event: str, *, exit_code: int,
                     reason_redacted: str | None = None,
                     result_path: str | None = None) -> dict[str, Any]:
    if event not in TERMINAL_EVENTS:
        raise RegistrationError("terminal event must be SUCCEEDED, FAILED, CANCELLED, or EXPIRED")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise RegistrationError("exit_code must be an integer")
    if event == "SUCCEEDED" and exit_code != 0:
        raise RegistrationError("SUCCEEDED requires exit_code 0")
    events = _events_for_job(path, job_id)
    latest_event = events[-1].get("event")
    if event == "SUCCEEDED" and latest_event != "RUNNING":
        raise RegistrationError("SUCCEEDED requires a preceding RUNNING event")
    if event != "SUCCEEDED" and latest_event not in {"REGISTERED", "RUNNING"}:
        raise RegistrationError("terminal event cannot follow another terminal event")
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "job_id": job_id,
        "event": event,
        "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "exit_code": exit_code,
        "binding": events[0]["binding"],
        "status": event,
    }
    if reason_redacted is not None:
        record["reason_redacted"] = reason_redacted
    if result_path is not None:
        record["result_path"] = result_path
    _reject_credentials(record)
    _validate_json_value(record, "terminal event")
    _append_line(path, record)
    latest = _events_for_job(path, job_id)[-1]
    if latest != record:
        raise RegistrationError("terminal event readback mismatch")
    return latest


def _append_terminal(path: Path, job_id: str, event: str, *, exit_code: int,
                     reason_redacted: str | None = None,
                     result_path: str | None = None) -> dict[str, Any]:
    with _state_transition_lock(path):
        return _append_terminal_unlocked(
            path, job_id, event, exit_code=exit_code,
            reason_redacted=reason_redacted, result_path=result_path,
        )


def append_terminal(job_id: str, event: str, *, exit_code: int,
                    reason_redacted: str | None = None,
                    result_path: str | None = None) -> dict[str, Any]:
    """Append a terminal event. Existing history is never replaced."""
    return _append_terminal(REGISTRATION_PATH, job_id, event, exit_code=exit_code,
                            reason_redacted=reason_redacted, result_path=result_path)
