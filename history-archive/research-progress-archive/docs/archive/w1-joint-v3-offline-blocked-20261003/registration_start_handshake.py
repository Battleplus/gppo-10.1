"""Two-step REGISTERED receipt and explicit START_ACK handshake."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import time
from collections.abc import Callable, Mapping
from typing import Any, TextIO


RECEIPT_SCHEMA = "w1-registration-start-receipt/1.0.0"
ACK_SCHEMA = "w1-registration-start-ack/1.0.0"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_LINE_BYTES = 16 * 1024
_RECEIPT_FIELDS = frozenset({
    "schema", "event", "job_id", "real_name", "name_id", "registration_path",
    "manifest_sha256", "hashes_sha256", "budget_sha256",
})
_ACK_FIELDS = frozenset({
    "schema", "event", "job_id", "manifest_sha256", "hashes_sha256",
})


class RegistrationStartHandshakeError(RuntimeError):
    """Raised when the registration receipt or explicit start acknowledgement fails."""

    def __init__(self, message: str, *, registered_job: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.registered_job = registered_job


def budget_binding_sha256(budget: Mapping[str, Any]) -> str:
    """Return the same canonical budget digest stored by server_registration."""
    if not isinstance(budget, Mapping) or not budget:
        raise RegistrationStartHandshakeError("BUDGET_BINDING_INVALID")
    try:
        encoded = json.dumps(dict(budget), ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RegistrationStartHandshakeError("BUDGET_BINDING_INVALID") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise RegistrationStartHandshakeError("HANDSHAKE_DIGEST_INVALID:" + field)
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RegistrationStartHandshakeError("HANDSHAKE_JSON_DUPLICATE_FIELD")
        result[key] = value
    return result


def _decode_object(line: str | bytes, *, label: str) -> dict[str, Any]:
    if isinstance(line, bytes):
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RegistrationStartHandshakeError(label + "_INVALID_ENCODING") from exc
    if not isinstance(line, str) or not line.endswith("\n"):
        raise RegistrationStartHandshakeError(label + "_INCOMPLETE_LINE")
    try:
        value = json.loads(line, object_pairs_hook=_unique_object)
    except RegistrationStartHandshakeError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RegistrationStartHandshakeError(label + "_INVALID_JSON") from exc
    if not isinstance(value, dict):
        raise RegistrationStartHandshakeError(label + "_OBJECT_REQUIRED")
    return value


def _write_line(stream: Any, value: Mapping[str, Any]) -> None:
    line = json.dumps(dict(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False) + "\n"
    try:
        stream.write(line)
    except TypeError:
        stream.write(line.encode("utf-8"))
    stream.flush()


def _validate_registered_job(job: Any, *, real_name: str, name_id: str,
                             manifest_sha256: str, hashes_sha256: str,
                             budget: Mapping[str, Any]) -> tuple[str, str]:
    if not isinstance(job, Mapping):
        raise RegistrationStartHandshakeError("REGISTERED_RECORD_INVALID")
    job_id = job.get("job_id")
    binding = job.get("binding")
    budget_sha256 = budget_binding_sha256(budget)
    if (not isinstance(job_id, str) or not job_id.strip()
            or job.get("event") != "REGISTERED" or job.get("status") != "REGISTERED"
            or job.get("real_name") != real_name or job.get("name_id") != name_id
            or not isinstance(binding, Mapping)
            or binding.get("manifest_sha256") != _require_digest(manifest_sha256, "manifest_sha256")
            or binding.get("hashes_sha256") != _require_digest(hashes_sha256, "hashes_sha256")
            or binding.get("budget_sha256") != budget_sha256
            or binding.get("budget") != dict(budget)):
        raise RegistrationStartHandshakeError("REGISTERED_RECORD_BINDING_MISMATCH")
    return job_id, budget_sha256


def _default_timed_line_reader(stream: Any, timeout_seconds: float) -> bytes:
    """Read one bounded line from a selectable native stdin without blocking past timeout."""
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError) as exc:
        raise RegistrationStartHandshakeError("START_ACK_INPUT_NOT_SELECTABLE") from exc
    deadline = time.monotonic() + timeout_seconds
    line = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RegistrationStartHandshakeError("START_ACK_TIMEOUT")
        try:
            readable, _, _ = select.select([fd], [], [], remaining)
        except (OSError, ValueError) as exc:
            raise RegistrationStartHandshakeError("START_ACK_INPUT_UNAVAILABLE") from exc
        if not readable:
            raise RegistrationStartHandshakeError("START_ACK_TIMEOUT")
        try:
            chunk = os.read(fd, 1)
        except OSError as exc:
            raise RegistrationStartHandshakeError("START_ACK_INPUT_UNAVAILABLE") from exc
        if not chunk:
            raise RegistrationStartHandshakeError("START_ACK_EOF")
        line.extend(chunk)
        if len(line) > _MAX_LINE_BYTES:
            raise RegistrationStartHandshakeError("START_ACK_LINE_TOO_LARGE")
        if chunk == b"\n":
            return bytes(line)


def register_then_wait_for_start_ack(
    *,
    register: Callable[[], Mapping[str, Any]],
    real_name: str,
    name_id: str,
    registration_path: str | Path,
    manifest_sha256: str,
    hashes_sha256: str,
    budget: Mapping[str, Any],
    stdin: Any,
    stdout: Any,
    timeout_seconds: float = 30.0,
    read_line_with_timeout: Callable[[Any, float], str | bytes] | None = None,
    on_registered: Callable[[Mapping[str, Any]], Any] | None = None,
) -> Mapping[str, Any]:
    """Register and return only after a job/manifest/hash-bound START_ACK arrives.

    The caller may invoke RUNNING registration, CUDA initialization, or any worker
    startup only after this function returns. `read_line_with_timeout` is a narrow
    stream adapter for tests; production uses selectable native stdin.
    """
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0):
        raise RegistrationStartHandshakeError("START_ACK_TIMEOUT_INVALID")
    manifest_sha256 = _require_digest(manifest_sha256, "manifest_sha256")
    hashes_sha256 = _require_digest(hashes_sha256, "hashes_sha256")
    if not isinstance(real_name, str) or not real_name or not isinstance(name_id, str) or not name_id:
        raise RegistrationStartHandshakeError("REGISTERED_IDENTITY_INVALID")
    path = str(registration_path)
    if not path:
        raise RegistrationStartHandshakeError("REGISTRATION_PATH_INVALID")

    job = register()
    try:
        job_id, budget_sha256 = _validate_registered_job(
            job, real_name=real_name, name_id=name_id,
            manifest_sha256=manifest_sha256, hashes_sha256=hashes_sha256,
            budget=budget,
        )
        if on_registered is not None:
            on_registered(job)
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "event": "REGISTERED",
            "job_id": job_id,
            "real_name": real_name,
            "name_id": name_id,
            "registration_path": path,
            "manifest_sha256": manifest_sha256,
            "hashes_sha256": hashes_sha256,
            "budget_sha256": budget_sha256,
        }
        _write_line(stdout, receipt)

        deadline = time.monotonic() + timeout_seconds
        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0:
            raise RegistrationStartHandshakeError("START_ACK_TIMEOUT")
        reader = read_line_with_timeout or _default_timed_line_reader
        try:
            raw_ack = reader(stdin, remaining)
        except RegistrationStartHandshakeError:
            raise
        except TimeoutError as exc:
            raise RegistrationStartHandshakeError("START_ACK_TIMEOUT") from exc
        except (EOFError, BrokenPipeError) as exc:
            raise RegistrationStartHandshakeError("START_ACK_EOF") from exc
        if time.monotonic() > deadline:
            raise RegistrationStartHandshakeError("START_ACK_TIMEOUT")
        ack = _decode_object(raw_ack, label="START_ACK")
        if set(ack) != _ACK_FIELDS:
            raise RegistrationStartHandshakeError("START_ACK_FIELDS_INVALID")
        if (ack.get("schema") != ACK_SCHEMA or ack.get("event") != "START_ACK"
                or ack.get("job_id") != job_id
                or ack.get("manifest_sha256") != manifest_sha256
                or ack.get("hashes_sha256") != hashes_sha256):
            raise RegistrationStartHandshakeError("START_ACK_BINDING_MISMATCH")
        return job
    except RegistrationStartHandshakeError as exc:
        exc.registered_job = job if isinstance(job, Mapping) else None
        raise
    except BaseException as exc:
        raise RegistrationStartHandshakeError(
            "START_ACK_HANDSHAKE_FAILED",
            registered_job=job if isinstance(job, Mapping) else None,
        ) from exc


def _receive_receipt(channel: Any, deadline: float) -> dict[str, Any]:
    line = bytearray()
    while True:
        if channel.recv_ready():
            chunk = channel.recv(4096)
            if not chunk:
                raise RegistrationStartHandshakeError("REGISTERED_RECEIPT_EOF")
            newline = chunk.find(b"\n")
            if newline >= 0:
                line.extend(chunk[:newline + 1])
                if newline + 1 != len(chunk):
                    raise RegistrationStartHandshakeError("REGISTERED_RECEIPT_TRAILING_DATA")
                return _decode_object(bytes(line), label="REGISTERED_RECEIPT")
            line.extend(chunk)
            if len(line) > _MAX_LINE_BYTES:
                raise RegistrationStartHandshakeError("REGISTERED_RECEIPT_LINE_TOO_LARGE")
        else:
            if getattr(channel, "closed", False) or channel.exit_status_ready():
                raise RegistrationStartHandshakeError("REGISTERED_RECEIPT_EOF")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RegistrationStartHandshakeError("REGISTERED_RECEIPT_TIMEOUT")
            time.sleep(min(0.01, remaining))


def controller_receive_report_and_ack(
    *,
    channel: Any,
    remote_stdin: Any,
    local_stdout: TextIO,
    real_name: str,
    name_id: str,
    registration_path: str | Path,
    manifest_sha256: str,
    hashes_sha256: str,
    budget: Mapping[str, Any],
    acknowledger: Callable[[Mapping[str, Any]], bool],
    reporter: Callable[[Mapping[str, Any]], Any] | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Stream one SSH-channel receipt, report it locally, then send explicit ACK.

    The acknowledger must obtain an explicit start decision and return literal
    `True`; cancellation, false values, and callback failures never send ACK.
    """
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0):
        raise RegistrationStartHandshakeError("START_ACK_TIMEOUT_INVALID")
    expected = {
        "schema": RECEIPT_SCHEMA,
        "event": "REGISTERED",
        "real_name": real_name,
        "name_id": name_id,
        "registration_path": str(registration_path),
        "manifest_sha256": _require_digest(manifest_sha256, "manifest_sha256"),
        "hashes_sha256": _require_digest(hashes_sha256, "hashes_sha256"),
        "budget_sha256": budget_binding_sha256(budget),
    }
    deadline = time.monotonic() + timeout_seconds
    receipt = _receive_receipt(channel, deadline)
    if set(receipt) != _RECEIPT_FIELDS:
        raise RegistrationStartHandshakeError("REGISTERED_RECEIPT_FIELDS_INVALID")
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RegistrationStartHandshakeError("REGISTERED_RECEIPT_BINDING_MISMATCH")
    job_id = receipt.get("job_id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise RegistrationStartHandshakeError("REGISTERED_RECEIPT_JOB_ID_INVALID")

    line = json.dumps(receipt, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)
    local_stdout.write("REGISTERED " + line + "\n")
    local_stdout.flush()
    if reporter is not None:
        reporter(dict(receipt))
    if time.monotonic() >= deadline:
        raise RegistrationStartHandshakeError("START_ACK_TIMEOUT")
    if acknowledger(receipt) is not True:
        raise RegistrationStartHandshakeError("START_NOT_ACKNOWLEDGED")
    if time.monotonic() >= deadline:
        raise RegistrationStartHandshakeError("START_ACK_TIMEOUT")

    ack = {
        "schema": ACK_SCHEMA,
        "event": "START_ACK",
        "job_id": job_id,
        "manifest_sha256": expected["manifest_sha256"],
        "hashes_sha256": expected["hashes_sha256"],
    }
    try:
        _write_line(remote_stdin, ack)
    except (OSError, EOFError, BrokenPipeError) as exc:
        raise RegistrationStartHandshakeError("START_ACK_WRITE_FAILED") from exc
    return receipt
