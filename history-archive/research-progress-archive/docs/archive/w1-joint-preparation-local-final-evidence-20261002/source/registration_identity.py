"""Deterministic server-registration identity derived from a supplied real name."""
from __future__ import annotations

import hashlib
import re
import unicodedata


_LOWERCASE_ENGLISH = re.compile(r"[a-z]+\Z")
_SHARED_NAMES = frozenset({"admin", "admin123", "root", "user", "user1", "ubuntu", "test", "example"})


def is_supported_real_name(value: object) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if _LOWERCASE_ENGLISH.fullmatch(value):
        return True
    return all(
        "CJK UNIFIED IDEOGRAPH" in unicodedata.name(char, "")
        or char in {"〇", "·"}
        for char in value
    ) and any("CJK UNIFIED IDEOGRAPH" in unicodedata.name(char, "") for char in value)


def derive_name_id(real_name: str) -> str:
    """Return stable lowercase ASCII without requiring a runtime transliterator.

    English input is already an acceptable identity. Chinese input receives a
    deterministic, non-secret ASCII identity derived from its UTF-8 digest;
    the real name remains the human-readable registration field.
    """
    if not is_supported_real_name(real_name):
        raise ValueError("REAL_NAME_FORMAT_INVALID")
    if real_name.casefold() in _SHARED_NAMES:
        raise ValueError("SHARED_ACCOUNT_NAME_NOT_ALLOWED")
    if _LOWERCASE_ENGLISH.fullmatch(real_name):
        return real_name
    digest = hashlib.sha256(real_name.encode("utf-8")).digest()
    letters = "".join(chr(ord("a") + (byte % 26)) for byte in digest)
    return "u" + letters
