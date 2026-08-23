"""Share canonical serialization and immutable-artifact file rules."""

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeGuard


def canonical_json_bytes(value: Any) -> bytes:
    """Return one stable byte encoding for content addressing."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one private JSON artifact atomically by exclusive creation."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.chmod(path, 0o600)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write one private JSON Lines artifact by exclusive creation."""
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            json.dump(row, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")
    os.chmod(path, 0o600)


def file_hashes(root: Path) -> Mapping[str, str]:
    """Return one digest for every file below the root."""
    return {
        path.relative_to(root).as_posix(): sha256_hex(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def sha256_hex(payload: bytes) -> str:
    """Return one hexadecimal SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def is_digest(value: object) -> TypeGuard[str]:
    """Return true for one lowercase hexadecimal SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Return one linearly interpolated quantile of sorted values."""
    if not sorted_values:
        raise ValueError("A quantile requires at least one value")
    position = q * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = position - low
    return sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction


def utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(UTC)
