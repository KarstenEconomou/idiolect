"""Share command-line path and process helpers."""

import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from idiolect.artifact import is_digest
from idiolect.config import ConfigError


def artifact_path(value: str | Path, root: Path | None, child: str | None = None) -> Path:
    """Resolve one content ID below its configured artifact root."""
    path = Path(value)
    if len(path.parts) != 1 or not is_digest(path.name) or root is None:
        return path
    return root / child / path if child is not None else root / path


@contextmanager
def keep_awake() -> Iterator[None]:
    """Prevent idle sleep during one long local operation on macOS."""
    process: subprocess.Popen[bytes] | None = None
    if sys.platform == "darwin":
        try:
            process = subprocess.Popen(
                ("caffeinate", "-i", "-w", str(os.getpid())),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as error:
            raise ConfigError(f"Cannot start macOS sleep assertion: {error}") from error
    try:
        yield
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
