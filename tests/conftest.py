"""Provide synthetic test data."""

from pathlib import Path

import pytest


@pytest.fixture
def signal_events() -> Path:
    """Return the synthetic Signal event file."""
    return Path(__file__).parent / "fixtures" / "signal" / "events.jsonl"


@pytest.fixture
def signal_delete() -> Path:
    """Return the synthetic Signal delete file."""
    return Path(__file__).parent / "fixtures" / "signal" / "delete.jsonl"


@pytest.fixture
def signal_mentions() -> Path:
    """Return Signal events with target mentions and replies."""
    return Path(__file__).parent / "fixtures" / "signal" / "mentions.jsonl"


@pytest.fixture
def configured_signal_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set explicit synthetic Signal values for a CLI test."""
    monkeypatch.setenv("IDIOLECT_SIGNAL_ACCOUNT", "+10000000000")
    monkeypatch.setenv("IDIOLECT_SIGNAL_BIN", "signal-cli-test")
    monkeypatch.setenv("IDIOLECT_SIGNAL_DATA_DIR", "safe-signal-data")
    monkeypatch.setenv("IDIOLECT_SIGNAL_CHATS", '["group-allowed"]')


@pytest.fixture
def local_config(tmp_path: Path) -> Path:
    """Create a safe configuration file for one test."""
    path = tmp_path / "test.toml"
    path.write_text(
        f"""
[signal]
[store]
root = "{tmp_path.as_posix()}"
database = "test.duckdb"

[data]
output = "{(tmp_path / "data").as_posix()}"
context = 4
valid_ratio = 0.0
test_ratio = 0.0
""",
        encoding="utf-8",
    )
    return path
