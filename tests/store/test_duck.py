"""Test the DuckDB message store."""

from datetime import UTC, datetime
from pathlib import Path

from idiolect.ingest import harvest
from idiolect.ingest.signal import SignalFileSource, SignalParser
from idiolect.store.duck import DuckRepository
from idiolect.types import ChatId

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_harvest_is_atomic_and_idempotent(signal_events: Path, tmp_path: Path) -> None:
    """Check filtering, updates, reactions, and duplicate events."""
    repository = DuckRepository(tmp_path / "data" / "test.duckdb")
    parser = SignalParser((ChatId("group-allowed"),))

    first = harvest(
        SignalFileSource(signal_events, clock=lambda: _NOW),
        parser,
        repository,
    )
    second = harvest(
        SignalFileSource(signal_events, clock=lambda: _NOW),
        parser,
        repository,
    )

    assert first.received == 6
    assert first.stored == 4
    assert first.messages == 3
    assert first.reactions == 1
    assert first.skipped == 2
    assert second.duplicates == 4
    assert repository.stats().events == 4
    assert repository.stats().messages == 2
    assert repository.stats().reactions == 1
    messages = repository.messages()
    assert [message.text for message in messages] == ["Edited message", "My reply"]
    assert messages[0].reactions[0].value == "👍"
