"""Test the DuckDB message store."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from idiolect.ingest import harvest, reindex
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


def test_store_round_trips_mentions_and_quote_snapshots(
    signal_mentions: Path,
    tmp_path: Path,
) -> None:
    """Check storage of addressing data in the required schema."""
    path = tmp_path / "test.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            CREATE TABLE messages (
                id VARCHAR PRIMARY KEY,
                event_id VARCHAR NOT NULL,
                chat_id VARCHAR NOT NULL,
                author_id VARCHAR NOT NULL,
                sent_at TIMESTAMPTZ NOT NULL,
                text VARCHAR,
                reply_to VARCHAR,
                edited_at TIMESTAMPTZ,
                deleted_at TIMESTAMPTZ,
                revision_at TIMESTAMPTZ NOT NULL
            )
            """
        )
    repository = DuckRepository(path)
    parser = SignalParser((ChatId("group-allowed"),))
    harvest(
        SignalFileSource(signal_mentions, clock=lambda: _NOW),
        parser,
        repository,
    )

    target, tagged, plain = repository.messages()

    assert tagged.mentions[0].person_id == target.author_id
    assert tagged.reply_to == target.id
    assert tagged.quote is not None
    assert tagged.quote.text == "Maybe"
    assert plain.mentions == ()


def test_reindex_refreshes_addressing_from_raw_events(
    signal_mentions: Path,
    tmp_path: Path,
) -> None:
    """Check that reindex rebuilds addressing from stored events."""
    repository = DuckRepository(tmp_path / "test.duckdb")
    parser = SignalParser((ChatId("group-allowed"),))
    events = tuple(SignalFileSource(signal_mentions, clock=lambda: _NOW).events())
    for index, event in enumerate(events):
        message = next(iter(parser.records(event)))
        if index == 1:
            message = replace(
                message,
                text="😀 ￼ are you coming?",
                mentions=(),
                quote=None,
            )
        repository.save(event, (message,))

    result = reindex(parser, repository)
    _, tagged, _ = repository.messages()

    assert (result.scanned, result.updated, result.messages) == (3, 3, 3)
    assert tagged.text == "😀 ￼ are you coming?"
    assert tagged.mentions
    assert tagged.quote is not None
