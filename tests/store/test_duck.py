"""Test the DuckDB message store."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from idiolect.ingest import harvest, reindex
from idiolect.ingest.signal import SignalFileSource, SignalParser
from idiolect.store.duck import DuckRepository, StoreError
from idiolect.types import (
    Attachment,
    ChatId,
    Event,
    EventId,
    Message,
    MessageId,
    PersonId,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_harvest_filters_updates_and_is_idempotent(
    signal_events: Path, tmp_path: Path
) -> None:
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


def test_store_keeps_the_newest_message_revision(tmp_path: Path) -> None:
    """Check an older source revision cannot replace a newer one."""
    repository = DuckRepository(tmp_path / "test.duckdb")
    older_at = _NOW + timedelta(seconds=10)
    newer_at = _NOW + timedelta(seconds=20)
    newer_event = Event(EventId("event-new"), "synthetic", "new", newer_at, b"new")
    older_event = Event(EventId("event-old"), "synthetic", "old", older_at, b"old")

    repository.save(newer_event, (_message(newer_event.id, "new", newer_at),))
    repository.save(older_event, (_message(older_event.id, "old", older_at),))

    messages = repository.messages()
    assert len(messages) == 1
    assert messages[0].text == "new"
    assert messages[0].event_id == newer_event.id


def test_store_rolls_back_event_when_record_persistence_fails(tmp_path: Path) -> None:
    """Check a failed record write does not leave a partial source event."""
    repository = DuckRepository(tmp_path / "test.duckdb")
    event = Event(EventId("event-failed"), "synthetic", "failed", _NOW, b"failed")
    message = _message(
        event.id,
        "will not persist",
        _NOW,
        attachments=(Attachment("duplicate"), Attachment("duplicate")),
    )

    with pytest.raises(StoreError, match="Cannot save event"):
        repository.save(event, (message,))

    assert repository.stats().events == 0
    assert repository.stats().messages == 0
    assert repository.messages() == ()


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


def _message(
    event_id: EventId,
    text: str,
    revision_at: datetime,
    *,
    attachments: tuple[Attachment, ...] = (),
) -> Message:
    """Create one synthetic message with an explicit revision timestamp."""
    return Message(
        id=MessageId("message-1"),
        event_id=event_id,
        chat_id=ChatId("chat-1"),
        author_id=PersonId("person-1"),
        sent_at=_NOW,
        text=text,
        edited_at=revision_at,
        attachments=attachments,
    )
