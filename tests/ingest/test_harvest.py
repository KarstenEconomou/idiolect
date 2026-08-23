"""Test event harvest behavior."""

import json
from datetime import UTC, datetime
from pathlib import Path

from idiolect.ingest import harvest
from idiolect.ingest.signal import SignalParser
from idiolect.store.duck import DuckRepository
from idiolect.types import ChatId, Event, EventId

_NOW = datetime(2026, 1, 1, tzinfo=UTC)

_BAD_EVENT = {
    "envelope": {
        "dataMessage": {
            "groupInfo": {"groupId": "group-allowed"},
            "message": "no author",
            "timestamp": 1,
        }
    }
}

_GOOD_EVENT = {
    "envelope": {
        "sourceNumber": "+12223334444",
        "timestamp": 2,
        "dataMessage": {
            "groupInfo": {"groupId": "group-allowed"},
            "message": "stored",
        },
    }
}


class FakeSource:
    """Return two fixed events, one of which cannot be normalized."""

    def __init__(self) -> None:
        self.events_returned = tuple(
            Event(
                id=EventId(f"fake:{index}"),
                source="fake",
                source_id=str(index),
                received_at=_NOW,
                payload=json.dumps(value).encode(),
            )
            for index, value in enumerate((_BAD_EVENT, _GOOD_EVENT))
        )

    def events(self):
        return self.events_returned


def test_harvest_skips_one_invalid_event_and_keeps_the_rest(tmp_path: Path) -> None:
    """Check that a malformed event does not discard later drained events."""
    result = harvest(
        FakeSource(),
        SignalParser((ChatId("group-allowed"),)),
        DuckRepository(tmp_path / "test.duckdb"),
    )

    assert (result.received, result.stored, result.skipped) == (2, 1, 1)
    assert result.messages == 1
    messages = DuckRepository(tmp_path / "test.duckdb").messages()
    assert [message.text for message in messages] == ["stored"]
