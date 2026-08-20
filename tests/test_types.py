"""Test the shared data records."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from idiolect.types import ChatId, EventId, Message, MessageId, PersonId


def test_message_is_fixed() -> None:
    """Check that a message cannot change."""
    message = Message(
        id=MessageId("message"),
        event_id=EventId("event"),
        chat_id=ChatId("chat"),
        author_id=PersonId("person"),
        sent_at=datetime.now(UTC),
        text="Hello.",
    )

    with pytest.raises(FrozenInstanceError):
        message.text = "Changed."  # ty: ignore[invalid-assignment]
