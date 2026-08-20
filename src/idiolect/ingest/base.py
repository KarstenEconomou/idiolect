"""Define the message input ports."""

from collections.abc import Iterable
from typing import Protocol

from idiolect.types import Event, Message


class Source(Protocol):
    """Read events from one source."""

    def events(self) -> Iterable[Event]:
        """Return new source events."""
        ...


class Parser(Protocol):
    """Convert source events to messages."""

    def messages(self, event: Event) -> Iterable[Message]:
        """Return messages from one event."""
        ...
