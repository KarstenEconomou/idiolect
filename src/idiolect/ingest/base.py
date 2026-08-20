"""Define the message input ports."""

from collections.abc import Iterable
from typing import Protocol

from idiolect.types import Event, Record


class Source(Protocol):
    """Read events from one source."""

    def events(self) -> Iterable[Event]:
        """Return new source events."""
        ...


class Parser(Protocol):
    """Convert source events to records."""

    def records(self, event: Event) -> Iterable[Record]:
        """Return records from one event."""
        ...
