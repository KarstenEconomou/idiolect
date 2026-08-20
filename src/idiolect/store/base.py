"""Define the data store ports."""

from collections.abc import Iterable
from typing import Protocol

from idiolect.types import DatasetRef, Event, Example, Message, PersonId, RunRef, Split


class Repository(Protocol):
    """Store events and normalized messages."""

    def save_events(self, events: Iterable[Event]) -> None:
        """Save source events."""
        ...

    def save_messages(self, messages: Iterable[Message]) -> None:
        """Save normalized messages."""
        ...

    def messages(self, person_id: PersonId | None = None) -> Iterable[Message]:
        """Return messages in time order."""
        ...


class DatasetStore(Protocol):
    """Store fixed model datasets."""

    def save(self, dataset: DatasetRef, split: Split, examples: Iterable[Example]) -> None:
        """Save one dataset part."""
        ...

    def load(self, dataset: DatasetRef, split: Split) -> Iterable[Example]:
        """Load one dataset part."""
        ...


class RunStore(Protocol):
    """Store model run artifacts."""

    def start(self, run: RunRef) -> None:
        """Create one run location."""
        ...
