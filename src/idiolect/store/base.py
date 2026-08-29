"""Define the data store ports."""

from collections.abc import Iterable
from typing import Protocol

from idiolect.types import (
    DatasetRef,
    Event,
    Example,
    Message,
    PersonId,
    Record,
    RunRef,
    Split,
    StoreStats,
)


class Repository(Protocol):
    """Store events and normalized messages."""

    def save(self, event: Event, records: Iterable[Record]) -> bool:
        """Save one event and return true for a new event."""
        ...

    def events(self) -> Iterable[Event]:
        """Return stored source events in storage order."""
        ...

    def replace(self, event: Event, records: Iterable[Record]) -> None:
        """Replace normalized records from one stored event."""
        ...

    def messages(self, person_id: PersonId | None = None) -> Iterable[Message]:
        """Return messages in time order."""
        ...

    def stats(self) -> StoreStats:
        """Return record counts from the store."""
        ...


class DatasetStore(Protocol):
    """Store fixed model datasets."""

    def save(
        self, dataset: DatasetRef, split: Split, examples: Iterable[Example]
    ) -> None:
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
