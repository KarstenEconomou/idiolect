"""Store normalized records from one input source."""

from dataclasses import dataclass

from idiolect.ingest.base import Parser, Source
from idiolect.store.base import Repository
from idiolect.types import Message, Reaction


@dataclass(frozen=True, slots=True)
class HarvestResult:
    """Keep counts from one harvest operation."""

    received: int = 0
    stored: int = 0
    messages: int = 0
    reactions: int = 0
    skipped: int = 0
    duplicates: int = 0


@dataclass(frozen=True, slots=True)
class ReindexResult:
    """Keep counts from one normalization refresh."""

    scanned: int = 0
    updated: int = 0
    messages: int = 0
    reactions: int = 0
    skipped: int = 0


def harvest(source: Source, parser: Parser, repository: Repository) -> HarvestResult:
    """Read, normalize, and store source events."""
    received = stored = messages = reactions = skipped = duplicates = 0
    for event in source.events():
        received += 1
        # One event that cannot be normalized must not discard the drained
        # events after it.
        try:
            records = tuple(parser.records(event))
        except Exception:  # noqa: BLE001 - tolerate one bad event, count it.
            skipped += 1
            continue
        if not records:
            skipped += 1
            continue
        if not repository.save(event, records):
            duplicates += 1
            continue
        stored += 1
        messages += sum(isinstance(record, Message) for record in records)
        reactions += sum(isinstance(record, Reaction) for record in records)
    return HarvestResult(received, stored, messages, reactions, skipped, duplicates)


def reindex(parser: Parser, repository: Repository) -> ReindexResult:
    """Refresh normalized records from stored source events."""
    scanned = updated = messages = reactions = skipped = 0
    for event in repository.events():
        scanned += 1
        records = tuple(parser.records(event))
        if not records:
            skipped += 1
            continue
        repository.replace(event, records)
        updated += 1
        messages += sum(isinstance(record, Message) for record in records)
        reactions += sum(isinstance(record, Reaction) for record in records)
    return ReindexResult(scanned, updated, messages, reactions, skipped)
