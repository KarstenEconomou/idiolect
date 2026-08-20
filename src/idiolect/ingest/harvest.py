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


def harvest(source: Source, parser: Parser, repository: Repository) -> HarvestResult:
    """Read, normalize, and store source events."""
    received = stored = messages = reactions = skipped = duplicates = 0
    for event in source.events():
        received += 1
        records = tuple(parser.records(event))
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
