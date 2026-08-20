"""Define records for Idiolect data."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import NewType

ChatId = NewType("ChatId", str)
DatasetId = NewType("DatasetId", str)
EventId = NewType("EventId", str)
MessageId = NewType("MessageId", str)
PersonId = NewType("PersonId", str)
RunId = NewType("RunId", str)


class Split(StrEnum):
    """Name one dataset part."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class Event:
    """Keep one source event."""

    id: EventId
    source: str
    source_id: str
    received_at: datetime
    payload: bytes


@dataclass(frozen=True, slots=True)
class Attachment:
    """Keep data about one attachment."""

    id: str
    media_type: str | None = None
    name: str | None = None
    size: int | None = None


@dataclass(frozen=True, slots=True)
class Reaction:
    """Keep one message reaction."""

    author_id: PersonId
    value: str
    sent_at: datetime


@dataclass(frozen=True, slots=True)
class Message:
    """Keep one normalized message."""

    id: MessageId
    chat_id: ChatId
    author_id: PersonId
    sent_at: datetime
    text: str | None = None
    reply_to: MessageId | None = None
    edited_at: datetime | None = None
    deleted_at: datetime | None = None
    reactions: tuple[Reaction, ...] = ()
    attachments: tuple[Attachment, ...] = ()


@dataclass(frozen=True, slots=True)
class Example:
    """Keep one model example."""

    context: tuple[Message, ...]
    target: Message


@dataclass(frozen=True, slots=True)
class DatasetRef:
    """Point to one fixed dataset."""

    id: DatasetId
    person_id: PersonId
    path: Path
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RunRef:
    """Point to one model run."""

    id: RunId
    dataset_id: DatasetId
    path: Path
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Metric:
    """Keep one model test result."""

    name: str
    value: float
