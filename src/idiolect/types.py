"""Define records for Idiolect data."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, NewType

ChatId = NewType("ChatId", str)
DatasetId = NewType("DatasetId", str)
EventId = NewType("EventId", str)
InferenceId = NewType("InferenceId", str)
EvaluationId = NewType("EvaluationId", str)
JudgmentId = NewType("JudgmentId", str)
PanelId = NewType("PanelId", str)
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
class Mention:
    """Keep one identity-linked text mention."""

    person_id: PersonId
    start_utf16: int
    length_utf16: int
    name: str | None = None


@dataclass(frozen=True, slots=True)
class Quote:
    """Keep the source snapshot for one reply."""

    author_id: PersonId
    sent_at: datetime
    text: str | None = None
    mentions: tuple[Mention, ...] = ()


@dataclass(frozen=True, slots=True)
class Reaction:
    """Keep one message reaction."""

    event_id: EventId
    message_id: MessageId
    chat_id: ChatId
    author_id: PersonId
    value: str
    sent_at: datetime
    removed: bool = False


@dataclass(frozen=True, slots=True)
class Message:
    """Keep one normalized message."""

    id: MessageId
    event_id: EventId
    chat_id: ChatId
    author_id: PersonId
    sent_at: datetime
    author_name: str | None = None
    is_self: bool = False
    text: str | None = None
    reply_to: MessageId | None = None
    edited_at: datetime | None = None
    deleted_at: datetime | None = None
    reactions: tuple[Reaction, ...] = ()
    attachments: tuple[Attachment, ...] = ()
    mentions: tuple[Mention, ...] = ()
    quote: Quote | None = None


type Record = Message | Reaction


@dataclass(frozen=True, slots=True)
class StoreStats:
    """Keep counts from the local store."""

    events: int
    messages: int
    reactions: int


@dataclass(frozen=True, slots=True)
class Example:
    """Keep one model example."""

    context: tuple[Message, ...]
    target: Message


@dataclass(frozen=True, slots=True)
class ChatExample:
    """Keep one rendered chat training example."""

    prompt: str
    completion: str


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
class TrainResult:
    """Keep the completed runs for one configured experiment."""

    runs: tuple[RunRef, ...]


@dataclass(frozen=True, slots=True)
class InferenceRef:
    """Point to one fixed inference batch."""

    id: InferenceId
    path: Path
    created_at: datetime
    predictions: int


@dataclass(frozen=True, slots=True)
class Metric:
    """Keep one model test result."""

    name: str
    value: float
    target: str
    unit: str
    samples: int
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True, slots=True)
class Interval:
    """Keep one estimate and its confidence interval."""

    value: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class GateResult:
    """Keep one evaluation eligibility gate result."""

    value: float
    limit: float
    passed: bool


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Keep one typed automatic evaluation report."""

    suite: str
    eligible: bool
    examples: int
    training_runs: int
    generation_seeds: int
    gates: Mapping[str, GateResult]
    likelihood: Mapping[str, Any]
    voice_profiles: Mapping[str, Any]
    behavior: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EvaluationRef:
    """Point to one fixed automatic evaluation."""

    id: EvaluationId
    path: Path
    created_at: datetime
    eligible: bool


@dataclass(frozen=True, slots=True)
class JudgmentRef:
    """Point to one fixed familiar-rater judgment set."""

    id: JudgmentId
    evaluation_id: EvaluationId
    path: Path
    created_at: datetime
    judgments: int


@dataclass(frozen=True, slots=True)
class PanelRef:
    """Point to one fixed familiar-panel report."""

    id: PanelId
    evaluation_id: EvaluationId
    path: Path
    created_at: datetime
    complete: bool
