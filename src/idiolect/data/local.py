"""Build local target-specific chat datasets."""

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from idiolect.config import DataConfig
from idiolect.data.render import render_example
from idiolect.store.base import Repository
from idiolect.types import (
    ChatExample,
    DatasetId,
    DatasetRef,
    Example,
    Message,
    PersonId,
    Split,
)

_SCHEMA_VERSION = 1
_RENDER_VERSION = 1


class DataError(ValueError):
    """Report an invalid dataset operation."""


@dataclass(frozen=True, slots=True)
class PersonSummary:
    """Keep local data that identifies one message author."""

    id: PersonId
    name: str | None
    messages: int
    first_at: datetime
    last_at: datetime
    is_self: bool


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Keep one dataset reference and its split counts."""

    dataset: DatasetRef
    counts: Mapping[Split, int]


class LocalBuilder:
    """Build immutable MLX-LM files from local messages."""

    def __init__(
        self,
        repository: Repository,
        root: Path,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Set the message source and output directory."""
        self._repository = repository
        self._root = root
        self._clock = _utc_now if clock is None else clock

    def build(self, person_id: PersonId, name: str, config: DataConfig) -> BuildResult:
        """Build or return one content-addressed dataset."""
        _validate_config(config)
        messages = tuple(self._repository.messages())
        targets = tuple(
            message
            for message in messages
            if message.author_id == person_id
            and message.text is not None
            and message.deleted_at is None
        )
        if not targets:
            raise DataError("The target person has no usable messages")

        split_targets = _split_targets(targets, config)
        pseudonyms = _pseudonyms(messages, person_id)
        rendered = _render_splits(
            messages,
            split_targets,
            person_id,
            name,
            pseudonyms,
            config.context,
        )
        source_digest = _source_digest(messages)
        recipe = {
            "schema_version": _SCHEMA_VERSION,
            "render_version": _RENDER_VERSION,
            "target_id": str(person_id),
            "target_name": name,
            "context": config.context,
            "valid_ratio": config.valid_ratio,
            "test_ratio": config.test_ratio,
            "split": "chronological-purged-context-v1",
            "format": "mlx-lm-completion-jsonl",
            "source_digest": source_digest,
        }
        digest = hashlib.sha256(_json_bytes(recipe)).hexdigest()
        dataset_id = DatasetId(digest)
        destination = self._root / digest
        if destination.exists():
            return _existing_result(destination, dataset_id)

        created_at = self._clock()
        counts = {split: len(values) for split, values in rendered.items()}
        files = {
            f"{split.value}.jsonl": _jsonl_bytes(examples)
            for split, examples in rendered.items()
            if examples
        }
        manifest = {
            "dataset_id": str(dataset_id),
            "created_at": created_at.isoformat(),
            "recipe": recipe,
            "counts": {split.value: count for split, count in counts.items()},
            "files": {
                name: hashlib.sha256(content).hexdigest()
                for name, content in files.items()
            },
            "pseudonyms": {str(key): value for key, value in pseudonyms.items()},
        }
        self._write(destination, files, manifest)
        return BuildResult(
            DatasetRef(dataset_id, person_id, destination, created_at),
            counts,
        )

    def _write(
        self,
        destination: Path,
        files: Mapping[str, bytes],
        manifest: Mapping[str, Any],
    ) -> None:
        """Write one dataset with an atomic directory move."""
        self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".build-", dir=self._root))
        try:
            for name, content in files.items():
                path = temporary / name
                with path.open("xb") as stream:
                    stream.write(content)
                os.chmod(path, 0o600)
            manifest_path = temporary / "manifest.json"
            with manifest_path.open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
            os.chmod(manifest_path, 0o600)
            temporary.rename(destination)
        except (OSError, TypeError, ValueError) as error:
            shutil.rmtree(temporary, ignore_errors=True)
            raise DataError(f"Cannot write dataset: {destination}") from error


def summarize_people(messages: Iterable[Message]) -> tuple[PersonSummary, ...]:
    """Return one summary for each message author."""
    values: dict[PersonId, list[Message]] = {}
    for message in messages:
        values.setdefault(message.author_id, []).append(message)
    summaries = []
    for person_id, person_messages in values.items():
        ordered = sorted(person_messages, key=_message_key)
        names = [message.author_name for message in ordered if message.author_name]
        summaries.append(
            PersonSummary(
                id=person_id,
                name=names[-1] if names else None,
                messages=len(ordered),
                first_at=ordered[0].sent_at,
                last_at=ordered[-1].sent_at,
                is_self=any(message.is_self for message in ordered),
            )
        )
    return tuple(sorted(summaries, key=lambda item: (-item.messages, str(item.id))))


def resolve_self(people: Iterable[PersonSummary]) -> PersonId:
    """Return the one author marked as the local Signal account."""
    candidates = tuple(person.id for person in people if person.is_self)
    if len(candidates) != 1:
        raise DataError(f"Expected one self identity, found {len(candidates)}")
    return candidates[0]


def _validate_config(config: DataConfig) -> None:
    if config.context < 1:
        raise DataError("Dataset context must be greater than zero")
    if not 0 <= config.valid_ratio < 1:
        raise DataError("Dataset valid_ratio must be at least zero and less than one")
    if not 0 <= config.test_ratio < 1:
        raise DataError("Dataset test_ratio must be at least zero and less than one")
    if config.valid_ratio + config.test_ratio >= 1:
        raise DataError("Dataset holdout ratios must have a sum less than one")


def _split_targets(
    targets: Sequence[Message],
    config: DataConfig,
) -> Mapping[Split, tuple[Message, ...]]:
    ordered = tuple(sorted(targets, key=_message_key))
    valid = _holdout_count(len(ordered), config.valid_ratio)
    test = _holdout_count(len(ordered), config.test_ratio)
    train = len(ordered) - valid - test
    if train < 1:
        raise DataError("The target has too few messages for the requested splits")
    train_end = train
    valid_end = train + valid
    return {
        Split.TRAIN: ordered[:train_end],
        Split.VALID: ordered[train_end:valid_end],
        Split.TEST: ordered[valid_end:],
    }


def _holdout_count(total: int, ratio: float) -> int:
    if ratio == 0:
        return 0
    return max(1, int(total * ratio))


def _render_splits(
    messages: Sequence[Message],
    split_targets: Mapping[Split, tuple[Message, ...]],
    person_id: PersonId,
    name: str,
    pseudonyms: Mapping[PersonId, str],
    context_size: int,
) -> Mapping[Split, tuple[ChatExample, ...]]:
    ordered = tuple(sorted(messages, key=_message_key))
    lower_bound: tuple[datetime, str] | None = None
    result: dict[Split, tuple[ChatExample, ...]] = {}
    for split in (Split.TRAIN, Split.VALID, Split.TEST):
        targets = split_targets[split]
        examples = []
        for target in targets:
            target_key = _message_key(target)
            context = tuple(
                message
                for message in ordered
                if message.chat_id == target.chat_id
                and (lower_bound is None or _message_key(message) > lower_bound)
                and _message_key(message) < target_key
            )[-context_size:]
            examples.append(
                render_example(Example(context, target), name, pseudonyms)
            )
        result[split] = tuple(examples)
        if targets:
            lower_bound = _message_key(targets[-1])
    return result


def _pseudonyms(
    messages: Sequence[Message],
    target_id: PersonId,
) -> Mapping[PersonId, str]:
    people: set[PersonId] = set()
    for message in messages:
        people.add(message.author_id)
        people.update(mention.person_id for mention in message.mentions)
        if message.quote is not None:
            people.add(message.quote.author_id)
            people.update(mention.person_id for mention in message.quote.mentions)
    people.discard(target_id)
    return {
        person_id: f"person_{index:02d}"
        for index, person_id in enumerate(sorted(people, key=str), start=1)
    }


def _source_digest(messages: Sequence[Message]) -> str:
    values = [_message_value(message) for message in sorted(messages, key=_message_key)]
    return hashlib.sha256(_json_bytes(values)).hexdigest()


def _message_value(message: Message) -> Mapping[str, Any]:
    quote = None
    if message.quote is not None:
        quote = {
            "author_id": str(message.quote.author_id),
            "sent_at": message.quote.sent_at.isoformat(),
            "text": message.quote.text,
            "mentions": [_mention_value(value) for value in message.quote.mentions],
        }
    return {
        "id": str(message.id),
        "event_id": str(message.event_id),
        "chat_id": str(message.chat_id),
        "author_id": str(message.author_id),
        "sent_at": message.sent_at.isoformat(),
        "author_name": message.author_name,
        "is_self": message.is_self,
        "text": message.text,
        "reply_to": str(message.reply_to) if message.reply_to is not None else None,
        "edited_at": message.edited_at.isoformat() if message.edited_at else None,
        "deleted_at": message.deleted_at.isoformat() if message.deleted_at else None,
        "mentions": [_mention_value(value) for value in message.mentions],
        "quote": quote,
        "attachments": [value.id for value in message.attachments],
    }


def _mention_value(mention: Any) -> Mapping[str, Any]:
    return {
        "person_id": str(mention.person_id),
        "start_utf16": mention.start_utf16,
        "length_utf16": mention.length_utf16,
        "name": mention.name,
    }


def _existing_result(path: Path, dataset_id: DatasetId) -> BuildResult:
    try:
        value = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if value.get("dataset_id") != str(dataset_id):
            raise DataError(f"Dataset manifest does not match its path: {path}")
        recipe = value["recipe"]
        if hashlib.sha256(_json_bytes(recipe)).hexdigest() != str(dataset_id):
            raise DataError(f"Dataset recipe does not match its ID: {path}")
        for name, expected in value["files"].items():
            actual = hashlib.sha256((path / name).read_bytes()).hexdigest()
            if actual != expected:
                raise DataError(f"Dataset file does not match its manifest: {path / name}")
        counts = {Split(key): int(count) for key, count in value["counts"].items()}
        created_at = datetime.fromisoformat(value["created_at"])
        target_id = PersonId(recipe["target_id"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, DataError):
            raise
        raise DataError(f"Cannot read existing dataset: {path}") from error
    return BuildResult(DatasetRef(dataset_id, target_id, path, created_at), counts)


def _message_key(message: Message) -> tuple[datetime, str]:
    return message.sent_at, str(message.id)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _jsonl_bytes(examples: Sequence[ChatExample]) -> bytes:
    lines = []
    for example in examples:
        value = {"prompt": example.prompt, "completion": example.completion}
        lines.append(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode()


def _utc_now() -> datetime:
    return datetime.now(UTC)
