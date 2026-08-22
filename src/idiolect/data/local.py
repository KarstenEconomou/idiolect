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
from idiolect.data.render import (
    RenderError,
    normalize_person_name,
    render_example,
    validate_mentions,
)
from idiolect.store.base import Repository
from idiolect.types import (
    ChatExample,
    DatasetId,
    DatasetRef,
    Example,
    Message,
    MessageId,
    PersonId,
    Reaction,
    Split,
)

_SCHEMA_VERSION = 2
_RENDER_VERSION = 2


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


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Keep verified metadata needed outside dataset construction."""

    dataset: DatasetRef
    target_name: str
    context_messages: int
    counts: Mapping[Split, int]


@dataclass(frozen=True, slots=True)
class _RenderedExample:
    """Keep one model row and its private source record."""

    value: ChatExample
    source: Example


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
        _validate_messages(messages)
        target_name = normalize_person_name(name)
        targets, selection = _select_targets(messages, person_id)
        if not targets:
            raise DataError("The target person has no usable messages")

        split_targets = _split_targets(targets, config)
        pseudonyms = _pseudonyms(messages, person_id)
        rendered = _render_splits(
            messages,
            split_targets,
            person_id,
            target_name,
            pseudonyms,
            config.context,
        )
        source_digest = _source_digest(messages)
        recipe = {
            "schema_version": _SCHEMA_VERSION,
            "render_version": _RENDER_VERSION,
            "target_id": str(person_id),
            "target_name": target_name,
            "context": config.context,
            "valid_ratio": config.valid_ratio,
            "test_ratio": config.test_ratio,
            "split": "chronological-purged-causal-context-v2",
            "target_policy": "unedited-text-only-no-attachments-v1",
            "format": "mlx-lm-completion-jsonl",
            "source_digest": source_digest,
        }
        counts = {split: len(values) for split, values in rendered.items()}
        files = {
            f"{split.value}.jsonl": _jsonl_bytes(examples)
            for split, examples in rendered.items()
            if examples
        }
        files["index.jsonl"] = _index_jsonl_bytes(rendered)
        identity = {
            "recipe": recipe,
            "counts": {split.value: count for split, count in counts.items()},
            "selection": selection,
            "files": {
                name: hashlib.sha256(content).hexdigest()
                for name, content in files.items()
            },
            "pseudonyms": {str(key): value for key, value in pseudonyms.items()},
        }
        digest = hashlib.sha256(_json_bytes(identity)).hexdigest()
        dataset_id = DatasetId(digest)
        destination = self._root / digest
        if destination.exists():
            return _existing_result(destination, dataset_id)

        created_at = self._clock()
        manifest = {
            "dataset_id": str(dataset_id),
            "created_at": created_at.isoformat(),
            **identity,
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


def load_dataset(path: Path) -> BuildResult:
    """Load and verify one immutable dataset."""
    try:
        dataset_id = DatasetId(path.name)
    except (TypeError, ValueError) as error:
        raise DataError(f"Dataset path does not contain an ID: {path}") from error
    return _existing_result(path, dataset_id)


def load_dataset_metadata(path: Path) -> DatasetMetadata:
    """Load verified target, context, identity, and split metadata."""
    result = load_dataset(path)
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        recipe = manifest["recipe"]
        target_name = recipe["target_name"]
        context = recipe["context"]
        if (
            not isinstance(target_name, str)
            or not target_name
            or not isinstance(context, int)
            or isinstance(context, bool)
            or context < 1
        ):
            raise TypeError
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DataError(f"Cannot read dataset metadata: {path}") from error
    return DatasetMetadata(result.dataset, target_name, context, result.counts)


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


def _validate_messages(messages: Sequence[Message]) -> None:
    identifiers: set[MessageId] = set()
    reaction_ids: set[str] = set()
    for message in messages:
        if message.id in identifiers:
            raise DataError(f"Source messages contain a duplicate ID: {message.id}")
        identifiers.add(message.id)
        values = (message.sent_at, message.edited_at, message.deleted_at)
        if any(value is not None and value.utcoffset() is None for value in values):
            raise DataError(f"Source message has a timestamp without a time zone: {message.id}")
        if message.edited_at is not None and message.edited_at < message.sent_at:
            raise DataError(f"Source message edit is before its send time: {message.id}")
        if message.deleted_at is not None and message.deleted_at < message.sent_at:
            raise DataError(f"Source message deletion is before its send time: {message.id}")
        try:
            validate_mentions(message.text, message.mentions)
            if message.quote is not None:
                validate_mentions(message.quote.text, message.quote.mentions)
        except RenderError as error:
            raise DataError(f"Source message has invalid mentions: {message.id}") from error
        if message.quote is not None and (
            message.reply_to is None
            or message.quote.sent_at.utcoffset() is None
            or message.quote.sent_at >= message.sent_at
        ):
            raise DataError(f"Source message has an invalid quote: {message.id}")
        for reaction in message.reactions:
            if str(reaction.event_id) in reaction_ids:
                raise DataError(
                    f"Source messages contain a duplicate reaction event: {reaction.event_id}"
                )
            reaction_ids.add(str(reaction.event_id))
            if (
                reaction.message_id != message.id
                or reaction.chat_id != message.chat_id
                or reaction.sent_at.utcoffset() is None
                or reaction.sent_at < message.sent_at
            ):
                raise DataError(f"Source message has an invalid reaction: {message.id}")
    by_id = {message.id: message for message in messages}
    for message in messages:
        if message.reply_to is None or message.reply_to not in by_id:
            continue
        original = by_id[message.reply_to]
        if original.chat_id != message.chat_id or original.sent_at >= message.sent_at:
            raise DataError(f"Source message has an invalid reply target: {message.id}")
        if message.quote is not None and (
            message.quote.author_id != original.author_id
            or message.quote.sent_at != original.sent_at
        ):
            raise DataError(f"Source message quote does not match its target: {message.id}")


def _select_targets(
    messages: Sequence[Message],
    person_id: PersonId,
) -> tuple[tuple[Message, ...], Mapping[str, int]]:
    reasons = {
        "attachment": 0,
        "deleted": 0,
        "edited": 0,
        "no_text": 0,
        "no_visible_text": 0,
    }
    targets = []
    total = 0
    for message in messages:
        if message.author_id != person_id:
            continue
        total += 1
        reason = _target_exclusion(message)
        if reason is None:
            targets.append(message)
        else:
            reasons[reason] += 1
    return tuple(targets), {
        "target_messages": total,
        "included": len(targets),
        "excluded": total - len(targets),
        **reasons,
    }


def _target_exclusion(message: Message) -> str | None:
    if message.deleted_at is not None:
        return "deleted"
    if message.edited_at is not None:
        return "edited"
    if message.attachments:
        return "attachment"
    if message.text is None:
        return "no_text"
    visible = any(
        not character.isspace() and character != "\ufffc" for character in message.text
    )
    if not visible and not message.mentions:
        return "no_visible_text"
    return None


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
) -> Mapping[Split, tuple[_RenderedExample, ...]]:
    ordered = tuple(sorted(messages, key=_message_key))
    lower_bound: tuple[datetime, str] | None = None
    result: dict[Split, tuple[_RenderedExample, ...]] = {}
    for split in (Split.TRAIN, Split.VALID, Split.TEST):
        targets = split_targets[split]
        examples = []
        for target in targets:
            context = tuple(
                message
                for message in ordered
                if message.chat_id == target.chat_id
                and (lower_bound is None or _message_key(message) > lower_bound)
                and message.sent_at < target.sent_at
                and _available_at(message, target.sent_at)
            )[-context_size:]
            source = Example(context, target)
            try:
                value = render_example(source, name, pseudonyms)
            except RenderError as error:
                raise DataError(f"Cannot render target message: {target.id}") from error
            examples.append(_RenderedExample(value, source))
        result[split] = tuple(examples)
        if targets:
            lower_bound = _message_key(targets[-1])
    return result


def _available_at(message: Message, observed_at: datetime) -> bool:
    if message.edited_at is not None and message.edited_at >= observed_at:
        return False
    return message.deleted_at is None or message.deleted_at < observed_at


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
        people.update(reaction.author_id for reaction in message.reactions)
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
        "reactions": [
            {
                "event_id": str(value.event_id),
                "message_id": str(value.message_id),
                "chat_id": str(value.chat_id),
                "author_id": str(value.author_id),
                "value": value.value,
                "sent_at": value.sent_at.isoformat(),
                "removed": value.removed,
            }
            for value in sorted(message.reactions, key=_reaction_key)
        ],
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
        if len(path.name) != 64 or any(
            character not in "0123456789abcdef" for character in path.name
        ):
            raise DataError(f"Dataset path does not contain an ID: {path}")
        value = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if value.get("dataset_id") != str(dataset_id):
            raise DataError(f"Dataset manifest does not match its path: {path}")
        recipe = value["recipe"]
        if not isinstance(recipe, dict):
            raise TypeError
        if recipe.get("schema_version") == _SCHEMA_VERSION:
            identity = {
                key: value[key]
                for key in ("recipe", "counts", "selection", "files", "pseudonyms")
            }
        else:
            identity = recipe
        if hashlib.sha256(_json_bytes(identity)).hexdigest() != str(dataset_id):
            raise DataError(f"Dataset identity does not match its ID: {path}")
        files = value["files"]
        if not isinstance(files, dict):
            raise TypeError
        actual_names = {
            item.name for item in path.iterdir() if item.is_file()
        }
        if actual_names != {"manifest.json", *files}:
            raise DataError(f"Dataset files do not match its manifest: {path}")
        for name, expected in files.items():
            file_path = _dataset_file(path, name)
            if not _is_digest(expected):
                raise TypeError
            actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if actual != expected:
                raise DataError(f"Dataset file does not match its manifest: {file_path}")
        raw_counts = value["counts"]
        if not isinstance(raw_counts, dict):
            raise TypeError
        if recipe.get("schema_version") == _SCHEMA_VERSION and any(
            not isinstance(count, int) or isinstance(count, bool)
            for count in raw_counts.values()
        ):
            raise TypeError
        counts = {Split(key): int(count) for key, count in raw_counts.items()}
        if any(count < 0 for count in counts.values()):
            raise TypeError
        raw_target_id = recipe["target_id"]
        if recipe.get("schema_version") == _SCHEMA_VERSION:
            if set(counts) != set(Split):
                raise TypeError
            for split, expected in counts.items():
                split_path = path / f"{split.value}.jsonl"
                if (expected > 0) != split_path.is_file():
                    raise DataError(
                        f"Dataset split count does not match its file: {split_path}"
                    )
                if expected > 0 and len(
                    split_path.read_text(encoding="utf-8").splitlines()
                ) != expected:
                    raise DataError(
                        f"Dataset split count does not match its file: {split_path}"
                    )
                if expected > 0:
                    _validate_split_rows(split_path)
            _validate_selection(value["selection"], counts)
            _validate_pseudonyms(value["pseudonyms"], target_id=raw_target_id)
            _validate_index(path / "index.jsonl", counts)
        created_at = datetime.fromisoformat(value["created_at"])
        if created_at.utcoffset() is None:
            raise TypeError
        target_id = PersonId(raw_target_id)
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


def _jsonl_bytes(examples: Sequence[_RenderedExample]) -> bytes:
    lines = []
    for example in examples:
        value = {
            "prompt": example.value.prompt,
            "completion": example.value.completion,
        }
        lines.append(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode()


def _index_jsonl_bytes(
    rendered: Mapping[Split, tuple[_RenderedExample, ...]],
) -> bytes:
    lines = []
    for split in Split:
        for index, example in enumerate(rendered[split]):
            source = example.source
            value = {
                "split": split.value,
                "index": index,
                "chat_id": str(source.target.chat_id),
                "target_message_id": str(source.target.id),
                "target_sent_at": source.target.sent_at.isoformat(),
                "context_message_ids": [str(message.id) for message in source.context],
                "context_reaction_event_ids": [
                    str(reaction.event_id)
                    for reaction in sorted(
                        (
                            reaction
                            for message in source.context
                            for reaction in message.reactions
                            if reaction.sent_at < source.target.sent_at
                        ),
                        key=_reaction_key,
                    )
                ],
            }
            lines.append(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode()


def _dataset_file(path: Path, name: Any) -> Path:
    allowed = {"index.jsonl", *(f"{split.value}.jsonl" for split in Split)}
    if not isinstance(name, str) or name not in allowed:
        raise DataError(f"Dataset manifest contains an invalid file name: {path}")
    return path / name


def _validate_split_rows(path: Path) -> None:
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise DataError(f"Dataset row is not valid: {path}:{line_number}") from error
        if (
            not isinstance(value, dict)
            or set(value) != {"prompt", "completion"}
            or not isinstance(value["prompt"], str)
            or not isinstance(value["completion"], str)
        ):
            raise DataError(f"Dataset row is not valid: {path}:{line_number}")


def _validate_selection(value: Any, counts: Mapping[Split, int]) -> None:
    keys = {
        "attachment",
        "deleted",
        "edited",
        "excluded",
        "included",
        "no_text",
        "no_visible_text",
        "target_messages",
    }
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in value.values()
        )
        or value["included"] != sum(counts.values())
        or value["target_messages"] != value["included"] + value["excluded"]
        or value["excluded"]
        != sum(value[key] for key in keys - {"target_messages", "included", "excluded"})
    ):
        raise DataError("Dataset target selection counts are not valid")


def _validate_pseudonyms(value: Any, target_id: Any) -> None:
    if (
        not isinstance(target_id, str)
        or not isinstance(value, dict)
        or target_id in value
        or any(not isinstance(key, str) for key in value)
        or any(not isinstance(name, str) or not name for name in value.values())
        or len(set(value.values())) != len(value)
    ):
        raise DataError("Dataset pseudonyms are not valid")


def _validate_index(path: Path, counts: Mapping[Split, int]) -> None:
    expected = [
        (split, index)
        for split in Split
        for index in range(counts[split])
    ]
    rows = path.read_text(encoding="utf-8").splitlines()
    if len(rows) != len(expected):
        raise DataError(f"Dataset index count does not match its splits: {path}")
    sources: dict[Split, set[str]] = {split: set() for split in Split}
    target_ids: set[str] = set()
    previous_target: tuple[datetime, str] | None = None
    keys = {
        "split",
        "index",
        "chat_id",
        "target_message_id",
        "target_sent_at",
        "context_message_ids",
        "context_reaction_event_ids",
    }
    for line_number, (line, (expected_split, expected_index)) in enumerate(
        zip(rows, expected, strict=True), start=1
    ):
        try:
            value = json.loads(line)
            sent_at = datetime.fromisoformat(value["target_sent_at"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise DataError(f"Dataset index row is not valid: {path}:{line_number}") from error
        context_ids = value.get("context_message_ids")
        reaction_ids = value.get("context_reaction_event_ids")
        target_id = value.get("target_message_id")
        if (
            not isinstance(value, dict)
            or set(value) != keys
            or value.get("split") != expected_split.value
            or value.get("index") != expected_index
            or not isinstance(value.get("chat_id"), str)
            or not isinstance(target_id, str)
            or not isinstance(context_ids, list)
            or not all(isinstance(item, str) for item in context_ids)
            or not isinstance(reaction_ids, list)
            or not all(isinstance(item, str) for item in reaction_ids)
            or sent_at.utcoffset() is None
            or target_id in target_ids
            or target_id in context_ids
            or len(set(context_ids)) != len(context_ids)
            or len(set(reaction_ids)) != len(reaction_ids)
            or (
                previous_target is not None
                and (sent_at, target_id) <= previous_target
            )
        ):
            raise DataError(f"Dataset index row is not valid: {path}:{line_number}")
        target_ids.add(target_id)
        sources[expected_split].add(target_id)
        sources[expected_split].update(context_ids)
        sources[expected_split].update(reaction_ids)
        previous_target = sent_at, target_id
    for index, split in enumerate(Split):
        for other in tuple(Split)[index + 1 :]:
            if not sources[split].isdisjoint(sources[other]):
                raise DataError(f"Dataset source message crosses split boundaries: {path}")


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reaction_key(reaction: Reaction) -> tuple[datetime, str]:
    return reaction.sent_at, str(reaction.event_id)


def _utc_now() -> datetime:
    return datetime.now(UTC)
