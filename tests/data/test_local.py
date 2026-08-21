"""Test local dataset construction."""

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from idiolect.config import DataConfig
from idiolect.data.local import (
    DataError,
    LocalBuilder,
    load_dataset,
    resolve_self,
    summarize_people,
)
from idiolect.types import (
    Attachment,
    ChatId,
    EventId,
    Mention,
    Message,
    MessageId,
    PersonId,
    Split,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_CHAT = ChatId("chat")
_TARGET = PersonId("target-id")
_FRIEND = PersonId("friend-id")


class FakeRepository:
    """Return fixed normalized messages."""

    def __init__(self, messages: tuple[Message, ...]) -> None:
        """Set the fixed messages."""
        self._messages = messages

    def messages(self, person_id: PersonId | None = None) -> tuple[Message, ...]:
        """Return all or one person's messages."""
        if person_id is None:
            return self._messages
        return tuple(message for message in self._messages if message.author_id == person_id)


def test_builder_writes_immutable_leakage_safe_mlx_data(tmp_path: Path) -> None:
    """Check target rendering, chronological splits, and content addressing."""
    messages = _conversation(10)
    builder = LocalBuilder(
        FakeRepository(messages),  # ty: ignore[invalid-argument-type]
        tmp_path / "data",
        clock=lambda: _NOW,
    )
    config = DataConfig(context=4, valid_ratio=0.2, test_ratio=0.2)

    first = builder.build(_TARGET, "Karsten", config)
    second = builder.build(_TARGET, "Karsten", config)

    assert first.dataset == second.dataset
    assert first.counts == {Split.TRAIN: 6, Split.VALID: 2, Split.TEST: 2}
    values = {
        split: _read_jsonl(first.dataset.path / f"{split.value}.jsonl")
        for split in Split
    }
    assert [item["completion"] for item in values[Split.TRAIN]] == [
        f"target-{index:02d}" for index in range(6)
    ]
    assert [item["completion"] for item in values[Split.VALID]] == [
        "target-06",
        "target-07",
    ]
    assert [item["completion"] for item in values[Split.TEST]] == [
        "target-08",
        "target-09",
    ]
    split_tokens = {
        split: set(re.findall(r"(?:friend|target)-\d{2}", json.dumps(items)))
        for split, items in values.items()
    }
    assert split_tokens[Split.TRAIN].isdisjoint(split_tokens[Split.VALID])
    assert split_tokens[Split.TRAIN].isdisjoint(split_tokens[Split.TEST])
    assert split_tokens[Split.VALID].isdisjoint(split_tokens[Split.TEST])
    assert "mentions @Karsten" in values[Split.TRAIN][0]["prompt"]
    assert "@Karsten ping friend-00" in values[Split.TRAIN][0]["prompt"]
    assert "￼" not in json.dumps(values, ensure_ascii=False)
    assert "target-id" not in json.dumps(values)
    manifest = json.loads(
        (first.dataset.path / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["recipe"]["split"] == "chronological-purged-causal-context-v2"
    assert manifest["counts"] == {"test": 2, "train": 6, "valid": 2}
    index = _read_jsonl(first.dataset.path / "index.jsonl")
    assert len(index) == 10
    assert index[0]["target_message_id"] == "target-message-00"

    (first.dataset.path / "train.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(DataError, match="does not match its manifest"):
        builder.build(_TARGET, "Karsten", config)


def test_builder_uses_only_clean_targets_and_causal_context(tmp_path: Path) -> None:
    """Check target eligibility and future-revision exclusion."""
    future = _NOW + timedelta(seconds=20)
    messages = (
        Message(
            id=MessageId("future-edit"),
            event_id=EventId("event-future-edit"),
            chat_id=_CHAT,
            author_id=_FRIEND,
            sent_at=_NOW,
            text="future edit text",
            edited_at=future,
        ),
        Message(
            id=MessageId("future-delete"),
            event_id=EventId("event-future-delete"),
            chat_id=_CHAT,
            author_id=_FRIEND,
            sent_at=_NOW + timedelta(seconds=1),
            deleted_at=future,
        ),
        Message(
            id=MessageId("same-time-before-by-id"),
            event_id=EventId("event-same-time"),
            chat_id=_CHAT,
            author_id=_FRIEND,
            sent_at=_NOW + timedelta(seconds=5),
            text="ambiguous same-time text",
        ),
        Message(
            id=MessageId("target-clean-one"),
            event_id=EventId("event-clean-one"),
            chat_id=_CHAT,
            author_id=_TARGET,
            sent_at=_NOW + timedelta(seconds=5),
            text="clean one",
        ),
        Message(
            id=MessageId("target-edited"),
            event_id=EventId("event-edited"),
            chat_id=_CHAT,
            author_id=_TARGET,
            sent_at=_NOW + timedelta(seconds=6),
            text="edited label",
            edited_at=_NOW + timedelta(seconds=7),
        ),
        Message(
            id=MessageId("target-attachment"),
            event_id=EventId("event-attachment"),
            chat_id=_CHAT,
            author_id=_TARGET,
            sent_at=_NOW + timedelta(seconds=8),
            text="caption",
            attachments=(Attachment("attachment"),),
        ),
        Message(
            id=MessageId("target-placeholder"),
            event_id=EventId("event-placeholder"),
            chat_id=_CHAT,
            author_id=_TARGET,
            sent_at=_NOW + timedelta(seconds=9),
            text=" \ufffc ",
        ),
        Message(
            id=MessageId("target-clean-two"),
            event_id=EventId("event-clean-two"),
            chat_id=_CHAT,
            author_id=_TARGET,
            sent_at=_NOW + timedelta(seconds=10),
            text="clean two",
        ),
    )
    builder = LocalBuilder(
        FakeRepository(messages),  # ty: ignore[invalid-argument-type]
        tmp_path / "data",
        clock=lambda: _NOW,
    )

    result = builder.build(
        _TARGET,
        " @Karsten ",
        DataConfig(context=8, valid_ratio=0, test_ratio=0),
    )

    rows = _read_jsonl(result.dataset.path / "train.jsonl")
    assert [row["completion"] for row in rows] == ["clean one", "clean two"]
    assert "future edit text" not in rows[0]["prompt"]
    assert "ambiguous same-time text" not in rows[0]["prompt"]
    assert "future edit text" not in rows[1]["prompt"]
    manifest = json.loads(
        (result.dataset.path / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["recipe"]["target_name"] == "Karsten"
    assert manifest["selection"] == {
        "attachment": 1,
        "deleted": 0,
        "edited": 1,
        "excluded": 3,
        "included": 2,
        "no_text": 0,
        "no_visible_text": 1,
        "target_messages": 5,
    }


def test_dataset_loader_rejects_unrecorded_files(tmp_path: Path) -> None:
    """Check that extra files cannot enter an immutable dataset."""
    builder = LocalBuilder(
        FakeRepository(_conversation(1)),  # ty: ignore[invalid-argument-type]
        tmp_path / "data",
        clock=lambda: _NOW,
    )
    result = builder.build(
        _TARGET,
        "Karsten",
        DataConfig(context=1, valid_ratio=0, test_ratio=0),
    )
    (result.dataset.path / "extra.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(DataError, match="files do not match"):
        load_dataset(result.dataset.path)


def test_dataset_id_commits_to_canonical_rows(tmp_path: Path) -> None:
    """Check a coordinated file and manifest change against the dataset ID."""
    builder = LocalBuilder(
        FakeRepository(_conversation(1)),  # ty: ignore[invalid-argument-type]
        tmp_path / "data",
        clock=lambda: _NOW,
    )
    result = builder.build(
        _TARGET,
        "Karsten",
        DataConfig(context=1, valid_ratio=0, test_ratio=0),
    )
    split_path = result.dataset.path / "train.jsonl"
    split_path.write_text(
        '{"prompt":"changed","completion":"changed"}\n',
        encoding="utf-8",
    )
    manifest_path = result.dataset.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["train.jsonl"] = hashlib.sha256(
        split_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DataError, match="identity does not match"):
        load_dataset(result.dataset.path)


def test_people_find_the_one_local_account() -> None:
    """Check target discovery without a phone number or display name."""
    messages = _conversation(2)

    people = summarize_people(messages)
    target = next(person for person in people if person.is_self)

    assert target.messages == 2
    assert target.name == "Karsten old"
    assert resolve_self(people) == _TARGET


def _conversation(target_count: int) -> tuple[Message, ...]:
    """Make one fixed conversation with a message before each target."""
    messages = []
    for index in range(target_count):
        friend_time = _NOW + timedelta(seconds=index * 2)
        target_time = friend_time + timedelta(seconds=1)
        messages.append(
            Message(
                id=MessageId(f"friend-message-{index:02d}"),
                event_id=EventId(f"friend-event-{index:02d}"),
                chat_id=_CHAT,
                author_id=_FRIEND,
                sent_at=friend_time,
                author_name="Friend",
                text=f"￼ ping friend-{index:02d}",
                mentions=(Mention(_TARGET, 0, 1, "Karsten old"),),
            )
        )
        messages.append(
            Message(
                id=MessageId(f"target-message-{index:02d}"),
                event_id=EventId(f"target-event-{index:02d}"),
                chat_id=_CHAT,
                author_id=_TARGET,
                sent_at=target_time,
                author_name="Karsten old",
                is_self=True,
                text=f"target-{index:02d}",
            )
        )
    return tuple(messages)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one generated JSON Lines file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
