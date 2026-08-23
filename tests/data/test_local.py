"""Test local response-episode dataset construction."""

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
from idiolect.prompt import MESSAGE_BOUNDARY, split_bubbles
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
    """Check episode rendering, chronological splits, and content addressing."""
    messages = _conversation(10)
    builder = LocalBuilder(
        FakeRepository(messages),  # ty: ignore[invalid-argument-type]
        tmp_path / "data",
        clock=lambda: _NOW,
    )
    config = DataConfig(context=4, valid_ratio=0.2, test_ratio=0.2)

    first = builder.build(_TARGET, "DIXIE", config)
    second = builder.build(_TARGET, "DIXIE", config)

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
    for items in values.values():
        for item in items:
            assert MESSAGE_BOUNDARY not in item["completion"]
    split_tokens = {
        split: set(re.findall(r"(?:friend|target)-\d{2}", json.dumps(items)))
        for split, items in values.items()
    }
    assert split_tokens[Split.TRAIN].isdisjoint(split_tokens[Split.VALID])
    assert split_tokens[Split.TRAIN].isdisjoint(split_tokens[Split.TEST])
    assert split_tokens[Split.VALID].isdisjoint(split_tokens[Split.TEST])
    assert "mentions @DIXIE" in values[Split.TRAIN][0]["prompt"]
    assert "@DIXIE ping friend-00" in values[Split.TRAIN][0]["prompt"]
    assert "[next response]" in values[Split.TRAIN][0]["prompt"]
    assert "reply to" not in values[Split.TRAIN][0]["prompt"]
    assert "￼" not in json.dumps(values, ensure_ascii=False)
    assert "target-id" not in json.dumps(values)
    manifest = json.loads(
        (first.dataset.path / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["recipe"]["unit"] == "response-episode-v1"
    assert manifest["recipe"]["burst_gap_seconds"] == 120.0
    assert manifest["recipe"]["bubble_boundary"] == MESSAGE_BOUNDARY
    assert manifest["recipe"]["diagnostics"]["response_episodes"] >= 10
    assert manifest["counts"] == {"test": 2, "train": 6, "valid": 2}
    index = _read_jsonl(first.dataset.path / "index.jsonl")
    assert len(index) == 10
    assert index[0]["target_message_ids"] == ["target-message-00"]

    (first.dataset.path / "train.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(DataError, match="does not match its manifest"):
        builder.build(_TARGET, "DIXIE", config)


def test_builder_builds_one_example_per_response_episode(tmp_path: Path) -> None:
    """Check that one three-bubble burst trains as one row."""
    question = _message("question", _FRIEND, 10, text="what do you think?")
    burst = tuple(
        _message(f"bubble-{index}", _TARGET, 11 + index, text=f"part-{index}")
        for index in range(3)
    )
    builder = LocalBuilder(
        FakeRepository((question, *burst)),  # ty: ignore[invalid-argument-type]
        tmp_path / "data",
        clock=lambda: _NOW,
    )

    result = builder.build(
        _TARGET,
        "DIXIE",
        DataConfig(context=4, valid_ratio=0, test_ratio=0),
    )

    rows = _read_jsonl(result.dataset.path / "train.jsonl")
    assert result.counts == {Split.TRAIN: 1, Split.VALID: 0, Split.TEST: 0}
    assert [row["completion"] for row in rows] == [
        f"part-0\n{MESSAGE_BOUNDARY}\npart-1\n{MESSAGE_BOUNDARY}\npart-2"
    ]
    assert split_bubbles(rows[0]["completion"]) == ("part-0", "part-1", "part-2")
    index = _read_jsonl(result.dataset.path / "index.jsonl")[0]
    assert index["target_message_ids"] == [
        "bubble-0",
        "bubble-1",
        "bubble-2",
    ]
    assert index["context_message_ids"] == ["question"]
    assert index["reply_parent_message_id"] is None


def test_burst_gap_configuration_governs_grouping_and_identity(tmp_path: Path) -> None:
    """Check that the configured threshold changes grouping and dataset ID."""
    messages = (
        _message("one", _TARGET, 0, text="one"),
        _message("two", _TARGET, 130, text="two"),
    )

    def build(burst_gap_seconds: float, root: str) -> Any:
        return LocalBuilder(
            FakeRepository(messages),  # ty: ignore[invalid-argument-type]
            tmp_path / root,
            clock=lambda: _NOW,
        ).build(
            _TARGET,
            "DIXIE",
            DataConfig(
                context=4,
                valid_ratio=0,
                test_ratio=0,
                burst_gap_seconds=burst_gap_seconds,
            ),
        )

    split_result = build(120, "split")
    merged_result = build(200, "merged")

    assert sum(split_result.counts.values()) == 2
    assert sum(merged_result.counts.values()) == 1
    assert split_result.dataset.id != merged_result.dataset.id


def test_builder_excludes_unusable_episodes_whole(tmp_path: Path) -> None:
    """Check eligibility and future-revision exclusion at episode level."""
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
        " @DIXIE ",
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
    assert manifest["recipe"]["target_name"] == "DIXIE"
    # One unusable bubble ends the current clean run without poisoning the
    # surrounding bursts: the usable bubbles remain separate episodes.
    assert manifest["selection"] == {
        "attachment": 1,
        "deleted": 0,
        "edited": 1,
        "no_text": 0,
        "no_visible_text": 1,
        "target_episodes": 1,
        "included": 2,
        "unusable_episodes": 0,
        "authored_messages": 5,
        "episode_messages_included": 2,
        "episode_messages_excluded": 3,
    }


def test_delayed_native_reply_keeps_its_parent_in_context(tmp_path: Path) -> None:
    """Check that hours-old reply parents survive recency truncation."""
    parent = _message("parent", _FRIEND, 0, text="should we use gemma?")
    filler = tuple(
        _message(f"filler-{index:02d}", PersonId(f"person-{index:02d}"), 60 + index * 60)
        for index in range(8)
    )
    reply = _message(
        "reply",
        _TARGET,
        60 + 8 * 60 + 3 * 3600,
        text="probably not",
        reply_to=MessageId("parent"),
    )
    builder = LocalBuilder(
        FakeRepository((parent, *filler, reply)),  # ty: ignore[invalid-argument-type]
        tmp_path / "data",
        clock=lambda: _NOW,
    )

    result = builder.build(
        _TARGET,
        "DIXIE",
        DataConfig(context=2, valid_ratio=0, test_ratio=0),
    )

    row = _read_jsonl(result.dataset.path / "train.jsonl")[0]
    index_row = _read_jsonl(result.dataset.path / "index.jsonl")[0]
    # The recency window holds only the two newest episodes; the hours-old
    # reply parent is anchored into the context ahead of them.
    assert "should we use gemma?" in row["prompt"]
    assert index_row["context_message_ids"] == [
        "parent",
        "filler-06",
        "filler-07",
    ]
    assert index_row["reply_parent_message_id"] == "parent"
    assert index_row["thread_anchor_message_ids"] == ["parent"]


def test_recency_budget_keeps_whole_episodes_and_drops_older_context(
    tmp_path: Path,
) -> None:
    """Check that the message budget never divides a context episode."""
    anchor = _message("anchor", _FRIEND, 0, text="anchor question")
    answer = _message(
        "answer",
        _TARGET,
        30,
        text="first answer",
        reply_to=MessageId("anchor"),
    )
    unrelated = tuple(
        _message(f"noise-{index}", PersonId(f"noise-{index}"), 60 + index)
        for index in range(3)
    )
    second = _message("second", _TARGET, 100, text="second answer")
    builder = LocalBuilder(
        FakeRepository((anchor, answer, *unrelated, second)),  # ty: ignore[invalid-argument-type]
        tmp_path / "data",
        clock=lambda: _NOW,
    )

    result = builder.build(
        _TARGET,
        "DIXIE",
        DataConfig(context=2, valid_ratio=0, test_ratio=0),
    )

    index_rows = _read_jsonl(result.dataset.path / "index.jsonl")
    by_episode = {row["episode_id"]: row for row in index_rows}
    # The budget holds two single-message episodes; the anchored reply parent
    # bypasses the budget while unrelated older episodes are dropped first.
    assert by_episode["answer"]["context_message_ids"] == ["anchor"]
    assert by_episode["second"]["context_message_ids"] == ["noise-1", "noise-2"]
    assert by_episode["second"]["thread_anchor_message_ids"] == []


def test_context_never_depends_on_target_text(tmp_path: Path) -> None:
    """Check that changing only target text cannot change selected context."""
    distant = _message(
        "distant",
        _FRIEND,
        0,
        text="the calibration constant is xylophone-seven",
    )
    filler = tuple(
        _message(f"gap-{index}", _FRIEND, 3600 + index * 3600, text=f"filler {index}")
        for index in range(4)
    )
    plain_target = _message("target", _TARGET, 20000, text="a normal reply")
    leaking_target = _message(
        "target",
        _TARGET,
        20000,
        text="about xylophone-seven calibration constants",
    )

    def build(target: Message, root: str) -> list[list[str]]:
        builder = LocalBuilder(
            FakeRepository((distant, *filler, target)),  # ty: ignore[invalid-argument-type]
            tmp_path / root,
            clock=lambda: _NOW,
        )
        result = builder.build(
            _TARGET,
            "DIXIE",
            DataConfig(context=2, valid_ratio=0, test_ratio=0),
        )
        return [row["context_message_ids"] for row in _read_jsonl(
            result.dataset.path / "index.jsonl"
        )]

    neutral_contexts = build(plain_target, "neutral")
    loaded_contexts = build(leaking_target, "loaded")

    # Identical metadata must select identical context regardless of the
    # completion text, so lexical retrieval from Y cannot hide inside X.
    assert neutral_contexts == loaded_contexts
    assert "distant" not in neutral_contexts[0]


def test_split_boundaries_never_divide_a_response_episode(tmp_path: Path) -> None:
    """Check atomic split assignment for multi-message target episodes."""
    bursts = []
    for index in range(6):
        base = index * 1000
        bursts.append(_message(f"q-{index}", _FRIEND, base))
        bursts.extend(
            _message(f"b-{index}-{position}", _TARGET, base + 1 + position)
            for position in range(3)
        )
    builder = LocalBuilder(
        FakeRepository(tuple(bursts)),  # ty: ignore[invalid-argument-type]
        tmp_path / "data",
        clock=lambda: _NOW,
    )
    config = DataConfig(context=4, valid_ratio=0.34, test_ratio=0.33)

    result = builder.build(_TARGET, "DIXIE", config)

    index_rows = _read_jsonl(result.dataset.path / "index.jsonl")
    assignment: dict[str, str] = {}
    for row in index_rows:
        for message_id in row["target_message_ids"]:
            owner = assignment.setdefault(message_id, row["split"])
            assert owner == row["split"], "A response episode crossed a split boundary"
    multi = [row for row in index_rows if len(row["target_message_ids"]) == 3]
    assert len(multi) == len(index_rows)
    assert {row["split"] for row in index_rows} == {"train", "valid", "test"}


def test_identical_inputs_rebuild_byte_identical_artifacts(tmp_path: Path) -> None:
    """Check deterministic rebuilds across output directories."""
    messages = _conversation(6)
    config = DataConfig(context=4, valid_ratio=0.2, test_ratio=0.2)

    first = LocalBuilder(
        FakeRepository(messages),  # ty: ignore[invalid-argument-type]
        tmp_path / "one",
        clock=lambda: _NOW,
    ).build(_TARGET, "DIXIE", config)
    second = LocalBuilder(
        FakeRepository(messages),  # ty: ignore[invalid-argument-type]
        tmp_path / "two",
        clock=lambda: _NOW,
    ).build(_TARGET, "DIXIE", config)

    assert first.dataset.id == second.dataset.id
    first_bytes = {
        path.name: path.read_bytes() for path in sorted(first.dataset.path.iterdir())
    }
    second_bytes = {
        path.name: path.read_bytes() for path in sorted(second.dataset.path.iterdir())
    }
    assert first_bytes == second_bytes


def test_dataset_loader_rejects_unrecorded_files(tmp_path: Path) -> None:
    """Check that extra files cannot enter an immutable dataset."""
    builder = LocalBuilder(
        FakeRepository(_conversation(1)),  # ty: ignore[invalid-argument-type]
        tmp_path / "data",
        clock=lambda: _NOW,
    )
    result = builder.build(
        _TARGET,
        "DIXIE",
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
        "DIXIE",
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


def test_invalid_data_configuration_is_rejected(tmp_path: Path) -> None:
    """Check boundary validation of episode construction settings."""
    builder = LocalBuilder(
        FakeRepository(_conversation(2)),  # ty: ignore[invalid-argument-type]
        tmp_path / "data",
        clock=lambda: _NOW,
    )

    with pytest.raises(DataError, match="burst_gap_seconds"):
        builder.build(
            _TARGET,
            "DIXIE",
            DataConfig(valid_ratio=0, test_ratio=0, burst_gap_seconds=0),
        )


def test_people_find_the_one_local_account() -> None:
    """Check target discovery without a phone number or display name."""
    messages = _conversation(2)

    people = summarize_people(messages)
    target = next(person for person in people if person.is_self)

    assert target.messages == 2
    assert target.name == "DIXIE old"
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
                mentions=(Mention(_TARGET, 0, 1, "DIXIE old"),),
            )
        )
        messages.append(
            Message(
                id=MessageId(f"target-message-{index:02d}"),
                event_id=EventId(f"target-event-{index:02d}"),
                chat_id=_CHAT,
                author_id=_TARGET,
                sent_at=target_time,
                author_name="DIXIE old",
                is_self=True,
                text=f"target-{index:02d}",
            )
        )
    return tuple(messages)


def _message(
    value: str,
    author: PersonId,
    offset: int,
    *,
    text: str | None = "text",
    reply_to: MessageId | None = None,
) -> Message:
    """Make one fixed synthetic message."""
    return Message(
        id=MessageId(value),
        event_id=EventId(f"event-{value}"),
        chat_id=_CHAT,
        author_id=author,
        sent_at=_NOW + timedelta(seconds=offset),
        text=text,
        reply_to=reply_to,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one generated JSON Lines file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
