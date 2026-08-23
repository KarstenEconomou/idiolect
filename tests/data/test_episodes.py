"""Test structural response-episode construction."""

from datetime import UTC, datetime, timedelta

import pytest

from idiolect.data.episodes import (
    EpisodeError,
    build_episodes,
    burst_gap_samples,
    gap_diagnostics,
)
from idiolect.types import ChatId, EventId, Message, MessageId, PersonId

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_CHAT = ChatId("chat")
_TARGET = PersonId("target-id")
_FRIEND = PersonId("friend-id")


def test_burst_of_one_author_becomes_one_episode() -> None:
    """Check that consecutive same-author messages form one response episode."""
    messages = (
        _message("question", _FRIEND, 0),
        _message("one", _TARGET, 1),
        _message("two", _TARGET, 2),
        _message("three", _TARGET, 3),
    )

    episodes = build_episodes(messages, 120)

    assert [(episode.author_id, len(episode.messages)) for episode in episodes] == [
        (_FRIEND, 1),
        (_TARGET, 3),
    ]
    burst = episodes[-1]
    assert burst.message_ids == tuple(
        MessageId(value) for value in ("one", "two", "three")
    )
    assert burst.start_at == _NOW + timedelta(seconds=1)
    assert burst.end_at == _NOW + timedelta(seconds=3)


def test_message_from_another_participant_terminates_the_episode() -> None:
    """Check that an intervening message always splits the target response."""
    messages = (
        _message("one", _TARGET, 0),
        _message("ack", _FRIEND, 1),
        _message("two", _TARGET, 2),
    )

    episodes = build_episodes(messages, 120)

    assert [episode.author_id for episode in episodes] == [
        _TARGET,
        _FRIEND,
        _TARGET,
    ]
    assert all(len(episode.messages) == 1 for episode in episodes)


def test_burst_gap_exceeded_starts_a_new_episode() -> None:
    """Check that a long same-author delay is not one invocation."""
    messages = (
        _message("one", _TARGET, 0),
        _message("two", _TARGET, 121),
    )

    episodes = build_episodes(messages, 120)

    assert len(episodes) == 2
    assert len(build_episodes(messages, 121)) == 1


def test_explicit_reply_to_another_antecedent_prevents_merging() -> None:
    """Check that incompatible native reply edges split rapid bubbles."""
    antecedent = _message("antecedent", PersonId("other"), 0)
    first = _message("one", _TARGET, 1)
    second = _message(
        "two",
        _TARGET,
        2,
        reply_to=MessageId("antecedent"),
    )

    episodes = build_episodes((antecedent, first, second), 120)

    targets = [episode for episode in episodes if episode.author_id == _TARGET]
    assert len(targets) == 2


def test_reply_inside_the_current_episode_stays_merged() -> None:
    """Check that a self-referencing bubble remains part of one episode."""
    one = _message("one", _TARGET, 0)
    two = _message("two", _TARGET, 1, reply_to=MessageId("one"))

    episodes = build_episodes((one, two), 120)

    assert len(episodes) == 1
    assert len(episodes[0].messages) == 2


def test_episodes_are_separate_per_chat_and_chronological() -> None:
    """Check chat isolation and deterministic within-chat order."""
    other_chat = ChatId("chat-two")
    messages = (
        _message("late", _TARGET, 300),
        _message("early", _TARGET, 0),
        _message("other-late", _TARGET, 310, chat=other_chat),
        _message("other-early", _TARGET, 10, chat=other_chat),
    )

    episodes = build_episodes(messages, 120)

    starts: dict[ChatId, list[datetime]] = {}
    for episode in episodes:
        starts.setdefault(episode.chat_id, []).append(episode.start_at)
    assert starts[_CHAT] == [_NOW, _NOW + timedelta(seconds=300)]
    assert starts[other_chat] == [
        _NOW + timedelta(seconds=10),
        _NOW + timedelta(seconds=310),
    ]


def test_gap_samples_cover_only_adjacent_same_author_pairs() -> None:
    """Check the calibration sample definition."""
    messages = (
        _message("one", _TARGET, 0),
        _message("two", _TARGET, 30),
        _message("interrupt", _FRIEND, 40),
        _message("three", _TARGET, 50),
    )

    samples = burst_gap_samples(messages)

    assert samples == (30.0,)
    diagnostics = gap_diagnostics(samples)
    assert diagnostics.samples == 1
    assert diagnostics.median_seconds == 30.0
    assert diagnostics.maximum_seconds == 30.0
    assert gap_diagnostics(()).samples == 0


def test_non_positive_burst_gap_is_rejected() -> None:
    """Check that the threshold must be a positive duration."""
    with pytest.raises(EpisodeError):
        build_episodes((_message("one", _TARGET, 0),), 0)


def _message(
    value: str,
    author: PersonId,
    offset: int,
    *,
    chat: ChatId = _CHAT,
    reply_to: MessageId | None = None,
) -> Message:
    """Make one fixed text message."""
    return Message(
        id=MessageId(value),
        event_id=EventId(f"event-{value}"),
        chat_id=chat,
        author_id=author,
        sent_at=_NOW + timedelta(seconds=offset),
        text=f"text-{value}",
        reply_to=reply_to,
    )
