"""Construct response episodes from normalized messages.

One response episode is one invocation-worth of behavior by one speaker. It
groups one or more consecutive Signal messages of that speaker into a single
conversational contribution while preserving every message boundary.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from idiolect.types import ChatId, Message, ResponseEpisode


class EpisodeError(ValueError):
    """Report invalid episode construction input."""


@dataclass(frozen=True, slots=True)
class BurstGapDiagnostics:
    """Keep the empirical distribution of same-author message gaps."""

    samples: int
    minimum_seconds: float | None
    maximum_seconds: float | None
    median_seconds: float | None
    p90_seconds: float | None
    p99_seconds: float | None


def build_episodes(
    messages: Sequence[Message],
    burst_gap_seconds: float,
) -> tuple[ResponseEpisode, ...]:
    """Group ordered messages into response episodes.

    Messages share an episode when they have one chat and one author, the gap
    to the previous message is at most ``burst_gap_seconds``, and the message
    does not carry a native reply to a message outside the current episode.
    Any message from another participant terminates the current episode.
    """
    _validate_gap(burst_gap_seconds)
    ordered = sorted(messages, key=_message_key)
    chats: dict[ChatId, list[ResponseEpisode]] = {}
    current: ResponseEpisode | None = None
    for message in ordered:
        members = () if current is None else current.messages
        if (
            current is not None
            and message.chat_id == current.chat_id
            and message.author_id == current.author_id
            and _gap_seconds(members[-1].sent_at, message.sent_at)
            <= burst_gap_seconds
            and (
                message.reply_to is None
                or message.reply_to in {item.id for item in members}
            )
        ):
            current = ResponseEpisode(
                current.chat_id,
                current.author_id,
                (*members, message),
            )
            chats[current.chat_id][-1] = current
        else:
            current = ResponseEpisode(
                message.chat_id,
                message.author_id,
                (message,),
            )
            chats.setdefault(message.chat_id, []).append(current)
    return tuple(
        episode
        for chat_id in sorted(chats, key=str)
        for episode in chats[chat_id]
    )


def burst_gap_samples(messages: Sequence[Message]) -> tuple[float, ...]:
    """Return sorted same-author gaps for messages adjacent in one chat.

    One sample is the delay between two consecutive messages of one author
    with no intervening message from another participant. These are exactly
    the delays the episode rules compare against ``burst_gap_seconds``.
    """
    ordered = sorted(messages, key=_message_key)
    samples: list[float] = []
    previous: Message | None = None
    for message in ordered:
        if (
            previous is not None
            and previous.chat_id == message.chat_id
            and previous.author_id == message.author_id
        ):
            samples.append(_gap_seconds(previous.sent_at, message.sent_at))
        previous = message
    return tuple(sorted(samples))


def gap_diagnostics(samples: Sequence[float]) -> BurstGapDiagnostics:
    """Summarize same-author gap samples for threshold calibration."""
    if not samples:
        return BurstGapDiagnostics(0, None, None, None, None, None)
    ordered = sorted(samples)

    def quantile(fraction: float) -> float:
        """Return one order-statistic quantile of the samples."""
        position = min(len(ordered) - 1, int(fraction * len(ordered)))
        return ordered[position]

    return BurstGapDiagnostics(
        len(ordered),
        ordered[0],
        ordered[-1],
        quantile(0.5),
        quantile(0.9),
        quantile(0.99),
    )


def _validate_gap(burst_gap_seconds: float) -> None:
    if not burst_gap_seconds > 0:
        raise EpisodeError("The burst gap must be greater than zero")


def _gap_seconds(first: datetime, second: datetime) -> float:
    return (second - first).total_seconds()


def _message_key(message: Message) -> tuple[datetime, str]:
    return message.sent_at, str(message.id)
