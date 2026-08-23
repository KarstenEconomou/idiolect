"""Test target-relative response-episode rendering."""

from datetime import UTC, datetime, timedelta

import pytest

from idiolect.data.render import RenderError, render_example
from idiolect.prompt import MESSAGE_BOUNDARY, split_bubbles
from idiolect.types import (
    Attachment,
    ChatId,
    EventId,
    Example,
    Mention,
    Message,
    MessageId,
    PersonId,
    Quote,
    Reaction,
    ResponseEpisode,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_CHAT = ChatId("chat")
_TARGET = PersonId("target")
_FRIEND = PersonId("friend")


def test_renderer_serializes_one_episode_with_message_boundaries() -> None:
    """Check that one target episode keeps its three Signal bubbles."""
    question = _message("question", _FRIEND, "what do you think?", 0)
    bubbles = (
        _message("bubble-one", _TARGET, "I don't know", 1),
        _message("bubble-two", _TARGET, "seems overfit to me", 2),
        _message("bubble-three", _TARGET, "especially on the style evals", 3),
    )
    example = Example((_episode((question,)),), _episode(bubbles))

    rendered = render_example(
        example,
        "DIXIE",
        {_FRIEND: "friend"},
    )

    assert rendered.completion == (
        "I don't know"
        f"\n{MESSAGE_BOUNDARY}\n"
        "seems overfit to me"
        f"\n{MESSAGE_BOUNDARY}\n"
        "especially on the style evals"
    )
    assert split_bubbles(rendered.completion) == (
        "I don't know",
        "seems overfit to me",
        "especially on the style evals",
    )
    assert "[next response]" in rendered.prompt
    assert "reply to" not in rendered.prompt


def test_renderer_keeps_a_single_bubble_unchanged() -> None:
    """Check that a normal one-message response stays one message."""
    context = _message("context", _FRIEND, "hello", 0)
    target = _episode((_message("target-message", _TARGET, "hi", 1),))

    rendered = render_example(
        Example((_episode((context,)),), target),
        "DIXIE",
        {_FRIEND: "friend"},
    )

    assert rendered.completion == "hi"
    assert MESSAGE_BOUNDARY not in rendered.completion


def test_renderer_groups_one_context_episode_into_one_entry() -> None:
    """Check speaker change, episode boundary, and bubble boundary differ."""
    first = _message("friend-one", _FRIEND, "one", 0)
    second = _message("friend-two", _FRIEND, "two", 60)
    other = _message("other", PersonId("other"), "intervening", 70)
    target = _message("target-message", _TARGET, "done", 80)

    rendered = render_example(
        Example(
            (_episode((first, second)), _episode((other,))),
            _episode((target,)),
        ),
        "DIXIE",
        {_FRIEND: "friend", PersonId("other"): "person_01"},
    )

    assert "[friend]\none\n[new message]\ntwo" in rendered.prompt
    assert "[person_01]\nintervening" in rendered.prompt


def test_renderer_preserves_name_and_native_addressing() -> None:
    """Check that plain names and native tags have related text and distinct data."""
    prior = _message("prior", _TARGET, "I'll check", 0)
    tagged = _message(
        "tagged",
        _FRIEND,
        "Hey @Old name, coming?",
        1,
        mentions=(Mention(_TARGET, 4, 9, "Old name"),),
        reply_to=prior.id,
        quote=Quote(_TARGET, prior.sent_at, "I'll check"),
    )
    plain = _message("plain", _FRIEND, "DIXIE, can you answer?", 2)
    target = _message(
        "target-message",
        _TARGET,
        "yeah @Old friend",
        3,
        mentions=(Mention(_FRIEND, 5, 11, "Old friend"),),
        reply_to=tagged.id,
        quote=Quote(
            _FRIEND,
            tagged.sent_at,
            tagged.text,
            tagged.mentions,
        ),
    )

    rendered = render_example(
        Example(
            (
                _episode((prior,)),
                _episode((tagged, plain)),
            ),
            _episode((target,)),
        ),
        "DIXIE",
        {_FRIEND: "friend"},
    )

    assert "You are DIXIE." in rendered.prompt
    assert (
        "[friend | mentions @DIXIE | reply to DIXIE: \"I'll check\"]" in rendered.prompt
    )
    assert "Hey @DIXIE, coming?" in rendered.prompt
    assert "DIXIE, can you answer?" in rendered.prompt
    assert rendered.completion == "yeah @friend"


def test_renderer_reports_the_reply_edge_of_the_episode_start() -> None:
    """Check that only the antecedent of the first bubble labels the entry."""
    anchor = _message("anchor", _TARGET, "the question", 0)
    first = _message(
        "reply-one",
        _FRIEND,
        "first part",
        1,
        reply_to=anchor.id,
        quote=Quote(_TARGET, anchor.sent_at, anchor.text),
    )
    second = _message("reply-two", _FRIEND, "second part", 2)
    target = _message("target-message", _TARGET, "done", 3)

    rendered = render_example(
        Example(
            (_episode((anchor,)), _episode((first, second))),
            _episode((target,)),
        ),
        "DIXIE",
        {_FRIEND: "friend"},
    )

    assert "[friend | reply to DIXIE: \"the question\"]" in rendered.prompt
    assert "first part\n[new message]\nsecond part" in rendered.prompt


def test_renderer_uses_identity_when_display_names_match() -> None:
    """Check that a same-name person does not become the target."""
    other = PersonId("other-dixie")
    context = _message(
        "context",
        _FRIEND,
        "Ask @DIXIE or DIXIE",
        0,
        mentions=(Mention(other, 4, 6, "DIXIE"),),
    )
    target = _message("target-message", _TARGET, "which one", 1)

    rendered = render_example(
        Example((_episode((context,)),), _episode((target,))),
        "DIXIE",
        {_FRIEND: "friend", other: "other_dixie"},
    )

    assert "mentions @DIXIE" not in rendered.prompt
    assert "Ask @other_dixie" in rendered.prompt
    assert "or DIXIE" in rendered.prompt


def test_renderer_requires_stable_non_target_names() -> None:
    """Check that a dataset cannot create unstable person labels."""
    context = _message("context", _FRIEND, "hello", 0)
    target = _message("target-message", _TARGET, "hi", 1)

    with pytest.raises(RenderError, match="stable pseudonym"):
        render_example(
            Example((_episode((context,)),), _episode((target,))),
            "DIXIE",
            {},
        )


def test_renderer_marks_media_that_accompanies_context_text() -> None:
    """Check that a caption does not hide its attachment context."""
    context = Message(
        id=MessageId("context"),
        event_id=EventId("event-context"),
        chat_id=_CHAT,
        author_id=_FRIEND,
        sent_at=_NOW,
        text="look at this",
        attachments=(Attachment("attachment"),),
    )
    target = _message("target-message", _TARGET, "wow", 1)

    rendered = render_example(
        Example((_episode((context,)),), _episode((target,))),
        "DIXIE",
        {_FRIEND: "friend"},
    )

    assert "[friend | 1 attachment]\nlook at this" in rendered.prompt


def test_renderer_interleaves_only_causal_reactions() -> None:
    """Check reaction timing and target-relative identity labels."""
    context_id = MessageId("context")
    context = Message(
        id=context_id,
        event_id=EventId("event-context"),
        chat_id=_CHAT,
        author_id=_FRIEND,
        sent_at=_NOW,
        text="news",
        reactions=(
            Reaction(
                EventId("reaction-before"),
                context_id,
                _CHAT,
                _TARGET,
                "👍",
                _NOW + timedelta(seconds=1),
            ),
            Reaction(
                EventId("reaction-after"),
                context_id,
                _CHAT,
                _FRIEND,
                "❌",
                _NOW + timedelta(seconds=30),
            ),
        ),
    )
    mid_bubble = _message("mid-bubble", _FRIEND, "more news", 5)
    target = _message("target-message", _TARGET, "nice", 10)

    rendered = render_example(
        Example(
            (_episode((context, mid_bubble)),),
            _episode((target,)),
        ),
        "DIXIE",
        {_FRIEND: "friend"},
    )

    assert "[DIXIE reacted \"👍\" to friend's message]" in rendered.prompt
    assert rendered.prompt.index('reacted "👍"') < rendered.prompt.index("more news")
    assert "❌" not in rendered.prompt


def test_renderer_rejects_reserved_boundary_text() -> None:
    """Check that source text cannot forge the serialization contract."""
    target = _message("target-message", _TARGET, f"one\n{MESSAGE_BOUNDARY}\ntwo", 0)

    with pytest.raises(RenderError, match="reserved boundary"):
        render_example(
            Example((), _episode((target,))),
            "DIXIE",
            {},
        )


def test_renderer_requires_text_in_every_target_bubble() -> None:
    """Check that media-only episodes cannot become completions."""
    target = Message(
        id=MessageId("target-message"),
        event_id=EventId("event-target-message"),
        chat_id=_CHAT,
        author_id=_TARGET,
        sent_at=_NOW,
        attachments=(Attachment("attachment"),),
    )

    with pytest.raises(RenderError, match="only text"):
        render_example(
            Example((), _episode((target,))),
            "DIXIE",
            {},
        )


def _message(
    value: str,
    author: PersonId,
    text: str | None,
    offset: int,
    *,
    mentions: tuple[Mention, ...] = (),
    reply_to: MessageId | None = None,
    quote: Quote | None = None,
) -> Message:
    """Make one fixed message for a rendering test."""
    return Message(
        id=MessageId(value),
        event_id=EventId(f"event-{value}"),
        chat_id=_CHAT,
        author_id=author,
        sent_at=_NOW + timedelta(seconds=offset),
        text=text,
        reply_to=reply_to,
        mentions=mentions,
        quote=quote,
    )


def _episode(messages: tuple[Message, ...]) -> ResponseEpisode:
    """Make one fixed response episode."""
    return ResponseEpisode(_CHAT, messages[0].author_id, messages)
