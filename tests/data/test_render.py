"""Test target-relative chat rendering."""

from datetime import UTC, datetime, timedelta

import pytest

from idiolect.data.render import RenderError, render_example
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
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_CHAT = ChatId("chat")
_TARGET = PersonId("target")
_FRIEND = PersonId("friend")


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
        Example((prior, tagged, plain), target),
        "DIXIE",
        {_FRIEND: "friend"},
    )

    assert "You are DIXIE." in rendered.prompt
    assert "[friend | mentions @DIXIE | reply to DIXIE: \"I'll check\"]" in rendered.prompt
    assert "Hey @DIXIE, coming?" in rendered.prompt
    assert "[friend]\nDIXIE, can you answer?" in rendered.prompt
    assert "[next response | reply to friend: \"Hey @DIXIE, coming?\"]" in rendered.prompt
    assert rendered.completion == "yeah @friend"


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
        Example((context,), target),
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
        render_example(Example((context,), target), "DIXIE", {})


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
        Example((context,), target),
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
                _NOW + timedelta(seconds=3),
            ),
        ),
    )
    target = _message("target-message", _TARGET, "nice", 2)

    rendered = render_example(
        Example((context,), target),
        "DIXIE",
        {_FRIEND: "friend"},
    )

    assert "[DIXIE reacted \"👍\" to friend's message]" in rendered.prompt
    assert "❌" not in rendered.prompt


def _message(
    value: str,
    author: PersonId,
    text: str,
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
