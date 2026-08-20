"""Test target-relative chat rendering."""

from datetime import UTC, datetime, timedelta

import pytest

from idiolect.data.render import RenderError, render_example
from idiolect.types import (
    ChatId,
    EventId,
    Example,
    Mention,
    Message,
    MessageId,
    PersonId,
    Quote,
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
    plain = _message("plain", _FRIEND, "Karsten, can you answer?", 2)
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
        "Karsten",
        {_FRIEND: "friend"},
    )

    assert "You are Karsten." in rendered.prompt
    assert "[friend | mentions @Karsten | reply to Karsten: \"I'll check\"]" in rendered.prompt
    assert "Hey @Karsten, coming?" in rendered.prompt
    assert "[friend]\nKarsten, can you answer?" in rendered.prompt
    assert "[next response | reply to friend: \"Hey @Karsten, coming?\"]" in rendered.prompt
    assert rendered.completion == "yeah @friend"


def test_renderer_uses_identity_when_display_names_match() -> None:
    """Check that a same-name person does not become the target."""
    other = PersonId("other-karsten")
    context = _message(
        "context",
        _FRIEND,
        "Ask @Karsten or Karsten",
        0,
        mentions=(Mention(other, 4, 8, "Karsten"),),
    )
    target = _message("target-message", _TARGET, "which one", 1)

    rendered = render_example(
        Example((context,), target),
        "Karsten",
        {_FRIEND: "friend", other: "other_karsten"},
    )

    assert "mentions @Karsten" not in rendered.prompt
    assert "Ask @other_karsten" in rendered.prompt
    assert "or Karsten" in rendered.prompt


def test_renderer_requires_stable_non_target_names() -> None:
    """Check that a dataset cannot create unstable person labels."""
    context = _message("context", _FRIEND, "hello", 0)
    target = _message("target-message", _TARGET, "hi", 1)

    with pytest.raises(RenderError, match="stable pseudonym"):
        render_example(Example((context,), target), "Karsten", {})


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
