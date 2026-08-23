"""Render target-relative chat examples.

One example renders one target response episode. The context arrives as whole
response episodes, so model text distinguishes a speaker change, a
response-episode boundary (a new bracketed entry), and a Signal message
boundary inside one episode (the reserved message-boundary line).
"""

import json
from collections.abc import Mapping, Sequence
from datetime import datetime

from idiolect.prompt import (
    ConversationEntry,
    PromptError,
    join_bubbles,
    render_conversation,
    reply_metadata,
)
from idiolect.types import (
    ChatExample,
    Example,
    Mention,
    Message,
    MessageId,
    PersonId,
    Reaction,
    ResponseEpisode,
)


class RenderError(ValueError):
    """Report invalid example data."""


def render_example(
    example: Example,
    target_name: str,
    person_names: Mapping[PersonId, str],
) -> ChatExample:
    """Render one example from the target person's view."""
    name = normalize_person_name(target_name)
    target_id = example.target.author_id
    messages = {
        message.id: message
        for episode in example.context
        for message in episode.messages
    }
    entries: list[ConversationEntry] = []
    for item in _timeline(example):
        if isinstance(item, ResponseEpisode):
            rendered = _episode_lines(item, target_id, name, person_names, messages)
        else:
            rendered = _reaction_lines(item, target_id, name, person_names, messages)
        entries.append(
            ConversationEntry(
                rendered[1][1:-1], rendered[2] if len(rendered) == 3 else None
            )
        )

    bubbles = []
    for message in example.target.messages:
        if message.text is None:
            raise RenderError("The target episode must contain only text messages")
        bubbles.append(
            _render_text(message.text, message.mentions, target_id, name, person_names)
        )
    prompt = render_conversation(name, tuple(entries))
    return ChatExample(prompt=prompt, completion=_join_bubbles(bubbles))


def _join_bubbles(texts: Sequence[str]) -> str:
    try:
        return join_bubbles(tuple(texts))
    except PromptError as error:
        raise RenderError(str(error)) from error


def _episode_lines(
    episode: ResponseEpisode,
    target_id: PersonId,
    target_name: str,
    person_names: Mapping[PersonId, str],
    messages: Mapping[MessageId, Message],
) -> tuple[str, ...]:
    author = _person_name(episode.author_id, target_id, target_name, person_names)
    meta = [author]
    mentions = [mention for item in episode.messages for mention in item.mentions]
    if any(mention.person_id == target_id for mention in mentions):
        meta.append(f"mentions @{target_name}")
    attachments = sum(len(item.attachments) for item in episode.messages)
    has_text = any(item.text is not None for item in episode.messages)
    if attachments and has_text:
        label = "attachment" if attachments == 1 else "attachments"
        meta.append(f"{attachments} {label}")
    reply = _reply_text(episode.first, target_id, target_name, person_names, messages)
    if reply is not None:
        meta.append(reply)
    body = _join_bubbles(
        [
            _message_text(item, target_id, target_name, person_names)
            for item in episode.messages
        ]
    )
    return "", f"[{' | '.join(meta)}]", body


def _reaction_lines(
    reaction: Reaction,
    target_id: PersonId,
    target_name: str,
    person_names: Mapping[PersonId, str],
    messages: Mapping[MessageId, Message],
) -> tuple[str, ...]:
    author = _person_name(
        reaction.author_id,
        target_id,
        target_name,
        person_names,
    )
    message = messages.get(reaction.message_id)
    if message is None:
        subject = "an earlier message"
    else:
        subject_author = _person_name(
            message.author_id,
            target_id,
            target_name,
            person_names,
        )
        subject = f"{subject_author}'s message"
    value = json.dumps(reaction.value, ensure_ascii=False)
    if reaction.removed:
        action = f"removed {value} reaction from"
    else:
        action = f"reacted {value} to"
    return "", f"[{author} {action} {subject}]"


def _timeline(example: Example) -> tuple[ResponseEpisode | Reaction, ...]:
    context_messages = [
        message for episode in example.context for message in episode.messages
    ]
    context_ids = {message.id for message in context_messages}
    reactions = [
        reaction
        for message in context_messages
        for reaction in message.reactions
        if reaction.message_id in context_ids
        and reaction.sent_at < example.target.start_at
    ]
    values: list[ResponseEpisode | Reaction] = [*example.context, *reactions]
    return tuple(sorted(values, key=_timeline_key))


def _timeline_key(value: ResponseEpisode | Reaction) -> tuple[datetime, int, str]:
    # One key per item orders an episode by its end so that a reaction observed
    # during the episode renders before that whole contribution.
    if isinstance(value, ResponseEpisode):
        return value.end_at, 1, str(value.first.id)
    return value.sent_at, 0, str(value.event_id)


def _message_text(
    message: Message,
    target_id: PersonId,
    target_name: str,
    person_names: Mapping[PersonId, str],
) -> str:
    if message.deleted_at is not None:
        return "[deleted message]"
    if message.text is not None:
        return _render_text(
            message.text,
            message.mentions,
            target_id,
            target_name,
            person_names,
        )
    if message.attachments:
        return "[attachment]"
    return "[empty message]"


def _reply_text(
    message: Message,
    target_id: PersonId,
    target_name: str,
    person_names: Mapping[PersonId, str],
    messages: Mapping[MessageId, Message],
) -> str | None:
    if message.reply_to is None:
        return None
    original = messages.get(message.reply_to)
    author_id = message.quote.author_id if message.quote is not None else None
    if author_id is None and original is not None:
        author_id = original.author_id
    author = (
        "an earlier message"
        if author_id is None
        else _person_name(author_id, target_id, target_name, person_names)
    )

    quote_text = message.quote.text if message.quote is not None else None
    quote_mentions = message.quote.mentions if message.quote is not None else ()
    if quote_text is None and original is not None:
        quote_text = original.text
        quote_mentions = original.mentions
    if quote_text is not None:
        rendered = _render_text(
            quote_text,
            quote_mentions,
            target_id,
            target_name,
            person_names,
        )
        return reply_metadata(author, rendered)
    return f"reply to {author}"


def _render_text(
    text: str,
    mentions: Sequence[Mention],
    target_id: PersonId,
    target_name: str,
    person_names: Mapping[PersonId, str],
) -> str:
    validate_mentions(text, mentions)
    parts: list[str] = []
    cursor = 0
    for mention in sorted(mentions, key=lambda item: item.start_utf16):
        start = _utf16_index(text, mention.start_utf16)
        end = _utf16_index(text, mention.start_utf16 + mention.length_utf16)
        prefix = text[cursor:start]
        source = text[start:end]
        if prefix.endswith("@") and not source.startswith("@"):
            prefix = prefix[:-1]
        parts.append(prefix)
        name = _person_name(mention.person_id, target_id, target_name, person_names)
        parts.append(f"@{name}")
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _person_name(
    person_id: PersonId,
    target_id: PersonId,
    target_name: str,
    person_names: Mapping[PersonId, str],
) -> str:
    if person_id == target_id:
        return target_name
    configured = person_names.get(person_id)
    if configured is not None:
        return normalize_person_name(configured)
    raise RenderError("Add a stable pseudonym for each non-target person")


def normalize_person_name(value: str) -> str:
    """Return one safe and stable person name for model text."""
    name = " ".join(value.strip().lstrip("@").split())
    name = name.replace("[", "(").replace("]", ")").replace("|", "/")
    if not name:
        raise RenderError("A person name must contain text")
    return name


def validate_mentions(text: str | None, mentions: Sequence[Mention]) -> None:
    """Verify native mention ranges for one source text."""
    if not mentions:
        return
    if text is None:
        raise RenderError("Mentions require message text")
    cursor = 0
    for mention in sorted(mentions, key=lambda item: item.start_utf16):
        if mention.start_utf16 < 0 or mention.length_utf16 < 1:
            raise RenderError("A mention range is not valid for its message")
        start = _utf16_index(text, mention.start_utf16)
        end = _utf16_index(text, mention.start_utf16 + mention.length_utf16)
        if start < cursor:
            raise RenderError("A mention range is not valid for its message")
        cursor = end


def _utf16_index(text: str, units: int) -> int:
    used = 0
    for index, character in enumerate(text):
        if used == units:
            return index
        used += 2 if ord(character) > 0xFFFF else 1
        if used > units:
            raise RenderError("A mention range splits one Unicode character")
    if used == units:
        return len(text)
    raise RenderError("A mention range is outside its message")
