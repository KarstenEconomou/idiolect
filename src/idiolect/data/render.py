"""Render target-relative chat examples."""

import json
from collections.abc import Mapping, Sequence
from datetime import datetime

from idiolect.prompt import ConversationEntry, render_conversation
from idiolect.types import (
    ChatExample,
    Example,
    Mention,
    Message,
    MessageId,
    PersonId,
    Reaction,
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
    messages = {message.id: message for message in example.context}
    entries: list[ConversationEntry] = []
    for item in _timeline(example):
        if isinstance(item, Message):
            rendered = _message_lines(item, target_id, name, person_names, messages)
        else:
            rendered = _reaction_lines(item, target_id, name, person_names, messages)
        entries.append(
            ConversationEntry(
                rendered[1][1:-1], rendered[2] if len(rendered) == 3 else None
            )
        )

    target_text = example.target.text
    if target_text is None:
        raise RenderError("The target message must contain text")
    target_meta = ["next response"]
    reply = _reply_text(example.target, target_id, name, person_names, messages)
    if reply is not None:
        target_meta.append(reply)
    prompt = render_conversation(
        name,
        tuple(entries),
        next_response=" | ".join(target_meta),
    )
    completion = _render_text(
        target_text,
        example.target.mentions,
        target_id,
        name,
        person_names,
    )
    return ChatExample(prompt=prompt, completion=completion)


def _message_lines(
    message: Message,
    target_id: PersonId,
    target_name: str,
    person_names: Mapping[PersonId, str],
    messages: Mapping[MessageId, Message],
) -> tuple[str, ...]:
    author = _person_name(message.author_id, target_id, target_name, person_names)
    meta = [author]
    if any(mention.person_id == target_id for mention in message.mentions):
        meta.append(f"mentions @{target_name}")
    if message.attachments and message.text is not None:
        label = "attachment" if len(message.attachments) == 1 else "attachments"
        meta.append(f"{len(message.attachments)} {label}")
    reply = _reply_text(message, target_id, target_name, person_names, messages)
    if reply is not None:
        meta.append(reply)
    body = _message_text(message, target_id, target_name, person_names)
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


def _timeline(example: Example) -> tuple[Message | Reaction, ...]:
    values: list[Message | Reaction] = list(example.context)
    context_ids = {message.id for message in example.context}
    for message in example.context:
        values.extend(
            reaction
            for reaction in message.reactions
            if reaction.message_id in context_ids
            and reaction.sent_at < example.target.sent_at
        )
    return tuple(sorted(values, key=_timeline_key))


def _timeline_key(value: Message | Reaction) -> tuple[datetime, int, str]:
    if isinstance(value, Message):
        return value.sent_at, 0, str(value.id)
    return value.sent_at, 1, str(value.event_id)


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
    if author_id is None:
        value = "reply to an earlier message"
    else:
        author = _person_name(author_id, target_id, target_name, person_names)
        value = f"reply to {author}"

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
        value = f"{value}: {json.dumps(rendered, ensure_ascii=False)}"
    return value


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
