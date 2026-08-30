"""Define model conversation meaning without model token rules."""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from idiolect.config import TrainDataConfig


class PromptError(ValueError):
    """Report an invalid model text format."""


CONVERSATION_HEADER = "Conversation:"
NEXT_RESPONSE_MARKER = "next response"
MESSAGE_BOUNDARY = "[new message]"
BUBBLE_DELIMITER = f"\n{MESSAGE_BOUNDARY}\n"
_BOUNDARY_LINE = re.compile(
    rf"^[^\S\r\n]*{re.escape(MESSAGE_BOUNDARY)}[^\S\r\n]*\r?$",
    re.MULTILINE,
)
_BOUNDARY_DELIMITER = re.compile(
    rf"(?:\A|\r\n|\r|\n)[^\S\r\n]*{re.escape(MESSAGE_BOUNDARY)}"
    rf"[^\S\r\n]*(?=\r\n|\r|\n|\Z)"
)


@dataclass(frozen=True, slots=True)
class Turn:
    """Keep one model chat turn."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelInput:
    """Keep one formatted model input."""

    turns: tuple[Turn, ...]
    has_prefill: bool
    completion_role: str = "assistant"


@dataclass(frozen=True, slots=True)
class ModelExample:
    """Keep one formatted prompt and its target completion."""

    prompt: ModelInput
    completion: str


@dataclass(frozen=True, slots=True)
class ConversationEntry:
    """Keep one neutral rendered conversation entry."""

    header: str
    content: str | None = None


def conversation_instruction(target_name: str) -> str:
    """Return the fixed target instruction."""
    return f"You are {target_name}. Write only {target_name}'s next response."


def join_bubbles(texts: Sequence[str]) -> str:
    """Join one response episode's message bubbles into model text."""
    validate_bubbles(texts)
    return BUBBLE_DELIMITER.join(texts)


def split_bubbles(text: str) -> tuple[str, ...]:
    """Split one response episode at standalone message boundary lines."""
    segments = _BOUNDARY_DELIMITER.split(text)
    return tuple(
        segment if index == 0 else _remove_leading_line_end(segment)
        for index, segment in enumerate(segments)
    )


def _remove_leading_line_end(text: str) -> str:
    """Remove the line end that follows one message boundary."""
    if text.startswith("\r\n"):
        return text[2:]
    if text.startswith(("\r", "\n")):
        return text[1:]
    return text


def validate_bubbles(texts: Sequence[str]) -> None:
    """Verify that one episode serialization splits without ambiguity."""
    for text in texts:
        if _BOUNDARY_LINE.search(text):
            raise PromptError(f"Message text contains the reserved boundary: {text!r}")


def render_conversation(
    target_name: str,
    entries: tuple[ConversationEntry, ...],
    *,
    next_response: str = NEXT_RESPONSE_MARKER,
) -> str:
    """Render entries with the fixed training conversation grammar."""
    lines = [conversation_instruction(target_name), "", CONVERSATION_HEADER]
    for entry in entries:
        lines.extend(("", f"[{entry.header}]"))
        if entry.content is not None:
            lines.append(entry.content)
    lines.extend(("", f"[{next_response}]"))
    return "\n".join(lines)


def reply_metadata(author: str, content: str) -> str:
    """Return the fixed Signal-style reply header metadata."""
    return f"reply to {author}: {json.dumps(content, ensure_ascii=False)}"


def format_prompt(prompt: str, config: TrainDataConfig) -> ModelInput:
    """Format one prompt with the training data policy."""
    validate_prompt_config(config)
    content = f"{config.prompt_prefix}{prompt}{config.prompt_suffix}"
    turns = []
    if config.format == "completion":
        turns.append(Turn("user", content))
        completion_role = "assistant"
    else:
        if config.system_prompt:
            turns.append(Turn("system", config.system_prompt))
        turns.append(Turn(config.prompt_role, content))
        completion_role = config.completion_role
    has_prefill = bool(config.completion_prefix)
    if has_prefill:
        turns.append(Turn(completion_role, config.completion_prefix))
    return ModelInput(tuple(turns), has_prefill, completion_role)


def format_example(
    prompt: str,
    completion: str,
    config: TrainDataConfig,
) -> ModelExample:
    """Format one prompt and target with the model text policy."""
    return ModelExample(
        format_prompt(prompt, config),
        f"{completion}{config.completion_suffix}",
    )


def completed_turns(example: ModelExample) -> tuple[Turn, ...]:
    """Return the conversation turns for one completed example."""
    turns = list(example.prompt.turns)
    if example.prompt.has_prefill:
        final = turns[-1]
        turns[-1] = Turn(final.role, f"{final.content}{example.completion}")
    else:
        turns.append(Turn(example.prompt.completion_role, example.completion))
    return tuple(turns)


def validate_prompt_config(config: TrainDataConfig) -> None:
    """Verify one model text format."""
    if config.format not in {"chat", "completion"}:
        raise PromptError("Model text format must be chat or completion")
    if config.format == "chat":
        roles = {"system", "user", "assistant"}
        if config.prompt_role not in roles:
            raise PromptError("Model prompt role is not valid")
        if config.completion_role not in roles:
            raise PromptError("Model completion role is not valid")
        return
    if config.system_prompt or config.prompt_role or config.completion_role:
        raise PromptError(
            "Completion text format cannot contain chat system or role values"
        )
