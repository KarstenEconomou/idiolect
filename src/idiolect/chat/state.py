"""Manage one in-memory interactive chat transcript."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Literal

from idiolect.chat.discovery import Assistant
from idiolect.config import ChatConfig, GenerationConfig
from idiolect.prompt import (
    ConversationEntry,
    ModelInput,
    format_prompt,
    render_conversation,
)


class ChatStateError(ValueError):
    """Report an invalid interactive chat state change."""


@dataclass(frozen=True, slots=True)
class TurnTelemetry:
    """Keep measured values for one assistant turn."""

    prompt_tokens: int
    generated_tokens: int
    prompt_throughput: float | None = None
    generation_throughput: float | None = None
    time_to_first_token: float | None = None
    generation_time: float | None = None
    peak_memory: float | None = None


@dataclass(frozen=True, slots=True)
class ChatTurn:
    """Keep one literal transcript turn."""

    role: Literal["user", "assistant"]
    content: str
    attempt: int = 0
    finish_reason: str | None = None
    seed: int | None = None
    telemetry: TurnTelemetry | None = None


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    """Keep one fitted and token-counted model prompt."""

    value: ModelInput
    prompt: str
    prompt_digest: str
    prompt_tokens: int
    seed: int
    dropped_messages: int


class ChatSession:
    """Own the mutable state for one local conversation."""

    def __init__(
        self,
        assistant: Assistant,
        chat: ChatConfig,
        generation: GenerationConfig,
        turns: tuple[ChatTurn, ...] = (),
        saved_chat_id: str | None = None,
        title: str | None = None,
    ) -> None:
        """Create one memory-only transcript."""
        _validate_turn_order(turns)
        self.assistant = assistant
        self.chat = chat
        self.generation = generation
        self.turns = list(turns)
        self.saved_chat_id = saved_chat_id
        self.title = title
        self.generating = False
        self._saved_fingerprint = self.fingerprint

    @property
    def dirty(self) -> bool:
        """Return true when the transcript differs from its saved state."""
        return self.fingerprint != self._saved_fingerprint

    @property
    def fingerprint(self) -> str:
        """Return one digest for the current transcript."""
        value = [asdict(turn) for turn in self.turns]
        return hashlib.sha256(_json_bytes(value)).hexdigest()

    def add_user(self, content: str) -> None:
        """Append one user message when no generation is active."""
        if self.generating:
            raise ChatStateError("A reply is already generating")
        if not content.strip():
            raise ChatStateError("A message must contain text")
        if self.turns and self.turns[-1].role != "assistant":
            raise ChatStateError(
                "The pending user message requires /retry before another message"
            )
        self.turns.append(ChatTurn("user", content))

    def begin_generation(self) -> int:
        """Start generation and return the active attempt number."""
        if self.generating:
            raise ChatStateError("A reply is already generating")
        if not self.turns or self.turns[-1].role != "user":
            raise ChatStateError("A reply requires a newest user message")
        self.generating = True
        return 0

    def finish_generation(
        self,
        content: str,
        finish_reason: str,
        seed: int,
        telemetry: TurnTelemetry,
        *,
        attempt: int = 0,
    ) -> None:
        """Append one complete or cancelled assistant reply."""
        if not self.generating:
            raise ChatStateError("No reply is generating")
        self.turns.append(
            ChatTurn(
                "assistant",
                content,
                attempt,
                finish_reason,
                seed,
                telemetry,
            )
        )
        self.generating = False

    def retry(self) -> int:
        """Remove the latest assistant reply and return its next attempt."""
        if self.generating:
            raise ChatStateError("Stop generation before retry")
        if not self.turns or self.turns[-1].role != "assistant":
            raise ChatStateError("There is no assistant reply to retry")
        previous = self.turns.pop()
        self.generating = True
        return previous.attempt + 1

    def mark_saved(self, chat_id: str, title: str | None) -> None:
        """Record the snapshot that matches the current transcript."""
        self.saved_chat_id = chat_id
        self.title = title
        self._saved_fingerprint = self.fingerprint


def prepare_prompt(
    state: ChatSession,
    tokenizer: Callable[[ModelInput], int],
    attempt: int,
) -> PreparedPrompt:
    """Build one training-shaped prompt and remove oldest whole messages."""
    if not state.turns or state.turns[-1].role != "user":
        raise ChatStateError("A prompt requires a newest user message")
    limit = state.assistant.context_messages
    selected = state.turns[-limit:]
    dropped = len(state.turns) - len(selected)
    while selected:
        prompt = render_conversation(
            state.assistant.target_name,
            tuple(
                ConversationEntry(
                    state.chat.participant_name
                    if turn.role == "user"
                    else state.assistant.target_name,
                    turn.content,
                )
                for turn in selected
            ),
        )
        model_input = format_prompt(prompt, state.assistant.run.data)
        token_count = tokenizer(model_input)
        if token_count <= state.generation.max_prompt_tokens:
            digest = _model_input_digest(model_input)
            return PreparedPrompt(
                model_input,
                prompt,
                digest,
                token_count,
                derive_seed(state.chat.seed, digest, attempt),
                dropped,
            )
        if len(selected) == 1:
            raise ChatStateError(
                "The newest user message exceeds max_prompt_tokens by itself"
            )
        selected = selected[1:]
        dropped += 1
    raise ChatStateError("A prompt requires a newest user message")


def derive_seed(chat_seed: int, prompt_digest: str, attempt: int) -> int:
    """Derive one stable 31-bit seed for a prompt attempt."""
    value = hashlib.sha256(f"{chat_seed}:{prompt_digest}:{attempt}".encode()).digest()
    return int.from_bytes(value[:8], "big") & 0x7FFF_FFFF


def replace_partial(turn: ChatTurn, content: str) -> ChatTurn:
    """Return an assistant turn with changed literal text."""
    if turn.role != "assistant":
        raise ChatStateError("Only an assistant reply can change")
    return replace(turn, content=content)


def _model_input_digest(value: ModelInput) -> str:
    payload = {
        "turns": [asdict(turn) for turn in value.turns],
        "has_prefill": value.has_prefill,
        "completion_role": value.completion_role,
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _validate_turn_order(turns: tuple[ChatTurn, ...]) -> None:
    if any(turn.role not in {"user", "assistant"} for turn in turns):
        raise ChatStateError("A transcript contains an invalid role")
    if turns and turns[0].role != "user":
        raise ChatStateError("A transcript must start with a user message")
    if any(
        turn.role == turns[index - 1].role
        for index, turn in enumerate(turns[1:], 1)
    ):
        raise ChatStateError("Transcript roles must alternate")


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
