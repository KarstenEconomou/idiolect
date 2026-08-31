"""Manage one in-memory interactive chat transcript."""

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Literal

from idiolect.artifact import canonical_json_bytes
from idiolect.chat.discovery import Assistant
from idiolect.config import ChatConfig, GenerationConfig
from idiolect.inference.base import TargetMode
from idiolect.prompt import (
    ConversationEntry,
    ModelInput,
    format_prompt,
    render_conversation,
    reply_metadata,
    split_bubbles,
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

    role: Literal["user", "assistant", "env"]
    content: str
    attempt: int = 0
    finish_reason: str | None = None
    seed: int | None = None
    telemetry: TurnTelemetry | None = None
    reference: int | None = None


@dataclass(frozen=True, slots=True)
class ChatBubble:
    """Keep one numbered message bubble in the chat transcript."""

    index: int
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    """Keep one fitted and token-counted model prompt."""

    value: ModelInput
    prompt: str
    prompt_digest: str
    prompt_tokens: int
    seed: int
    dropped_messages: int
    active_turns: int
    active_references: tuple[ChatBubble, ...]
    system_tokens: int = 0
    history_tokens: int = 0
    input_tokens: int = 0
    evicted_tokens: int = 0


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
        return hashlib.sha256(canonical_json_bytes(value)).hexdigest()

    @property
    def participant_name_editable(self) -> bool:
        """Return true before the first model-visible turn."""
        return not any(turn.role != "env" for turn in self.turns)

    def set_participant_name(self, value: str) -> None:
        """Set the prompt participant name before the conversation starts."""
        name = value.strip()
        if not self.participant_name_editable:
            raise ChatStateError("OP name cannot change after first turn")
        if not name:
            raise ChatStateError("OP name must contain text")
        if any(character in "[]|\r\n" for character in name):
            raise ChatStateError("OP name contains a reserved character")
        self.chat = replace(self.chat, participant_name=name)

    def add_user(self, content: str, reference: int | None = None) -> None:
        """Append one user message when no generation is active."""
        if self.generating:
            raise ChatStateError("A reply is already generating")
        if not content.strip():
            raise ChatStateError("A message must contain text")
        previous = _last_model_turn(self.turns)
        if previous is not None and previous.role != "assistant":
            raise ChatStateError(
                "The pending user message requires a retry before another message"
            )
        if reference is not None and (
            not isinstance(reference, int)
            or isinstance(reference, bool)
            or reference < 0
            or not any(
                bubble.index == reference for bubble in enumerate_bubbles(self.turns)
            )
        ):
            raise ChatStateError("A reference must target an earlier chat bubble")
        self.turns.append(ChatTurn("user", content, reference=reference))

    def add_env(self, content: str) -> None:
        """Append one local environment message outside model context."""
        if self.generating:
            raise ChatStateError("A reply is already generating")
        if not content.strip():
            raise ChatStateError("An ENV message must contain text")
        previous = _last_model_turn(self.turns)
        if previous is not None and previous.role == "user":
            raise ChatStateError(
                "The pending user message requires a reply before ENV output"
            )
        self.turns.append(ChatTurn("env", content))

    def begin_generation(self) -> int:
        """Start generation and return the active attempt number."""
        if self.generating:
            raise ChatStateError("A reply is already generating")
        previous = _last_model_turn(self.turns)
        if previous is None or previous.role != "user":
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
    if limit < 1:
        raise ChatStateError("The recorded context window is empty")
    model_turns = tuple(turn for turn in state.turns if turn.role != "env")
    selected = model_turns[-limit:]
    dropped = len(model_turns) - len(selected)
    bubbles = {bubble.index: bubble for bubble in enumerate_bubbles(state.turns)}
    while selected:
        prompt, model_input = _prompt_for_turns(state, selected, bubbles)
        token_count = tokenizer(model_input)
        if token_count <= state.generation.max_prompt_tokens:
            digest = _model_input_digest(model_input)
            _, system_input = _prompt_for_turns(state, (), bubbles)
            _, history_input = _prompt_for_turns(state, selected[:-1], bubbles)
            system_tokens = min(tokenizer(system_input), token_count)
            system_history_tokens = min(
                max(tokenizer(history_input), system_tokens),
                token_count,
            )
            evicted_tokens = 0
            if dropped:
                _, untrimmed_input = _prompt_for_turns(state, model_turns, bubbles)
                evicted_tokens = max(0, tokenizer(untrimmed_input) - token_count)
            return PreparedPrompt(
                value=model_input,
                prompt=prompt,
                prompt_digest=digest,
                prompt_tokens=token_count,
                seed=derive_seed(state.chat.seed, digest, attempt),
                dropped_messages=dropped,
                active_turns=len(selected),
                active_references=_active_references(state.turns, dropped),
                system_tokens=system_tokens,
                history_tokens=system_history_tokens - system_tokens,
                input_tokens=token_count - system_history_tokens,
                evicted_tokens=evicted_tokens,
            )
        if len(selected) == 1:
            raise ChatStateError(
                "The newest user message exceeds max_prompt_tokens by itself"
            )
        selected = selected[1:]
        dropped += 1
    raise ChatStateError("A prompt requires a newest user message")


def _prompt_for_turns(
    state: ChatSession,
    turns: tuple[ChatTurn, ...],
    bubbles: dict[int, ChatBubble],
) -> tuple[str, ModelInput]:
    """Return one formatted prompt for the specified model turns."""
    prompt = render_conversation(
        state.assistant.target_name,
        tuple(
            ConversationEntry(
                _entry_header(state, turn, bubbles),
                turn.content,
            )
            for turn in turns
        ),
    )
    return prompt, format_prompt(prompt, state.assistant.data)


def derive_seed(chat_seed: int, prompt_digest: str, attempt: int) -> int:
    """Derive one stable 31-bit seed for a prompt attempt."""
    value = hashlib.sha256(f"{chat_seed}:{prompt_digest}:{attempt}".encode()).digest()
    return int.from_bytes(value[:8], "big") & 0x7FFF_FFFF


def replace_partial(turn: ChatTurn, content: str) -> ChatTurn:
    """Return an assistant turn with changed literal text."""
    if turn.role != "assistant":
        raise ChatStateError("Only an assistant reply can change")
    return replace(turn, content=content)


def enumerate_bubbles(
    turns: tuple[ChatTurn, ...] | list[ChatTurn],
) -> tuple[ChatBubble, ...]:
    """Return stored transcript bubbles in chronological display order."""
    bubbles: list[ChatBubble] = []
    for turn in turns:
        if turn.role == "env":
            continue
        segments = (
            (turn.content,)
            if turn.role == "user"
            else tuple(
                segment for segment in split_bubbles(turn.content) if segment.strip()
            )
        )
        if not segments:
            segments = (turn.content,)
        bubbles.extend(
            ChatBubble(len(bubbles), turn.role, content) for content in segments
        )
    return tuple(bubbles)


def _active_references(
    turns: tuple[ChatTurn, ...] | list[ChatTurn],
    dropped_messages: int,
) -> tuple[ChatBubble, ...]:
    """Return numbered bubbles in one fitted model-message window."""
    active: list[ChatBubble] = []
    model_index = 0
    bubble_index = 0
    for turn in turns:
        if turn.role == "env":
            continue
        segments = (
            (turn.content,)
            if turn.role == "user"
            else tuple(
                segment for segment in split_bubbles(turn.content) if segment.strip()
            )
        )
        if not segments:
            segments = (turn.content,)
        if model_index >= dropped_messages:
            active.extend(
                ChatBubble(bubble_index + offset, turn.role, content)
                for offset, content in enumerate(segments)
            )
        bubble_index += len(segments)
        model_index += 1
    return tuple(active)


def _model_input_digest(value: ModelInput) -> str:
    payload = {
        "turns": [asdict(turn) for turn in value.turns],
        "has_prefill": value.has_prefill,
        "completion_role": value.completion_role,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _validate_turn_order(turns: tuple[ChatTurn, ...]) -> None:
    if any(turn.role not in {"user", "assistant", "env"} for turn in turns):
        raise ChatStateError("A transcript contains an invalid role")
    model_turns = tuple(turn for turn in turns if turn.role != "env")
    if model_turns and model_turns[0].role != "user":
        raise ChatStateError("A transcript must start with a user message")
    if any(
        turn.role == model_turns[index - 1].role
        for index, turn in enumerate(model_turns[1:], 1)
    ):
        raise ChatStateError("Transcript roles must alternate")
    bubbles = enumerate_bubbles(turns)
    indexes = {bubble.index for bubble in bubbles}
    current = 0
    for index, turn in enumerate(turns):
        if turn.role == "env":
            previous = _last_model_turn(turns[:index])
            if previous is not None and previous.role == "user":
                raise ChatStateError("An ENV turn cannot follow a pending user message")
            if (
                turn.attempt != 0
                or turn.finish_reason is not None
                or turn.seed is not None
                or turn.telemetry is not None
                or turn.reference is not None
            ):
                raise ChatStateError("An ENV turn cannot contain model metadata")
            continue
        if turn.role == "assistant" and turn.reference is not None:
            raise ChatStateError("Only a user turn can contain a reference")
        if turn.reference is not None and (
            not isinstance(turn.reference, int)
            or isinstance(turn.reference, bool)
            or turn.reference not in indexes
            or turn.reference >= current
        ):
            raise ChatStateError("A reference must target an earlier chat bubble")
        if turn.role == "user":
            current += 1
        else:
            segments = tuple(
                segment for segment in split_bubbles(turn.content) if segment.strip()
            )
            current += len(segments) or 1


def _last_model_turn(turns: list[ChatTurn] | tuple[ChatTurn, ...]) -> ChatTurn | None:
    """Return the newest user or assistant turn."""
    return next((turn for turn in reversed(turns) if turn.role != "env"), None)


def _entry_header(
    state: ChatSession,
    turn: ChatTurn,
    bubbles: dict[int, ChatBubble],
) -> str:
    """Return one prompt header with optional live Signal reply metadata."""
    author = (
        state.chat.participant_name
        if turn.role == "user"
        else state.assistant.target_name
    )
    if turn.reference is None or state.assistant.mode != TargetMode.RUN_ADAPTER:
        return author
    referenced = bubbles[turn.reference]
    referenced_author = (
        state.chat.participant_name
        if referenced.role == "user"
        else state.assistant.target_name
    )
    return f"{author} | {reply_metadata(referenced_author, referenced.content)}"
