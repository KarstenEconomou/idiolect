"""Coordinate chat state with the supervised model worker."""

from collections.abc import Iterator
from dataclasses import dataclass

from idiolect.chat.discovery import Assistant
from idiolect.chat.state import ChatSession, TurnTelemetry, prepare_prompt
from idiolect.chat.worker import (
    CompleteEvent,
    DeltaEvent,
    DiagnosticEvent,
    FailureEvent,
    GenerateCommand,
    LoadCommand,
    ProbeCommand,
    ProbeEvent,
    StateEvent,
    WorkerError,
    WorkerState,
    WorkerSupervisor,
)
from idiolect.config import ChatConfig, GenerationConfig, InferConfig

_CHAT_KEYS = frozenset(
    {"output", "seed", "participant_name", "context_policy", "history"}
)
_GENERATION_KEYS = frozenset(
    {
        "backend",
        "max_prompt_tokens",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "min_tokens_to_keep",
        "repetition_penalty",
        "repetition_context_size",
    }
)


class ChatError(RuntimeError):
    """Report an invalid or failed chat operation."""


@dataclass(frozen=True, slots=True)
class RuntimeStats:
    """Keep measured session-level chat telemetry."""

    user_turns: int
    assistant_turns: int
    prompt_tokens: int
    generated_tokens: int
    weighted_generation_throughput: float | None


def validate_chat_policy(chat: ChatConfig, infer: InferConfig) -> None:
    """Verify complete chat and generation policy at the chat boundary."""
    if chat.unknown:
        raise ChatError(
            "Chat configuration has unknown values: " + ", ".join(sorted(chat.unknown))
        )
    missing = sorted(_CHAT_KEYS - chat.specified)
    missing.extend(
        f"infer.{name}" for name in sorted(_GENERATION_KEYS - infer.specified)
    )
    if missing:
        raise ChatError(f"Chat configuration is incomplete: {', '.join(missing)}")
    if chat.output is None:
        raise ChatError("Chat output is not configured")
    if chat.seed < 0:
        raise ChatError("Chat seed must not be negative")
    if not chat.participant_name.strip():
        raise ChatError("Chat participant_name must contain text")
    if any(character in "[]|\r\n" for character in chat.participant_name):
        raise ChatError("Chat participant_name contains a reserved character")
    if chat.context_policy != "recorded-window-drop-oldest":
        raise ChatError("Chat context_policy is not supported")
    if chat.history != "explicit-save":
        raise ChatError("Chat history policy is not supported")
    if infer.backend != "mlx-lm":
        raise ChatError("Chat backend must be mlx-lm")
    if infer.max_prompt_tokens < 1 or infer.max_tokens < 1:
        raise ChatError("Chat token limits must be greater than zero")


class ChatRuntime:
    """Run one assistant at a time with one isolated model worker."""

    def __init__(
        self,
        chat: ChatConfig,
        generation: GenerationConfig,
        worker_factory=WorkerSupervisor,
    ) -> None:
        """Set policies and the worker factory without loading a model."""
        self._configured_chat = chat
        self._configured_generation = generation
        self.chat = chat
        self.generation = generation
        self._worker_factory = worker_factory
        self.worker = None
        self.session: ChatSession | None = None
        self.state = WorkerState.PROBING
        self.probe: dict[str, object] = {}
        self.diagnostics: list[str] = []

    def select(self, assistant: Assistant) -> ChatSession:
        """Switch to one assistant and load its recorded model once."""
        if self.worker is not None:
            self.worker.shutdown()
        self.worker = self._worker_factory()
        self.worker.send(ProbeCommand())
        self.worker.send(LoadCommand(str(assistant.run.ref.path)))
        self.chat = self._configured_chat
        self.generation = self._configured_generation
        self.session = ChatSession(assistant, self.chat, self.generation)
        self._wait_ready()
        return self.session

    def attach(self, session: ChatSession) -> None:
        """Load the assistant for one resumed in-memory session."""
        if self.worker is not None:
            self.worker.shutdown()
        self.worker = self._worker_factory()
        self.worker.send(ProbeCommand())
        self.worker.send(LoadCommand(str(session.assistant.run.ref.path)))
        self.chat = session.chat
        self.generation = session.generation
        self.session = session
        self._wait_ready()

    def generate(self, attempt: int = 0) -> Iterator[str]:
        """Yield one reply and commit its final measured turn."""
        if self.worker is None or self.session is None:
            raise ChatError("Select an assistant before generation")
        if not self.session.generating:
            self.session.begin_generation()
        prepared = prepare_prompt(self.session, self.worker.count_tokens, attempt)
        self.worker.send(
            GenerateCommand(prepared.value, prepared.seed, self.generation)
        )
        pieces = []
        try:
            while True:
                event = self.worker.receive(1.0)
                if isinstance(event, DeltaEvent):
                    pieces.append(event.text)
                    yield event.text
                elif isinstance(event, StateEvent):
                    self.state = event.state
                elif isinstance(event, ProbeEvent):
                    self.probe.update(event.values)
                elif isinstance(event, DiagnosticEvent):
                    self.diagnostics.append(event.text)
                elif isinstance(event, FailureEvent):
                    raise ChatError(event.message)
                elif isinstance(event, CompleteEvent):
                    result = event.result
                    telemetry = TurnTelemetry(
                        result.prompt_tokens,
                        result.generated_tokens,
                        result.prompt_throughput,
                        result.generation_throughput,
                        event.time_to_first_token,
                        event.elapsed,
                        result.peak_memory,
                    )
                    self.session.finish_generation(
                        "".join(pieces),
                        result.finish_reason,
                        prepared.seed,
                        telemetry,
                        attempt=attempt,
                    )
                    return
        except BaseException:
            self.session.generating = False
            self.state = WorkerState.FAILED
            raise

    def cancel(self) -> None:
        """Request cancellation of the active reply."""
        if self.worker is not None:
            self.worker.cancel()

    def reload(self) -> None:
        """Restart and reload the selected assistant after failure."""
        if self.session is None:
            raise ChatError("Select an assistant before reload")
        self.attach(self.session)

    def close(self) -> None:
        """Release the worker process."""
        if self.worker is not None:
            self.worker.shutdown()
            self.worker = None

    @property
    def stats(self) -> RuntimeStats:
        """Return measured aggregate values for the current transcript."""
        turns = () if self.session is None else tuple(self.session.turns)
        telemetry = [turn.telemetry for turn in turns if turn.telemetry is not None]
        generated = sum(value.generated_tokens for value in telemetry)
        weighted = None
        rates = [
            value for value in telemetry if value.generation_throughput is not None
        ]
        if rates and generated:
            rated_tokens = sum(value.generated_tokens for value in rates)
            if rated_tokens:
                weighted = (
                    sum(
                        value.generated_tokens * value.generation_throughput
                        for value in rates
                        if value.generation_throughput is not None
                    )
                    / rated_tokens
                )
        return RuntimeStats(
            sum(turn.role == "user" for turn in turns),
            sum(turn.role == "assistant" for turn in turns),
            sum(value.prompt_tokens for value in telemetry),
            generated,
            weighted,
        )

    def _wait_ready(self) -> None:
        if self.worker is None:
            raise ChatError("The model worker is not running")
        while True:
            event = self.worker.receive(1.0)
            if isinstance(event, StateEvent):
                self.state = event.state
                if event.state == WorkerState.READY:
                    return
            elif isinstance(event, ProbeEvent):
                self.probe.update(event.values)
            elif isinstance(event, DiagnosticEvent):
                self.diagnostics.append(event.text)
            elif isinstance(event, FailureEvent):
                raise ChatError(event.message)
            elif event is None and not self.worker.alive:
                raise WorkerError("The model worker stopped during load")
