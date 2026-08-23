"""Coordinate chat state with the supervised model worker."""

import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, replace

from idiolect.chat.discovery import Assistant
from idiolect.chat.state import ChatSession, TurnTelemetry, prepare_prompt
from idiolect.chat.worker import (
    CompleteEvent,
    DeltaEvent,
    DiagnosticEvent,
    FailureEvent,
    GenerateCommand,
    LoadBaseCommand,
    LoadCommand,
    PrefillEvent,
    ProbeCommand,
    ProbeEvent,
    StateEvent,
    WorkerError,
    WorkerState,
    WorkerSupervisor,
)
from idiolect.config import ChatConfig, GenerationConfig, InferenceConfig, TrainConfig
from idiolect.prompt import PromptError, validate_prompt_config

_CHAT_KEYS = frozenset(
    {
        "output",
        "seed",
        "participant_name",
        "context_policy",
        "history",
        "default_model",
        "default_name",
        "default_context_messages",
        "default_system_prompt",
    }
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


def validate_chat_policy(
    chat: ChatConfig,
    inference: InferenceConfig,
    train: TrainConfig,
) -> None:
    """Verify complete chat and generation policy at the chat boundary."""
    if chat.unknown:
        raise ChatError(
            "Chat configuration has unknown values: " + ", ".join(sorted(chat.unknown))
        )
    missing = sorted(_CHAT_KEYS - chat.specified)
    missing.extend(
        f"inference.{name}" for name in sorted(_GENERATION_KEYS - inference.specified)
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
    if chat.default_model != "train-base":
        raise ChatError("Chat default_model must be train-base")
    if not chat.default_name.strip():
        raise ChatError("Chat default_name must contain text")
    if any(character in "[]|@\r\n" for character in chat.default_name):
        raise ChatError("Chat default_name contains a reserved character")
    if chat.default_context_messages < 1:
        raise ChatError("Chat default_context_messages must be greater than zero")
    if not chat.default_system_prompt.strip():
        raise ChatError("Chat default_system_prompt must contain text")
    if not train.base_model or not train.model_source:
        raise ChatError("Chat default model is not configured in [train]")
    if train.model_source not in {"hub", "path"}:
        raise ChatError("Chat default model_source must be hub or path")
    if train.model_source == "hub" and not train.model_revision:
        raise ChatError("Chat default hub model requires model_revision")
    if train.model_source == "hub" and train.model_cache is None:
        raise ChatError("Chat default hub model requires model_cache")
    try:
        validate_prompt_config(
            replace(train.data, system_prompt=chat.default_system_prompt)
        )
    except PromptError as error:
        raise ChatError(str(error)) from error
    if inference.backend != "mlx-lm":
        raise ChatError("Chat backend must be mlx-lm")
    if inference.max_prompt_tokens < 1 or inference.max_tokens < 1:
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
        # Start the process before Textual takes control of the terminal. Model
        # resolution and loading still start only after assistant selection.
        self.worker = self._worker_factory()
        self.session: ChatSession | None = None
        self.state = WorkerState.PROBING
        self.probe: dict[str, object] = {}
        self.diagnostics: list[str] = []

    def select(self, assistant: Assistant) -> ChatSession:
        """Switch to one assistant and load its recorded model once."""
        self._ensure_worker()
        if self.worker is None:
            raise ChatError("The model worker is not running")
        self.worker.send(ProbeCommand())
        self.worker.send(_load_command(assistant))
        self.chat = self._configured_chat
        self.generation = self._configured_generation
        self.session = ChatSession(assistant, self.chat, self.generation)
        self._wait_ready()
        return self.session

    def attach(self, session: ChatSession) -> None:
        """Load the assistant for one resumed in-memory session."""
        self._ensure_worker()
        if self.worker is None:
            raise ChatError("The model worker is not running")
        self.worker.send(ProbeCommand())
        self.worker.send(_load_command(session.assistant))
        self.chat = session.chat
        self.generation = session.generation
        self.session = session
        self._wait_ready()

    def generate(
        self,
        attempt: int = 0,
        prompt_progress: Callable[[int, int], None] | None = None,
    ) -> Generator[str]:
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
                elif isinstance(event, PrefillEvent):
                    if prompt_progress is not None:
                        prompt_progress(event.current, event.total)
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
        except ChatError:
            self.session.generating = False
            self.state = WorkerState.FAILED
            raise
        except BaseException:
            # The consumer abandoned or interrupted this reply. Stop the worker
            # and drain its events so a later reply cannot consume them.
            self.session.generating = False
            self._abandon_reply()
            raise

    def cancel(self) -> None:
        """Request cancellation of the active reply."""
        if self.worker is not None:
            self.worker.cancel()

    def _abandon_reply(self, timeout: float = 30.0) -> None:
        """Cancel the active reply and drain leftover worker events."""
        if self.worker is None:
            return
        self.worker.cancel()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            event = self.worker.receive(1.0)
            if isinstance(event, CompleteEvent):
                self.state = WorkerState.CANCELLED
                return
            if isinstance(event, FailureEvent):
                break
            if event is None and not self.worker.alive:
                break
        self.state = WorkerState.FAILED

    def reload(self) -> None:
        """Restart and reload the selected assistant after failure."""
        if self.session is None:
            raise ChatError("Select an assistant before reload")
        self.attach(self.session)

    def ensure_worker(self) -> None:
        """Start a replacement worker before background model loading starts."""
        self._ensure_worker()

    def close(self) -> None:
        """Release the worker process."""
        if self.worker is not None:
            self.worker.shutdown()
            self.worker = None

    @property
    def backend_versions(self) -> dict[str, str | None]:
        """Return the recorded backend runtime versions, when probed."""
        return {
            name: value
            for name in ("mlx_version", "mlx_lm_version")
            if (value := self.probe.get(name)) is None
            or isinstance(value, str)
        }

    def _ensure_worker(self) -> None:
        if self.worker is None or not self.worker.alive:
            self.worker = self._worker_factory()

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
                    self._record_model_digest()
                    return
            elif isinstance(event, ProbeEvent):
                self.probe.update(event.values)
            elif isinstance(event, DiagnosticEvent):
                self.diagnostics.append(event.text)
            elif isinstance(event, FailureEvent):
                raise ChatError(event.message)
            elif event is None and not self.worker.alive:
                raise WorkerError("The model worker stopped during load")

    def _record_model_digest(self) -> None:
        if self.session is None or self.session.assistant.base_model is None:
            return
        value = self.probe.get("model_digest")
        if isinstance(value, str):
            self.session.assistant = replace(
                self.session.assistant,
                base_model_digest=value,
            )


def _load_command(assistant: Assistant) -> LoadCommand | LoadBaseCommand:
    if assistant.run is not None:
        return LoadCommand(str(assistant.run.ref.path))
    if assistant.base_model is None:
        raise ChatError("The assistant model is not configured")
    return LoadBaseCommand(
        assistant.base_model,
        assistant.data,
        assistant.base_model_digest,
    )
