"""Supervise the isolated local model worker process."""

import contextlib
import io
import multiprocessing
import queue
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from idiolect.config import GenerationConfig, TrainConfig, TrainDataConfig
from idiolect.inference.base import BackendResult
from idiolect.model import ModelSpec
from idiolect.prompt import ModelInput, Turn


class WorkerError(RuntimeError):
    """Report a failed or unavailable model worker."""


class WorkerState(StrEnum):
    """Name one visible worker state."""

    PROBING = "probing"
    RESOLVING = "resolving"
    VERIFYING = "verifying"
    LOADING = "loading"
    READY = "ready"
    GENERATING = "generating"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ProbeCommand:
    """Request local runtime information."""


@dataclass(frozen=True, slots=True)
class LoadCommand:
    """Request one verified adapter load."""

    run_path: str


@dataclass(frozen=True, slots=True)
class LoadBaseCommand:
    """Request one configured base-model load."""

    model: ModelSpec
    data: TrainDataConfig
    expected_digest: str | None = None


@dataclass(frozen=True, slots=True)
class GenerateCommand:
    """Request one streaming generation."""

    prompt: ModelInput
    seed: int
    config: GenerationConfig


@dataclass(frozen=True, slots=True)
class CountCommand:
    """Request the formatted token count for one prompt."""

    prompt: ModelInput


@dataclass(frozen=True, slots=True)
class CancelCommand:
    """Request cooperative generation cancellation."""


@dataclass(frozen=True, slots=True)
class UnloadCommand:
    """Request release of the loaded model."""


@dataclass(frozen=True, slots=True)
class ShutdownCommand:
    """Request a graceful worker shutdown."""


type WorkerCommand = (
    ProbeCommand
    | LoadCommand
    | LoadBaseCommand
    | CountCommand
    | GenerateCommand
    | CancelCommand
    | UnloadCommand
    | ShutdownCommand
)


@dataclass(frozen=True, slots=True)
class StateEvent:
    """Report one worker state change."""

    state: WorkerState


@dataclass(frozen=True, slots=True)
class RuntimeProbe:
    """Keep measured local hardware and runtime details."""

    mlx_version: str
    mlx_lm_version: str
    device: str
    architecture: str
    device_properties: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class LoadProbe:
    """Keep measurements for one verified model load."""

    model_digest: str
    model_size: int
    adapter_size: int | None
    load_duration: float


@dataclass(frozen=True, slots=True)
class ProbeEvent:
    """Report measured local hardware and runtime details."""

    probe: RuntimeProbe


@dataclass(frozen=True, slots=True)
class LoadEvent:
    """Report measurements for the current model load."""

    probe: LoadProbe


@dataclass(frozen=True, slots=True)
class DeltaEvent:
    """Report one generated text delta."""

    text: str


@dataclass(frozen=True, slots=True)
class PrefillEvent:
    """Report measured prompt prefill progress."""

    current: int
    total: int


@dataclass(frozen=True, slots=True)
class CountEvent:
    """Report one formatted prompt token count."""

    tokens: int


@dataclass(frozen=True, slots=True)
class CompleteEvent:
    """Report final generation metrics."""

    result: BackendResult
    time_to_first_token: float | None
    elapsed: float


@dataclass(frozen=True, slots=True)
class DiagnosticEvent:
    """Report captured backend diagnostics."""

    text: str


@dataclass(frozen=True, slots=True)
class FailureEvent:
    """Report one controlled or unexpected worker failure."""

    message: str


type WorkerEvent = (
    StateEvent
    | ProbeEvent
    | LoadEvent
    | CountEvent
    | PrefillEvent
    | DeltaEvent
    | CompleteEvent
    | DiagnosticEvent
    | FailureEvent
)


class WorkerSupervisor:
    """Own one spawned model worker and its communication queues."""

    def __init__(self, context: Any | None = None) -> None:
        """Start one worker without loading a model."""
        self._context = (
            multiprocessing.get_context("spawn") if context is None else context
        )
        self._commands = self._context.Queue()
        self._events = self._context.Queue()
        self._cancel = self._context.Event()
        self._process = self._context.Process(
            target=worker_main,
            args=(self._commands, self._events, self._cancel),
            daemon=True,
        )
        self._process.start()
        self._reported_exit = False

    @property
    def alive(self) -> bool:
        """Return true while the child process runs."""
        return self._process.is_alive()

    def send(self, command: WorkerCommand) -> None:
        """Send one typed command to the active worker."""
        if not self.alive:
            raise WorkerError("The model worker is not running")
        if isinstance(command, GenerateCommand):
            self._cancel.clear()
        if isinstance(command, CancelCommand):
            self._cancel.set()
        self._commands.put(command)

    def receive(self, timeout: float | None = None) -> WorkerEvent | None:
        """Return the next event or report an unexpected child exit."""
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            if not self.alive and not self._reported_exit:
                self._reported_exit = True
                return FailureEvent(
                    f"Model worker exited unexpectedly with code {self._process.exitcode}"
                )
            return None

    def cancel(self) -> None:
        """Request cancellation at the next generated token boundary."""
        self._cancel.set()

    def count_tokens(self, prompt: ModelInput, timeout: float = 30.0) -> int:
        """Return one formatted token count from the loaded worker."""
        self.send(CountCommand(prompt))
        deadline = time.monotonic() + timeout
        deferred = []
        while time.monotonic() < deadline:
            event = self.receive(max(0.01, deadline - time.monotonic()))
            if isinstance(event, CountEvent):
                for item in deferred:
                    self._events.put(item)
                return event.tokens
            if isinstance(event, FailureEvent):
                for item in deferred:
                    self._events.put(item)
                raise WorkerError(event.message)
            if event is not None:
                deferred.append(event)
        for item in deferred:
            self._events.put(item)
        raise WorkerError("The model worker did not return a prompt token count")

    def shutdown(self, timeout: float = 3.0) -> None:
        """Stop the worker gracefully and force it only after the timeout."""
        if self.alive:
            self._commands.put(ShutdownCommand())
            self._process.join(timeout)
        if self.alive:
            self._process.terminate()
            self._process.join(timeout)
        for channel in (self._commands, self._events):
            channel.close()


def worker_main(commands: Any, events: Any, cancel: Any) -> None:
    """Run the serial model command loop in the child process."""
    session = None
    while True:
        command = commands.get()
        diagnostics = io.StringIO()
        try:
            with (
                contextlib.redirect_stdout(diagnostics),
                contextlib.redirect_stderr(diagnostics),
            ):
                if isinstance(command, ShutdownCommand):
                    if session is not None:
                        session.close()
                    return
                if isinstance(command, ProbeCommand):
                    events.put(StateEvent(WorkerState.PROBING))
                    events.put(ProbeEvent(_probe()))
                    continue
                if isinstance(command, (LoadCommand, LoadBaseCommand)):
                    if session is not None:
                        session.close()
                    events.put(StateEvent(WorkerState.RESOLVING))
                    from idiolect.inference.local import (
                        configured_target,
                        recorded_target,
                    )
                    from idiolect.inference.mlx import MlxBackend
                    from idiolect.model import resolve_model

                    def resolve_for_load(spec: ModelSpec) -> Path:
                        """Resolve the model and report the next load state."""
                        path = resolve_model(spec)
                        events.put(StateEvent(WorkerState.VERIFYING))
                        return path

                    if isinstance(command, LoadCommand):
                        target = recorded_target(
                            Path(command.run_path),
                            adapter=True,
                            resolver=resolve_for_load,
                        )
                    else:
                        target = configured_target(
                            TrainConfig(
                                base_model=command.model.name,
                                model_source=command.model.source,
                                model_revision=command.model.revision,
                                model_cache=command.model.cache,
                                trust_remote_code=command.model.trust_remote_code,
                                data=command.data,
                            ),
                            resolver=resolve_for_load,
                            expected_digest=command.expected_digest,
                        )
                    backend = MlxBackend()
                    _ = backend.version
                    events.put(StateEvent(WorkerState.LOADING))
                    started = time.perf_counter()
                    session = backend.load(target)
                    events.put(
                        LoadEvent(
                            LoadProbe(
                                model_digest=target.model_digest,
                                model_size=_path_size(target.model_path),
                                adapter_size=(
                                    _path_size(target.adapter_path)
                                    if target.adapter_path is not None
                                    else None
                                ),
                                load_duration=time.perf_counter() - started,
                            )
                        )
                    )
                    events.put(StateEvent(WorkerState.READY))
                    continue
                if isinstance(command, UnloadCommand):
                    if session is not None:
                        session.close()
                        session = None
                    events.put(StateEvent(WorkerState.READY))
                    continue
                if isinstance(command, CancelCommand):
                    cancel.set()
                    continue
                if isinstance(command, CountCommand):
                    if session is None:
                        raise WorkerError("Load an assistant before token counting")
                    events.put(CountEvent(session.count_tokens(command.prompt)))
                    continue
                if isinstance(command, GenerateCommand):
                    if session is None:
                        raise WorkerError("Load an assistant before generation")
                    events.put(StateEvent(WorkerState.GENERATING))
                    started = time.perf_counter()
                    first_at = None
                    final = None
                    for event in session.stream(
                        command.prompt,
                        command.seed,
                        command.config,
                        cancel,
                        lambda current, total: events.put(
                            PrefillEvent(current, total)
                        ),
                    ):
                        if event.text:
                            if first_at is None:
                                first_at = time.perf_counter()
                            events.put(DeltaEvent(event.text))
                        if event.result is not None:
                            final = event.result
                    if final is None:
                        raise WorkerError(
                            "The model worker did not receive a final result"
                        )
                    elapsed = time.perf_counter() - started
                    ttft = None if first_at is None else first_at - started
                    state = (
                        WorkerState.CANCELLED
                        if final.finish_reason == "cancelled"
                        else WorkerState.READY
                    )
                    events.put(StateEvent(state))
                    events.put(CompleteEvent(final, ttft, elapsed))
                    continue
                raise WorkerError("The model worker received an unknown command")
        except Exception as error:  # noqa: BLE001 - isolate every backend failure.
            events.put(StateEvent(WorkerState.FAILED))
            value = diagnostics.getvalue()
            if value:
                events.put(DiagnosticEvent(value))
                diagnostics.seek(0)
                diagnostics.truncate(0)
            events.put(FailureEvent(_failure_message(error)))
        finally:
            value = diagnostics.getvalue()
            if value:
                events.put(DiagnosticEvent(value))


def command_value(command: WorkerCommand) -> dict[str, Any]:
    """Return one JSON-compatible command record."""
    if isinstance(command, GenerateCommand):
        return {
            "type": "generate",
            "prompt": {
                "turns": [asdict(turn) for turn in command.prompt.turns],
                "has_prefill": command.prompt.has_prefill,
                "completion_role": command.prompt.completion_role,
            },
            "seed": command.seed,
            "config": asdict(command.config),
        }
    if isinstance(command, CountCommand):
        return {"type": "count", "prompt": _prompt_value(command.prompt)}
    if isinstance(command, LoadBaseCommand):
        return {
            "type": "load-base",
            "model": {
                "name": command.model.name,
                "source": command.model.source,
                "revision": command.model.revision,
                "cache": (
                    str(command.model.cache)
                    if command.model.cache is not None
                    else None
                ),
                "trust_remote_code": command.model.trust_remote_code,
            },
            "data": asdict(command.data),
            "expected_digest": command.expected_digest,
        }
    if isinstance(command, ProbeCommand):
        value = {"type": "probe"}
    elif isinstance(command, LoadCommand):
        value = {"type": "load"}
    elif isinstance(command, CancelCommand):
        value = {"type": "cancel"}
    elif isinstance(command, UnloadCommand):
        value = {"type": "unload"}
    else:
        value = {"type": "shutdown"}
    if isinstance(command, LoadCommand):
        value["run_path"] = command.run_path
    return value


def command_from_value(value: dict[str, Any]) -> WorkerCommand:
    """Parse one JSON-compatible command record."""
    kind = value.get("type")
    if kind == "probe":
        return ProbeCommand()
    if kind == "load":
        return LoadCommand(value["run_path"])
    if kind == "load-base":
        model = value["model"]
        return LoadBaseCommand(
            ModelSpec(
                model["name"],
                model["source"],
                model["revision"],
                Path(model["cache"]) if model["cache"] is not None else None,
                model["trust_remote_code"],
            ),
            TrainDataConfig(**value["data"]),
            value["expected_digest"],
        )
    if kind == "cancel":
        return CancelCommand()
    if kind == "unload":
        return UnloadCommand()
    if kind == "shutdown":
        return ShutdownCommand()
    if kind == "generate":
        prompt = value["prompt"]
        return GenerateCommand(
            ModelInput(
                tuple(Turn(**turn) for turn in prompt["turns"]),
                prompt["has_prefill"],
                prompt["completion_role"],
            ),
            value["seed"],
            GenerationConfig(**value["config"]),
        )
    if kind == "count":
        return CountCommand(_prompt_from_value(value["prompt"]))
    raise WorkerError("Worker command is not valid")


def _prompt_value(prompt: ModelInput) -> dict[str, Any]:
    return {
        "turns": [asdict(turn) for turn in prompt.turns],
        "has_prefill": prompt.has_prefill,
        "completion_role": prompt.completion_role,
    }


def _prompt_from_value(value: dict[str, Any]) -> ModelInput:
    return ModelInput(
        tuple(Turn(**turn) for turn in value["turns"]),
        value["has_prefill"],
        value["completion_role"],
    )


def _probe() -> RuntimeProbe:
    import platform
    from importlib.metadata import version

    import mlx.core as mx

    info = tuple(sorted(mx.device_info().items()))
    return RuntimeProbe(
        mlx_version=version("mlx"),
        mlx_lm_version=version("mlx-lm"),
        device=str(mx.default_device()),
        architecture=platform.machine(),
        device_properties=info,
    )


def _failure_message(error: BaseException) -> str:
    name = type(error).__name__
    text = str(error).strip()
    return f"{name}: {text}" if text else name


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
