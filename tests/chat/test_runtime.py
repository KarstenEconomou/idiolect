"""Test chat runtime model selection behavior."""

from idiolect.chat.discovery import Assistant
from idiolect.chat.runtime import ChatRuntime
from idiolect.chat.worker import (
    CompleteEvent,
    DeltaEvent,
    LoadBaseCommand,
    PrefillEvent,
    ProbeEvent,
    StateEvent,
    WorkerEvent,
    WorkerState,
)
from idiolect.config import ChatConfig, GenerationConfig, TrainDataConfig
from idiolect.inference.base import BackendResult
from idiolect.model import ModelSpec


def test_base_assistant_load_records_verified_model_digest() -> None:
    """Check that the base load stays in the worker and records its digest."""
    worker = ReadyWorker()
    runtime = ChatRuntime(
        ChatConfig(),
        GenerationConfig(),
        worker_factory=lambda: worker,
    )
    assistant = _assistant()

    session = runtime.select(assistant)

    load = next(
        command for command in worker.commands if isinstance(command, LoadBaseCommand)
    )
    assert load.model.name == "org/model"
    assert load.data.system_prompt == "Be terse."
    assert session.assistant.model_digest == "a" * 64


def test_generation_reports_measured_prefill_progress() -> None:
    """Check prompt progress between the worker and chat caller."""
    worker = ReadyWorker()
    runtime = ChatRuntime(
        ChatConfig(participant_name="person_01"),
        GenerationConfig(max_prompt_tokens=100),
        worker_factory=lambda: worker,
    )
    session = runtime.select(_assistant())
    session.add_user("Hello")
    worker.events.extend(
        [
            StateEvent(WorkerState.GENERATING),
            PrefillEvent(0, 5),
            PrefillEvent(5, 5),
            DeltaEvent("Reply"),
            StateEvent(WorkerState.READY),
            CompleteEvent(BackendResult("", "stop", 5, 1), 0.1, 0.2),
        ]
    )
    progress = []

    pieces = list(
        runtime.generate(
            prompt_progress=lambda current, total: progress.append((current, total))
        )
    )

    assert pieces == ["Reply"]
    assert progress == [(0, 5), (5, 5)]
    assert session.turns[-1].content == "Reply"


class ReadyWorker:
    """Return a complete synthetic base-model load event sequence."""

    def __init__(self) -> None:
        """Create command and event records."""
        self.commands = []
        self.events: list[WorkerEvent] = [
            ProbeEvent({"model_digest": "a" * 64}),
            StateEvent(WorkerState.READY),
        ]
        self.alive = True

    def send(self, command) -> None:
        """Record one worker command."""
        self.commands.append(command)

    def receive(self, _timeout):
        """Return the next fixed worker event."""
        return self.events.pop(0)

    def count_tokens(self, _prompt) -> int:
        """Return one synthetic prompt token count."""
        return 5

    def shutdown(self) -> None:
        """Record no shutdown work for this test."""


def _assistant() -> Assistant:
    return Assistant(
        "IDIOLECT // DIXIE@BASE [model]",
        "DIXIE",
        "model",
        None,
        None,
        32,
        ModelSpec("org/model", "hub", "fixed", None, False),
        TrainDataConfig(
            format="chat",
            system_prompt="Be terse.",
            prompt_role="user",
            completion_role="assistant",
        ),
    )
