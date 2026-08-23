"""Test chat runtime model selection behavior."""

import time

from idiolect.chat.discovery import Assistant
from idiolect.chat.runtime import ChatRuntime
from idiolect.chat.worker import (
    CancelCommand,
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


def test_abandoned_reply_cancels_the_worker_and_drains_events() -> None:
    """Check that an abandoned reply cannot leak deltas into the next one."""
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
            DeltaEvent("Stale"),
            DeltaEvent(" text"),
            StateEvent(WorkerState.CANCELLED),
            CompleteEvent(BackendResult("Stale text", "cancelled", 5, 2), 0.1, 0.2),
        ]
    )

    generator = runtime.generate()
    assert next(generator) == "Stale"
    generator.close()

    assert any(
        isinstance(command, CancelCommand) for command in worker.commands
    )
    assert runtime.state == WorkerState.CANCELLED
    assert not session.generating
    assert [turn.role for turn in session.turns] == ["user"]

    worker.events.extend([DeltaEvent("Fresh"), CompleteEvent(BackendResult("", "stop", 6, 1), 0.1, 0.3)])
    pieces = list(runtime.generate())

    assert pieces == ["Fresh"]
    assert session.turns[-1].content == "Fresh"


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
        """Return the next fixed worker event or block on an empty queue."""
        while not self.events:
            time.sleep(0.01)
        return self.events.pop(0)

    def cancel(self) -> None:
        """Record one cancellation request."""
        self.commands.append(CancelCommand())

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
