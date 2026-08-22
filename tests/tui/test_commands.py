"""Test chat slash commands and the landing watermark."""

import asyncio
import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest
from textual.widgets import OptionList, Static

from idiolect.chat.discovery import Assistant
from idiolect.chat.runtime import ChatRuntime
from idiolect.chat.state import ChatSession
from idiolect.chat.storage import ChatStorageError, ChatStore
from idiolect.chat.worker import WorkerState
from idiolect.config import ChatConfig, GenerationConfig
from idiolect.tui.app import WATERMARK, ChatApp
from idiolect.tui.commands import CommandError, completions, parse_command


def test_commands_keep_save_title_and_complete_prefix() -> None:
    """Check exact command parsing and prefix completion."""
    command = parse_command('/save "Night city"')
    assert command is not None
    assert command.argument == "Night city"
    assert completions("/re") == ("/resume", "/retry")
    with pytest.raises(CommandError, match="Unknown"):
        parse_command("/remove")


def test_landing_uses_readme_watermark_as_literal_text(tmp_path) -> None:
    """Check the first screen without loading Textual model dependencies."""
    app = ChatApp(
        ChatConfig(output=tmp_path),
        GenerationConfig(),
        runtime_factory=cast(
            Callable[..., ChatRuntime], lambda *_args: SimpleRuntime()
        ),
    )

    async def verify() -> None:
        """Inspect the mounted landing screen."""
        async with app.run_test() as pilot:
            await pilot.pause()
            assert str(app.query_one("#watermark", Static).content) == WATERMARK

    asyncio.run(verify())


class SimpleRuntime:
    """Provide the unused runtime surface for one landing test."""

    session = None


def test_model_load_does_not_block_textual_event_processing(tmp_path) -> None:
    """Check that the landing screen remains mounted during a slow model load."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
    assistant = _assistant()
    runtime = BlockingRuntime(chat, generation)
    app = ChatApp(
        chat,
        generation,
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
        initial_assistant=assistant,
    )

    async def verify() -> None:
        """Inspect loading state before the fake model load can finish."""
        async with app.run_test() as pilot:
            assert await asyncio.to_thread(runtime.started.wait, 1)
            await asyncio.sleep(0.15)
            await pilot.pause()
            status = app.query_one("#load-status", Static)
            assert "LOADING" in str(status.content)
            assert app.query_one("#chooser", OptionList).disabled is True
            runtime.release.set()
            for _ in range(20):
                await pilot.pause()
                if app.query_one("#chat").display:
                    break
            assert app.query_one("#chat").display

    try:
        asyncio.run(verify())
    finally:
        runtime.release.set()


def test_failed_confirmation_save_keeps_memory_only_chat(tmp_path) -> None:
    """Check that a save error cannot close the app or replace its transcript."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
    runtime = ImmediateRuntime(chat, generation)
    store = FailingStore()
    app = ChatApp(
        chat,
        generation,
        store=cast(ChatStore, store),
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
        initial_assistant=_assistant(),
    )

    async def verify() -> None:
        """Choose Save in the quit confirmation and inspect retained state."""
        async with app.run_test() as pilot:
            for _ in range(20):
                await pilot.pause()
                if app.query_one("#chat").display:
                    break
            session = runtime.session
            assert session is not None
            session.add_user("unsaved")
            await pilot.press("ctrl+c")
            await pilot.pause()
            await pilot.click("#save")
            await pilot.pause()
            assert runtime.closed is False
            assert runtime.session is session
            assert runtime.session.turns[-1].content == "unsaved"
            assert app.query_one("#chat").display

    asyncio.run(verify())


class BlockingRuntime:
    """Hold one fake model load while the Textual loop continues."""

    def __init__(self, chat: ChatConfig, generation: GenerationConfig) -> None:
        """Create synchronization points and visible runtime state."""
        self.chat = chat
        self.generation = generation
        self.session = None
        self.state = WorkerState.PROBING
        self.started = threading.Event()
        self.release = threading.Event()
        self.probe = {}

    def select(self, assistant: Assistant) -> ChatSession:
        """Wait until the test releases the synthetic model load."""
        self.session = ChatSession(assistant, self.chat, self.generation)
        self.state = WorkerState.LOADING
        self.started.set()
        if not self.release.wait(2):
            raise RuntimeError("Synthetic model load timed out")
        self.state = WorkerState.READY
        return self.session

    def close(self) -> None:
        """Release a pending synthetic load."""
        self.release.set()


class ImmediateRuntime(BlockingRuntime):
    """Load immediately and record whether the runtime closes."""

    def __init__(self, chat: ChatConfig, generation: GenerationConfig) -> None:
        """Create one open fake runtime."""
        super().__init__(chat, generation)
        self.release.set()
        self.closed = False

    def close(self) -> None:
        """Record application shutdown."""
        self.closed = True


class FailingStore:
    """Reject every explicit save without changing transcript state."""

    def leaves(self):
        """Return no saved chooser rows."""
        return ()

    def save(self, _session, _title=None):
        """Report one controlled local storage failure."""
        raise ChatStorageError("Synthetic disk failure")


def _assistant() -> Assistant:
    """Return the minimal assistant identity needed by the chat screen."""
    return cast(
        Assistant,
        SimpleNamespace(name="IDIOLECT // K@aaaaaaaa [M]"),
    )
