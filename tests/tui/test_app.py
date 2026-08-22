"""Test the terminal chat application."""

import asyncio
import re
import threading
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import cast

from rich.text import Text
from textual import events
from textual.containers import VerticalScroll
from textual.pilot import Pilot
from textual.widgets import OptionList, Rule, Static

from idiolect.chat.discovery import Assistant, DiscoveryItem
from idiolect.chat.runtime import ChatRuntime
from idiolect.chat.state import ChatSession, ChatTurn, TurnTelemetry
from idiolect.chat.storage import ChatStorageError, ChatStore
from idiolect.chat.worker import WorkerError, WorkerState
from idiolect.config import ChatConfig, GenerationConfig
from idiolect.tui.app import WATERMARK, ChatApp
from idiolect.tui.widgets import Composer, LoadingStatus


def test_landing_shows_literal_watermark(tmp_path) -> None:
    """Check the registry watermark text."""
    app = ChatApp(
        ChatConfig(output=tmp_path),
        GenerationConfig(),
        runtime_factory=cast(
            Callable[..., ChatRuntime], lambda *_args: SimpleRuntime()
        ),
    )

    async def verify() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            assert str(app.query_one("#watermark", Static).content) == WATERMARK

    asyncio.run(verify())


def test_registry_opens_highlighted_assistant_from_keyboard(tmp_path) -> None:
    """Check registry focus, pointer blocking, and keyboard selection."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
    assistant = _assistant()
    runtime = ImmediateRuntime(chat, generation)
    app = ChatApp(
        chat,
        generation,
        assistants=(
            DiscoveryItem("Unavailable assistant", "failed", None, None, "invalid"),
            DiscoveryItem(assistant.name, "BASE", None, assistant),
        ),
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
    )

    async def verify() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            chooser = app.query_one("#chooser", OptionList)
            rule = app.query_one("#catalog-rule", Rule)
            assert str(app.query_one("#catalog-title", Static).content) == "REGISTRY"
            assert len(app.query("#search")) == 0
            assert rule.line_style == "solid"
            assert rule.content_region.x == chooser.content_region.x
            assert rule.content_region.width == chooser.content_region.width
            assert chooser.highlighted == 1
            assert chooser.has_focus
            highlight = chooser.get_component_rich_style(
                "option-list--option-highlighted"
            )
            assert highlight.color is not None
            assert highlight.color.number == 4
            assert highlight.bgcolor is not None
            assert highlight.bgcolor.name == "default"
            description = app.query_one("#catalog-description", Static)
            summary = app.query_one("#catalog-summary", Static)
            assert str(summary.content) == "1 available"
            assert description.region.y == summary.region.y
            assert app.query_one("#catalog-title", Static).region.y < summary.region.y
            prompt = chooser.get_option_at_index(1).prompt
            assert isinstance(prompt, Text)
            assert "READY" in prompt.plain

            await pilot.click(chooser, offset=(2, 1))
            await pilot.pause()
            assert app.query_one("#landing").display
            assert runtime.session is None

            await pilot.press("enter")
            await _wait_for_chat(app, pilot)
            assert runtime.session is not None
            assert runtime.session.assistant is assistant

    asyncio.run(verify())


def test_transcript_is_literal_and_scrollable(tmp_path) -> None:
    """Check transcript labels, identity, telemetry, and navigation."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig(max_prompt_tokens=1000)
    runtime = TranscriptRuntime(chat, generation)
    app = ChatApp(
        chat,
        generation,
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
        initial_assistant=_assistant(),
    )

    async def verify() -> None:
        async with app.run_test(size=(50, 18)) as pilot:
            await _wait_for_chat(app, pilot)
            scroller = app.query_one("#transcript-scroll", VerticalScroll)
            await pilot.pause()
            assert scroller.max_scroll_y > 0
            assert scroller.scroll_y == scroller.max_scroll_y
            assert scroller.styles.scrollbar_size_vertical == 0

            transcript = app.query_one("#transcript", Static)
            transcript_text = str(transcript.content)
            assert "USER:" in transcript_text
            assert "DIXIE:" in transcript_text
            assert "[bold]literal[/bold]" in transcript_text
            assert "IDIOLECT //" not in transcript_text
            assert str(app.query_one("#identity", Static).content) == (
                "IDIOLECT // DIXIE@BASE [M]"
            )
            assert str(app.query_one("#footer", Static).content) == (
                "CONTEXT 500/1000 (50%)    GENERATED 64"
            )
            assert app.query_one("#status", LoadingStatus).display is False

            bottom = scroller.scroll_y
            scroller.post_message(
                events.MouseScrollUp(
                    scroller,
                    0,
                    0,
                    0,
                    -1,
                    0,
                    False,
                    False,
                    False,
                )
            )
            await pilot.pause()
            assert scroller.scroll_y < bottom

            scroller.scroll_end(animate=False)
            await pilot.press("ctrl+up")
            await pilot.pause()
            assert scroller.scroll_y < bottom
            scrolled_up = scroller.scroll_y
            await pilot.press("ctrl+down")
            await pilot.pause()
            assert scroller.scroll_y > scrolled_up

    asyncio.run(verify())


def test_composer_submits_and_inserts_line_breaks(tmp_path) -> None:
    """Check composer submission and multiline input."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig(max_prompt_tokens=100)
    runtime = ImmediateRuntime(chat, generation)
    app = ChatApp(
        chat,
        generation,
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
        initial_assistant=_assistant(),
    )

    async def verify() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_for_chat(app, pilot)
            composer = app.query_one(Composer)
            composer.insert("first")
            await pilot.press("shift+enter")
            composer.insert("second")
            assert composer.text == "first\nsecond"

            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if "Synthetic reply" in str(
                    app.query_one("#transcript", Static).content
                ):
                    break
            transcript = str(app.query_one("#transcript", Static).content)
            assert "USER:\nfirst\nsecond" in transcript
            assert "DIXIE:\nSynthetic reply" in transcript
            assert composer.text == ""

    asyncio.run(verify())


def test_prefill_progress_appears_above_composer(tmp_path) -> None:
    """Check measured prefill status and prompt spacing."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig(max_prompt_tokens=100)
    runtime = ProgressRuntime(chat, generation)
    app = ChatApp(
        chat,
        generation,
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
        initial_assistant=_assistant(),
    )

    async def verify() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_for_chat(app, pilot)
            composer = app.query_one(Composer)
            composer.insert("show progress")
            await pilot.press("enter")
            assert await asyncio.to_thread(runtime.prefill_started.wait, 1)
            await pilot.pause()

            status = app.query_one("#status", LoadingStatus)
            footer = app.query_one("#footer", Static)
            rendered = status.render()
            assert status.state == "PREFILL 0/4"
            assert isinstance(rendered, Text)
            assert rendered.plain.endswith(" PREFILL 0/4")
            assert status.display is True
            assert status.styles.padding == footer.styles.padding
            assert footer.content_region.x == composer.region.x

            runtime.release_prefill.set()
            assert await asyncio.to_thread(runtime.generation_finished.wait, 1)
            for _ in range(20):
                await pilot.pause()
                reply_visible = "Progress reply" in str(
                    app.query_one("#transcript", Static).content
                )
                if reply_visible and not status.display:
                    break
            assert status.display is False

    try:
        asyncio.run(verify())
    finally:
        runtime.release_prefill.set()


def test_theme_uses_terminal_and_ansi_colors() -> None:
    """Check the terminal color contract."""
    app = ChatApp(ChatConfig(), GenerationConfig())

    assert app.native_ansi_color is True
    assert "ansi_default" in ChatApp.CSS
    assert "ansi_blue" in ChatApp.CSS
    assert "ansi_bright_black" in ChatApp.CSS
    assert "ansi_red" in ChatApp.CSS
    assert re.search(r"#[0-9a-fA-F]{3,8}\b", ChatApp.CSS) is None


def test_model_load_keeps_event_processing_active(tmp_path) -> None:
    """Check event processing during model loading."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
    runtime = BlockingRuntime(chat, generation)
    app = ChatApp(
        chat,
        generation,
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
        initial_assistant=_assistant(),
    )

    async def verify() -> None:
        async with app.run_test() as pilot:
            assert await asyncio.to_thread(runtime.started.wait, 1)
            await asyncio.sleep(0.15)
            await pilot.pause()
            status = app.query_one("#load-status", LoadingStatus)
            rendered = status.render()
            assert status.state == "LOADING"
            assert isinstance(rendered, Text)
            assert rendered.plain.endswith(" LOADING")
            assert "// MODEL SESSION" not in rendered.plain
            assert app.query_one("#chooser", OptionList).disabled is True
            runtime.release.set()
            await _wait_for_chat(app, pilot)

    try:
        asyncio.run(verify())
    finally:
        runtime.release.set()


def test_failed_confirmation_save_keeps_memory_only_chat(tmp_path) -> None:
    """Check transcript state after a confirmation save fails."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
    runtime = ImmediateRuntime(chat, generation)
    app = ChatApp(
        chat,
        generation,
        store=cast(ChatStore, FailingStore()),
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
        initial_assistant=_assistant(),
    )

    async def verify() -> None:
        async with app.run_test() as pilot:
            await _wait_for_chat(app, pilot)
            session = runtime.session
            assert session is not None
            session.add_user("unsaved")

            await pilot.press("ctrl+c")
            await pilot.pause()
            assert app.focused is not None
            assert app.focused.id == "save"
            await pilot.press("enter")
            await pilot.pause()

            assert runtime.closed is False
            assert runtime.session is session
            assert runtime.session.turns[-1].content == "unsaved"
            assert app.query_one("#chat").display

    asyncio.run(verify())


def test_retry_keeps_user_turn_after_generation_failure(tmp_path) -> None:
    """Check reload and retry after a generation failure."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig(max_prompt_tokens=100)
    runtime = RetryRuntime(chat, generation)
    app = ChatApp(
        chat,
        generation,
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
        initial_assistant=_assistant(),
    )

    async def verify() -> None:
        async with app.run_test() as pilot:
            await _wait_for_chat(app, pilot)
            composer = app.query_one(Composer)
            composer.insert("keep this input")
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if app.query_one("#status", LoadingStatus).state == "FAILED":
                    break

            session = runtime.session
            assert session is not None
            assert [(turn.role, turn.content) for turn in session.turns] == [
                ("user", "keep this input")
            ]

            composer.insert("/retry")
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if "Recovered reply" in str(
                    app.query_one("#transcript", Static).content
                ):
                    break

            assert runtime.attachments == 1
            assert [(turn.role, turn.content, turn.attempt) for turn in session.turns] == [
                ("user", "keep this input", 0),
                ("assistant", "Recovered reply", 1),
            ]

    asyncio.run(verify())


async def _wait_for_chat(app: ChatApp, pilot: Pilot[None]) -> None:
    for _ in range(20):
        await pilot.pause()
        if app.query_one("#chat").display:
            return
    raise AssertionError("The chat screen did not open")


class SimpleRuntime:
    """Provide runtime state for a registry test."""

    session = None
    state = WorkerState.PROBING

    def ensure_worker(self) -> None:
        """Keep the fake runtime ready."""


class BlockingRuntime:
    """Hold one fake model load."""

    def __init__(self, chat: ChatConfig, generation: GenerationConfig) -> None:
        """Create model-load synchronization points."""
        self.chat = chat
        self.generation = generation
        self.session: ChatSession | None = None
        self.state = WorkerState.PROBING
        self.started = threading.Event()
        self.release = threading.Event()
        self.probe: dict[str, object] = {}

    def ensure_worker(self) -> None:
        """Keep the fake runtime ready."""

    def select(self, assistant: Assistant) -> ChatSession:
        """Wait before the fake model becomes ready."""
        self.session = ChatSession(assistant, self.chat, self.generation)
        self.state = WorkerState.LOADING
        self.started.set()
        if not self.release.wait(2):
            raise RuntimeError("Synthetic model load timed out")
        self.state = WorkerState.READY
        return self.session

    def close(self) -> None:
        """Release a pending fake load."""
        self.release.set()


class ImmediateRuntime(BlockingRuntime):
    """Generate a reply without a model backend."""

    def __init__(self, chat: ChatConfig, generation: GenerationConfig) -> None:
        """Create one open fake runtime."""
        super().__init__(chat, generation)
        self.release.set()
        self.closed = False

    def generate(
        self,
        attempt: int = 0,
        prompt_progress: Callable[[int, int], None] | None = None,
    ) -> Iterator[str]:
        """Commit one synthetic assistant reply."""
        if self.session is None:
            raise RuntimeError("No fake chat session")
        self.session.begin_generation()
        yield "Synthetic reply"
        self.session.finish_generation(
            "Synthetic reply",
            "length",
            101 + attempt,
            TurnTelemetry(2, 2),
            attempt=attempt,
        )

    def close(self) -> None:
        """Record fake runtime shutdown."""
        self.closed = True


class TranscriptRuntime(ImmediateRuntime):
    """Load a synthetic transcript that needs scrolling."""

    def select(self, assistant: Assistant) -> ChatSession:
        """Return alternating user and assistant turns."""
        turns = tuple(
            turn
            for index in range(12)
            for turn in (
                ChatTurn("user", f"User message {index}\n[bold]literal[/bold]"),
                ChatTurn(
                    "assistant",
                    f"Assistant reply {index}\nwith a second line",
                    telemetry=TurnTelemetry(
                        prompt_tokens=500,
                        generated_tokens=64,
                        generation_throughput=12.3,
                        peak_memory=3.25,
                    ),
                ),
            )
        )
        self.session = ChatSession(assistant, self.chat, self.generation, turns)
        self.state = WorkerState.READY
        return self.session


class ProgressRuntime(ImmediateRuntime):
    """Hold generation while prompt progress is visible."""

    def __init__(self, chat: ChatConfig, generation: GenerationConfig) -> None:
        """Create prefill synchronization points."""
        super().__init__(chat, generation)
        self.prefill_started = threading.Event()
        self.release_prefill = threading.Event()
        self.generation_finished = threading.Event()

    def generate(
        self,
        attempt: int = 0,
        prompt_progress: Callable[[int, int], None] | None = None,
    ) -> Iterator[str]:
        """Report prompt progress before one synthetic reply."""
        if self.session is None or prompt_progress is None:
            raise RuntimeError("No fake chat session or progress callback")
        self.session.begin_generation()
        prompt_progress(0, 4)
        self.prefill_started.set()
        if not self.release_prefill.wait(2):
            raise RuntimeError("Synthetic prefill timed out")
        prompt_progress(4, 4)
        yield "Progress reply"
        self.state = WorkerState.READY
        self.session.finish_generation(
            "Progress reply",
            "length",
            101 + attempt,
            TurnTelemetry(4, 2),
            attempt=attempt,
        )
        self.generation_finished.set()


class RetryRuntime(ImmediateRuntime):
    """Fail one generation and complete its retry."""

    def __init__(self, chat: ChatConfig, generation: GenerationConfig) -> None:
        """Create a fake runtime with no completed retry."""
        super().__init__(chat, generation)
        self.attachments = 0

    def attach(self, session: ChatSession) -> None:
        """Record one fake model reload."""
        self.session = session
        self.state = WorkerState.READY
        self.attachments += 1

    def generate(
        self,
        attempt: int = 0,
        prompt_progress: Callable[[int, int], None] | None = None,
    ) -> Iterator[str]:
        """Fail attempt zero and complete attempt one."""
        if self.session is None:
            raise RuntimeError("No fake chat session")
        self.session.begin_generation()
        if attempt == 0:
            self.session.generating = False
            self.state = WorkerState.FAILED
            raise WorkerError("Synthetic generation failure")
        yield "Recovered reply"
        self.session.finish_generation(
            "Recovered reply",
            "length",
            101 + attempt,
            TurnTelemetry(2, 2),
            attempt=attempt,
        )


class FailingStore:
    """Reject explicit chat saves."""

    def leaves(self) -> tuple[()]:
        """Return no saved chat rows."""
        return ()

    def save(self, _session: ChatSession, _title: str | None = None) -> None:
        """Raise a controlled storage error."""
        raise ChatStorageError("Synthetic disk failure")


def _assistant() -> Assistant:
    return cast(
        Assistant,
        SimpleNamespace(
            name="IDIOLECT // DIXIE@BASE [M]",
            target_name="DIXIE",
            run=None,
            context_messages=32,
        ),
    )
