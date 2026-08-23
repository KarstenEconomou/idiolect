"""Test the terminal chat application."""

import asyncio
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import ClassVar, cast

from rich.console import Console, RenderableType
from rich.text import Text
from textual import events
from textual.containers import Horizontal, VerticalScroll
from textual.pilot import Pilot
from textual.widgets import Input, OptionList, Rule, Static

from idiolect.chat.discovery import Assistant, DiscoveryItem
from idiolect.chat.runtime import ChatRuntime
from idiolect.chat.state import ChatSession, ChatTurn, TurnTelemetry
from idiolect.chat.storage import ChatStorageError, ChatStore, SavedChat
from idiolect.chat.worker import WorkerState
from idiolect.config import ChatConfig, GenerationConfig, TrainDataConfig
from idiolect.model import ModelSpec
from idiolect.tui.app import ChatApp, _episode_segments
from idiolect.tui.specs import HalfCellScrollBarRender, SpecsDocument
from idiolect.tui.widgets import (
    CommandMenu,
    Composer,
    KeyboardButton,
    LoadingStatus,
    ReferenceBar,
    ReferenceMenu,
    SpecsScroll,
    Transcript,
)


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
            watermark = app.query_one("#watermark", Static).content
            assert isinstance(watermark, Text)
            console = Console()
            mark_style = watermark.get_style_at_offset(console, 0)
            tagline_style = watermark.get_style_at_offset(
                console,
                watermark.plain.index("Someone, reconstructed."),
            )
            assert mark_style.color is not None and mark_style.color.number == 2
            assert mark_style.bold
            assert tagline_style.color is not None
            assert tagline_style.color.number == 2
            assert not tagline_style.bold
            assert tagline_style.dim
            assert str(app.query_one("#catalog-title", Static).content) == "REGISTRY"
            assert str(app.query_one("#catalog-description", Static).content) == (
                "Connect to a BASE, CONSTRUCT, or TRACE."
            )
            assert app.query_one("#catalog-columns", Static).styles.color.ansi == 7
            columns = str(app.query_one("#catalog-columns", Static).content)
            assert "CONSTRUCT" in columns
            assert "BASE" in columns
            assert "TYPE" in str(app.query_one("#catalog-columns", Static).content)
            assert len(app.query("#catalog-summary")) == 0
            assert len(app.query("#search")) == 0
            assert app.query_one("#landing").region.x == 0
            assert app.query_one("#landing-box").region.width == 80
            assert app.query_one("#catalog-heading").content_region.x == 2
            assert chooser.content_region.x == 2
            assert app.query_one("#catalog-hints", Static).content_region.x == 2
            assert chooser.highlighted == 1
            assert chooser.has_focus
            prompt = chooser.get_option_at_index(1).prompt
            assert isinstance(prompt, Text)
            assert "DIXIE::BASE" in prompt.plain
            assert "M" in prompt.plain
            assert "READY" in prompt.plain
            assert str(app.query_one("#catalog-hints", Static).content) == (
                "↑↓ MOVE    ENTER CONNECT    S SPECS    CTRL+C QUIT"
            )

            await pilot.click(chooser, offset=(2, 1))
            await pilot.pause()
            assert app.query_one("#landing").display
            assert runtime.session is None

            await pilot.press("enter")
            await _wait_for_chat(app, pilot)
            assert runtime.session is not None
            assert runtime.session.assistant is assistant

    asyncio.run(verify())


def test_registry_theme_cycles_and_persists_across_pages(tmp_path) -> None:
    """Check the hidden theme key changes every page without entering input."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
    assistant = _assistant()
    runtime = ImmediateRuntime(chat, generation)
    app = ChatApp(
        chat,
        generation,
        assistants=(DiscoveryItem(assistant.name, "BASE", None, assistant),),
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
    )

    async def verify() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            chooser = app.query_one("#chooser", OptionList)
            for name, ansi in (
                ("yellow", 3),
                ("blue", 4),
                ("purple", 5),
                ("cyan", 6),
                ("green", 2),
            ):
                await pilot.press("t")
                await pilot.pause()
                assert app.has_class(f"-accent-{name}")
                selected = chooser.get_component_rich_style(
                    "option-list--option-highlighted"
                )
                assert selected.color is not None and selected.color.number == ansi
                watermark = app.query_one("#watermark", Static).content
                assert isinstance(watermark, Text)
                style = watermark.get_style_at_offset(Console(), 0)
                assert style.color is not None and style.color.number == ansi

            hints = str(app.query_one("#catalog-hints", Static).content)
            assert "THEME" not in hints
            await pilot.press("s")
            await pilot.pause()
            assert app.query_one("#specs-identity", Static).styles.color.ansi == 2
            assert app.query_one("#specs-eyes", Static).styles.color.ansi == 2

            await pilot.press("t")
            await pilot.pause()
            assert app.has_class("-accent-green")
            await pilot.press("escape", "enter")
            await _wait_for_chat(app, pilot)
            assert app.query_one("#identity", Static).styles.color.ansi == 2
            assert app.query_one("#chat-eyes", Static).styles.color.ansi == 2
            assert app.query_one("#composer-prompt", Static).styles.color.ansi == 2
            transcript = app.query_one("#transcript", Transcript)
            transcript.set_turns((("USER", "Theme check"),))
            segments = tuple(
                Console().render(cast(RenderableType, transcript.content))
            )
            label = next(segment for segment in segments if segment.text == "USER:")
            assert label.style is not None
            assert label.style.color is not None and label.style.color.number == 2

    asyncio.run(verify())


def test_registry_opens_specs_and_returns_to_the_same_row(tmp_path) -> None:
    """Check SPECS navigation without loading or changing the selection."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig(backend="mlx-lm", max_prompt_tokens=1920)
    assistant = _assistant()
    runtime = ImmediateRuntime(chat, generation)
    app = ChatApp(
        chat,
        generation,
        assistants=(DiscoveryItem(assistant.name, "BASE", None, assistant),),
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
    )

    async def verify() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            chooser = app.query_one("#chooser", OptionList)
            assert chooser.highlighted == 0

            await pilot.press("s")
            await pilot.pause()

            assert app.query_one("#landing").display is False
            assert app.query_one("#specs").display
            assert runtime.session is None
            assert str(app.query_one("#specs-identity", Static).content) == (
                assistant.name
            )
            specs_heading = app.query_one("#specs-heading", Horizontal)
            specs_eyes = app.query_one("#specs-eyes", Static)
            assert str(specs_eyes.content) == "· ·"
            assert (
                specs_eyes.content_region.right
                == specs_heading.content_region.right - 1
            )
            assert specs_eyes.region.right == specs_heading.content_region.right
            assert specs_eyes.styles.color.ansi == 2

            await pilot.press("escape")
            await pilot.pause()

            assert app.query_one("#landing").display
            assert app.query_one("#specs").display is False
            assert chooser.highlighted == 0
            assert chooser.has_focus
            assert runtime.session is None

    asyncio.run(verify())


def test_specs_page_uses_a_stable_half_cell_scrollbar(tmp_path) -> None:
    """Check the SPECS scrollbar width and interaction-state colors."""
    assistant = _assistant()
    app = ChatApp(
        ChatConfig(output=tmp_path),
        GenerationConfig(),
        assistants=(DiscoveryItem(assistant.name, "BASE", None, assistant),),
        runtime_factory=cast(
            Callable[..., ChatRuntime],
            lambda chat, generation: ImmediateRuntime(chat, generation),
        ),
    )

    async def verify() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("s")
            await pilot.pause()

            scroller = app.query_one("#specs-scroll", VerticalScroll)
            assert scroller.styles.scrollbar_size_vertical == 1
            assert scroller.vertical_scrollbar.renderer is HalfCellScrollBarRender
            assert (
                scroller.styles.scrollbar_color_hover
                == scroller.styles.scrollbar_color
            )
            assert (
                scroller.styles.scrollbar_color_active
                == scroller.styles.scrollbar_color
            )
            assert (
                scroller.styles.scrollbar_background_hover
                == scroller.styles.scrollbar_background
            )
            assert (
                scroller.styles.scrollbar_background_active
                == scroller.styles.scrollbar_background
            )

    asyncio.run(verify())


def test_specs_side_arrows_cycle_available_registry_rows(tmp_path) -> None:
    """Check SPECS wraps through READY rows and skips FAULT rows."""
    first = _assistant()
    first_model = first.base_model
    assert first_model is not None
    second = replace(
        first,
        name="IDIOLECT // MARGO::BASE [N]",
        target_name="MARGO",
        model_basename="N",
        base_model=replace(first_model, name="example/N"),
    )
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
    app = ChatApp(
        chat,
        generation,
        assistants=(
            DiscoveryItem("Unavailable assistant", "failed", None, None, "invalid"),
            DiscoveryItem(first.name, "BASE", None, first),
            DiscoveryItem(second.name, "BASE", None, second),
        ),
        runtime_factory=cast(
            Callable[..., ChatRuntime],
            lambda *_args: ImmediateRuntime(chat, generation),
        ),
    )

    async def verify() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            chooser = app.query_one("#chooser", OptionList)
            assert chooser.highlighted == 1
            await pilot.press("s")
            await pilot.pause()

            specs = app.query_one("#specs-scroll", SpecsScroll)
            assert specs.has_focus
            assert str(app.query_one("#specs-identity", Static).content) == first.name
            assert str(app.query_one("#specs-hints", Static).content) == (
                "↑↓ SCROLL    ←→ MODEL    ESC REGISTRY    CTRL+C QUIT"
            )

            await pilot.press("right")
            await pilot.pause()
            assert str(app.query_one("#specs-identity", Static).content) == second.name
            assert chooser.highlighted == 2

            await pilot.press("right")
            await pilot.pause()
            assert str(app.query_one("#specs-identity", Static).content) == first.name
            assert chooser.highlighted == 1

            await pilot.press("left", "escape")
            await pilot.pause()
            assert app.query_one("#landing").display
            assert chooser.highlighted == 2

    asyncio.run(verify())


def test_specs_prompt_wrap_uses_the_transcript_inset(tmp_path) -> None:
    """Check final viewport wrapping keeps every prompt line inset."""
    original = _assistant()
    base_data = original.base_data
    assert base_data is not None
    assistant = replace(
        original,
        base_data=replace(
            base_data,
            system_prompt=f"{'A' * 100}\n",
        ),
    )
    app = ChatApp(
        ChatConfig(output=tmp_path),
        GenerationConfig(),
        assistants=(DiscoveryItem(assistant.name, "BASE", None, assistant),),
    )

    async def verify() -> None:
        async with app.run_test(size=(48, 24)) as pilot:
            await pilot.press("s")
            await pilot.pause()

            body = app.query_one("#specs-body", Static)
            rendered = [
                body.render_line(y).text.rstrip()
                for y in range(body.virtual_size.height)
            ]
            start = rendered.index("SYSTEM PROMPT") + 1
            end = next(
                index
                for index in range(start, len(rendered))
                if rendered[index].startswith("PROMPT ROLE")
            )
            prompt_lines = rendered[start:end]

            assert len(prompt_lines) == 3
            assert all(line.startswith(" ") for line in prompt_lines)
            assert "".join(line[1:] for line in prompt_lines) == "A" * 100
            assert all(line != " " for line in prompt_lines)

    asyncio.run(verify())


def test_registry_opens_trace_specs_with_saved_lineage_and_policy(tmp_path) -> None:
    """Check TRACE registry wiring uses the saved model and generation policy."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig(temperature=0.7)
    assistant = _assistant()
    trace = SavedChat(
        "c" * 64,
        tmp_path / ("c" * 64),
        datetime(2026, 8, 22, tzinfo=UTC),
        "Night session",
        None,
        assistant,
        chat,
        GenerationConfig(temperature=0.3),
        (),
    )
    runtime = ImmediateRuntime(chat, generation)
    app = ChatApp(
        chat,
        generation,
        assistants=(DiscoveryItem(assistant.name, "BASE", None, assistant),),
        store=cast(ChatStore, RegistryStore(trace)),
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
    )

    async def verify() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("down", "s")
            await pilot.pause()

            content = app.query_one("#specs-body", Static).content
            assert isinstance(content, SpecsDocument)
            assert "TRACE\n" in content.plain
            assert trace.title in content.plain
            assert trace.id in content.plain
            assert "TEMPERATURE\n 0.3" in content.plain
            assert "NOT EVALUATED" in content.plain
            assert runtime.session is None

    asyncio.run(verify())


def test_registry_does_not_open_specs_for_a_fault(tmp_path) -> None:
    """Check that an unavailable registry entry has no SPECS action."""
    app = ChatApp(
        ChatConfig(output=tmp_path),
        GenerationConfig(),
        assistants=(
            DiscoveryItem("Unavailable assistant", "failed", None, None, "invalid"),
        ),
        runtime_factory=cast(
            Callable[..., ChatRuntime],
            lambda chat, generation: ImmediateRuntime(chat, generation),
        ),
    )

    async def verify() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            chooser = app.query_one("#chooser", OptionList)
            assert chooser.highlighted is None
            disabled = chooser.get_component_rich_style(
                "option-list--option-disabled"
            )
            assert disabled.color is not None and disabled.color.number == 8
            assert disabled.dim

            await pilot.press("s")
            await pilot.pause()

            assert app.query_one("#landing").display
            assert app.query_one("#specs").display is False

    asyncio.run(verify())


def test_chat_divider_matches_the_page_gutter(tmp_path) -> None:
    """Check the chat divider has the REGISTRY and SPECS horizontal inset."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
    app = ChatApp(
        chat,
        generation,
        runtime_factory=cast(
            Callable[..., ChatRuntime],
            lambda *_args: ImmediateRuntime(chat, generation),
        ),
        initial_assistant=_assistant(),
    )

    async def verify() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_for_chat(app, pilot)

            divider = app.query_one("#identity-rule", Rule)
            heading = app.query_one("#chat-heading", Horizontal)
            eyes = app.query_one("#chat-eyes", Static)
            assert divider.content_region.x == 2
            assert divider.content_region.width == app.size.width - 4
            assert str(eyes.content) == "· ·"
            assert eyes.content_region.right == heading.content_region.right - 1
            assert eyes.region.right == heading.content_region.right
            assert eyes.styles.color.ansi == 2

    asyncio.run(verify())


def test_registry_expands_and_collapses_trace_names(tmp_path) -> None:
    """Check trace hierarchy, entry emphasis, hints, and Space disclosure."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
    assistant = _assistant()
    saved = SavedChat(
        "a" * 64,
        tmp_path / ("a" * 64),
        datetime(2026, 8, 22, tzinfo=UTC),
        "Night session",
        None,
        assistant,
        chat,
        generation,
        (),
    )
    second = replace(
        saved,
        id="b" * 64,
        path=tmp_path / ("b" * 64),
        title="Morning session",
    )
    runtime = ImmediateRuntime(chat, generation)
    app = ChatApp(
        chat,
        generation,
        assistants=(DiscoveryItem(assistant.name, "BASE", None, assistant),),
        store=cast(ChatStore, RegistryStore(saved, second)),
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
    )

    async def verify() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            chooser = app.query_one("#chooser", OptionList)
            trace = chooser.get_option_at_index(1).prompt
            assert isinstance(trace, Text)
            assert trace.plain.startswith(
                f"{assistant.target_run} Night session"
            )
            assert "\n" not in trace.plain
            second_trace = chooser.get_option_at_index(2).prompt
            assert isinstance(second_trace, Text)
            assert "Morning session" in second_trace.plain
            assert "SPACE DETAILS" in str(
                app.query_one("#catalog-hints", Static).content
            )
            assert "BACKSPACE MANAGE" not in str(
                app.query_one("#catalog-hints", Static).content
            )
            unselected_entry = trace.get_style_at_offset(
                Console(), trace.plain.index("READY")
            )
            assert unselected_entry.color is not None
            assert unselected_entry.color.number == 8
            assert not unselected_entry.dim

            await pilot.press("space")
            await pilot.pause()
            collapsed_first = chooser.get_option_at_index(1).prompt
            collapsed_second = chooser.get_option_at_index(2).prompt
            assert isinstance(collapsed_first, Text)
            assert isinstance(collapsed_second, Text)
            assert "session" not in collapsed_first.plain
            assert "session" not in collapsed_second.plain
            assert "\n" not in collapsed_first.plain
            assert "\n" not in collapsed_second.plain
            assert runtime.session is None

            await pilot.press("space")
            await pilot.pause()
            expanded_first = chooser.get_option_at_index(1).prompt
            expanded_second = chooser.get_option_at_index(2).prompt
            assert isinstance(expanded_first, Text)
            assert isinstance(expanded_second, Text)
            assert "Night session" in expanded_first.plain
            assert "Morning session" in expanded_second.plain

            await pilot.press("down")
            await pilot.pause()
            trace = chooser.get_option_at_index(1).prompt
            assert isinstance(trace, Text)
            selected_entry = trace.get_style_at_offset(
                Console(), trace.plain.index("READY")
            )
            selected_name = trace.get_style_at_offset(
                Console(), trace.plain.index("Night session")
            )
            assert selected_entry.color is None
            assert selected_entry.dim
            assert selected_name.color is None
            assert selected_name.dim
            assert "SPACE DETAILS" in str(
                app.query_one("#catalog-hints", Static).content
            )
            assert "BACKSPACE MANAGE" in str(
                app.query_one("#catalog-hints", Static).content
            )

    asyncio.run(verify())


def test_registry_confirms_trace_erasure(tmp_path) -> None:
    """Check the TRACE erasure menu and safe default action."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
    assistant = _assistant()
    saved = SavedChat(
        "a" * 64,
        tmp_path / ("a" * 64),
        datetime(2026, 8, 22, tzinfo=UTC),
        "Night session",
        None,
        assistant,
        chat,
        generation,
        (),
    )
    runtime = ImmediateRuntime(chat, generation)
    store = RegistryStore(saved)
    app = ChatApp(
        chat,
        generation,
        assistants=(DiscoveryItem(assistant.name, "BASE", None, assistant),),
        store=cast(ChatStore, store),
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
    )

    async def verify() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("down", "backspace")
            await pilot.pause()

            heading = app.screen.query_one("#trace-message", Static).content
            assert isinstance(heading, Text)
            assert heading.plain == "TRACE Night session"
            metadata = heading.get_style_at_offset(
                Console(), heading.plain.index("Night session")
            )
            assert metadata.color is not None
            assert metadata.color.name == "bright_black"
            assert [
                str(button.label) for button in app.screen.query(KeyboardButton)
            ] == ["ERASE", "RENAME", "RETAIN"]
            assert app.focused is not None
            assert app.focused.id == "retain"
            trace_message = app.screen.query_one("#trace-message", Static)
            assert trace_message.content_region.x == app.query_one(
                "#catalog-hints", Static
            ).content_region.x
            assert (
                app.screen.query_one("#erase", KeyboardButton).content_region.x
                - trace_message.content_region.x
                == 1
            )
            assert str(app.query_one("#catalog-hints", Static).content) == (
                "←→ MOVE    ENTER SELECT    ESC RETAIN"
            )
            chooser = app.query_one("#chooser", OptionList)
            app._trace_blink_visible = True
            app._refresh_catalog_prompts(f"saved-{saved.id}")
            subject = chooser.get_option_at_index(1).prompt
            assert isinstance(subject, Text)
            assert "Night session" in subject.plain
            app._trace_blink_visible = False
            app._refresh_catalog_prompts(f"saved-{saved.id}")
            hidden_subject = chooser.get_option_at_index(1).prompt
            assert isinstance(hidden_subject, Text)
            assert "Night session" not in hidden_subject.plain
            assert "\n" not in hidden_subject.plain
            assert hidden_subject.plain.startswith(saved.assistant.target_run)
            assert hidden_subject.plain.endswith("READY")
            app._trace_blink_visible = True
            app._refresh_catalog_prompts(f"saved-{saved.id}")

            await pilot.press("left", "left", "enter")
            await pilot.pause()

            assert store.erased == [saved.id]
            assert len(app.query_one("#chooser", OptionList).options) == 1

    asyncio.run(verify())


def test_registry_renames_trace_with_current_name_as_default(tmp_path) -> None:
    """Check registry TRACE renaming through the styled name field."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
    assistant = _assistant()
    saved = SavedChat(
        "a" * 64,
        tmp_path / ("a" * 64),
        datetime(2026, 8, 22, tzinfo=UTC),
        "Night session",
        None,
        assistant,
        chat,
        generation,
        (),
    )
    runtime = ImmediateRuntime(chat, generation)
    store = RegistryStore(saved)
    app = ChatApp(
        chat,
        generation,
        store=cast(ChatStore, store),
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
    )

    async def verify() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("backspace", "left", "enter")
            await pilot.pause()

            name = app.screen.query_one("#trace-name", Input)
            assert name.has_focus
            assert name.placeholder == "Night session"
            trace_name_message = app.screen.query_one(
                "#trace-name-message", Static
            )
            assert trace_name_message.content_region.x == app.query_one(
                "#catalog-hints", Static
            ).content_region.x
            assert name.content_region.x - trace_name_message.content_region.x == 1
            assert str(app.query_one("#catalog-hints", Static).content) == (
                "ENTER NAME    ESC RETAIN"
            )
            assert app._trace_blink_timer is not None
            app._trace_blink_visible = False
            app._refresh_catalog_prompts(f"saved-{saved.id}")
            subject = app.query_one("#chooser", OptionList).get_option_at_index(0).prompt
            assert isinstance(subject, Text)
            assert "Night session" not in subject.plain
            app._trace_blink_visible = True
            app._refresh_catalog_prompts(f"saved-{saved.id}")
            name.value = "Morning session"
            await pilot.press("enter")
            await pilot.pause()

            assert store.renamed == [(saved.id, "Morning session")]
            trace = app.query_one("#chooser", OptionList).get_option_at_index(0).prompt
            assert isinstance(trace, Text)
            assert "Morning session" in trace.plain
            assert app._trace_blink_timer is None

    asyncio.run(verify())


def test_assistant_episode_displays_as_distinct_message_bubbles() -> None:
    """Check serving interprets serialization boundaries as new messages."""
    segments = _episode_segments(
        "DIXIE",
        "first bubble\n[new message]\nsecond bubble",
    )

    assert segments == (
        ("DIXIE", "first bubble"),
        ("DIXIE", "second bubble"),
    )
    # A reply without boundaries stays one displayed message.
    assert _episode_segments("DIXIE", "plain reply") == (("DIXIE", "plain reply"),)
    # Blank serialization segments are never shown.
    assert _episode_segments("DIXIE", "\n[new message]\nreal") == (("DIXIE", "real"),)


def test_transcript_formats_markdown_and_remains_scrollable(tmp_path) -> None:
    """Check transcript Markdown, identity, telemetry, and navigation."""
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

            transcript = app.query_one("#transcript", Transcript)
            transcript_text = transcript.plain
            assert "USER:" in transcript_text
            assert "DIXIE:" in transcript_text
            assert "USER:\n User **message** 0\n [bold]literal[/bold]" in transcript_text
            assert "IDIOLECT //" not in transcript_text
            console = Console(width=20, color_system=None)
            with console.capture() as capture:
                console.print(transcript.content)
            rendered_lines = capture.get().splitlines()
            assert rendered_lines[0] == "USER:"
            assert rendered_lines[1].strip() == "User message 0"
            assert rendered_lines[1].startswith(" ")
            assert rendered_lines[2].startswith(" ")
            assert rendered_lines[3].startswith(" ")
            assert runtime.session is not None
            assert runtime.session.turns[0].content == (
                "User **message** 0\n[bold]literal[/bold]"
            )
            segments = tuple(console.render(cast(RenderableType, transcript.content)))
            assert any(
                segment.text == "message"
                and segment.style is not None
                and segment.style.bold
                for segment in segments
            )
            assert any(
                segment.text == "reply"
                and segment.style is not None
                and segment.style.italic
                for segment in segments
            )
            assert str(app.query_one("#identity", Static).content) == (
                "IDIOLECT // DIXIE::BASE [M]"
            )
            assert str(app.query_one("#footer", Static).content) == (
                "CTX 500/1,000 (50%)    GEN 64 TOK @ 12.3 TOK/S"
            )
            footer = app.query_one("#footer", Static)
            assert footer.styles.color.ansi == 8
            assert not footer.styles.text_style.bold
            composer_bar = app.query_one("#composer-bar", Horizontal)
            assert footer.region.y == composer_bar.region.bottom
            assert footer.content_region.x == scroller.content_region.x
            assert footer.styles.padding.top == 0
            assert footer.styles.padding.bottom == 0
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

            scroller.scroll_home(animate=False)
            app.query_one(Composer).insert("/")
            await pilot.pause()
            await pilot.pause()
            assert scroller.scroll_y == scroller.max_scroll_y
            await pilot.press("escape")
            await pilot.pause()
            await pilot.pause()
            assert scroller.scroll_y == scroller.max_scroll_y

    asyncio.run(verify())


def test_footer_discloses_secondary_telemetry_when_space_allows(tmp_path) -> None:
    """Check telemetry priority and responsive disclosure."""
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
            footer = app.query_one("#footer", Static)
            assert str(footer.content) == (
                "CTX 500/1,000 (50%)    GEN 64 TOK @ 12.3 TOK/S"
            )

            await pilot.resize_terminal(20, 18)
            await pilot.pause()
            assert str(footer.content) == "CTX 50%"

            await pilot.resize_terminal(65, 18)
            await pilot.pause()
            assert str(footer.content) == (
                "CTX 500/1,000 (50%)    GEN 64 TOK @ 12.3 TOK/S    TTFT 0.42 S"
            )

            await pilot.resize_terminal(80, 18)
            await pilot.pause()
            assert str(footer.content) == (
                "CTX 500/1,000 (50%)    GEN 64 TOK @ 12.3 TOK/S"
                "    TTFT 0.42 S    MEM 3.25 GB"
            )

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
            prompt = app.query_one("#composer-prompt", Static)
            assert str(prompt.content) == ">"
            assert prompt.region.x < composer.region.x
            composer.insert("first")
            await pilot.press("shift+enter")
            composer.insert("second")
            assert composer.text == "first\nsecond"
            assert prompt.region.y == composer.region.y

            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if "Synthetic reply" in app.query_one(Transcript).plain:
                    break
            transcript = app.query_one(Transcript).plain
            assert "USER:\n first\n second" in transcript
            assert "DIXIE:\n Synthetic reply" in transcript
            assert composer.text == ""

    asyncio.run(verify())


def test_transcript_link_opens_validated_web_destination(tmp_path, monkeypatch) -> None:
    """Check pointer activation for one rendered HTTP link."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig(max_prompt_tokens=100)
    runtime = ImmediateRuntime(chat, generation)
    app = ChatApp(
        chat,
        generation,
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
        initial_assistant=_assistant(),
    )
    opened: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        app,
        "open_url",
        lambda url, *, new_tab=True: opened.append((url, new_tab)),
    )

    async def verify() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_for_chat(app, pilot)
            app.query_one(Composer).insert("[SITE](https://example.test/guide)")
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if "Synthetic reply" in app.query_one(Transcript).plain:
                    break

            transcript = app.query_one(Transcript)
            await pilot.click(transcript, offset=(2, 1))
            await pilot.pause()
            assert opened == [("https://example.test/guide", True)]

            app.action_open_link("mailto:user@example.test")
            assert opened == [("https://example.test/guide", True)]

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
            scroller = app.query_one("#transcript-scroll", VerticalScroll)
            rendered = status.render()
            assert status.state == "PREFILL 0/4"
            assert isinstance(rendered, Text)
            assert rendered.plain.endswith(" PREFILL 0/4")
            assert status.display is True
            assert status.content_region.x == scroller.content_region.x
            assert status.styles.color == app.query_one("#catalog-subtitle").styles.color

            runtime.release_prefill.set()
            assert await asyncio.to_thread(runtime.generation_finished.wait, 1)
            for _ in range(20):
                await pilot.pause()
                reply_visible = "Progress reply" in app.query_one(Transcript).plain
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
            status = app.query_one("#load-status", LoadingStatus)
            for _ in range(20):
                await pilot.pause()
                if status.state == "LOADING":
                    break
            else:
                raise AssertionError("The loading state did not appear")
            rendered = status.render()
            assert isinstance(rendered, Text)
            assert rendered.plain.endswith(" LOADING")
            assert "// MODEL SESSION" not in rendered.plain
            assert status.styles.color == app.query_one("#catalog-subtitle").styles.color
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
            assert app.focused.id == "discard"
            assert str(app.screen.query_one("#confirm-message", Static).content) == (
                "CONNECTION"
            )
            assert app.screen.query_one("#confirm-message", Static).styles.color.ansi == 7
            assert app.screen.query_one(
                "#confirm-message", Static
            ).styles.text_style.bold
            assert (
                app.screen.query_one("#discard", KeyboardButton).content_region.x
                - app.screen.query_one("#confirm-message", Static).content_region.x
                == 1
            )
            assert [
                str(button.label) for button in app.screen.query(KeyboardButton)
            ] == ["DISCONNECT", "SAVE", "RESUME"]
            await pilot.press("right", "enter")
            await pilot.pause()
            assert app.screen.query_one("#trace-name", Input).has_focus
            assert str(app.query_one("#footer", Static).content) == (
                "ENTER SAVE    ESC RESUME"
            )
            await pilot.press("enter")
            await pilot.pause()

            assert runtime.closed is False
            assert runtime.session is session
            assert runtime.session.turns[-1].content == "unsaved"
            assert app.query_one("#chat").display

    asyncio.run(verify())


def test_save_requests_trace_name_and_uses_default_for_blank(tmp_path) -> None:
    """Check explicit and blank trace names before registry navigation."""

    async def verify(value: str, expected: str | None) -> None:
        chat = ChatConfig(output=tmp_path)
        generation = GenerationConfig()
        runtime = ImmediateRuntime(chat, generation)
        store = RecordingStore()
        app = ChatApp(
            chat,
            generation,
            store=cast(ChatStore, store),
            runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
            initial_assistant=_assistant(),
        )
        async with app.run_test() as pilot:
            await _wait_for_chat(app, pilot)
            assert runtime.session is not None
            runtime.session.add_user("default trace name")
            app.query_one(Composer).insert("/registry")
            await pilot.press("enter", "right", "enter")
            await pilot.pause()

            name = app.screen.query_one("#trace-name", Input)
            assert name.has_focus
            assert name.placeholder == "default trace name"
            name.value = value
            await pilot.press("enter")
            await pilot.pause()

            assert store.titles == [expected]
            assert app.query_one("#landing").display

    asyncio.run(verify("Named trace", "Named trace"))
    asyncio.run(verify("   ", None))


def test_command_menu_filters_navigates_and_returns_to_registry(tmp_path) -> None:
    """Check keyboard command discovery, dismissal, and selection."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
    runtime = ImmediateRuntime(chat, generation)
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
            composer.insert("/")
            await pilot.pause()

            menu = app.query_one("#command-menu", CommandMenu)
            exit_button = menu.query_one("#command-exit", Horizontal)
            registry_button = menu.query_one("#command-registry", Horizontal)
            save_button = menu.query_one("#command-save", Horizontal)
            specs_button = menu.query_one("#command-specs", Horizontal)
            assert menu.display is True
            assert str(menu.query_one("#command-message", Static).content) == "COMMAND"
            assert menu.query_one("#command-message", Static).styles.color.ansi == 7
            assert menu.query_one("#command-message", Static).styles.text_style.bold
            assert (
                exit_button.query_one(".command-name", Static).content_region.x
                - menu.query_one("#command-message", Static).content_region.x
                == 1
            )
            assert str(
                exit_button.query_one(".command-description", Static).content
            ) == "Exit IDIOLECT."
            assert str(
                registry_button.query_one(".command-description", Static).content
            ) == "Return to REGISTRY."
            assert str(
                save_button.query_one(".command-description", Static).content
            ) == "Save TRACE."
            assert str(
                specs_button.query_one(".command-description", Static).content
            ) == "View MODEL SPECS."
            assert specs_button.display is False
            assert save_button.has_class("-disabled")
            assert save_button.has_class("-selected") is False
            assert save_button.query_one(".command-name", Static).styles.color == (
                app.query_one("#catalog-subtitle").styles.color
            )
            assert registry_button.region.y == exit_button.region.y + 2
            assert exit_button.has_class("-selected")
            assert registry_button.has_class("-selected") is False
            selected_name = exit_button.query_one(".command-name", Static)
            selected_description = exit_button.query_one(
                ".command-description", Static
            )
            assert selected_description.styles.color == selected_name.styles.color
            assert selected_description.styles.text_style.dim
            assert not selected_description.styles.text_style.bold
            assert composer.has_focus
            assert str(app.query_one("#footer", Static).content) == (
                "↑↓ MOVE    ENTER COMMAND    ESC CLOSE"
            )

            await pilot.press("down")
            assert menu.query_one("#command-echo", Horizontal).has_class(
                "-selected"
            )
            await pilot.press("down")
            assert registry_button.has_class("-selected")
            await pilot.press("down")
            assert specs_button.has_class("-selected")
            assert specs_button.display
            assert exit_button.display is False
            await pilot.press("down")
            assert exit_button.has_class("-selected")
            await pilot.press("up")
            assert specs_button.has_class("-selected")
            await pilot.press("up")
            assert registry_button.has_class("-selected")

            await pilot.click("#command-registry")
            await pilot.pause()
            assert app.query_one("#chat").display

            await pilot.press("escape")
            assert menu.display is False
            assert composer.text == "/"
            assert str(app.query_one("#footer", Static).content) == ""

            composer.clear()
            composer.insert("prompt")
            composer.move_cursor((0, 0))
            composer.insert("/ ")
            await pilot.pause()
            assert menu.display
            assert composer.cursor_location == (0, 2)
            await pilot.press("right")
            await pilot.pause()
            assert composer.cursor_location == (0, 3)
            assert not menu.display
            await pilot.press("left")
            await pilot.pause()
            assert composer.cursor_location == (0, 2)
            assert menu.display
            await pilot.press("escape")
            await pilot.pause()
            assert menu.display is False
            assert composer.text == "/ prompt"

            composer.clear()
            composer.insert("/")
            await pilot.pause()
            composer.insert("r")
            await pilot.pause()
            assert exit_button.display is False
            assert registry_button.display is True

            await pilot.press("tab")
            await pilot.pause()
            assert composer.text == "/r"
            assert menu.display
            assert registry_button.has_class("-selected")
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one("#landing").display
            assert composer.text == ""

    asyncio.run(verify())


def test_echo_command_uses_an_env_turn_and_argument_bar(tmp_path) -> None:
    """Check argument commands activate a bar and stay out of model context."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
    runtime = ImmediateRuntime(chat, generation)
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
            composer.insert("/echo")
            await pilot.pause()

            menu = app.query_one("#command-menu", CommandMenu)
            assert menu.display
            assert menu.query_one("#command-echo", Horizontal).has_class(
                "-selected"
            )
            await pilot.press("enter")
            await pilot.pause()

            command_bar = app.query_one("#command-bar", Static)
            assert app._command_selected
            assert command_bar.display
            assert isinstance(command_bar.content, Text)
            assert command_bar.content.plain == "/ ECHO ENV echo."
            command_style = command_bar.content.get_style_at_offset(Console(), 0)
            description_style = command_bar.content.get_style_at_offset(
                Console(), command_bar.content.plain.index("ENV")
            )
            assert command_style.dim
            assert command_style.color is not None
            assert command_style.color.name == "green"
            assert description_style.color is not None
            assert description_style.color.name == "bright_black"
            assert composer.text == ""
            composer.insert("@")
            await pilot.pause()
            assert not app.query_one("#reference-menu", ReferenceMenu).display

            composer.insert("hello")
            await pilot.press("enter")
            await pilot.pause()

            assert not command_bar.display
            assert runtime.session is not None
            assert runtime.session.turns[-1].role == "env"
            assert runtime.session.turns[-1].content == "@hello"
            transcript = app.query_one(Transcript)
            assert "ENV:\n @hello" in transcript.plain
            segments = tuple(
                Console().render(cast(RenderableType, transcript.content))
            )
            env_label = next(segment for segment in segments if segment.text == "ENV:")
            env_text = next(segment for segment in segments if "@hello" in segment.text)
            assert env_label.style is not None and env_label.style.dim
            assert env_label.style.color is not None
            assert env_label.style.color.name == "green"
            assert env_text.style is not None and env_text.style.dim
            assert env_text.style.color is not None
            assert env_text.style.color.name == "bright_black"

            runtime.session.add_user("next")
            assert [turn.role for turn in runtime.session.turns] == ["env", "user"]

    asyncio.run(verify())


def test_echo_command_can_be_selected_after_prefilled_arguments(tmp_path) -> None:
    """Check slash-token selection still works when the cursor follows args."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
    runtime = ImmediateRuntime(chat, generation)
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
            composer.insert("/echo hello")
            await pilot.press("enter")
            await pilot.pause()

            assert app._command_selected
            assert composer.text == "hello"
            assert app.query_one("#command-bar", Static).display

            await pilot.press("enter")
            await pilot.pause()
            assert runtime.session is not None
            assert runtime.session.turns[-1].content == "hello"

    asyncio.run(verify())


def test_slash_command_menu_opens_for_a_token_inside_prompt(tmp_path) -> None:
    """Check slash commands are discoverable away from prompt offset zero."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
    runtime = ImmediateRuntime(chat, generation)
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
            composer.insert("prompt /ec")
            await pilot.pause()

            menu = app.query_one("#command-menu", CommandMenu)
            assert menu.display
            assert menu.query_one("#command-echo", Horizontal).has_class(
                "-selected"
            )
            await pilot.press("enter")
            await pilot.pause()

            assert app._command_selected
            assert composer.text == "prompt "
            assert app.query_one("#command-bar", Static).display

    asyncio.run(verify())


def test_reference_menu_selects_bubble_and_escape_clears_reference(tmp_path) -> None:
    """Check reference selection, replacement, and Escape semantics."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
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
            assert runtime.session is not None
            runtime.session.add_user("first")
            runtime.session.begin_generation()
            runtime.session.finish_generation(
                "one\n[new message]\ntwo",
                "stop",
                1,
                TurnTelemetry(2, 1),
            )
            app._render_transcript()

            composer = app.query_one(Composer)
            composer.insert("@")
            await pilot.pause()

            menu = app.query_one("#reference-menu", ReferenceMenu)
            assert menu.display
            assert str(app.query_one("#reference-message", Static).content) == "REF"
            assert str(menu.query_one("#reference-2 .reference-name", Static).content) == "DIXIE:02"
            assert menu.query_one("#reference-2").has_class("-selected")
            assert str(app.query_one("#footer", Static).content) == (
                "↑↓ MOVE    ENTER REF    ESC CLOSE"
            )

            composer_bar = app.query_one("#composer-bar", Horizontal)
            before_geometry = (
                composer_bar.region.x,
                composer_bar.region.width,
                app.query_one("#composer-prompt", Static).region.x,
                composer.region.x,
                composer_bar.styles.border_top,
            )
            assert composer_bar.styles.border_top[1] == (
                app.query_one("#composer-prompt", Static).styles.color
            )
            await pilot.press("up", "enter")
            await pilot.pause()
            bar = app.query_one("#reference-bar", ReferenceBar)
            assert bar.display
            assert isinstance(bar.content, Text)
            assert "@ DIXIE:01" in bar.content.plain
            at_style = bar.content.get_style_at_offset(Console(), 0)
            assert at_style.dim
            assert at_style.color is not None
            assert at_style.color.name == "green"
            assert bar.region.y + bar.region.height == composer_bar.region.y
            assert bar.styles.border_top[1].ansi == (
                app.query_one("#composer-prompt", Static).styles.color.ansi
            )
            border_style = bar.get_style_at(1, 0)
            assert border_style.dim
            assert border_style.color is not None
            assert border_style.color.name == "green"
            bottom_border_style = bar.get_style_at(1, bar.size.height - 1)
            assert bottom_border_style.dim
            assert bottom_border_style.color is not None
            assert bottom_border_style.color.name == "green"
            assert bar.get_style_at(0, 1).dim
            assert composer.text == ""
            reference_style = bar.content.get_style_at_offset(Console(), 2)
            assert reference_style.dim
            assert reference_style.color is not None
            assert reference_style.color.name == "green"
            assert (
                composer_bar.region.x,
                composer_bar.region.width,
                app.query_one("#composer-prompt", Static).region.x,
                composer.region.x,
                composer_bar.styles.border_top,
            ) == before_geometry

            composer.insert("follow-up")
            await pilot.pause()
            assert app._reference_selected
            assert composer.reference_selected
            assert composer.has_focus
            assert not composer.command_menu_active
            assert not composer.reference_menu_active
            await pilot.press("escape")
            await pilot.pause()
            assert not bar.display
            assert composer.text == "follow-up"

            composer.clear()
            composer.insert("@D")
            await pilot.pause()
            assert menu.display
            assert str(menu.query_one("#reference-0 .reference-name", Static).content) == "DIXIE:01"
            assert str(menu.query_one("#reference-1 .reference-name", Static).content) == "DIXIE:02"
            assert menu.query_one("#reference-2").display is False
            await pilot.press("enter")
            await pilot.pause()
            assert composer.text == ""

            composer.insert("prompt")
            composer.insert("@U")
            await pilot.pause()
            assert menu.display
            assert str(menu.query_one("#reference-0 .reference-name", Static).content) == "USER:00"
            await pilot.press("enter")
            await pilot.pause()
            assert composer.text == "prompt"
            assert "@ USER:00" in bar.content.plain
            assert app._reference_selected

            composer.clear()
            composer.insert("@first")
            await pilot.pause()
            assert not menu.display

            composer.insert("x")
            await pilot.pause()
            assert not menu.display
            await pilot.press("backspace")
            await pilot.pause()
            assert not menu.display

            composer.clear()
            composer.insert("prompt")
            composer.move_cursor((0, 0))
            composer.insert("@ ")
            await pilot.pause()
            assert menu.display
            assert str(menu.query_one("#reference-2 .reference-name", Static).content) == "DIXIE:02"
            assert composer.cursor_location == (0, 2)
            await pilot.press("right")
            await pilot.pause()
            assert composer.cursor_location == (0, 3)
            assert not menu.display
            await pilot.press("left")
            await pilot.pause()
            assert composer.cursor_location == (0, 2)
            assert menu.display
            await pilot.press("enter")
            await pilot.pause()
            assert composer.text == "prompt"
            assert app._reference_selected

            runtime.session.add_user("sent", reference=2)
            app._render_transcript()
            transcript = app.query_one(Transcript)
            assert "USER:\n REF @DIXIE:02\n sent" in transcript.plain
            segments = tuple(
                Console().render(cast(RenderableType, transcript.content))
            )
            annotation = next(
                segment for segment in segments if "REF @DIXIE:02" in segment.text
            )
            assert annotation.style is not None
            assert annotation.style.dim
            assert annotation.style.color is not None
            assert annotation.style.color.name == "green"
            speaker = next(segment for segment in segments if segment.text == "USER:")
            assert speaker.style is not None
            assert not speaker.style.dim
            assert speaker.style.color is not None
            assert speaker.style.color.name == "green"

    asyncio.run(verify())


def test_specs_command_restores_the_unchanged_trace_chat(tmp_path) -> None:
    """Check temporary TRACE details preserve the live chat state."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig(temperature=0.3, max_prompt_tokens=100)
    assistant = _assistant()
    trace = SavedChat(
        "c" * 64,
        tmp_path / ("c" * 64),
        datetime(2026, 8, 22, tzinfo=UTC),
        "Night session",
        None,
        assistant,
        chat,
        generation,
        (),
    )
    runtime = ImmediateRuntime(chat, generation)
    app = ChatApp(
        chat,
        GenerationConfig(temperature=0.9),
        initial_chat=trace,
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
    )

    async def verify() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_for_chat(app, pilot)
            composer = app.query_one(Composer)
            composer.insert("Keep this turn")
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if "Synthetic reply" in app.query_one(Transcript).plain:
                    break

            session = runtime.session
            assert session is not None
            assert session.dirty
            turns = tuple(session.turns)
            fingerprint = session.fingerprint
            transcript = app.query_one(Transcript)
            transcript_text = transcript.plain

            composer.insert("/specs")
            await pilot.press("enter")
            await pilot.pause()

            assert app.query_one("#chat").display is False
            assert app.query_one("#specs").display
            assert str(app.query_one("#specs-hints", Static).content) == (
                "↑↓ SCROLL    ESC CHAT    CTRL+C QUIT"
            )
            content = app.query_one("#specs-body", Static).content
            assert isinstance(content, SpecsDocument)
            assert "TYPE\n TRACE\n" in content.plain
            assert f"TRACE ID\n {trace.id}\n" in content.plain
            assert "TEMPERATURE\n 0.3\n" in content.plain

            await pilot.press("right")
            await pilot.pause()
            assert f"TRACE ID\n {trace.id}\n" in content.plain
            await pilot.press("escape")
            await pilot.pause()

            assert app.query_one("#chat").display
            assert app.query_one("#specs").display is False
            assert composer.has_focus
            assert composer.text == ""
            assert runtime.session is session
            assert tuple(session.turns) == turns
            assert session.fingerprint == fingerprint
            assert session.dirty
            assert transcript.plain == transcript_text

    asyncio.run(verify())


def test_save_command_checkpoints_only_new_trace_data(tmp_path) -> None:
    """Check checkpoint naming, clean-state disabling, and chat continuity."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
    runtime = ImmediateRuntime(chat, generation)
    store = RecordingStore()
    app = ChatApp(
        chat,
        generation,
        store=cast(ChatStore, store),
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
        initial_assistant=_assistant(),
    )

    async def verify() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_for_chat(app, pilot)
            composer = app.query_one(Composer)
            assert runtime.session is not None
            runtime.session.add_user("checkpoint data")
            composer.insert("/save")
            await pilot.press("enter")
            await pilot.pause()

            name = app.screen.query_one("#trace-name", Input)
            assert name.has_focus
            assert name.placeholder == "checkpoint data"
            await pilot.press("enter")
            await pilot.pause()

            assert store.titles == [None]
            assert app.query_one("#chat").display
            assert runtime.closed is False
            assert runtime.session.dirty is False
            alert = app.query_one("#chat-alert", LoadingStatus)
            composer_bar = app.query_one("#composer-bar", Horizontal)
            assert alert.state == "Saved aaaaaaaa — default trace name."
            assert alert.display
            assert alert.region.bottom == composer_bar.region.y
            assert alert.styles.text_align == "right"
            assert alert.styles.color.ansi == 8
            assert not alert.has_class("-error")

            composer.insert("/save")
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.screen.query("#trace-name")) == 0
            assert store.titles == [None]
            assert app.query_one("#chat-alert", LoadingStatus).state == (
                "The TRACE has no new data to save."
            )
            assert alert.styles.color.ansi == 1
            assert alert.has_class("-error")

    asyncio.run(verify())


def test_chat_errors_align_right_above_composer(tmp_path) -> None:
    """Check inline failure placement and loading-status visual language."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
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
            app._loading = True
            composer.insert("message during load")
            await pilot.press("enter")
            await pilot.pause()

            failure = app.query_one("#chat-alert", LoadingStatus)
            assert failure.state == "CONNECTION is not ready."
            assert failure.display

            app._loading = False
            composer.clear()
            composer.insert("/unknown")
            await pilot.press("enter")
            await pilot.pause()

            loading = app.query_one("#status", LoadingStatus)
            composer_bar = app.query_one("#composer-bar", Horizontal)
            assert failure.state == "Unknown chat command."
            assert failure.display
            assert failure.region.bottom == composer_bar.region.y
            assert failure.content_region.x == loading.content_region.x
            assert failure.styles.text_align == "right"
            assert failure.styles.color.ansi == 1
            assert failure.has_class("-error")

    asyncio.run(verify())


def test_dirty_slash_commands_open_connection_confirmation(tmp_path) -> None:
    """Check that both navigation commands protect unsaved turns."""

    async def verify(command: str) -> None:
        chat = ChatConfig(output=tmp_path)
        generation = GenerationConfig()
        runtime = ImmediateRuntime(chat, generation)
        app = ChatApp(
            chat,
            generation,
            runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
            initial_assistant=_assistant(),
        )
        async with app.run_test() as pilot:
            await _wait_for_chat(app, pilot)
            assert runtime.session is not None
            runtime.session.add_user("unsaved")
            app.query_one(Composer).insert(command)
            await pilot.press("enter")
            await pilot.pause()

            assert str(app.screen.query_one("#confirm-message", Static).content) == (
                "CONNECTION"
            )
            assert app.focused is not None
            assert app.focused.id == "discard"
            assert str(app.query_one("#footer", Static).content) == (
                "←→ MOVE    ENTER SELECT    ESC RESUME"
            )
            assert (
                app.query_one("#transcript-scroll", VerticalScroll)
                .styles.padding.bottom
                == 3
            )

            await pilot.press("escape")
            await pilot.pause()
            assert app.query_one("#chat").display
            assert str(app.query_one("#footer", Static).content) == ""
            assert (
                app.query_one("#transcript-scroll", VerticalScroll)
                .styles.padding.bottom
                == 1
            )

    asyncio.run(verify("/exit"))
    asyncio.run(verify("/registry"))


def test_commands_follow_generation_navigation_rules(tmp_path) -> None:
    """Check that exit cancels and registry stays unavailable during generation."""
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
        async with app.run_test() as pilot:
            await _wait_for_chat(app, pilot)
            composer = app.query_one(Composer)
            composer.insert("hold reply")
            await pilot.press("enter")
            assert await asyncio.to_thread(runtime.prefill_started.wait, 1)

            composer.insert("/registry")
            await pilot.pause()
            registry = app.query_one("#command-registry", Horizontal)
            assert registry.has_class("-disabled")
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one("#chat").display
            assert runtime.cancelled.is_set() is False

            composer.insert("/exit")
            await pilot.press("enter")
            assert await asyncio.to_thread(runtime.cancelled.wait, 1)
            assert await asyncio.to_thread(runtime.generation_finished.wait, 1)
            await pilot.pause()
            assert app.query_one("#chat").display
            assert runtime.closed is False

    try:
        asyncio.run(verify())
    finally:
        runtime.release_prefill.set()


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

    backend_versions: ClassVar[dict[str, str | None]] = {}

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

    def attach(self, session: ChatSession) -> None:
        """Wait before attaching one existing fake session."""
        self.session = session
        self.state = WorkerState.LOADING
        self.started.set()
        if not self.release.wait(2):
            raise RuntimeError("Synthetic model attach timed out")
        self.state = WorkerState.READY

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
                ChatTurn("user", f"User **message** {index}\n[bold]literal[/bold]"),
                ChatTurn(
                    "assistant",
                    f"Assistant *reply* {index}\nwith a second line",
                    telemetry=TurnTelemetry(
                        prompt_tokens=500,
                        generated_tokens=64,
                        generation_throughput=12.3,
                        time_to_first_token=0.42,
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
        self.cancelled = threading.Event()

    def cancel(self) -> None:
        """Release generation after recording cancellation."""
        self.cancelled.set()
        self.release_prefill.set()

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


class FailingStore:
    """Reject explicit chat saves."""

    def leaves(self) -> tuple[()]:
        """Return no saved chat rows."""
        return ()

    def save(
        self,
        _session: ChatSession,
        _title: str | None = None,
        _backend_versions: dict[str, str | None] | None = None,
    ) -> None:
        """Raise a controlled storage error."""
        raise ChatStorageError("Synthetic disk failure")


class RecordingStore:
    """Record requested trace titles without writing private artifacts."""

    def __init__(self) -> None:
        """Create an empty title record."""
        self.titles: list[str | None] = []

    def leaves(self) -> tuple[()]:
        """Return no saved chat rows."""
        return ()

    def save(
        self,
        session: ChatSession,
        title: str | None = None,
        backend_versions: dict[str, str | None] | None = None,
    ) -> SimpleNamespace:
        """Record the requested title and return one synthetic trace."""
        del backend_versions
        """Record the requested title and return one synthetic trace."""
        self.titles.append(title)
        session.mark_saved("a" * 64, title or "default trace name")
        return SimpleNamespace(id="a" * 64, title=title or "default trace name")


class RegistryStore:
    """Return fixed saved traces for registry tests."""

    def __init__(self, *saved: SavedChat) -> None:
        """Set the fixed trace rows."""
        self.saved = list(saved)
        self.erased: list[str] = []
        self.renamed: list[tuple[str, str]] = []

    def leaves(self) -> tuple[SavedChat, ...]:
        """Return the fixed trace rows."""
        return tuple(self.saved)

    def erase(self, chat_id: str) -> None:
        """Remove one fixed trace row."""
        self.erased.append(chat_id)
        self.saved = [saved for saved in self.saved if saved.id != chat_id]

    def rename(self, chat_id: str, title: str) -> SavedChat:
        """Replace one fixed trace row with its renamed form."""
        self.renamed.append((chat_id, title))
        current = next(saved for saved in self.saved if saved.id == chat_id)
        renamed = replace(current, id="b" * 64, title=title)
        self.saved = [renamed if saved.id == chat_id else saved for saved in self.saved]
        return renamed


def _assistant() -> Assistant:
    return Assistant(
        name="IDIOLECT // DIXIE::BASE [M]",
        target_name="DIXIE",
        model_basename="M",
        run=None,
        dataset=None,
        context_messages=32,
        base_model=ModelSpec(
            "example/M",
            "hub",
            "revision-1",
            None,
            False,
        ),
        base_data=TrainDataConfig(
            format="chat-template",
            system_prompt="Speak with terse technical precision.",
            prompt_role="user",
            completion_role="assistant",
        ),
    )
