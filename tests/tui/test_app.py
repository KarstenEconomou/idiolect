"""Test the terminal chat application."""

import asyncio
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import ClassVar, cast

from rich.color import ColorTriplet
from rich.console import Console, RenderableType
from rich.text import Text
from textual import events
from textual.containers import Horizontal, HorizontalScroll, Vertical, VerticalScroll
from textual.pilot import Pilot
from textual.widgets import Input, OptionList, Rule, Static

from idiolect.chat.discovery import Assistant, DiscoveryItem
from idiolect.chat.runtime import ChatRuntime
from idiolect.chat.state import ChatSession, ChatTurn, TurnTelemetry, prepare_prompt
from idiolect.chat.storage import ChatStorageError, ChatStore, SavedChat
from idiolect.chat.worker import LoadProbe, RuntimeProbe, WorkerState
from idiolect.config import ChatConfig, GenerationConfig, TrainDataConfig
from idiolect.model import ModelSpec
from idiolect.tui.app import ChatApp
from idiolect.tui.specs import SpecsDocument
from idiolect.tui.widgets import (
    CommandMenu,
    Composer,
    ControlSheet,
    KeyboardButton,
    ReferenceBar,
    ReferenceMenu,
    SpecsScroll,
    StatusLine,
    TraceMenuModal,
    Transcript,
)

_HACKER_RGB = ColorTriplet(128, 255, 0)


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
            assert mark_style.color is not None
            mark_color = mark_style.color.get_truecolor()
            assert (mark_color.red, mark_color.green, mark_color.blue) == (128, 255, 0)
            assert mark_style.bold
            assert tagline_style.color is not None
            tagline_color = tagline_style.color.get_truecolor()
            assert (tagline_color.red, tagline_color.green, tagline_color.blue) == (
                128,
                255,
                0,
            )
            assert not tagline_style.bold
            assert tagline_style.dim
            watermark_lines = watermark.plain.splitlines()
            assert watermark_lines[3].index("Someone, reconstructed.") == (
                watermark_lines[2].index("IDIOLECT") + 1
            )
            assert str(app.query_one("#catalog-title", Static).content) == "REGISTRY"
            assert str(app.query_one("#catalog-description", Static).content) == (
                "Select a CONSTRUCT to establish a LINK."
            )
            assert app.query_one("#catalog-columns", Static).styles.color.ansi == 7
            columns = str(app.query_one("#catalog-columns", Static).content)
            assert "CONSTRUCT" in columns
            assert "BASE" in columns
            assert "TYPE" in str(app.query_one("#catalog-columns", Static).content)
            assert "STATUS" in columns
            assert "ENTRY" not in columns
            assert len(app.query("#catalog-summary")) == 0
            assert len(app.query("#search")) == 0
            assert app.query_one("#landing").region.x == 0
            assert app.query_one("#landing-box").region.width == 80
            assert app.query_one("#catalog-heading").content_region.x == 2
            assert app.query_one("#catalog-description").content_region.x == 3
            assert chooser.content_region.x == 2
            assert chooser.get_component_styles("option-list--option").padding.left == 1
            assert app.query_one("#catalog-hints", Static).content_region.x == 2
            assert chooser.highlighted == 1
            assert chooser.has_focus
            prompt = chooser.get_option_at_index(1).prompt
            assert isinstance(prompt, Text)
            assert "DIXIE::BASE" in prompt.plain
            assert "M" in prompt.plain
            assert "READY" in prompt.plain
            assert str(app.query_one("#catalog-hints", Static).content) == (
                "↑↓ MOVE    ↩ CONNECT    S SPECS    C CHROMA    ⌃C TERMINATE"
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


def test_registry_chroma_menu_previews_all_themes_and_persists(tmp_path) -> None:
    """Check CHROMA navigation, live preview, selection, and persistence."""
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
            assert app.has_class("-accent-green")
            await pilot.press("c")
            await pilot.pause()
            dialog = app.screen.query_one("#chroma-dialog", Vertical)
            assert not dialog.has_class("-unplaced")
            assert (
                dialog.region.bottom == app.query_one("#catalog-hints", Static).region.y
            )
            assert str(app.screen.query_one("#chroma-message", Static).content) == (
                "CHROMA"
            )
            assert [
                str(button.label) for button in app.screen.query(KeyboardButton)
            ] == [
                "LOCKSMITH",
                "LOOKOUT",
                "PICKPOCKET",
                "CLEANER",
                "MOLE",
                "GENTLEMAN",
                "HACKER",
                "REDHEAD",
            ]
            assert app.focused is not None
            assert app.focused.id == "chroma-green"
            heading = app.screen.query_one("#chroma-message", Static)
            blue = app.screen.query_one("#chroma-blue", KeyboardButton)
            assert blue.render_line(0).text.startswith("LOCKSMITH")
            assert blue.content_region.x - heading.content_region.x == 1
            assert (
                app.screen.query_one(
                    "#chroma-orange",
                    KeyboardButton,
                ).region.right
                <= dialog.content_region.right
            )
            assert str(app.query_one("#catalog-hints", Static).content) == (
                "←→ MOVE    ↩ EQUIP    ⎋ CANCEL"
            )

            await pilot.press("up", "down")
            assert app.focused is not None
            assert app.focused.id == "chroma-green"
            assert app.has_class("-accent-green")

            await pilot.resize_terminal(24, 24)
            await pilot.pause()
            actions = app.screen.query_one("#chroma-actions", HorizontalScroll)
            assert actions.scroll_x > 0
            assert actions.content_region.x - heading.content_region.x == 1
            await pilot.press("right")
            orange = app.screen.query_one("#chroma-orange", KeyboardButton)
            assert orange.render_line(0).text == "REDHEAD"
            assert orange.region.right == actions.content_region.right
            await pilot.press("left")
            await pilot.resize_terminal(80, 24)

            await pilot.click("#chroma-blue")
            await pilot.pause()
            assert app.focused is not None
            assert app.focused.id == "chroma-green"
            assert app.has_class("-accent-green")
            assert len(app.screen.query("#chroma-dialog")) == 1

            for name, accent in (
                ("orange", "#FF5500"),
                ("blue", "#00AAFF"),
                ("red", "#FF002B"),
                ("yellow", "#FFD500"),
                ("pink", "#FF00D5"),
                ("violet", "#AA00FF"),
                ("teal", "#00FFFF"),
                ("green", "#80FF00"),
            ):
                await pilot.press("right")
                await pilot.pause()
                assert app.has_class(f"-accent-{name}")
                selected = chooser.get_component_rich_style(
                    "option-list--option-highlighted"
                )
                assert selected.color is not None
                selected_color = selected.color.get_truecolor()
                expected = tuple(bytes.fromhex(accent[1:]))
                assert (
                    selected_color.red,
                    selected_color.green,
                    selected_color.blue,
                ) == expected
                watermark = app.query_one("#watermark", Static).content
                assert isinstance(watermark, Text)
                style = watermark.get_style_at_offset(Console(), 0)
                assert style.color is not None
                truecolor = style.color.get_truecolor()
                assert (truecolor.red, truecolor.green, truecolor.blue) == expected

            hints = str(app.query_one("#catalog-hints", Static).content)
            assert hints == "←→ MOVE    ↩ EQUIP    ⎋ CANCEL"
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one("#catalog-alert", StatusLine).state == (
                "SYS: ACK HACKER equipped."
            )
            assert str(app.query_one("#catalog-hints", Static).content) == (
                "↑↓ MOVE    ↩ CONNECT    S SPECS    C CHROMA    ⌃C TERMINATE"
            )
            await pilot.press("s")
            await pilot.pause()
            assert app.query_one("#specs-identity", Static).styles.color.hex == (
                "#80FF00"
            )
            assert app.query_one("#specs-link", Static).display is False

            assert app.has_class("-accent-green")
            await pilot.press("escape", "enter")
            await _wait_for_chat(app, pilot)
            assert app.query_one("#identity", Static).styles.color.hex == "#80FF00"
            chat_link = app.query_one("#chat-link", Static)
            assert re.fullmatch(r"LINK#[0-9A-F]{6}", str(chat_link.content))
            assert chat_link.styles.color.hex == "#80FF00"
            assert app.query_one("#composer-prompt", Static).styles.color.hex == (
                "#80FF00"
            )
            transcript = app.query_one("#transcript", Transcript)
            transcript.set_turns((("OP", "Theme check"),))
            segments = tuple(Console().render(cast(RenderableType, transcript.content)))
            label = next(segment for segment in segments if segment.text == "OP:")
            assert label.style is not None
            assert label.style.color is not None
            label_color = label.style.color.get_truecolor()
            assert (label_color.red, label_color.green, label_color.blue) == (
                128,
                255,
                0,
            )

    asyncio.run(verify())


def test_chroma_command_opens_menu_in_chat(tmp_path) -> None:
    """Check that /chroma opens and restores the chat theme menu."""
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
            composer.insert("Visible dialogue")
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if "Synthetic reply" in app.query_one(Transcript).plain:
                    break
            footer_before = str(app.query_one("#footer", Static).content)
            composer.insert("/chroma")
            await pilot.press("enter")
            await pilot.pause()

            dialog = app.screen.query_one("#chroma-dialog", Vertical)
            assert app.query_one("#chat").display
            assert (
                dialog.region.bottom
                == app.query_one("#composer-bar", Horizontal).region.y
            )
            assert app.query_one(Transcript).region.bottom <= dialog.region.y
            assert app.focused is not None
            assert app.focused.id == "chroma-green"
            assert str(app.query_one("#footer", Static).content) == (
                "←→ MOVE    ↩ EQUIP    ⎋ CANCEL"
            )
            assert (
                app.query_one(
                    "#transcript-scroll", VerticalScroll
                ).styles.padding.bottom
                == 3
            )

            await pilot.press("right")
            await pilot.pause()
            assert app.has_class("-accent-orange")

            await pilot.press("escape")
            await pilot.pause()
            assert app.query_one("#chat").display
            assert app.has_class("-accent-green")
            assert str(app.query_one("#footer", Static).content) == footer_before
            assert (
                app.query_one(
                    "#transcript-scroll", VerticalScroll
                ).styles.padding.bottom
                == 1
            )

            composer.insert("/chroma")
            await pilot.press("enter", "right", "enter")
            await pilot.pause()
            assert app.has_class("-accent-orange")
            assert app.query_one("#chat-alert", StatusLine).state == (
                "SYS: ACK REDHEAD equipped."
            )
            assert str(app.query_one("#footer", Static).content) == footer_before
            assert (
                app.query_one(
                    "#transcript-scroll", VerticalScroll
                ).styles.padding.bottom
                == 1
            )

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
            chat_heading = app.query_one("#chat-heading", Horizontal)
            specs_link = app.query_one("#specs-link", Static)
            assert specs_heading.region.y == chat_heading.region.y
            assert specs_link.display is False

            await pilot.press("escape")
            await pilot.pause()

            assert app.query_one("#landing").display
            assert app.query_one("#specs").display is False
            assert chooser.highlighted == 0
            assert chooser.has_focus
            assert runtime.session is None

    asyncio.run(verify())


def test_specs_connects_to_the_selected_registry_entry(tmp_path) -> None:
    """Check that Enter connects from registry-launched SPECS."""
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
            await pilot.press("s")
            await pilot.pause()
            await pilot.press("enter")
            await _wait_for_chat(app, pilot)

            assert app.query_one("#specs").display is False
            assert app.query_one("#chat").display
            assert runtime.session is not None
            assert runtime.session.assistant is assistant

    asyncio.run(verify())


def test_specs_page_uses_a_narrow_scrollbar(tmp_path) -> None:
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
            assert (
                scroller.styles.scrollbar_color_hover == scroller.styles.scrollbar_color
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
                "↑↓ SCROLL    ←→ CONSTRUCT    ↩ CONNECT    ⎋ REGISTRY    ⌃C TERMINATE"
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
            start = rendered.index("SYSTEM") + 1
            end = next(
                index for index in range(start, len(rendered)) if rendered[index] == ""
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
            disabled = chooser.get_component_rich_style("option-list--option-disabled")
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
    assistant = replace(_assistant(), base_model_digest="f" * 64)
    app = ChatApp(
        chat,
        generation,
        runtime_factory=cast(
            Callable[..., ChatRuntime],
            lambda *_args: ImmediateRuntime(chat, generation),
        ),
        initial_assistant=assistant,
    )

    async def verify() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_for_chat(app, pilot)

            divider = app.query_one("#identity-rule", Rule)
            heading = app.query_one("#chat-heading", Horizontal)
            link = app.query_one("#chat-link", Static)
            assert divider.content_region.x == 2
            assert divider.content_region.width == app.size.width - 4
            assert re.fullmatch(r"LINK#[0-9A-F]{6}", str(link.content))
            assert link.content_region.right == heading.content_region.right - 1
            assert link.region.right == heading.content_region.right
            assert link.styles.color.hex == "#80FF00"

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
            assert trace.plain.startswith(f"{assistant.target_run} Night session")
            assert "\n" not in trace.plain
            second_trace = chooser.get_option_at_index(2).prompt
            assert isinstance(second_trace, Text)
            assert "Morning session" in second_trace.plain
            assert "⎵ DETAILS" in str(
                app.query_one("#catalog-hints", Static).content
            )
            assert "⌫ MANAGE" not in str(
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
            assert "⎵ DETAILS" in str(
                app.query_one("#catalog-hints", Static).content
            )
            assert "⌫ MANAGE" in str(
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

            assert isinstance(app.screen, TraceMenuModal)
            assert app.screen.hidden_until_placed
            assert not app.screen.query_one("#trace-dialog").has_class("-unplaced")
            heading = app.screen.query_one("#trace-message", Static).content
            assert str(heading) == "TRACE MANAGE"
            assert [
                str(button.label) for button in app.screen.query(KeyboardButton)
            ] == ["ERASE", "RENAME", "RETAIN"]
            assert app.focused is not None
            assert app.focused.id == "retain"
            await pilot.press("up", "down")
            assert app.focused is not None
            assert app.focused.id == "retain"
            trace_message = app.screen.query_one("#trace-message", Static)
            assert (
                trace_message.content_region.x
                == app.query_one("#catalog-hints", Static).content_region.x
            )
            assert (
                app.screen.query_one("#erase", KeyboardButton).content_region.x
                - trace_message.content_region.x
                == 0
            )
            assert str(app.query_one("#catalog-hints", Static).content) == (
                "←→ MOVE    ↩ SELECT    ⎋ RETAIN"
            )
            chooser = app.query_one("#chooser", OptionList)
            subject = chooser.get_option_at_index(1).prompt
            assert isinstance(subject, Text)
            assert "Night session" in subject.plain

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
            assert app.screen.query_one("#trace-name-dialog").region.bottom == (
                app.query_one("#catalog-hints").region.y
            )
            trace_name_message = app.screen.query_one("#trace-name-message", Static)
            assert (
                trace_name_message.content_region.x
                == app.query_one("#catalog-hints", Static).content_region.x
            )
            assert name.content_region.x - trace_name_message.content_region.x == 1
            assert str(app.query_one("#catalog-hints", Static).content) == (
                "↩ NAME    ⎋ RETAIN"
            )
            name.value = "Morning session"
            await pilot.press("enter")
            await pilot.pause()

            assert store.renamed == [(saved.id, "Morning session")]
            trace = app.query_one("#chooser", OptionList).get_option_at_index(0).prompt
            assert isinstance(trace, Text)
            assert "Morning session" in trace.plain

    asyncio.run(verify())


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
            assert "OP:" in transcript_text
            assert "DIXIE:" in transcript_text
            assert "OP:\n User **message** 0\n [bold]literal[/bold]" in transcript_text
            assert "IDIOLECT //" not in transcript_text
            console = Console(width=20, color_system=None)
            with console.capture() as capture:
                console.print(transcript.content)
            rendered_lines = capture.get().splitlines()
            assert rendered_lines[0] == "OP:"
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
            assert app.query_one("#status", StatusLine).display is False

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


def test_loaded_trace_starts_at_the_bottom_of_chat_history(tmp_path) -> None:
    """Check a loaded TRACE follows its newest turn after chat layout settles."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
    turns = tuple(
        turn
        for index in range(8)
        for turn in (
            ChatTurn("user", f"User message {index}"),
            ChatTurn("assistant", f"Assistant reply {index}"),
        )
    )
    trace = SavedChat(
        "c" * 64,
        tmp_path / ("c" * 64),
        datetime(2026, 8, 22, tzinfo=UTC),
        "Loaded trace",
        None,
        _assistant(),
        chat,
        generation,
        turns,
    )
    runtime = ImmediateRuntime(chat, generation)
    app = ChatApp(
        chat,
        generation,
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
        initial_chat=trace,
    )

    async def verify() -> None:
        async with app.run_test(size=(50, 18)) as pilot:
            await _wait_for_chat(app, pilot)
            scroller = app.query_one("#transcript-scroll", VerticalScroll)
            await pilot.pause()
            assert scroller.max_scroll_y > 0

            scroller.scroll_home(animate=False)
            composer = app.query_one(Composer)
            await pilot.press("up")
            assert composer.text == "User message 7"
            await pilot.press("up")
            assert composer.text == "User message 6"
            composer.clear()
            composer.insert("/specs")
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("escape")
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
    """Check multiline submission keeps a followed transcript at the bottom."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig(max_prompt_tokens=100)
    runtime = MultilineRuntime(chat, generation)
    app = ChatApp(
        chat,
        generation,
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
        initial_assistant=_assistant(),
    )

    async def verify() -> None:
        async with app.run_test(size=(50, 18)) as pilot:
            await _wait_for_chat(app, pilot)
            composer = app.query_one(Composer)
            prompt = app.query_one("#composer-prompt", Static)
            scroller = app.query_one("#transcript-scroll", VerticalScroll)
            assert str(prompt.content) == ">"
            assert prompt.region.x < composer.region.x
            assert scroller.max_scroll_y > 0
            assert scroller.scroll_y == scroller.max_scroll_y
            lines = tuple(f"line {index}" for index in range(8))
            composer.insert("\n".join(lines))
            await pilot.pause()
            assert composer.text == "\n".join(lines)
            assert prompt.region.y == composer.region.y
            assert scroller.scroll_y == scroller.max_scroll_y

            await pilot.press("enter")
            assert await asyncio.to_thread(runtime.generation_started.wait, 1)
            await pilot.pause()
            transcript = app.query_one(Transcript).plain
            assert "OP:\n line 0\n line 1" in transcript
            assert composer.text == ""
            assert scroller.scroll_y == scroller.max_scroll_y

    try:
        asyncio.run(verify())
    finally:
        runtime.release_generation.set()


def test_blank_question_mark_opens_static_composer_controls(tmp_path) -> None:
    """Check CONTROL discovery, transitions, history, and literal punctuation."""
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
        async with app.run_test(size=(60, 24)) as pilot:
            await _wait_for_chat(app, pilot)
            composer = app.query_one(Composer)
            prompt = app.query_one("#composer-prompt", Static)
            sheet = app.query_one("#control-sheet", ControlSheet)
            footer = app.query_one("#footer", Static)
            scroller = app.query_one("#transcript-scroll", VerticalScroll)
            telemetry = str(footer.content)

            scroller.scroll_home(animate=False)
            await pilot.press("?")
            await pilot.pause()

            assert sheet.display
            assert composer.text == ""
            assert composer.has_focus
            assert str(prompt.content) == "?"
            assert str(footer.content) == telemetry
            assert scroller.scroll_y == scroller.max_scroll_y
            heading = sheet.query_one(".menu-heading", Static)
            assert str(heading.content) == "CONTROL"
            assert heading.styles.color.ansi == 7
            assert heading.styles.text_style.bold
            rows = tuple(
                (
                    str(row.query_one(".control-key", Static).content),
                    str(row.query_one(".control-description", Static).content),
                )
                for row in sheet.query(".control-action")
            )
            assert rows == (
                ("/", "COMMAND"),
                ("@", "REFERENCE"),
                ("?", "CONTROL"),
                ("↩", "TRANSMIT"),
                ("⇧↩", "NEWLINE"),
                ("↑↓", "HISTORY"),
                ("⎋", "CANCEL"),
            )
            assert all(row.region.height == 1 for row in sheet.query(".control-action"))
            first_row = sheet.query(".control-action").first()
            first_key = first_row.query_one(".control-key", Static)
            first_description = first_row.query_one(".control-description", Static)
            assert first_key.content_region.x - heading.content_region.x == 1
            assert first_description.styles.color == footer.styles.color

            await pilot.press("x")
            await pilot.pause()
            assert not sheet.display
            assert str(prompt.content) == ">"
            assert composer.text == "x"
            await pilot.press("?")
            assert composer.text == "x?"
            assert not sheet.display

            composer.clear()
            await pilot.pause()
            await pilot.press("?", "?")
            await pilot.pause()
            assert composer.text == "?"
            assert not sheet.display

            composer.clear()
            await pilot.pause()
            await pilot.press("?", "/")
            await pilot.pause()
            assert composer.text == "/"
            assert app.query_one("#command-menu", CommandMenu).display
            assert not sheet.display

            composer.clear()
            await pilot.pause()
            await pilot.press("?", "@")
            await pilot.pause()
            assert composer.text == "@"
            assert app.query_one("#reference-menu", ReferenceMenu).display
            assert not sheet.display
            await pilot.press("enter", "?")
            await pilot.pause()
            assert composer.reference_selected
            assert composer.text == "?"
            assert not sheet.display
            await pilot.press("escape")

            composer.clear()
            await pilot.pause()
            composer.insert("/echo")
            await pilot.pause()
            await pilot.press("enter", "?")
            await pilot.pause()
            assert composer.command_selected
            assert composer.text == "?"
            assert not sheet.display
            await pilot.press("escape")

            composer.clear()
            await pilot.pause()
            composer.set_history(("?recalled",))
            await pilot.press("?", "up")
            await pilot.pause()
            assert composer.text == "?recalled"
            assert not sheet.display
            await pilot.press("?")
            assert composer.text == "?recalled?"

            composer.clear()
            await pilot.pause()
            await pilot.press("?", "shift+enter")
            await pilot.pause()
            assert composer.text == "\n"
            assert not sheet.display
            assert str(prompt.content) == ">"

            composer.clear()
            await pilot.pause()
            await pilot.press("?", "escape")
            await pilot.pause()
            assert not sheet.display
            assert composer.text == ""
            assert str(prompt.content) == ">"

            await pilot.press("?", "backspace")
            await pilot.pause()
            assert not sheet.display
            assert composer.text == ""
            assert str(prompt.content) == ">"

    asyncio.run(verify())


def test_control_escape_cancels_active_generation(tmp_path) -> None:
    """Check CONTROL dismissal keeps the existing generation cancel action."""
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

            await pilot.press("?", "escape")
            assert await asyncio.to_thread(runtime.cancelled.wait, 1)
            await pilot.pause()

            assert not app.query_one("#control-sheet", ControlSheet).display
            assert str(app.query_one("#composer-prompt", Static).content) == ">"
            assert composer.text == ""

    try:
        asyncio.run(verify())
    finally:
        runtime.release_prefill.set()


def test_blank_composer_recalls_accepted_submission_history(tmp_path) -> None:
    """Check bounded history navigation starts only from a blank composer."""
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
        async with app.run_test(size=(50, 18)) as pilot:
            await _wait_for_chat(app, pilot)
            composer = app.query_one(Composer)
            transcript = app.query_one(Transcript)
            for value, reply_count in (("first", 1), ("second", 2)):
                composer.insert(value)
                await pilot.press("enter")
                for _ in range(20):
                    await pilot.pause()
                    if transcript.plain.count("Synthetic reply") == reply_count:
                        break

            composer.insert("draft")
            await pilot.press("up")
            assert composer.text == "draft"

            composer.clear()
            await pilot.press("up")
            assert composer.text == "second"
            await pilot.press("up")
            assert composer.text == "first"
            await pilot.press("up")
            assert composer.text == "first"
            await pilot.press("down")
            assert composer.text == "second"
            await pilot.press("down")
            assert composer.text == ""
            await pilot.press("down")
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

            status = app.query_one("#status", StatusLine)
            scroller = app.query_one("#transcript-scroll", VerticalScroll)
            rendered = status.render()
            assert status.state == "PREFILL 0/4 TOK"
            assert isinstance(rendered, Text)
            assert rendered.plain.endswith(" PREFILL 0/4 TOK")
            assert status.display is True
            assert status.content_region.x == scroller.content_region.x
            assert (
                status.styles.color == app.query_one("#catalog-subtitle").styles.color
            )

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
            assert app.query_one("#chat").display
            assert app.query_one("#landing").display is False
            status = app.query_one("#status", StatusLine)
            for _ in range(20):
                await pilot.pause()
                if status.state == "LINK LOADING":
                    break
            else:
                raise AssertionError("The loading state did not appear")
            rendered = status.render()
            assert isinstance(rendered, Text)
            assert rendered.plain.endswith(" LINK LOADING")
            assert "// MODEL SESSION" not in rendered.plain
            assert (
                status.styles.color == app.query_one("#catalog-subtitle").styles.color
            )
            assert app.query_one("#chooser", OptionList).disabled is True
            runtime.release.set()
            await _wait_for_chat(app, pilot)
            assert app.query_one("#chat-alert", StatusLine).state == (
                "SYS: ACK LINK established."
            )

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
                "LINK DIRTY"
            )
            assert (
                app.screen.query_one("#confirm-message", Static).styles.color.ansi == 7
            )
            assert app.screen.query_one(
                "#confirm-message", Static
            ).styles.text_style.bold
            assert (
                app.screen.query_one("#discard", KeyboardButton).content_region.x
                - app.screen.query_one("#confirm-message", Static).content_region.x
                == 0
            )
            assert [
                str(button.label) for button in app.screen.query(KeyboardButton)
            ] == ["DISCONNECT", "TRACE", "RESUME"]
            await pilot.press("right", "enter")
            await pilot.pause()
            assert app.screen.query_one("#trace-name", Input).has_focus
            assert str(app.query_one("#footer", Static).content) == (
                "↩ TRACE    ⎋ RESUME"
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
            app.query_one(Composer).insert("/disconnect")
            await pilot.press("enter", "right", "enter")
            await pilot.pause()

            name = app.screen.query_one("#trace-name", Input)
            assert name.has_focus
            assert name.placeholder == "default trace name"
            assert app.screen.query_one("#trace-name-dialog").region.bottom == (
                app.query_one("#composer-bar").region.y
            )
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
            terminate_button = menu.query_one("#command-terminate", Horizontal)
            disconnect_button = menu.query_one("#command-disconnect", Horizontal)
            trace_button = menu.query_one("#command-trace", Horizontal)
            specs_button = menu.query_one("#command-specs", Horizontal)
            probe_button = menu.query_one("#command-probe", Horizontal)
            buffer_button = menu.query_one("#command-buffer", Horizontal)
            chroma_button = menu.query_one("#command-chroma", Horizontal)
            assert menu.display is True
            assert str(menu.query_one("#command-message", Static).content) == "COMMAND"
            assert menu.query_one("#command-message", Static).styles.color.ansi == 7
            assert menu.query_one("#command-message", Static).styles.text_style.bold
            terminate_name = terminate_button.query_one(".command-name", Static)
            assert str(terminate_name.content) == "/terminate"
            disconnect_name = disconnect_button.query_one(".command-name", Static)
            assert disconnect_name.content_region.width >= len("/disconnect")
            assert (
                disconnect_name.content_region.x
                - menu.query_one("#command-message", Static).content_region.x
                == 1
            )
            assert (
                str(terminate_button.query_one(".command-description", Static).content)
                == "Terminate IDIOLECT."
            )
            assert (
                str(disconnect_button.query_one(".command-description", Static).content)
                == "Disconnect active LINK."
            )
            assert (
                str(trace_button.query_one(".command-description", Static).content)
                == "Save current TRACE."
            )
            assert (
                str(specs_button.query_one(".command-description", Static).content)
                == "View CONSTRUCT specifications."
            )
            assert (
                str(probe_button.query_one(".command-description", Static).content)
                == "View active LINK."
            )
            assert (
                str(buffer_button.query_one(".command-description", Static).content)
                == "View context BUFFER."
            )
            assert (
                str(chroma_button.query_one(".command-description", Static).content)
                == "Equip CHROMA."
            )
            assert terminate_button.display is False
            assert trace_button.has_class("-disabled")
            assert trace_button.has_class("-selected") is False
            assert trace_button.query_one(".command-name", Static).styles.color == (
                app.query_one("#catalog-subtitle").styles.color
            )
            assert chroma_button.region.y == buffer_button.region.y + 1
            assert disconnect_button.region.y == chroma_button.region.y + 1
            assert buffer_button.has_class("-selected")
            assert disconnect_button.has_class("-selected") is False
            selected_name = buffer_button.query_one(".command-name", Static)
            selected_description = buffer_button.query_one(
                ".command-description", Static
            )
            assert selected_description.styles.color == selected_name.styles.color
            assert selected_description.styles.text_style.dim
            assert not selected_description.styles.text_style.bold
            assert composer.has_focus
            assert str(app.query_one("#footer", Static).content) == (
                "↑↓ MOVE    ↩ COMMAND    ⎋ CLOSE"
            )
            await pilot.press("down")
            assert chroma_button.has_class("-selected")
            await pilot.press("down")
            assert disconnect_button.has_class("-selected")
            await pilot.press("down")
            echo_button = menu.query_one("#command-echo", Horizontal)
            assert echo_button.has_class("-selected")
            assert echo_button.display
            assert buffer_button.display is False
            await pilot.press("down")
            assert probe_button.has_class("-selected")
            assert probe_button.display
            await pilot.press("down")
            assert specs_button.has_class("-selected")
            assert specs_button.display
            await pilot.press("down")
            assert terminate_button.has_class("-selected")
            assert terminate_button.display
            await pilot.press("down")
            assert buffer_button.has_class("-selected")
            await pilot.press("up")
            assert terminate_button.has_class("-selected")
            await pilot.press("up")
            assert specs_button.has_class("-selected")
            await pilot.press("up")
            assert probe_button.has_class("-selected")
            await pilot.press("up")
            assert echo_button.has_class("-selected")
            await pilot.press("up")
            assert disconnect_button.has_class("-selected")

            await pilot.click("#command-disconnect")
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
            composer.insert("d")
            await pilot.pause()
            assert terminate_button.display is False
            assert disconnect_button.display is True

            await pilot.press("tab")
            await pilot.pause()
            assert composer.text == "/d"
            assert menu.display
            assert disconnect_button.has_class("-selected")
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
            assert menu.query_one("#command-echo", Horizontal).has_class("-selected")
            await pilot.press("enter")
            await pilot.pause()

            command_bar = app.query_one("#command-bar", Static)
            assert composer.command_selected
            assert command_bar.display
            assert isinstance(command_bar.content, Text)
            assert command_bar.content.plain == "ECHO SYS echo."
            assert str(app.query_one("#composer-prompt", Static).content) == "/"
            command_style = command_bar.content.get_style_at_offset(Console(), 0)
            description_style = command_bar.content.get_style_at_offset(
                Console(), command_bar.content.plain.index("SYS")
            )
            assert command_style.dim
            assert command_style.color is not None
            assert command_style.color.get_truecolor() == _HACKER_RGB
            assert description_style.color is not None
            assert description_style.color.name == "bright_black"
            assert composer.text == ""

            await pilot.press("backspace")
            await pilot.pause()
            assert not composer.command_selected
            assert not command_bar.display
            assert str(app.query_one("#composer-prompt", Static).content) == ">"

            composer.insert("/echo")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert composer.command_selected
            composer.insert("@")
            await pilot.pause()
            assert not app.query_one("#reference-menu", ReferenceMenu).display

            composer.insert("hello")
            await pilot.press("enter")
            await pilot.pause()

            assert not command_bar.display
            assert str(app.query_one("#composer-prompt", Static).content) == ">"
            assert runtime.session is not None
            assert runtime.session.turns[-1].role == "env"
            assert runtime.session.turns[-1].content == "@hello"
            transcript = app.query_one(Transcript)
            assert "SYS:\n @hello" in transcript.plain
            segments = tuple(Console().render(cast(RenderableType, transcript.content)))
            env_label = next(segment for segment in segments if segment.text == "SYS:")
            env_text = next(segment for segment in segments if "@hello" in segment.text)
            assert env_label.style is not None and env_label.style.dim
            assert env_label.style.color is not None
            assert env_label.style.color.get_truecolor() == _HACKER_RGB
            assert env_text.style is not None and not env_text.style.dim
            assert env_text.style.color is not None
            assert env_text.style.color.name == "bright_black"

            runtime.session.add_user("next")
            assert [turn.role for turn in runtime.session.turns] == ["env", "user"]

    asyncio.run(verify())


def test_command_argument_errors_use_generic_messages(tmp_path) -> None:
    """Check standardized missing and unexpected argument errors."""
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
            await pilot.press("enter", "enter")
            await pilot.pause()
            assert app.query_one("#chat-alert", StatusLine).state == (
                "SYS: ERR COMMAND argument missing."
            )
            command_bar = app.query_one("#command-bar", Static)
            alert = app.query_one("#chat-alert", StatusLine)
            activity = app.query_one("#activity-row", Horizontal)
            assert alert.region.y == command_bar.content_region.y
            assert command_bar.region.right <= alert.region.x
            assert alert.region.right == activity.region.right

            await pilot.press("escape")
            composer.clear()
            await pilot.pause()
            composer.insert("/chroma extra")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one("#chat-alert", StatusLine).state == (
                "SYS: ERR COMMAND argument unexpected."
            )

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

            assert composer.command_selected
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
            assert menu.query_one("#command-echo", Horizontal).has_class("-selected")
            await pilot.press("enter")
            await pilot.pause()

            assert composer.command_selected
            assert composer.text == "prompt "
            assert app.query_one("#command-bar", Static).display

    asyncio.run(verify())


def test_reference_menu_selects_bubble_and_escape_clears_reference(tmp_path) -> None:
    """Check reference selection, replacement, and Escape semantics."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig()
    runtime = ReferenceRuntime(chat, generation)
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
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if "two" in app.query_one(Transcript).plain:
                    break

            transcript = app.query_one(Transcript).plain
            assert "[new message]" not in transcript
            assert "DIXIE:\n one\n\nDIXIE:\n two" in transcript

            composer.insert("@")
            await pilot.pause()

            menu = app.query_one("#reference-menu", ReferenceMenu)
            assert menu.display
            assert str(app.query_one("#reference-message", Static).content) == "REF"
            assert (
                str(menu.query_one("#reference-2 .reference-name", Static).content)
                == "DIXIE:02"
            )
            assert menu.query_one("#reference-2").has_class("-selected")
            assert str(app.query_one("#footer", Static).content) == (
                "↑↓ MOVE    ↩ REF    ⎋ CLOSE"
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
            assert bar.content.plain.startswith("DIXIE:01 ")
            assert str(app.query_one("#composer-prompt", Static).content) == "@"
            at_style = bar.content.get_style_at_offset(Console(), 0)
            assert at_style.dim
            assert at_style.color is not None
            assert at_style.color.get_truecolor() == _HACKER_RGB
            assert bar.region.y + bar.region.height == composer_bar.region.y
            assert bar.styles.border_top[1].hex == (
                app.query_one("#composer-prompt", Static).styles.color.hex
            )
            border_style = bar.get_style_at(1, 0)
            assert border_style.dim
            assert border_style.color is not None
            assert border_style.color.get_truecolor() == _HACKER_RGB
            bottom_border_style = bar.get_style_at(1, bar.size.height - 1)
            assert bottom_border_style.dim
            assert bottom_border_style.color is not None
            assert bottom_border_style.color.get_truecolor() == _HACKER_RGB
            assert bar.get_style_at(0, 1).dim
            assert composer.text == ""
            reference_style = bar.content.get_style_at_offset(Console(), 0)
            assert reference_style.dim
            assert reference_style.color is not None
            assert reference_style.color.get_truecolor() == _HACKER_RGB
            assert (
                composer_bar.region.x,
                composer_bar.region.width,
                app.query_one("#composer-prompt", Static).region.x,
                composer.region.x,
                composer_bar.styles.border_top,
            ) == before_geometry

            await pilot.press("backspace")
            await pilot.pause()
            assert not composer.reference_selected
            assert not bar.display
            assert str(app.query_one("#composer-prompt", Static).content) == ">"

            composer.insert("@")
            await pilot.pause()
            await pilot.press("up", "enter")
            await pilot.pause()
            assert composer.reference_selected
            assert str(app.query_one("#composer-prompt", Static).content) == "@"

            composer.insert("follow-up")
            await pilot.pause()
            assert composer.reference_selected
            assert composer.has_focus
            assert not composer.command_menu_active
            assert not composer.reference_menu_active
            await pilot.press("escape")
            await pilot.pause()
            assert not bar.display
            assert composer.text == "follow-up"
            assert str(app.query_one("#composer-prompt", Static).content) == ">"

            composer.clear()
            composer.insert("@D")
            await pilot.pause()
            assert menu.display
            assert (
                str(menu.query_one("#reference-0 .reference-name", Static).content)
                == "DIXIE:01"
            )
            assert (
                str(menu.query_one("#reference-1 .reference-name", Static).content)
                == "DIXIE:02"
            )
            assert menu.query_one("#reference-2").display is False
            await pilot.press("enter")
            await pilot.pause()
            assert composer.text == ""

            composer.insert("prompt")
            composer.insert("@O")
            await pilot.pause()
            assert menu.display
            assert (
                str(menu.query_one("#reference-0 .reference-name", Static).content)
                == "OP:00"
            )
            await pilot.press("enter")
            await pilot.pause()
            assert composer.text == "prompt"
            assert bar.content.plain.startswith("OP:00 ")
            assert str(app.query_one("#composer-prompt", Static).content) == "@"

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
            assert (
                str(menu.query_one("#reference-2 .reference-name", Static).content)
                == "DIXIE:02"
            )
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
            assert composer.reference_selected

            composer.clear()
            composer.insert("sent")
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if "REF @DIXIE:02" in app.query_one(Transcript).plain:
                    break
            transcript = app.query_one(Transcript)
            assert "OP:\n REF @DIXIE:02\n sent" in transcript.plain
            segments = tuple(Console().render(cast(RenderableType, transcript.content)))
            annotation = next(
                segment for segment in segments if "REF @DIXIE:02" in segment.text
            )
            assert annotation.style is not None
            assert annotation.style.dim
            assert annotation.style.color is not None
            assert annotation.style.color.get_truecolor() == _HACKER_RGB
            speaker = next(segment for segment in segments if segment.text == "OP:")
            assert speaker.style is not None
            assert not speaker.style.dim
            assert speaker.style.color is not None
            assert speaker.style.color.get_truecolor() == _HACKER_RGB

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
                "↑↓ SCROLL    ⎋ LINK    ⌃C TERMINATE"
            )
            content = app.query_one("#specs-body", Static).content
            assert isinstance(content, SpecsDocument)
            assert "TYPE\n TRACE\n" in content.plain
            assert f"ID\n {trace.id.upper()}\n" in content.plain
            assert "TEMPERATURE\n 0.3\n" in content.plain

            await pilot.press("ctrl+c")
            await pilot.pause()

            assert app.query_one("#chat").display
            assert app.query_one("#specs").display is False
            assert (
                str(app.screen.query_one("#confirm-message", Static).content)
                == "LINK DIRTY"
            )
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


def test_probe_command_shows_live_details_and_restores_chat(tmp_path) -> None:
    """Check the live probe adds telemetry without context or session changes."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig(temperature=0.3)
    runtime = TranscriptRuntime(chat, generation)
    app = ChatApp(
        chat,
        generation,
        initial_assistant=_assistant(),
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
    )

    async def verify() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_for_chat(app, pilot)
            session = runtime.session
            assert session is not None
            fingerprint = session.fingerprint
            composer = app.query_one(Composer)
            composer.insert("/probe")
            await pilot.press("enter")
            await pilot.pause()

            assert app.query_one("#specs").display
            assert str(app.query_one("#specs-identity", Static).content) == (
                session.assistant.name
            )
            content = app.query_one("#specs-body", Static).content
            assert isinstance(content, SpecsDocument)
            assert "RUNTIME\nSTATE\n READY\nMLX\n 0.32.1\n" in content.plain
            assert "STACK\n" not in content.plain
            assert "DEVICE\nTYPE\n GPU\n" in content.plain
            assert "MODEL\n" in content.plain
            assert "MLX VERSION\n" not in content.plain
            assert "SIZE\n 8.00 GiB\n" in content.plain
            assert "DIGEST\n" not in content.plain
            assert "ADAPTER SIZE\n" not in content.plain
            assert "TELEMETRY\n" in content.plain
            assert "OUTPUT\n 64 TOK\n" in content.plain
            assert "PREFILL THROUGHPUT\n —\n" in content.plain
            assert "DECODE THROUGHPUT\n 12.3 TOK/S\n" in content.plain
            assert "TIME TO FIRST TOKEN\n 0.420 S\n" in content.plain
            assert "INFERENCE LATENCY\n —\n" in content.plain
            assert "PEAK MEMORY\n 3.25 GB\n" in content.plain
            assert "PROMPT TOKENS\n" not in content.plain
            assert "UTILIZATION\n" not in content.plain
            assert "RESIDENT\n" not in content.plain
            assert "IDENTITY\n" not in content.plain
            assert "GENERATION\n" not in content.plain
            assert "FIDELITY\n" not in content.plain

            await pilot.press("escape")
            await pilot.pause()

            assert app.query_one("#chat").display
            assert not app.query_one("#specs").display
            assert runtime.session is session
            assert session.fingerprint == fingerprint
            assert composer.has_focus

    asyncio.run(verify())


def test_buffer_command_shows_fitted_context_and_restores_chat(tmp_path) -> None:
    """Check the context sheet lists active references without changing chat."""
    chat = ChatConfig(
        output=tmp_path,
        context_policy="recorded-window-drop-oldest",
        participant_name="person_01",
    )
    generation = GenerationConfig(max_prompt_tokens=100)
    runtime = BufferRuntime(chat, generation)
    assistant = _assistant()
    assert assistant.base_data is not None
    assistant = replace(
        assistant,
        base_data=replace(assistant.base_data, format="chat"),
    )
    app = ChatApp(
        chat,
        generation,
        initial_assistant=assistant,
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
    )

    async def verify() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_for_chat(app, pilot)
            composer = app.query_one(Composer)
            composer.insert("Keep this context")
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if runtime.last_prompt is not None:
                    break

            session = runtime.session
            assert session is not None
            fingerprint = session.fingerprint
            composer.insert("/buffer")
            await pilot.press("enter")
            await pilot.pause()

            assert app.query_one("#specs").display
            assert str(app.query_one("#specs-identity", Static).content) == (
                session.assistant.name
            )
            content = app.query_one("#specs-body", Static).content
            assert isinstance(content, SpecsDocument)
            assert "PROMPT\n" in content.plain
            assert "TOKENS\nPROMPT\n 2 TOK\n" in content.plain
            assert "LIMIT\n 100 TOK\n" in content.plain
            assert "UTILIZATION\n 2.0%\n" in content.plain
            assert "TURNS\nACTIVE\n 1\n" in content.plain
            assert "CAPACITY\n 32\n" in content.plain
            assert "EVICTED\n 0\n" in content.plain
            assert "HEAD\n @OP:00\n" in content.plain
            assert "RESIDENT\n" in content.plain
            assert "REFS\n @OP:00\n" in content.plain
            assert "@OP:00\n" in content.plain
            assert "Keep this context" not in content.plain

            await pilot.press("escape")
            await pilot.pause()

            assert app.query_one("#chat").display
            assert not app.query_one("#specs").display
            assert runtime.session is session
            assert session.fingerprint == fingerprint
            assert composer.has_focus

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
            composer.insert("/trace")
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
            alert = app.query_one("#chat-alert", StatusLine)
            composer_bar = app.query_one("#composer-bar", Horizontal)
            assert alert.state == "SYS: ACK TRACE AAAAAA saved."
            assert re.fullmatch(
                r"LINK#[0-9A-F]{6}",
                str(app.query_one("#chat-link", Static).content),
            )
            assert alert.display
            assert alert.region.bottom == composer_bar.region.y
            assert alert.styles.text_align == "right"
            rendered = alert.render()
            assert isinstance(rendered, Text)
            prefix_style = rendered.get_style_at_offset(Console(), 0)
            message_style = rendered.get_style_at_offset(
                Console(), rendered.plain.index("ACK")
            )
            assert prefix_style.color is not None
            assert prefix_style.color.get_truecolor() == _HACKER_RGB
            assert prefix_style.dim
            assert message_style.color is not None
            assert message_style.color.name == "bright_black"
            assert not message_style.dim

            composer.insert("/trace")
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.screen.query("#trace-name")) == 0
            assert store.titles == [None]
            assert app.query_one("#chat-alert", StatusLine).state == (
                "SYS: ACK TRACE AAAAAA exists."
            )

    asyncio.run(verify())


def test_chat_errors_align_right_above_composer(tmp_path) -> None:
    """Check inline failure placement and loading-status visual language."""
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
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_for_chat(app, pilot)
            assert await asyncio.to_thread(runtime.started.wait, 1)
            composer = app.query_one(Composer)
            composer.insert("message during load")
            await pilot.press("enter")
            await pilot.pause()

            failure = app.query_one("#chat-alert", StatusLine)
            loading = app.query_one("#status", StatusLine)
            activity = app.query_one("#activity-row", Horizontal)
            assert failure.state == "SYS: ERR LINK not established."
            assert failure.display
            assert loading.display
            assert loading.region.y == failure.region.y
            assert failure.region.right == activity.region.right

            runtime.release.set()
            for _ in range(20):
                await pilot.pause()
                if not loading.display:
                    break
            composer.clear()
            composer.insert("/unknown")
            await pilot.press("enter")
            await pilot.pause()

            composer_bar = app.query_one("#composer-bar", Horizontal)
            assert failure.state == "SYS: ERR COMMAND unknown."
            assert failure.display
            assert failure.region.bottom == composer_bar.region.y
            assert failure.region.right == activity.region.right
            assert failure.styles.text_align == "right"
            rendered = failure.render()
            assert isinstance(rendered, Text)
            prefix_style = rendered.get_style_at_offset(Console(), 0)
            message_style = rendered.get_style_at_offset(
                Console(), rendered.plain.index("ERR")
            )
            assert prefix_style.color is not None
            assert prefix_style.color.get_truecolor() == _HACKER_RGB
            assert prefix_style.dim
            assert message_style.color is not None
            assert message_style.color.name == "bright_black"
            assert not message_style.dim

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
                "LINK DIRTY"
            )
            assert app.focused is not None
            assert app.focused.id == "discard"
            await pilot.press("up", "down")
            assert app.focused is not None
            assert app.focused.id == "discard"
            assert str(app.query_one("#footer", Static).content) == (
                "←→ MOVE    ↩ SELECT    ⎋ RESUME"
            )
            assert (
                app.query_one(
                    "#transcript-scroll", VerticalScroll
                ).styles.padding.bottom
                == 3
            )

            await pilot.press("escape")
            await pilot.pause()
            assert app.query_one("#chat").display
            assert str(app.query_one("#footer", Static).content) == ""
            assert (
                app.query_one(
                    "#transcript-scroll", VerticalScroll
                ).styles.padding.bottom
                == 1
            )

    asyncio.run(verify("/terminate"))
    asyncio.run(verify("/disconnect"))


def test_commands_follow_generation_navigation_rules(tmp_path) -> None:
    """Check that terminate cancels and disconnect stays unavailable during generation."""
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

            composer.insert("/disconnect")
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one("#chat-alert", StatusLine).state == (
                "SYS: ERR CONSTRUCT is generating."
            )

            composer.clear()
            composer.insert("/disconnect")
            await pilot.pause()
            disconnect = app.query_one("#command-disconnect", Horizontal)
            assert disconnect.has_class("-disabled")
            disabled_name = disconnect.query_one(".command-name", Static)
            disabled_description = disconnect.query_one(
                ".command-description",
                Static,
            )
            assert (
                disabled_name.styles.color
                == app.query_one("#catalog-subtitle").styles.color
            )
            assert disabled_description.styles.color == disabled_name.styles.color
            assert disabled_name.styles.text_style.dim
            assert disabled_description.styles.text_style.dim
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one("#chat").display
            assert runtime.cancelled.is_set() is False

            composer.insert("/terminate")
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


def test_message_during_generation_reports_error_and_retains_text(tmp_path) -> None:
    """Check a second message is rejected visibly without losing its text."""
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

            composer.insert("send later")
            await pilot.press("enter")
            await pilot.pause()

            assert app.query_one("#chat-alert", StatusLine).state == (
                "SYS: ERR CONSTRUCT is generating."
            )
            assert composer.text == "send later"
            session = runtime.session
            assert session is not None
            assert [turn.content for turn in session.turns if turn.role == "user"] == [
                "hold reply"
            ]

    try:
        asyncio.run(verify())
    finally:
        runtime.release_prefill.set()


def test_retry_replaces_a_cancelled_reply(tmp_path) -> None:
    """Check retry removes a cancelled reply and advances its attempt."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig(max_prompt_tokens=100)
    turns = (
        ChatTurn("user", "Recover this"),
        ChatTurn(
            "assistant",
            "Partial reply",
            attempt=0,
            finish_reason="cancelled",
            seed=101,
            telemetry=TurnTelemetry(2, 1),
        ),
    )
    trace = SavedChat(
        "d" * 64,
        tmp_path / ("d" * 64),
        datetime(2026, 8, 29, tzinfo=UTC),
        "Cancelled trace",
        None,
        _assistant(),
        chat,
        generation,
        turns,
    )
    runtime = ImmediateRuntime(chat, generation)
    app = ChatApp(
        chat,
        generation,
        runtime_factory=cast(Callable[..., ChatRuntime], lambda *_args: runtime),
        initial_chat=trace,
    )

    async def verify() -> None:
        async with app.run_test() as pilot:
            await _wait_for_chat(app, pilot)
            composer = app.query_one(Composer)
            composer.insert("/retry")
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if "Synthetic reply" in app.query_one(Transcript).plain:
                    break

            session = runtime.session
            assert session is not None
            assert [turn.content for turn in session.turns] == [
                "Recover this",
                "Synthetic reply",
            ]
            assert session.turns[-1].attempt == 1

            composer.insert("/retry")
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one("#chat-alert", StatusLine).state == (
                "SYS: ERR GENERATION not retryable."
            )
            assert [turn.content for turn in session.turns] == [
                "Recover this",
                "Synthetic reply",
            ]

    asyncio.run(verify())


def test_retry_reloads_after_a_failed_generation(tmp_path) -> None:
    """Check retry reloads a failed worker and regenerates the pending turn."""
    chat = ChatConfig(output=tmp_path)
    generation = GenerationConfig(max_prompt_tokens=100)
    runtime = FailureThenRetryRuntime(chat, generation)
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
            composer.insert("Recover this")
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if app.query_one("#chat-alert", StatusLine).state.endswith(
                    "Synthetic generation failure."
                ):
                    break

            composer.insert("/retry")
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                if "Recovered reply" in app.query_one(Transcript).plain:
                    break

            session = runtime.session
            assert session is not None
            assert runtime.reloads == 1
            assert runtime.attempts == [0, 1]
            assert [turn.content for turn in session.turns] == [
                "Recover this",
                "Recovered reply",
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

    backend_versions: ClassVar[dict[str, str | None]] = {}

    def __init__(self, chat: ChatConfig, generation: GenerationConfig) -> None:
        """Create model-load synchronization points."""
        self.chat = chat
        self.generation = generation
        self.session: ChatSession | None = None
        self.state = WorkerState.PROBING
        self.started = threading.Event()
        self.release = threading.Event()
        self.runtime_probe = RuntimeProbe(
            "0.32.1",
            "0.31.3",
            "Device(gpu, 0)",
            "arm64",
            (("max_buffer_size", 5 * 1024**3), ("unified_memory", True)),
        )
        self.load_probe = LoadProbe("f" * 64, 8 * 1024**3, None, 1.25)
        self.last_prompt = None

    def ensure_worker(self) -> None:
        """Keep the fake runtime ready."""

    def select(self, assistant: Assistant) -> ChatSession:
        """Wait before the fake model becomes ready."""
        self.last_prompt = None
        self.session = ChatSession(assistant, self.chat, self.generation)
        self.state = WorkerState.LOADING
        self.started.set()
        if not self.release.wait(2):
            raise RuntimeError("Synthetic model load timed out")
        self.state = WorkerState.READY
        return self.session

    def attach(self, session: ChatSession) -> None:
        """Wait before attaching one existing fake session."""
        self.last_prompt = None
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
        if not self.session.generating:
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


class FailureThenRetryRuntime(ImmediateRuntime):
    """Fail one generation and recover after a synthetic reload."""

    def __init__(self, chat: ChatConfig, generation: GenerationConfig) -> None:
        """Create retry records."""
        super().__init__(chat, generation)
        self.attempts: list[int] = []
        self.reloads = 0

    def reload(self) -> None:
        """Record one successful worker reload."""
        self.reloads += 1
        self.state = WorkerState.READY

    def generate(
        self,
        attempt: int = 0,
        prompt_progress: Callable[[int, int], None] | None = None,
    ) -> Iterator[str]:
        """Fail the first attempt and complete the second attempt."""
        del prompt_progress
        if self.session is None:
            raise RuntimeError("No fake chat session")
        self.attempts.append(attempt)
        if len(self.attempts) == 1:
            self.session.begin_generation()
            self.session.generating = False
            self.state = WorkerState.FAILED
            raise RuntimeError("Synthetic generation failure")
        self.session.begin_generation()
        yield "Recovered reply"
        self.session.finish_generation(
            "Recovered reply",
            "stop",
            101 + attempt,
            TurnTelemetry(2, 2),
            attempt=attempt,
        )


class ReferenceRuntime(ImmediateRuntime):
    """Generate an assistant episode with two serialized bubbles."""

    def generate(
        self,
        attempt: int = 0,
        prompt_progress: Callable[[int, int], None] | None = None,
    ) -> Iterator[str]:
        """Commit one synthetic two-bubble assistant reply."""
        del prompt_progress
        if self.session is None:
            raise RuntimeError("No fake chat session")
        self.session.begin_generation()
        pieces = ("one\r\n  [new", " message]\t\r\n", "two")
        yield from pieces
        value = "".join(pieces)
        self.session.finish_generation(
            value,
            "stop",
            101 + attempt,
            TurnTelemetry(2, 1),
            attempt=attempt,
        )


class BufferRuntime(ImmediateRuntime):
    """Record one fitted prompt for BUFFER page tests."""

    def generate(
        self,
        attempt: int = 0,
        prompt_progress: Callable[[int, int], None] | None = None,
    ) -> Iterator[str]:
        """Record context and commit one synthetic assistant reply."""
        del prompt_progress
        if self.session is None:
            raise RuntimeError("No fake chat session")
        self.session.begin_generation()
        self.last_prompt = prepare_prompt(self.session, lambda _value: 2, attempt)
        yield "Synthetic reply"
        self.session.finish_generation(
            "Synthetic reply",
            "length",
            101 + attempt,
            TurnTelemetry(2, 2),
            attempt=attempt,
        )


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


class MultilineRuntime(TranscriptRuntime):
    """Hold generation after one multiline user message is stored."""

    def __init__(self, chat: ChatConfig, generation: GenerationConfig) -> None:
        """Create generation synchronization points."""
        super().__init__(chat, generation)
        self.generation_started = threading.Event()
        self.release_generation = threading.Event()

    def generate(
        self,
        attempt: int = 0,
        prompt_progress: Callable[[int, int], None] | None = None,
    ) -> Iterator[str]:
        """Wait with one pending multiline user message."""
        del prompt_progress
        if self.session is None:
            raise RuntimeError("No fake chat session")
        self.session.begin_generation()
        self.generation_started.set()
        if not self.release_generation.wait(2):
            raise RuntimeError("Synthetic generation timed out")
        yield "Synthetic reply"
        self.session.finish_generation(
            "Synthetic reply",
            "length",
            101 + attempt,
            TurnTelemetry(2, 2),
            attempt=attempt,
        )


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
