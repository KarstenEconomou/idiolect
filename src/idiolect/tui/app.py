"""Render the local Idiolect chat terminal interface."""

from __future__ import annotations

import secrets
import threading
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, cast

from rich.style import Style
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
from textual.widgets import OptionList, Rule, Static, TextArea
from textual.widgets.option_list import Option

from idiolect.chat.discovery import Assistant, DiscoveryItem
from idiolect.chat.runtime import ChatError, ChatRuntime
from idiolect.chat.state import (
    ChatBubble,
    ChatSession,
    TurnTelemetry,
    enumerate_bubbles,
)
from idiolect.chat.storage import (
    ChatStorageError,
    ChatStore,
    SavedChat,
    default_chat_title,
)
from idiolect.chat.worker import WorkerError, WorkerState
from idiolect.config import ChatConfig, GenerationConfig
from idiolect.prompt import split_bubbles
from idiolect.tui.catalog import CatalogLayout
from idiolect.tui.commands import (
    COMMAND_ARGUMENTS,
    COMMAND_DESCRIPTIONS,
    CommandError,
    completions,
    parse_command,
)
from idiolect.tui.markdown import is_web_link
from idiolect.tui.sheets import InfoSheet, SheetPage, SheetScroll
from idiolect.tui.specs import (
    HalfCellScrollBarRender,
    SheetDocument,
    render_buffer,
    render_probe,
    render_specs,
)
from idiolect.tui.widgets import (
    ChromaMenuModal,
    CommandBar,
    CommandMenu,
    Composer,
    ConfirmModal,
    KeyboardOptionList,
    ReferenceBar,
    ReferenceMenu,
    StatusLine,
    TraceMenuModal,
    TraceNameModal,
    Transcript,
)

_WATERMARK_SOURCE = """     ╭─╮
  ╭──╯ ╰──╮
  │  · ·  │    IDIOLECT
  ╰──╮ ╭──╯     Someone, reconstructed.
     ╰─╯"""
_TAGLINE = "Someone, reconstructed."
_UNSAVED_LINK_ID = "------"


@dataclass(frozen=True, slots=True)
class ChromaTheme:
    """Declare one stable CHROMA identity, accent value, and display name."""

    color: str
    accent: str
    name: str


_CHROMA_THEMES = (
    ChromaTheme("blue", "#00AAFF", "LOCKSMITH"),
    ChromaTheme("red", "#FF002B", "LOOKOUT"),
    ChromaTheme("yellow", "#FFD500", "PICKPOCKET"),
    ChromaTheme("pink", "#FF00D5", "CLEANER"),
    ChromaTheme("violet", "#AA00FF", "MOLE"),
    ChromaTheme("teal", "#00FFFF", "GENTLEMAN"),
    ChromaTheme("green", "#80FF00", "HACKER"),
    ChromaTheme("orange", "#FF5500", "REDHEAD"),
)
_DEFAULT_CHROMA = "green"


class DialogKind(Enum):
    """Identify the active modal dialog."""

    CHROMA = "chroma"
    CONFIRM = "confirm"
    TRACE_NAME = "trace-name"
    TRACE_MANAGE = "trace-manage"
    TRACE_RENAME = "trace-rename"


class SelectorKind(Enum):
    """Identify the active composer selector."""

    COMMAND = "command"
    REFERENCE = "reference"


class NoticeKind(Enum):
    """Identify one transient system notice."""

    ACK = "ACK"
    ERR = "ERR"


@dataclass(frozen=True, slots=True)
class ActiveDialog:
    """Keep the one active dialog and its domain context."""

    kind: DialogKind
    trace_id: str | None = None
    initial_theme_index: int | None = None


@dataclass(frozen=True, slots=True)
class ActiveSelector:
    """Keep the active selector and accessory state."""

    kind: SelectorKind
    menu_open: bool
    accessory_selected: bool


def _watermark(color: str = "#80FF00") -> Text:
    """Return the product watermark in one accent color."""
    value = Text(_WATERMARK_SOURCE, style=Style(color=color, bold=True))
    tagline_start = _WATERMARK_SOURCE.index(_TAGLINE)
    value.stylize(
        Style(color=color, bold=False, dim=True),
        tagline_start,
        tagline_start + len(_TAGLINE),
    )
    return value


def _link_label(identifier: str | None) -> str:
    """Return the compact label for one live link."""
    identifier = _UNSAVED_LINK_ID if identifier is None else identifier
    return f"LINK#{identifier}"


def _random_link_id() -> str:
    """Return a random six-digit hexadecimal link identifier."""
    return secrets.token_hex(3).upper()


def _css_accent(value: str) -> str:
    """Return one Textual CSS color from an ANSI name or hexadecimal value."""
    return value if value.startswith("#") else f"ansi_{value}"


def _accent_theme_css(
    themes: tuple[ChromaTheme, ...] = _CHROMA_THEMES,
) -> str:
    """Return screen-class overrides for each selectable CHROMA accent."""
    color_selectors = (
        "OptionList > .option-list--option-highlighted",
        "OptionList:focus > .option-list--option-highlighted",
        "OptionList > .option-list--option-hover",
        "#specs-identity",
        ".brand-link",
        "#identity",
        "#composer-prompt",
        ".command-action.-selected .command-name",
        ".command-action.-selected .command-description",
        ".reference-action.-selected .reference-name",
        ".reference-action.-selected .reference-preview",
        ".menu-action.-selected .menu-name",
        ".menu-action.-selected .menu-description",
        "#command-bar",
        ".selection-bar",
        "#confirm-actions Button:focus",
        "#confirm-actions Button.-active",
        "#trace-actions Button:focus",
        "#trace-actions Button.-active",
        "#chroma-actions Button:focus",
        "#chroma-actions Button.-active",
    )
    button_selectors = (
        "Button:hover",
        "Button:focus",
        "Button.-active",
    )
    rules = []
    for theme in themes:
        prefix = f".-accent-{theme.color} "
        accent = _css_accent(theme.accent)
        colored = ", ".join(prefix + selector for selector in color_selectors)
        buttons = ", ".join(prefix + selector for selector in button_selectors)
        rules.append(f"{colored} {{ color: {accent}; }}")
        rules.append(f"{prefix}#composer-bar {{ border: solid {accent}; }}")
        rules.append(f"{prefix}#reference-bar {{ border: solid {accent}; }}")
        rules.append(f"{prefix}#command-bar {{ border: solid {accent}; }}")
        rules.append(f"{buttons} {{ border: tall {accent}; }}")
    return "\n".join(rules)


WATERMARK = _watermark()

_FOOTER_GAP = "    "


def _episode_segments(
    name: str,
    content: str,
) -> tuple[tuple[str, str], ...]:
    """Split one assistant response episode into labeled message bubbles.

    Stored turn content stays exact; only the display separates the bubbles
    of one episode. Blank serialization segments are never shown.
    """
    segments = tuple(segment for segment in split_bubbles(content) if segment.strip())
    if not segments:
        return ((name, content),)
    return tuple((name, segment) for segment in segments)


def _telemetry_footer(
    telemetry: TurnTelemetry,
    max_prompt_tokens: int,
    available_width: int,
) -> str:
    """Format the measured values that fit in the footer."""
    max_prompt_tokens = max(1, max_prompt_tokens)
    pressure = 100 * telemetry.prompt_tokens / max_prompt_tokens
    context = f"CTX {telemetry.prompt_tokens:,}/{max_prompt_tokens:,} ({pressure:.0f}%)"
    if len(context) > available_width:
        compact_context = f"CTX {pressure:.0f}%"
        return compact_context if len(compact_context) <= available_width else ""
    generation = f"GEN {telemetry.generated_tokens:,} TOK"
    if telemetry.generation_throughput is not None:
        generation += f" @ {telemetry.generation_throughput:.1f} TOK/S"
    fields = [context, generation]
    if telemetry.time_to_first_token is not None:
        fields.append(f"TTFT {telemetry.time_to_first_token:.2f} S")
    if telemetry.peak_memory is not None:
        fields.append(f"MEM {telemetry.peak_memory:.2f} GB")

    visible = [fields[0]]
    for field in fields[1:]:
        candidate = _FOOTER_GAP.join((*visible, field))
        if len(candidate) > available_width:
            break
        visible.append(field)
    return _FOOTER_GAP.join(visible)


class ChatApp(App[None]):
    """Run the assistant registry and local chat screen."""

    CSS = """
    $terminal: ansi_default;
    $accent: #80FF00;
    $metadata: ansi_bright_black;
    Screen { background: $terminal; color: $terminal; }
    #landing { align: left top; padding: 0; }
    #landing-box { width: 100%; max-width: 100%; height: 100%; background: $terminal; }
    #watermark { color: $accent; height: 5; padding: 0 2; text-align: left; }
    #catalog-heading { height: 1; margin-top: 1; padding: 0 2; }
    #catalog-title { text-style: bold; }
    #catalog-subtitle { height: 1; padding: 0 2; color: $metadata; }
    #catalog-description { width: 1fr; padding: 0 0 0 1; }
    #catalog-rule { height: 1; margin: 0; padding: 0 2; color: $metadata; }
    #catalog-columns { height: 1; padding: 0 2; color: ansi_white; text-style: bold; }
    #load-status { display: none; height: 1; padding: 0 2; color: $metadata; }
    #chooser { height: 1fr; padding: 0 2; border: none; color: $terminal; background: $terminal; background-tint: transparent; scrollbar-color: $metadata; scrollbar-background: $terminal; }
    OptionList > .option-list--option { padding: 0 0 0 1; color: $terminal; background: $terminal; }
    OptionList > .option-list--option-highlighted, OptionList:focus > .option-list--option-highlighted { color: $accent; background: $terminal; text-style: bold; }
    OptionList > .option-list--option-disabled { color: $metadata; text-style: dim; }
    OptionList > .option-list--option-hover { color: $accent; background: $terminal; text-style: bold; }
    #catalog-hints { height: 1; color: $metadata; background: $terminal; padding: 0 2; }
    #catalog-alert { display: none; height: 1; color: $accent; background: $terminal; padding: 0 2; text-align: right; text-style: none; }
    #footer { height: 1; color: $metadata; background: $terminal; padding: 0 2; }
    #specs { display: none; background: $terminal; }
    #specs-heading { height: auto; min-height: 1; margin-top: 0; padding: 0 2; }
    #specs-identity { width: 1fr; height: auto; min-height: 1; color: $accent; text-style: bold; }
    .brand-link { width: 12; height: 1; padding-right: 1; color: $accent; text-align: right; text-style: bold; }
    #specs-rule { height: 1; margin: 0; padding: 0 2; color: $metadata; }
    #specs-scroll { height: 1fr; padding: 0 2; background: $terminal; scrollbar-size-vertical: 1; scrollbar-color: $metadata; scrollbar-color-hover: $metadata; scrollbar-color-active: $metadata; scrollbar-background: $terminal; scrollbar-background-hover: $terminal; scrollbar-background-active: $terminal; }
    #specs-body { width: 100%; height: auto; color: $terminal; background: $terminal; }
    #specs-hints { height: 1; color: $metadata; background: $terminal; padding: 0 2; }
    #chat { display: none; }
    #chat-heading { height: 1; padding: 0 2; background: $terminal; }
    #identity { width: 1fr; min-width: 0; height: 1; color: $accent; background: $terminal; text-style: bold; }
    #identity-rule { height: 1; margin: 0; padding: 0 2; color: $metadata; }
    #transcript-scroll { height: 1fr; padding: 1 2; background: $terminal; scrollbar-size: 0 0; }
    #transcript { width: 100%; height: auto; background: $terminal; }
    #composer-bar { height: auto; min-height: 3; max-height: 10; border: solid $accent; margin: 0 1; padding: 0 1; background: $terminal; scrollbar-size: 0 0; }
    #composer-bar:focus-within { border: solid $accent; }
    #composer-prompt { width: 1; height: 1; color: $accent; }
    #composer { width: 1fr; min-width: 0; height: auto; min-height: 1; max-height: 10; border: none; margin: 0; padding: 0 1; background: $terminal; scrollbar-size: 0 0; }
    #composer .text-area--cursor, #composer .text-area--selection { color: $terminal; background: $terminal; text-style: reverse; }
    #composer .text-area--cursor-line, #composer .text-area--matching-bracket { background: $terminal; }
    #composer .text-area--gutter, #composer .text-area--suggestion, #composer .text-area--placeholder { color: $metadata; background: $terminal; }
    #command-menu { display: none; width: 100%; height: auto; max-height: 4; margin: 0 1; padding: 0 1; background: $terminal; }
    .menu-heading { height: 1; color: ansi_white; text-style: bold; }
    .menu-actions { height: auto; max-height: 3; }
    .menu-action { height: 1; }
    .menu-name { padding: 0 1; color: $terminal; }
    .menu-description { color: $metadata; }
    .menu-action.-selected .menu-name { color: $accent; text-style: bold; }
    .menu-action.-selected .menu-description { color: $accent; text-style: dim; }
    .menu-action.-disabled .menu-name, .menu-action.-disabled .menu-description, .menu-action.-muted .menu-name, .menu-action.-muted .menu-description { color: $metadata; text-style: dim; }
    #command-message { height: 1; color: ansi_white; text-style: bold; }
    #command-actions { height: auto; max-height: 3; }
    .command-action { height: 1; }
    .command-name { width: 13; height: 1; padding: 0 1; color: $terminal; }
    .command-description { width: 1fr; height: 1; color: $metadata; }
    .command-action.-selected .command-name { color: $accent; text-style: bold; }
    .command-action.-selected .command-description { color: $accent; text-style: dim; }
    .command-action.-disabled .command-name, .command-action.-disabled .command-description { color: $metadata; text-style: dim; }
    #activity-row { height: auto; min-height: 0; align: left middle; }
    #activity-primary { width: 1fr; min-width: 0; height: auto; min-height: 0; }
    #command-bar { display: none; width: 100%; height: auto; min-height: 3; margin: 0 1; padding: 0 1; border: solid $accent; background: $terminal; }
    #reference-menu { display: none; width: 100%; height: auto; max-height: 4; margin: 0 1; padding: 0 1; background: $terminal; }
    #reference-message { height: 1; color: ansi_white; text-style: bold; }
    #reference-actions { height: auto; max-height: 3; }
    .reference-action { height: 1; }
    .reference-name { width: 12; height: 1; padding: 0 1; color: $terminal; }
    .reference-preview { width: 1fr; height: 1; color: $metadata; }
    .reference-action.-selected .reference-name { color: $accent; text-style: bold; }
    .reference-action.-selected .reference-preview { color: $accent; text-style: dim; }
    #reference-bar { display: none; height: auto; min-height: 3; margin: 0 1; padding: 0 1; border: solid $accent; background: $terminal; }
    #status { display: none; height: 1; color: $metadata; background: $terminal; padding: 0 2; }
    #chat-alert { display: none; width: auto; max-width: 70%; height: 1; color: $accent; background: $terminal; padding: 0 2 0 0; text-align: right; text-style: none; text-overflow: ellipsis; }
    #confirm-dialog { width: 100%; height: 2; padding: 0 1; background: $terminal; border: none; }
    #confirm-message { height: 1; color: ansi_white; text-style: bold; }
    #confirm-actions { height: 1; }
    #confirm-actions Button { width: auto; min-width: 0; height: 1; padding: 0; border: none; background: $terminal; color: $metadata; text-style: none; }
    #confirm-actions Button:hover { border: none; background: $terminal; color: $metadata; text-style: none; }
    #confirm-actions Button:focus, #confirm-actions Button.-active { border: none; background: $terminal; color: $accent; text-style: bold; }
    #trace-name-dialog { width: 100%; height: 2; padding: 0 1; background: $terminal; border: none; }
    #trace-name-dialog.-unplaced { visibility: hidden; }
    #trace-name-message { height: 1; color: ansi_white; text-style: bold; }
    #trace-name { height: 1; border: none; padding: 0 1; background: $terminal; color: $terminal; }
    #trace-name:focus { border: none; }
    #trace-name > .input--placeholder { color: $metadata; }
    #trace-name > .input--cursor { color: $terminal; background: $terminal; text-style: reverse; }
    #trace-dialog { width: 100%; height: 2; padding: 0 1; background: $terminal; border: none; }
    #trace-message { height: 1; color: ansi_white; text-style: bold; }
    #trace-actions { height: 1; }
    #trace-actions Button { width: auto; min-width: 0; height: 1; padding: 0; border: none; background: $terminal; color: $metadata; text-style: none; }
    #trace-actions Button:hover { border: none; background: $terminal; color: $metadata; text-style: none; }
    #trace-actions Button:focus, #trace-actions Button.-active { border: none; background: $terminal; color: $accent; text-style: bold; }
    #chroma-dialog { width: 100%; height: 2; padding: 0 1; background: $terminal; border: none; }
    #chroma-dialog.-unplaced { visibility: hidden; }
    #chroma-message { height: 1; color: ansi_white; text-style: bold; }
    #chroma-actions { height: 1; padding: 0 0 0 1; }
    #chroma-actions Button { width: auto; min-width: 0; height: 1; padding: 0; border: none; background: $terminal; color: $metadata; text-style: none; }
    #chroma-actions Button:hover { border: none; background: $terminal; color: $metadata; text-style: none; }
    #chroma-actions Button:focus, #chroma-actions Button.-active { border: none; background: $terminal; color: $accent; text-style: bold; }
    .horizontal-menu { width: 100%; height: 2; padding: 0 1; background: $terminal; border: none; }
    .horizontal-menu.-unplaced { visibility: hidden; }
    .horizontal-menu-actions { height: 1; overflow-x: auto; overflow-y: hidden; scrollbar-size: 0 0; }
    .horizontal-menu-actions Button { width: auto; min-width: 0; height: 1; padding: 0; border: none; background: $terminal; color: $metadata; text-style: none; }
    .horizontal-menu-actions Button:hover { border: none; background: $terminal; color: $metadata; text-style: none; }
    .horizontal-menu-actions Button:focus, .horizontal-menu-actions Button.-active { border: none; background: $terminal; color: $accent; text-style: bold; }
    Button { border: tall $metadata; background: $terminal; color: $terminal; }
    Button:hover, Button:focus, Button.-active { border: tall $accent; background: $terminal; color: $terminal; background-tint: transparent; tint: transparent; text-style: reverse bold; }
    ModalScreen { background: transparent; }
    """ + _accent_theme_css()

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "stop", "Stop"),
        Binding("ctrl+c", "interrupt", "Stop or quit"),
        Binding(
            "ctrl+up",
            "scroll_transcript_up",
            "Scroll chat up",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+down",
            "scroll_transcript_down",
            "Scroll chat down",
            show=False,
            priority=True,
        ),
    ]

    def __init__(
        self,
        chat: ChatConfig,
        generation: GenerationConfig,
        assistants: Iterable[DiscoveryItem] = (),
        store: ChatStore | None = None,
        runtime_factory: Callable[..., ChatRuntime] = ChatRuntime,
        initial_assistant: Assistant | None = None,
        initial_chat: SavedChat | None = None,
    ) -> None:
        super().__init__(ansi_color=True)
        self.chat_policy = chat
        self.generation = generation
        self.assistants = tuple(assistants)
        self.store = ChatStore(chat.output) if store is None and chat.output else store
        self.runtime = runtime_factory(chat, generation)
        self.initial_assistant = initial_assistant
        self.initial_chat = initial_chat
        self._rows: dict[str, DiscoveryItem | SavedChat] = {}
        self._trace_details_expanded = True
        self._generating = False
        self._generation_attempt = 0
        self._loading = False
        self._stream_lock = threading.Lock()
        self._streaming_text = ""
        self._command_matches: tuple[str, ...] = ()
        self._command_index = 0
        self._dismissed_command_text: str | None = None
        self._command_selected = False
        self._command_name: str | None = None
        self._command_token_span: tuple[int, int] | None = None
        self._reference_bubbles: tuple[ChatBubble, ...] = ()
        self._reference_index: int | None = None
        self._reference_menu_index: int | None = None
        self._reference_token_span: tuple[int, int] | None = None
        self._reference_selected = False
        self._dismissed_reference_text: str | None = None
        self._active_dialog: ActiveDialog | None = None
        self._active_selector: ActiveSelector | None = None
        self._trace_blink_visible = True
        self._trace_blink_timer: Timer | None = None
        self._load_status_text: str | None = None
        self._status_text: str | None = None
        self._footer_text: str | None = None
        self._catalog_width: int | None = None
        self._loading_timer: Timer | None = None
        self._notice_timer: Timer | None = None
        self._selected_catalog_key: str | None = None
        self._specs_key: str | None = None
        self._active_sheet: SheetPage | None = None
        self._active_trace: SavedChat | None = None
        self._link_id: str | None = None
        self._accent_theme_index = next(
            index
            for index, theme in enumerate(_CHROMA_THEMES)
            if theme.color == _DEFAULT_CHROMA
        )

    def compose(self) -> ComposeResult:
        """Create the landing and chat screen widgets."""
        with Container(id="landing"), Vertical(id="landing-box"):
            yield Static(WATERMARK, markup=False, id="watermark")
            with Horizontal(id="catalog-heading"):
                yield Static("REGISTRY", markup=False, id="catalog-title")
            with Horizontal(id="catalog-subtitle"):
                yield Static(
                    "Select a CONSTRUCT to establish a LINK.",
                    markup=False,
                    id="catalog-description",
                )
            yield Rule(line_style="solid", id="catalog-rule")
            yield Static("", markup=False, id="catalog-columns")
            yield StatusLine(id="load-status")
            yield KeyboardOptionList(id="chooser")
            yield StatusLine(id="catalog-alert")
            yield Static(
                "↑↓ MOVE    ENTER CONNECT    CTRL+C TERMINATE",
                markup=False,
                id="catalog-hints",
            )
        yield InfoSheet(id="specs")
        with Container(id="chat"):
            with Horizontal(id="chat-heading"):
                yield Static("", markup=False, id="identity")
                yield Static(
                    _link_label(None),
                    markup=False,
                    id="chat-link",
                    classes="brand-link",
                )
            yield Rule(line_style="solid", id="identity-rule")
            with VerticalScroll(id="transcript-scroll"):
                yield Transcript(id="transcript")
            with Horizontal(id="activity-row"):
                with Container(id="activity-primary"):
                    yield CommandMenu(id="command-menu")
                    yield ReferenceMenu(id="reference-menu")
                    yield StatusLine(id="status")
                    yield CommandBar(id="command-bar", classes="selection-bar")
                yield StatusLine(id="chat-alert")
            yield ReferenceBar(id="reference-bar", classes="selection-bar")
            with Horizontal(id="composer-bar"):
                yield Static(">", markup=False, id="composer-prompt")
                yield Composer(id="composer", language=None)
            yield Static("", markup=False, id="footer")

    def on_mount(self) -> None:
        """Populate the chooser or open a direct selection."""
        specs_scroll = self.query_one("#specs-scroll", SheetScroll)
        cast(Any, specs_scroll.vertical_scrollbar).renderer = HalfCellScrollBarRender
        self._set_accent_theme(_DEFAULT_CHROMA)
        self._fill_chooser()
        self._loading_timer = self.set_interval(0.1, self._refresh_loading_state)
        if self.initial_chat is not None:
            self._begin_attach(
                self._saved_session(self.initial_chat),
                trace=self.initial_chat,
            )
        elif self.initial_assistant is not None:
            self._begin_select(self.initial_assistant)

    def on_unmount(self) -> None:
        """Stop the loading-state timer before the screen is removed."""
        if self._loading_timer is not None:
            self._loading_timer.stop()
            self._loading_timer = None
        if self._notice_timer is not None:
            self._notice_timer.stop()
            self._notice_timer = None
        if self._trace_blink_timer is not None:
            self._trace_blink_timer.stop()
            self._trace_blink_timer = None

    def on_resize(self, event: events.Resize) -> None:
        """Update catalog columns when the terminal width changes."""
        if (
            self._catalog_width is not None
            and event.size.width != self._catalog_width
            and self.query_one("#landing").display
        ):
            self.call_after_refresh(self._fill_chooser)
        elif self.query_one("#chat").display:
            self.call_after_refresh(self._update_footer)
            self.call_after_refresh(self._update_reference_menu)
            self.call_after_refresh(self._render_command)
            self.call_after_refresh(self._render_reference)
        elif self.query_one("#specs").display:
            self.call_after_refresh(self.query_one(InfoSheet).refresh_page)

    def on_key(self, event: events.Key) -> None:
        """Keep Escape reference behavior consistent with app bindings."""
        if (
            event.key == "escape"
            and (
                self._command_selected
                or self.query_one("#command-bar", CommandBar).display
                or self._reference_selected
                or self.query_one("#reference-bar", ReferenceBar).display
            )
            and not self.query_one("#command-menu", CommandMenu).display
            and not self.query_one("#reference-menu", ReferenceMenu).display
        ):
            if (
                self._command_selected
                or self.query_one("#command-bar", CommandBar).display
            ):
                self._escape_command()
            else:
                self._escape_reference()
            event.prevent_default()
            event.stop()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Open the selected assistant or saved chat."""
        if self._loading:
            return
        self._open_row(str(event.option.id))

    def on_keyboard_option_list_details_toggled(
        self,
        event: KeyboardOptionList.DetailsToggled,
    ) -> None:
        """Expand or collapse all trace names in the registry."""
        if not any(isinstance(row, SavedChat) for row in self._rows.values()):
            return
        self._trace_details_expanded = not self._trace_details_expanded
        self._refresh_catalog_prompts(event.key)

    def on_keyboard_option_list_erase_requested(
        self,
        event: KeyboardOptionList.EraseRequested,
    ) -> None:
        """Open the action menu when the highlighted row is a saved trace."""
        row = self._rows.get(event.key)
        if isinstance(row, SavedChat):
            self._active_dialog = ActiveDialog(DialogKind.TRACE_MANAGE, row.id)
            self._start_trace_blink()
            self._refresh_catalog_prompts(event.key)
            self.push_screen(
                TraceMenuModal(),
                lambda choice: self._after_trace_action(row, choice),
            )

    def on_keyboard_option_list_specs_requested(
        self,
        event: KeyboardOptionList.SpecsRequested,
    ) -> None:
        """Open model details for one available registry row."""
        if self._loading or self._active_dialog is not None:
            return
        row = self._rows.get(event.key)
        if isinstance(row, DiscoveryItem):
            if row.available and row.assistant is not None:
                self._show_specs(event.key)
        elif isinstance(row, SavedChat):
            self._show_specs(event.key)

    def on_keyboard_option_list_chroma_requested(
        self,
        event: KeyboardOptionList.ChromaRequested,
    ) -> None:
        """Open the registry's accent theme menu."""
        del event
        if self._loading or self._active_dialog is not None:
            return
        self._open_chroma()

    def _open_chroma(self) -> None:
        """Open the accent theme menu from the active screen."""
        self._active_dialog = ActiveDialog(
            DialogKind.CHROMA,
            initial_theme_index=self._accent_theme_index,
        )
        if self.query_one("#landing").display:
            self._update_catalog_hints()
        else:
            self._update_confirmation_spacing()
            self._update_footer()
        themes = tuple((theme.color, theme.name) for theme in _CHROMA_THEMES)
        current = _CHROMA_THEMES[self._accent_theme_index].color
        self.push_screen(
            ChromaMenuModal(themes, current, self._set_accent_theme),
            self._after_chroma,
        )

    def _after_chroma(self, choice: str | None) -> None:
        """Commit or cancel the CHROMA preview."""
        equipped = None
        initial = (
            self._active_dialog.initial_theme_index
            if self._active_dialog is not None
            and self._active_dialog.initial_theme_index is not None
            else self._accent_theme_index
        )
        if choice is None:
            name = _CHROMA_THEMES[initial].color
            self._set_accent_theme(name)
        else:
            self._set_accent_theme(choice)
            equipped = next(
                theme.name for theme in _CHROMA_THEMES if theme.color == choice
            )
        self._active_dialog = None
        if self.query_one("#landing").display:
            self._update_catalog_hints()
        else:
            self._update_confirmation_spacing()
            self._update_footer()
        if equipped is not None:
            self._show_ack(f"{equipped} equipped")

    def _set_accent_theme(self, name: str) -> None:
        """Apply one configured ANSI accent theme across the interface."""
        theme_index = next(
            index for index, theme in enumerate(_CHROMA_THEMES) if theme.color == name
        )
        self._accent_theme_index = theme_index
        theme = _CHROMA_THEMES[theme_index]
        for available in _CHROMA_THEMES:
            self.remove_class(f"-accent-{available.color}")
        self.add_class(f"-accent-{theme.color}")
        self.query_one("#watermark", Static).update(_watermark(theme.accent))
        self.query_one("#transcript", Transcript).set_accent(theme.accent)
        self.query_one("#command-bar", CommandBar).set_accent(theme.accent)
        self.query_one("#reference-bar", ReferenceBar).set_accent(theme.accent)
        if self._selected_catalog_key is not None:
            self._refresh_catalog_prompts(self._selected_catalog_key)

    def on_sheet_scroll_cycle_requested(
        self,
        event: SheetScroll.CycleRequested,
    ) -> None:
        """Show the adjacent available registry entry in SPECS."""
        self._cycle_specs(event.offset)

    def on_sheet_scroll_connect_requested(
        self,
        event: SheetScroll.ConnectRequested,
    ) -> None:
        """Connect to the selected registry entry from SPECS."""
        del event
        if (
            self._active_sheet is None
            or not self._active_sheet.connect
            or self._loading
            or self._specs_key is None
        ):
            return
        self.query_one("#specs").display = False
        self._open_row(self._specs_key)

    def _after_trace_action(self, trace: SavedChat, choice: str | None) -> None:
        if choice == "rename":
            self._active_dialog = ActiveDialog(DialogKind.TRACE_RENAME, trace.id)
            self._update_catalog_hints()
            self.push_screen(
                TraceNameModal(trace.title, registry=True),
                lambda title: self._after_trace_rename(trace, title),
            )
            return
        self._close_trace_menu()
        if choice != "erase":
            return
        if self.store is None:
            self._show_err("LINK not configured")
            return
        try:
            self.store.erase(trace.id)
        except ChatStorageError as error:
            self._show_err(str(error))
            return
        self._fill_chooser()

    def _after_trace_rename(self, trace: SavedChat, title: str | None) -> None:
        self._close_trace_menu()
        if title is None:
            return
        if self.store is None:
            self._show_err("LINK not configured")
            return
        try:
            self.store.rename(
                trace.id,
                title if title.strip() else trace.title,
            )
        except ChatStorageError as error:
            self._show_err(str(error))
            return
        self._fill_chooser()

    def _close_trace_menu(self) -> None:
        """Close TRACE controls and stop subject emphasis."""
        selected_key = self._selected_catalog_key
        if self._trace_blink_timer is not None:
            self._trace_blink_timer.stop()
            self._trace_blink_timer = None
        self._trace_blink_visible = True
        self._active_dialog = None
        if selected_key is not None:
            self._refresh_catalog_prompts(selected_key)
        else:
            self._update_catalog_hints()

    def _start_trace_blink(self) -> None:
        """Start the managed TRACE visibility pulse."""
        if self._trace_blink_timer is not None:
            self._trace_blink_timer.stop()
        self._trace_blink_visible = True
        self._trace_blink_timer = self.set_interval(0.45, self._toggle_trace_blink)

    def _toggle_trace_blink(self) -> None:
        """Toggle the managed TRACE name without changing row height."""
        if (
            self._active_dialog is None
            or self._active_dialog.trace_id is None
            or self._selected_catalog_key is None
        ):
            return
        self._trace_blink_visible = not self._trace_blink_visible
        self._refresh_catalog_prompts(self._selected_catalog_key)

    def _open_row(self, key: str) -> None:
        row = self._rows.get(key)
        if isinstance(row, DiscoveryItem) and row.assistant is not None:
            self._begin_select(row.assistant)
        elif isinstance(row, SavedChat):
            self._begin_attach(self._saved_session(row), trace=row)

    @staticmethod
    def _saved_session(saved: SavedChat) -> ChatSession:
        return ChatSession(
            saved.assistant,
            saved.chat,
            saved.generation,
            saved.turns,
            saved.id,
            saved.title,
        )

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Update slash and reference controls for one composer value."""
        value = event.text_area.text
        command_dismissed = value == self._dismissed_command_text
        reference_dismissed = value == self._dismissed_reference_text
        if not command_dismissed:
            self._dismissed_command_text = None
        if not reference_dismissed:
            self._dismissed_reference_text = None
        command_token = self._command_token_at_cursor(
            value,
            event.text_area.cursor_location,
        )
        self._command_matches = (
            ()
            if command_dismissed or command_token is None
            else completions(command_token[2])
        )
        self._command_index = 0
        self._update_command_menu()
        self._update_reference_menu()
        self._render_command()
        self._render_reference()

    def on_composer_resized(self, event: Composer.Resized) -> None:
        """Preserve bottom-follow when the composer makes the viewport shorter."""
        if event.height_delta <= 0:
            return
        scroller = self.query_one("#transcript-scroll", VerticalScroll)
        if scroller.scroll_y >= scroller.max_scroll_y - event.height_delta - 1:
            scroller.scroll_end(animate=False)

    def on_text_area_selection_changed(
        self,
        event: TextArea.SelectionChanged,
    ) -> None:
        """Update menus when the composer cursor moves."""
        if event.text_area is not self.query_one(Composer):
            return
        composer = self.query_one(Composer)
        command_token = self._command_token_at_cursor(
            composer.text,
            composer.cursor_location,
        )
        self._command_matches = (
            ()
            if composer.text == self._dismissed_command_text or command_token is None
            else completions(command_token[2])
        )
        self._update_command_menu()
        self._update_reference_menu()
        self._render_command()

    def on_composer_reference_moved(self, event: Composer.ReferenceMoved) -> None:
        """Move the highlighted reference with wrapping navigation."""
        if not self._reference_bubbles:
            return
        indexes = [bubble.index for bubble in self._reference_bubbles]
        current_index = (
            self._reference_menu_index
            if self._reference_selected
            else self._reference_index
        )
        current = (
            indexes.index(current_index)
            if current_index in indexes
            else len(indexes) - 1
        )
        selected = indexes[(current + event.offset) % len(indexes)]
        if self._reference_selected:
            self._reference_menu_index = selected
        else:
            self._reference_index = selected
        self._update_reference_menu()

    def on_composer_reference_dismissed(
        self,
        event: Composer.ReferenceDismissed,
    ) -> None:
        """Hide the reference menu until the composer changes."""
        del event
        self._dismissed_reference_text = self.query_one(Composer).text
        self._reference_token_span = None
        self._reference_bubbles = ()
        if self._reference_selected:
            self._reference_menu_index = None
        else:
            self._reference_index = None
        self._update_reference_menu()

    def on_composer_reference_selected(
        self,
        event: Composer.ReferenceSelected,
    ) -> None:
        """Keep the highlighted bubble as the composer reference."""
        del event
        selected_index = (
            self._reference_menu_index
            if self._reference_selected
            else self._reference_index
        )
        if selected_index is None:
            return
        composer = self.query_one(Composer)
        token_span = self._reference_token_span
        if token_span is None:
            return
        value = composer.text
        cursor_offset = self._cursor_offset(value, composer.cursor_location)
        start, end = token_span
        replacement = self._remove_reference_token(value, start, end)
        replacement_offset = self._reference_cursor_offset(
            value,
            start,
            end,
            cursor_offset,
            len(replacement),
        )
        self._reference_index = selected_index
        self._reference_selected = True
        self._reference_menu_index = None
        self._reference_token_span = None
        self._dismissed_reference_text = None
        composer.load_text(replacement)
        composer.move_cursor(self._cursor_location(replacement, replacement_offset))
        self._update_reference_menu()
        self._render_reference()
        self._update_footer()

    def on_composer_reference_escaped(
        self,
        event: Composer.ReferenceEscaped,
    ) -> None:
        """Turn the selected reference into literal composer text."""
        del event
        self._escape_reference()

    def on_composer_command_escaped(
        self,
        event: Composer.CommandEscaped,
    ) -> None:
        """Remove the selected argument command from the composer."""
        del event
        self._escape_command()

    def _escape_reference(self) -> None:
        """Keep the editable reference token as literal prompt text."""
        self._reference_selected = False
        self._reference_index = None
        self._reference_menu_index = None
        self._reference_token_span = None
        self._render_reference()
        self._update_reference_menu()
        self._update_footer()

    def _clear_reference_selection(self) -> None:
        """Clear a reference when a slash command takes control."""
        self._reference_selected = False
        self._reference_index = None
        self._reference_menu_index = None
        self._reference_token_span = None
        self._render_reference()

    def _escape_command(self) -> None:
        """Remove the selected command while keeping its argument text."""
        self._command_selected = False
        self._command_name = None
        self._command_token_span = None
        self._command_matches = ()
        self._render_command()
        self._update_command_menu()
        self._update_reference_menu()
        self._update_footer()

    def on_composer_command_moved(self, event: Composer.CommandMoved) -> None:
        """Move the command highlight with wrapping navigation."""
        enabled = self._enabled_command_indexes()
        if not enabled:
            return
        current = (
            enabled.index(self._command_index) if self._command_index in enabled else 0
        )
        self._command_index = enabled[(current + event.offset) % len(enabled)]
        self._update_command_menu()

    def on_composer_command_dismissed(self, event: Composer.CommandDismissed) -> None:
        """Hide the command menu until the composer value changes."""
        self._dismissed_command_text = self._composer_value(
            self.query_one(Composer).text
        )
        self._command_matches = ()
        self._update_command_menu()

    def _activate_command(self, selected: str) -> None:
        """Activate one selected command and keep any argument text."""
        composer = self.query_one(Composer)
        token = self._command_token_span
        if token is None:
            token_value = self._command_token_at_cursor(
                composer.text,
                composer.cursor_location,
            )
            if token_value is None:
                token_value = self._leading_command_token(composer.text)
            token = token_value[:2] if token_value is not None else None
        if token is None:
            return
        value = composer.text
        cursor_offset = self._cursor_offset(value, composer.cursor_location)
        start, end = token
        replacement = self._remove_command_token(value, start, end)
        replacement_offset = self._reference_cursor_offset(
            value,
            start,
            end,
            cursor_offset,
            len(replacement),
        )
        self._reference_selected = False
        self._reference_index = None
        self._reference_menu_index = None
        self._reference_token_span = None
        self._command_matches = ()
        self._dismissed_command_text = None
        if selected in COMMAND_ARGUMENTS:
            self._command_selected = True
            self._command_name = selected[1:]
            self._command_token_span = None
            composer.load_text(replacement)
            composer.move_cursor(self._cursor_location(replacement, replacement_offset))
            self._render_command()
            self._update_command_menu()
            self._update_reference_menu()
            self._render_reference()
            self._update_footer()
            return
        self._clear_command()
        composer.clear()
        self._command(selected[1:])

    @staticmethod
    def _remove_command_token(value: str, start: int, end: int) -> str:
        """Remove one slash token while keeping surrounding prompt text."""
        result = value[:start] + value[end:]
        if start == 0:
            return result.lstrip()
        if (
            start < len(result)
            and result[start - 1].isspace()
            and result[start].isspace()
        ):
            result = result[:start] + result[start + 1 :]
        return result

    def _clear_command(self) -> None:
        """Clear selected command state without changing composer text."""
        self._command_selected = False
        self._command_name = None
        self._command_token_span = None
        self._command_matches = ()
        self._render_command()

    def on_composer_submitted(self, event: Composer.Submitted) -> None:
        """Run one command or start one user turn."""
        value = self._composer_value(event.value)
        if not value.strip() and not self._command_selected:
            return
        if self._loading:
            self._show_err("LINK not established")
            return
        composer = self.query_one(Composer)
        try:
            selected = self._selected_command()
            if selected is not None:
                self._activate_command(selected)
                return
            if self._command_selected:
                if self._command_name is None:
                    return
                command = parse_command(f"/{self._command_name} {value}")
                if command is None:
                    return
                self._command(command.name, command.arguments)
                composer.record_submission(
                    f"/{self._command_name}" + (f" {value}" if value else "")
                )
                composer.clear()
                self._clear_command()
                return
            command = parse_command(value)
            if command is not None:
                if command.accepts_arguments:
                    self._activate_command(f"/{command.name}")
                    return
                self._clear_reference_selection()
                composer.clear()
                self._command(command.name, command.arguments)
                composer.record_submission(value)
                return
            if self._generating:
                self._show_err("CONSTRUCT is generating")
                return
            session = self._session()
            reference = self._reference_index if self._reference_selected else None
            session.add_user(value, reference)
            composer.record_submission(value)
            self._reference_selected = False
            self._reference_index = None
            composer.clear()
            self._render_reference()
            self._update_reference_menu()
            self._render_transcript()
            self._start_generation(0)
        except (
            ChatError,
            ChatStorageError,
            CommandError,
            ValueError,
            WorkerError,
        ) as error:
            self._show_err(str(error))

    def action_stop(self) -> None:
        """Stop active generation at a token boundary."""
        if self.query_one("#specs").display:
            if self._active_sheet is not None and self._active_sheet.origin == "chat":
                self._restore_chat_from_specs()
            else:
                self._show_registry()
        elif self._generating:
            self.runtime.cancel()
        elif (
            (
                self._command_selected
                or self.query_one("#command-bar", CommandBar).display
                or self._reference_selected
                or self.query_one("#reference-bar", ReferenceBar).display
            )
            and not self.query_one("#command-menu", CommandMenu).display
            and not self.query_one("#reference-menu", ReferenceMenu).display
        ):
            if (
                self._command_selected
                or self.query_one("#command-bar", CommandBar).display
            ):
                self._escape_command()
            else:
                self._escape_reference()

    def action_interrupt(self) -> None:
        """Stop active work or open the idle quit confirmation."""
        if (
            self.query_one("#specs").display
            and self._active_sheet is not None
            and self._active_sheet.origin == "chat"
        ):
            self._restore_chat_from_specs()
        if self._generating:
            self.runtime.cancel()
        else:
            self._request_quit()

    def action_scroll_transcript_up(self) -> None:
        """Move the chat viewport up without leaving the composer."""
        if self.query_one("#chat").display:
            self.query_one("#transcript-scroll", VerticalScroll).scroll_relative(
                y=-3,
                animate=False,
            )

    def action_scroll_transcript_down(self) -> None:
        """Move the chat viewport down without leaving the composer."""
        if self.query_one("#chat").display:
            self.query_one("#transcript-scroll", VerticalScroll).scroll_relative(
                y=3,
                animate=False,
            )

    def action_open_link(self, url: str) -> None:
        """Open one validated transcript link in the default browser."""
        if is_web_link(url):
            self.open_url(url)

    def _start_generation(self, attempt: int, *, reload_worker: bool = False) -> None:
        self._generating = True
        self._generation_attempt = attempt
        self._stream_reset()
        self._set_status("generating")
        self._update_command_menu()
        self._generate_thread(attempt, reload_worker)

    @work(thread=True, exclusive=True, group="generation")
    def _generate_thread(self, attempt: int, reload_worker: bool = False) -> None:
        last_render = 0.0
        try:
            if reload_worker:
                self.runtime.reload()
            for delta in self.runtime.generate(attempt, self._report_prefill):
                if self._stream_append(delta):
                    self.call_from_thread(self._set_status, "generating")
                now = time.monotonic()
                if now - last_render >= 1 / 12:
                    self.call_from_thread(self._render_transcript, True)
                    last_render = now
            self.call_from_thread(self._generation_done)
        except Exception as error:  # noqa: BLE001
            self.log(traceback.format_exc())
            self.call_from_thread(self._generation_failed, str(error))

    def _stream_append(self, delta: str) -> bool:
        """Append one delta and return true when the reply just started."""
        with self._stream_lock:
            empty = not self._streaming_text
            self._streaming_text += delta
        return empty

    def _stream_value(self) -> str:
        """Return one snapshot of the streaming reply text."""
        with self._stream_lock:
            return self._streaming_text

    def _stream_reset(self) -> None:
        """Clear the streaming reply buffer."""
        with self._stream_lock:
            self._streaming_text = ""

    def _generation_done(self) -> None:
        self._generating = False
        self._stream_reset()
        self._render_transcript()
        self._update_status()
        self._update_footer()
        self._update_command_menu()

    def _generation_failed(self, message: str) -> None:
        self._generating = False
        self._set_status(None)
        self._update_command_menu()
        self._show_err(message)

    def _report_prefill(self, current: int, total: int) -> None:
        self.call_from_thread(self._set_status, f"prefill {current}/{total} TOK")

    def _show_chat(self) -> None:
        session = self._session()
        self.chat_policy = session.chat
        self.generation = session.generation
        self.query_one("#landing").display = False
        self.query_one("#chat").display = True
        self.query_one("#identity", Static).update(session.assistant.name)
        self.query_one("#chat-link", Static).update(_link_label(self._link_id))
        self._render_transcript()
        self._render_command()
        self._render_reference()
        self._update_reference_menu()
        self._update_status()
        self._update_footer()
        self.query_one(Composer).focus()
        self.call_after_refresh(self._scroll_transcript_end)

    def _render_transcript(self, partial: bool = False) -> None:
        transcript = self.query_one("#transcript", Transcript)
        scroller = self.query_one("#transcript-scroll", VerticalScroll)
        follow_latest = scroller.scroll_y >= scroller.max_scroll_y - 1
        session = self._session()
        turns = []
        for turn in session.turns:
            if turn.role == "env":
                turns.append(("SYS", turn.content, True))
                continue
            name = "OP" if turn.role == "user" else self._chat_name(session)
            if turn.role == "user" and turn.reference is not None:
                referenced = self._reference_bubble(turn.reference)
                if referenced is not None:
                    name += (
                        f" [@{self._reference_name(referenced)}:{turn.reference:02d}]"
                    )
            if turn.role == "assistant":
                turns.extend(_episode_segments(name, turn.content))
            else:
                turns.append((name, turn.content))
        if partial and self._generating:
            turns.extend(
                _episode_segments(self._chat_name(session), self._stream_value() or "…")
            )
        transcript.set_turns(turns)
        if follow_latest:
            self.call_after_refresh(self._scroll_transcript_end)

    def _scroll_transcript_end(self) -> None:
        self.query_one("#transcript-scroll", VerticalScroll).scroll_end(animate=False)

    @staticmethod
    def _chat_name(session: ChatSession) -> str:
        return session.assistant.target_name.upper()

    def _reference_bubble(self, index: int) -> ChatBubble | None:
        """Return one stored bubble by its stable transcript index."""
        session = self.runtime.session
        if session is None:
            return None
        return next(
            (
                bubble
                for bubble in enumerate_bubbles(session.turns)
                if bubble.index == index
            ),
            None,
        )

    def _available_references(self) -> tuple[ChatBubble, ...]:
        """Return all stored bubbles that can receive a reply."""
        session = self.runtime.session
        return () if session is None else enumerate_bubbles(session.turns)

    def _reference_name(self, bubble: ChatBubble) -> str:
        """Return the display identity for one reference bubble."""
        if bubble.role == "user":
            return "OP"
        session = self.runtime.session
        return "ASSISTANT" if session is None else session.assistant.target_name.upper()

    def _reference_preview(self, content: str) -> str:
        """Flatten and width-limit one reference preview."""
        value = " ".join(content.split())
        width = max(8, self.size.width - 22)
        if len(value) <= width:
            return value
        return value[: max(1, width - 3)].rstrip() + "..."

    def _matching_references(self, query: str) -> tuple[ChatBubble, ...]:
        """Return references that match one editable at-sign query."""
        query = query.casefold()
        bubbles = self._available_references()
        if not query:
            return bubbles
        return tuple(
            bubble
            for bubble in bubbles
            if (
                self._reference_search_text(bubble).startswith(query)
                or (
                    f"{self._reference_name(bubble)}:{bubble.index}".casefold().startswith(
                        query
                    )
                )
                or f"{bubble.index:02d}".startswith(query)
                or str(bubble.index).startswith(query)
            )
        )

    def _reference_search_text(self, bubble: ChatBubble) -> str:
        """Return the searchable identity for one reference."""
        return f"{self._reference_name(bubble)}:{bubble.index:02d}".casefold()

    @staticmethod
    def _cursor_offset(value: str, location: tuple[int, int]) -> int:
        """Return one text offset for a TextArea cursor location."""
        row, column = location
        lines = value.split("\n")
        return sum(len(line) + 1 for line in lines[:row]) + min(
            column,
            len(lines[row]) if row < len(lines) else 0,
        )

    @staticmethod
    def _cursor_location(value: str, offset: int) -> tuple[int, int]:
        """Return a TextArea cursor location for one text offset."""
        remaining = max(0, min(offset, len(value)))
        lines = value.split("\n")
        for row, line in enumerate(lines):
            if remaining <= len(line):
                return row, remaining
            remaining -= len(line) + 1
        return len(lines) - 1, len(lines[-1])

    @staticmethod
    def _remove_reference_token(value: str, start: int, end: int) -> str:
        """Remove one reference token while keeping surrounding prompt text."""
        result = value[:start] + value[end:]
        if start == 0:
            return result.lstrip()
        if (
            start > 0
            and start < len(result)
            and result[start - 1].isspace()
            and result[start].isspace()
        ):
            result = result[:start] + result[start + 1 :]
        return result

    @staticmethod
    def _reference_cursor_offset(
        value: str,
        start: int,
        end: int,
        cursor_offset: int,
        replacement_length: int,
    ) -> int:
        """Map the cursor through removal of one reference token."""
        if cursor_offset <= start:
            return cursor_offset
        if start == 0:
            trailing = value[end:]
            trimmed = len(trailing) - len(trailing.lstrip())
            if cursor_offset <= end + trimmed:
                return 0
            return min(
                cursor_offset - (end - start) - trimmed,
                replacement_length,
            )
        if cursor_offset <= end:
            return start
        base = value[:start] + value[end:]
        extra = int(
            start < len(base) and base[start - 1].isspace() and base[start].isspace()
        )
        return min(
            cursor_offset - (end - start) - extra,
            replacement_length,
        )

    def _reference_token_at_cursor(
        self,
        value: str,
        location: tuple[int, int],
        allow_embedded: bool = False,
    ) -> tuple[int, int, str] | None:
        """Return the at-sign token under the composer cursor."""
        row, column = location
        lines = value.split("\n")
        if row < 0 or row >= len(lines):
            return None
        line = lines[row]
        column = min(max(column, 0), len(line))
        line_start = sum(len(item) + 1 for item in lines[:row])
        for marker in range(min(column, len(line)), -1, -1):
            if marker >= len(line) or line[marker] != "@":
                continue
            if not allow_embedded and marker and not line[marker - 1].isspace():
                continue
            end = next(
                (
                    index
                    for index, character in enumerate(line[marker + 1 :], marker + 1)
                    if character.isspace()
                ),
                len(line),
            )
            delimiter_end = end + 1 if end < len(line) else end
            if marker <= column <= delimiter_end:
                return line_start + marker, line_start + end, line[marker + 1 : end]
        return None

    def _command_token_at_cursor(
        self,
        value: str,
        location: tuple[int, int],
    ) -> tuple[int, int, str] | None:
        """Return a slash token under the composer cursor."""
        row, column = location
        lines = value.split("\n")
        if row < 0 or row >= len(lines):
            return None
        line = lines[row]
        column = min(max(column, 0), len(line))
        line_start = sum(len(item) + 1 for item in lines[:row])
        for marker in range(min(column, len(line)), -1, -1):
            if marker >= len(line) or line[marker] != "/":
                continue
            if marker and not line[marker - 1].isspace():
                continue
            end = next(
                (
                    index
                    for index, character in enumerate(
                        line[marker + 1 :],
                        marker + 1,
                    )
                    if character.isspace()
                ),
                len(line),
            )
            delimiter_end = end + 1 if end < len(line) else end
            if marker <= column <= delimiter_end:
                start = line_start + marker
                return start, line_start + end, line[marker:end]
        return None

    @staticmethod
    def _leading_command_token(value: str) -> tuple[int, int, str] | None:
        """Return the leading slash token independent of cursor location."""
        if not value.startswith("/"):
            return None
        end = next(
            (
                index
                for index, character in enumerate(value[1:], 1)
                if character.isspace()
            ),
            len(value),
        )
        return 0, end, value[:end]

    def _composer_value(self, value: str) -> str:
        """Return the prompt text, which excludes the selected reference."""
        return value

    def _render_command(self) -> None:
        """Refresh the selected argument command bar."""
        bar = self.query_one("#command-bar", CommandBar)
        active = self._command_selected and self._command_name is not None
        if not active:
            bar.display = False
            self._align_activity_notice()
            return
        assert self._command_name is not None
        command = f"/{self._command_name}"
        bar.set_command(self._command_name, COMMAND_DESCRIPTIONS[command])
        bar.display = True
        self._align_activity_notice()

    def _render_reference(self) -> None:
        """Refresh the selected reference bar."""
        bar = self.query_one("#reference-bar", ReferenceBar)
        bubble = (
            self._reference_bubble(self._reference_index)
            if self._reference_selected and self._reference_index is not None
            else None
        )
        if bubble is None:
            bar.display = False
            return
        name = self._reference_name(bubble)
        bar.set_reference(name, bubble.index, self._reference_preview(bubble.content))
        bar.display = True

    def _update_reference_menu(self) -> None:
        """Refresh the leading-@ reference selector."""
        composer = self.query_one(Composer)
        menu = self.query_one("#reference-menu", ReferenceMenu)
        if (
            self._command_selected
            or self.query_one("#command-menu", CommandMenu).display
        ):
            self._reference_token_span = None
            self._reference_bubbles = ()
            if not self._reference_selected:
                self._reference_index = None
            menu.set_references((), None)
            composer.reference_menu_active = False
            composer.reference_menu_escape_enabled = False
            composer.reference_selected = self._reference_selected
            self._sync_active_selector()
            self._update_footer()
            self._align_activity_notice()
            return
        token = self._reference_token_at_cursor(
            composer.text,
            composer.cursor_location,
            allow_embedded=self._reference_selected,
        )
        token_allowed = token is not None and (
            self._reference_selected or token[0] == 0
        )
        self._reference_token_span = token[:2] if token_allowed else None
        self._reference_bubbles = (
            self._matching_references(token[2])
            if token_allowed and token is not None
            else ()
        )
        active = (
            token_allowed
            and composer.text != self._dismissed_reference_text
            and bool(self._reference_bubbles)
        )
        if active:
            indexes = [bubble.index for bubble in self._reference_bubbles]
            if self._reference_selected:
                if self._reference_menu_index not in indexes:
                    self._reference_menu_index = indexes[-1]
                selected_index = indexes.index(self._reference_menu_index)
            else:
                if self._reference_index not in indexes:
                    self._reference_index = indexes[-1]
                selected_index = indexes.index(self._reference_index)
            start = min(
                max(selected_index - 2, 0),
                max(len(self._reference_bubbles) - 3, 0),
            )
            visible = self._reference_bubbles[start : start + 3]
            rows = tuple(
                (
                    f"{self._reference_name(bubble)}:{bubble.index:02d}",
                    self._reference_preview(bubble.content),
                )
                for bubble in visible
            )
            menu.set_references(rows, selected_index - start)
        else:
            if self._reference_selected:
                self._reference_menu_index = None
            else:
                self._reference_index = None
            menu.set_references((), None)
        composer.reference_menu_active = active
        composer.reference_menu_escape_enabled = active
        composer.reference_selected = self._reference_selected
        self._sync_active_selector()
        self._update_footer()
        self._align_activity_notice()
        if active:
            self.call_after_refresh(self._scroll_transcript_end)

    def _update_footer(self) -> None:
        if (
            self._active_dialog is not None
            and self._active_dialog.kind is DialogKind.CHROMA
        ):
            self._set_footer("←→ MOVE    ENTER EQUIP    ESC CANCEL")
            return
        if self._active_dialog is not None and self._active_dialog.kind in {
            DialogKind.CONFIRM,
            DialogKind.TRACE_NAME,
        }:
            self._set_footer(
                "ENTER TRACE    ESC RESUME"
                if self._active_dialog.kind is DialogKind.TRACE_NAME
                else "←→ MOVE    ENTER SELECT    ESC RESUME"
            )
            return
        if (
            self._active_selector is not None
            and self._active_selector.kind is SelectorKind.REFERENCE
            and self._active_selector.menu_open
        ):
            self._set_footer("↑↓ MOVE    ENTER REF    ESC CLOSE")
            return
        if (
            self._active_selector is not None
            and self._active_selector.kind is SelectorKind.COMMAND
            and self._active_selector.menu_open
        ):
            self._set_footer("↑↓ MOVE    ENTER COMMAND    ESC CLOSE")
            return
        session = self._session()
        last = next((turn for turn in reversed(session.turns) if turn.telemetry), None)
        if last is None or last.telemetry is None:
            value = ""
        else:
            telemetry = last.telemetry
            value = _telemetry_footer(
                telemetry,
                self.generation.max_prompt_tokens,
                max(0, self.size.width - 4),
            )
        self._set_footer(value)

    def _set_footer(self, value: str) -> None:
        value = value.upper()
        if value != self._footer_text:
            self.query_one("#footer", Static).update(value)
            self._footer_text = value

    def _update_status(self) -> None:
        state = self.runtime.state.value
        self._set_status(self._link_status_label(state) if self._loading else state)

    def _set_status(self, value: str | None) -> None:
        normalized = (
            ""
            if value is None or value.casefold() == "ready"
            else self._status_label(value)
        )
        if normalized != self._status_text:
            self.query_one("#status", StatusLine).set_state(
                normalized,
                animated=normalized not in {"CANCELLED", "FAILED"},
            )
            self._status_text = normalized

    def _command(self, name: str, arguments: str = "") -> None:
        """Run one selected command with its optional composer arguments."""
        if name == "echo":
            if not arguments.strip():
                raise CommandError("COMMAND argument missing")
            session = self._session()
            session.add_env(arguments)
            self._render_transcript()
            self._update_footer()
            return
        if name == "terminate":
            self.action_interrupt()
        elif name == "disconnect":
            if self._generating:
                self._show_err("CONSTRUCT is generating")
            else:
                self._return_to_landing()
        elif name == "specs":
            self._show_chat_specs()
        elif name == "probe":
            self._show_chat_probe()
        elif name == "retry":
            self._retry_generation()
        elif name == "buffer":
            self._show_chat_buffer()
        elif name == "chroma":
            self._open_chroma()
        elif name == "trace":
            session = self._session()
            if not session.dirty:
                trace_id = session.saved_chat_id
                message = (
                    "TRACE exists"
                    if trace_id is None
                    else f"TRACE {trace_id[:6].upper()} exists"
                )
                self._show_ack(message)
            elif self.store is None:
                self._show_err("LINK not configured")
            elif self._generating:
                self._show_err("CONSTRUCT is generating")
            else:
                self._active_dialog = ActiveDialog(DialogKind.TRACE_NAME)
                self._update_confirmation_spacing()
                self._update_footer()
                self.push_screen(
                    TraceNameModal(default_chat_title(session)),
                    self._after_checkpoint_name,
                )

    def _retry_generation(self) -> None:
        """Retry one cancelled reply or failed generation."""
        if not self._retry_enabled():
            raise CommandError("GENERATION not retryable")
        session = self._session()
        last = session.turns[-1]
        if last.role == "assistant":
            attempt = session.retry()
            self._render_transcript()
            self._start_generation(attempt)
            return
        self._start_generation(self._generation_attempt + 1, reload_worker=True)

    def _after_checkpoint_name(self, title: str | None) -> None:
        self._close_confirmation()
        if title is not None:
            self._save_from_confirmation(title)

    def _trace_enabled(self) -> bool:
        session = self.runtime.session
        return (
            self.store is not None
            and session is not None
            and session.dirty
            and not self._generating
        )

    def _retry_enabled(self) -> bool:
        """Return whether the latest generation can be retried."""
        session = self.runtime.session
        if self._generating or self._loading or session is None or not session.turns:
            return False
        last = session.turns[-1]
        return (
            last.role == "assistant" and last.finish_reason == "cancelled"
        ) or (last.role == "user" and self.runtime.state is WorkerState.FAILED)

    def _enabled_command_indexes(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, command in enumerate(self._command_matches)
            if (command != "/disconnect" or not self._generating)
            and (command != "/retry" or self._retry_enabled())
            and (command != "/trace" or self._trace_enabled())
        )

    def _command_cursor_in_block(self, composer: Composer) -> bool:
        """Return whether the composer cursor is inside its slash token."""
        return (
            self._command_token_at_cursor(
                composer.text,
                composer.cursor_location,
            )
            is not None
        )

    def _selected_command(self) -> str | None:
        if self._command_selected:
            return None
        if not self._command_cursor_in_block(self.query_one(Composer)):
            return None
        if (
            not self._command_matches
            or self._command_index not in self._enabled_command_indexes()
        ):
            return None
        return self._command_matches[self._command_index]

    def _update_command_menu(self) -> None:
        composer = self.query_one(Composer)
        token = self._command_token_at_cursor(
            composer.text,
            composer.cursor_location,
        )
        self._command_token_span = token[:2] if token is not None else None
        visible_matches = (
            ()
            if self._command_selected
            else self._command_matches
            if self._command_cursor_in_block(composer)
            else ()
        )
        enabled = self._enabled_command_indexes()
        if enabled and self._command_index not in enabled:
            self._command_index = enabled[0]
        selected = self._selected_command()
        menu = self.query_one("#command-menu", CommandMenu)
        was_visible = menu.display
        menu.set_commands(
            visible_matches,
            selected,
            registry_enabled=not self._generating,
            retry_enabled=self._retry_enabled(),
            trace_enabled=self._trace_enabled(),
        )
        composer.command_menu_active = bool(visible_matches)
        composer.command_menu_escape_enabled = not self._generating
        composer.command_selected = self._command_selected
        composer.reference_selected = self._reference_selected
        self._sync_active_selector()
        self._update_footer()
        self._align_activity_notice()
        if was_visible or visible_matches:
            self.call_after_refresh(self._scroll_transcript_end)

    def _return_to_landing(self) -> None:
        if self._session().dirty:
            self._push_confirmation(self._after_landing_confirm)
            return
        self.query_one("#chat").display = False
        self.query_one("#landing").display = True
        self._fill_chooser()

    def _after_landing_confirm(self, choice: str | None) -> None:
        if choice == "save":
            self._push_trace_name(self._after_landing_name)
            return
        self._close_confirmation()
        if choice == "discard":
            self.query_one("#chat").display = False
            self.query_one("#landing").display = True
            self._fill_chooser()

    def _after_landing_name(self, title: str | None) -> None:
        self._close_confirmation()
        if title is None or not self._save_from_confirmation(title):
            return
        self.query_one("#chat").display = False
        self.query_one("#landing").display = True
        self._fill_chooser()

    def _request_quit(self) -> None:
        if self.runtime.session is not None and self.runtime.session.dirty:
            self._push_confirmation(self._after_quit_confirm)
        else:
            self.runtime.close()
            self.exit()

    def _after_quit_confirm(self, choice: str | None) -> None:
        if choice == "save":
            self._push_trace_name(self._after_quit_name)
            return
        self._close_confirmation()
        if choice == "discard":
            self.runtime.close()
            self.exit()

    def _after_quit_name(self, title: str | None) -> None:
        self._close_confirmation()
        if title is None or not self._save_from_confirmation(title):
            return
        self.runtime.close()
        self.exit()

    def _push_confirmation(self, callback: Callable[[str | None], None]) -> None:
        self._active_dialog = ActiveDialog(DialogKind.CONFIRM)
        self._update_confirmation_spacing()
        self._update_footer()
        self.push_screen(ConfirmModal(), callback)

    def _push_trace_name(self, callback: Callable[[str | None], None]) -> None:
        self._active_dialog = ActiveDialog(DialogKind.TRACE_NAME)
        self._update_footer()
        self.push_screen(TraceNameModal(default_chat_title(self._session())), callback)

    def _close_confirmation(self) -> None:
        self._active_dialog = None
        self._update_confirmation_spacing()
        self._update_footer()

    def _update_confirmation_spacing(self) -> None:
        scroller = self.query_one("#transcript-scroll", VerticalScroll)
        overlay = self._active_dialog is not None and self._active_dialog.kind in {
            DialogKind.CHROMA,
            DialogKind.CONFIRM,
            DialogKind.TRACE_NAME,
        }
        bottom = 3 if overlay else 1
        self._sync_active_selector()
        if (
            self._active_selector is not None
            and self._active_selector.accessory_selected
        ):
            bottom += 2
        scroller.styles.padding = (1, 2, bottom, 2)
        self.call_after_refresh(self._scroll_transcript_end)

    def _sync_active_selector(self) -> None:
        """Derive the one active selector record from rendered controls."""
        command_menu = self.query_one("#command-menu", CommandMenu).display
        reference_menu = self.query_one("#reference-menu", ReferenceMenu).display
        command_selected = (
            self._command_selected or self.query_one("#command-bar", CommandBar).display
        )
        reference_selected = (
            self._reference_selected
            or self.query_one("#reference-bar", ReferenceBar).display
        )
        if command_menu or command_selected:
            self._active_selector = ActiveSelector(
                SelectorKind.COMMAND,
                command_menu,
                command_selected,
            )
        elif reference_menu or reference_selected:
            self._active_selector = ActiveSelector(
                SelectorKind.REFERENCE,
                reference_menu,
                reference_selected,
            )
        else:
            self._active_selector = None

    def _save_from_confirmation(self, title: str) -> bool:
        if self.store is None:
            self._show_err("LINK not configured")
            return False
        try:
            saved = self.store.save(
                self._session(),
                title if title.strip() else None,
                self.runtime.backend_versions,
            )
        except ChatStorageError as error:
            self._show_err(str(error))
            return False
        self._active_trace = saved
        self._show_ack(f"TRACE {saved.id[:6].upper()} saved")
        return True

    def _begin_select(self, assistant: Assistant) -> None:
        if not self._prepare_load():
            return
        self.query_one(Composer).set_history(())
        self._active_trace = None
        self._link_id = _random_link_id()
        self.runtime.session = ChatSession(
            assistant,
            self.chat_policy,
            self.generation,
        )
        self._set_loading(True)
        self._show_chat()
        self._select_thread(assistant)

    def _begin_attach(
        self,
        session: ChatSession,
        *,
        trace: SavedChat | None = None,
    ) -> None:
        if not self._prepare_load():
            return
        self.query_one(Composer).set_history(
            tuple(turn.content for turn in session.turns if turn.role == "user")
        )
        self._active_trace = trace
        self._link_id = _random_link_id()
        self.runtime.session = session
        self._set_loading(True)
        self._show_chat()
        self._attach_thread(session)

    def _prepare_load(self) -> bool:
        try:
            self.runtime.ensure_worker()
        except Exception as error:  # noqa: BLE001
            self.log(traceback.format_exc())
            self._load_failed(str(error))
            return False
        return True

    @work(thread=True, exclusive=True, group="model-load")
    def _select_thread(self, assistant: Assistant) -> None:
        try:
            self.runtime.select(assistant)
        except Exception as error:  # noqa: BLE001
            self.log(traceback.format_exc())
            self.call_from_thread(self._load_failed, str(error))
            return
        self.call_from_thread(self._load_done)

    @work(thread=True, exclusive=True, group="model-load")
    def _attach_thread(
        self,
        session: ChatSession,
    ) -> None:
        try:
            self.runtime.attach(session)
        except Exception as error:  # noqa: BLE001
            self.log(traceback.format_exc())
            self.call_from_thread(self._load_failed, str(error))
            return
        self.call_from_thread(self._load_done)

    def _load_done(self) -> None:
        self._set_loading(False)
        self._show_chat()
        self._show_ack("LINK established")

    def _load_failed(self, message: str) -> None:
        self._set_loading(False)
        if self.runtime.session is not None:
            self._show_chat()
        self._set_status(None)
        self._show_err(message)

    def _show_ack(self, message: str) -> None:
        """Show one transient acknowledgement."""
        self._show_notice(NoticeKind.ACK, message)

    def _show_err(self, message: str) -> None:
        """Show one transient error notice."""
        self._show_notice(NoticeKind.ERR, message)

    @staticmethod
    def _notice_body(message: str) -> str:
        """Normalize the first word in one notice message."""
        value = message.strip()
        start = next(
            (index for index, character in enumerate(value) if character.isalpha()),
            len(value),
        )
        if start == len(value):
            return value
        word = value[start:].split(maxsplit=1)[0].rstrip(".,:;!?")
        if word.isupper():
            return value
        return value[:start] + value[start].lower() + value[start + 1 :]

    def _show_notice(self, kind: NoticeKind, message: str) -> None:
        """Show one typed transient notice beside the active controls."""
        if self._notice_timer is not None:
            self._notice_timer.stop()
        identifier = (
            "#chat-alert" if self.query_one("#chat").display else "#catalog-alert"
        )
        other = "#catalog-alert" if identifier == "#chat-alert" else "#chat-alert"
        self.query_one(other, StatusLine).set_state("")
        body = self._notice_body(message)
        if body and not body.endswith((".", "!", "?")):
            body += "."
        value = f"SYS: {kind.value} {body}"
        accent = _CHROMA_THEMES[self._accent_theme_index].accent
        content = Text.assemble(
            ("SYS:", Style(color=accent, dim=True)),
            (f" {kind.value} {body}", Style(color="bright_black")),
        )
        notice = self.query_one(identifier, StatusLine)
        if identifier == "#chat-alert":
            self._align_activity_notice()
        notice.set_content(content, value)
        self._notice_timer = self.set_timer(5, self._clear_notice)

    def _align_activity_notice(self) -> None:
        """Align the chat notice with the active content's last text line."""
        if not self.is_mounted or len(self.query("#activity-row")) == 0:
            return
        command_bar = self.query_one("#command-bar", CommandBar)
        command_menu = self.query_one("#command-menu", CommandMenu)
        reference_menu = self.query_one("#reference-menu", ReferenceMenu)
        if command_bar.display:
            offset = 1
        elif command_menu.display:
            offset = sum(
                action.display for action in command_menu.query(".command-action")
            )
        elif reference_menu.display:
            offset = sum(
                action.display for action in reference_menu.query(".reference-action")
            )
        else:
            offset = 0
        self.query_one("#chat-alert", StatusLine).styles.margin = (
            offset,
            0,
            0,
            0,
        )

    def _clear_notice(self) -> None:
        """Clear the transient notice line."""
        for identifier in ("#chat-alert", "#catalog-alert"):
            self.query_one(identifier, StatusLine).set_state("")
        self._notice_timer = None

    def _set_loading(self, value: bool) -> None:
        self._loading = value
        self.query_one("#chooser", OptionList).disabled = value
        self._refresh_loading_state()

    def _refresh_loading_state(self) -> None:
        if not self.is_mounted or len(self.query("#load-status")) == 0:
            return
        state = self.runtime.state.value
        status = self._link_status_label(state) if self._loading else ""
        if status != self._load_status_text:
            self.query_one("#load-status", StatusLine).set_state(status)
            self._load_status_text = status
        if self._loading and self.query_one("#chat").display:
            self._update_status()

    @staticmethod
    def _status_label(value: str) -> str:
        """Return the uppercase label for one worker state."""
        return value.upper()

    @staticmethod
    def _link_status_label(value: str) -> str:
        """Return one loading label with its link prefix."""
        return f"LINK {value.upper()}"

    def _fill_chooser(self) -> None:
        self._catalog_width = self.size.width
        chooser = self.query_one("#chooser", OptionList)
        assert isinstance(chooser, KeyboardOptionList)
        chooser.selection_changed = self._refresh_catalog_prompts
        self._selected_catalog_key = None
        chooser.clear_options()
        self._rows.clear()
        options = []
        saved_chats = () if self.store is None else self.store.leaves()
        layout = CatalogLayout.for_terminal(self.size.width)
        self.query_one("#catalog-columns", Static).update(
            layout.line("CONSTRUCT", "BASE", "TYPE", "STATUS")
        )
        for index, row in enumerate(self.assistants):
            if row.available and row.assistant is not None:
                assistant = row.assistant
                if assistant.run is None:
                    kind = "BASE"
                else:
                    kind = "CONSTRUCT"
                text = layout.text(
                    assistant.target_run,
                    assistant.model_basename,
                    kind,
                    "READY",
                )
            else:
                text = layout.text(
                    row.label,
                    "",
                    "—",
                    "FAULT",
                    failed=True,
                )
            key = f"assistant-{index}"
            self._rows[key] = row
            options.append(Option(text, id=key, disabled=not row.available))
        for saved in saved_chats:
            text = layout.text(
                saved.assistant.target_run,
                saved.assistant.model_basename,
                "TRACE",
                "READY",
                trace_name=(saved.title if self._trace_details_expanded else None),
            )
            key = f"saved-{saved.id}"
            self._rows[key] = saved
            options.append(Option(text, id=key))
        chooser.add_options(options)
        for option_index, option in enumerate(chooser.options):
            if not option.disabled:
                chooser.highlighted = option_index
                chooser.focus()
                break
        self._update_catalog_hints()

    def _show_specs(self, key: str) -> None:
        """Show details for one registry entry without loading its model."""
        self._specs_key = key
        self._open_sheet(self._registry_sheet_page())

    def _show_chat_specs(self) -> None:
        """Show model details as a temporary view over the active chat."""
        session = self._session()
        self._specs_key = None
        self._open_sheet(
            SheetPage(
                session.assistant.name,
                self._render_chat_specs_document,
                "chat",
                _link_label(self._link_id),
                "↑↓ SCROLL    ESC LINK    CTRL+C TERMINATE",
            )
        )

    def _show_chat_probe(self) -> None:
        """Show live hardware and load details over the active chat."""
        session = self._session()
        self._specs_key = None
        self._open_sheet(
            SheetPage(
                session.assistant.name,
                lambda: render_probe(
                    self.runtime.runtime_probe,
                    self.runtime.load_probe,
                ),
                "chat",
                _link_label(self._link_id),
                "↑↓ SCROLL    ESC LINK    CTRL+C TERMINATE",
            )
        )

    def _show_chat_buffer(self) -> None:
        """Show context-window details over the active chat."""
        session = self._session()
        self._specs_key = None
        self._open_sheet(
            SheetPage(
                session.assistant.name,
                lambda: render_buffer(
                    session,
                    getattr(self.runtime, "last_prompt", None),
                ),
                "chat",
                _link_label(self._link_id),
                "↑↓ SCROLL    ESC LINK    CTRL+C TERMINATE",
            )
        )

    def _open_sheet(self, page: SheetPage) -> None:
        """Open one sheet through the shared sheet component."""
        self.query_one("#landing").display = False
        self.query_one("#chat").display = False
        self._active_sheet = page
        self.query_one(InfoSheet).open(page)

    def _restore_chat_from_specs(self) -> None:
        """Return from temporary model details without changing chat state."""
        self.query_one(InfoSheet).close()
        self.query_one("#chat").display = True
        self._active_sheet = None
        self.query_one(Composer).focus()

    def _show_registry(self) -> None:
        """Return from model details to the unchanged registry selection."""
        self.query_one(InfoSheet).close()
        self.query_one("#landing").display = True
        self._active_sheet = None
        self._specs_key = None
        self.query_one("#chooser", OptionList).focus()

    def _cycle_specs(self, offset: int) -> None:
        """Cycle SPECS through available registry rows with wrapping."""
        chooser = self.query_one("#chooser", OptionList)
        available = [
            (index, str(option.id))
            for index, option in enumerate(chooser.options)
            if option.id is not None and not option.disabled
        ]
        keys = [key for _, key in available]
        if self._specs_key not in keys or not available:
            return
        current = keys.index(self._specs_key)
        option_index, key = available[(current + offset) % len(available)]
        self._specs_key = key
        chooser.highlighted = option_index
        page = self._registry_sheet_page()
        self._active_sheet = page
        self.query_one(InfoSheet).open(page)

    def _registry_sheet_page(self) -> SheetPage:
        """Return the page declaration for the selected registry row."""
        row = self._rows.get(self._specs_key or "")
        if isinstance(row, DiscoveryItem):
            title = "SPECS" if row.assistant is None else row.assistant.name
        elif isinstance(row, SavedChat):
            title = row.assistant.name
        else:
            title = "SPECS"
        return SheetPage(
            title,
            self._render_registry_specs_document,
            "registry",
            None,
            "↑↓ SCROLL    ←→ CONSTRUCT    ENTER CONNECT    ESC REGISTRY    CTRL+C TERMINATE",
            cycle=True,
            connect=True,
        )

    def _render_registry_specs_document(self) -> SheetDocument:
        """Render the current registry SPECS document."""
        row = self._rows.get(self._specs_key or "")
        if isinstance(row, DiscoveryItem) and row.assistant is not None:
            assistant = row.assistant
            kind = "BASE" if assistant.run is None else "CONSTRUCT"
            generation = self.generation
            trace = None
        elif isinstance(row, SavedChat):
            assistant = row.assistant
            kind = "TRACE"
            generation = row.generation
            trace = row
        else:
            return SheetDocument()
        return render_specs(assistant, generation, kind, trace)

    def _render_chat_specs_document(self) -> SheetDocument:
        """Render the active chat SPECS document."""
        session = self._session()
        assistant = session.assistant
        trace = self._active_trace
        kind = (
            "TRACE"
            if trace is not None
            else "BASE"
            if assistant.run is None
            else "CONSTRUCT"
        )
        return render_specs(assistant, session.generation, kind, trace)

    def _refresh_catalog_prompts(self, selected_key: str) -> None:
        self._selected_catalog_key = selected_key
        chooser = self.query_one("#chooser", OptionList)
        layout = CatalogLayout.for_terminal(self.size.width)
        managed_trace_id = (
            None if self._active_dialog is None else self._active_dialog.trace_id
        )
        for option in chooser.options:
            if option.id is None:
                continue
            row = self._rows.get(option.id)
            if isinstance(row, DiscoveryItem):
                if row.available and row.assistant is not None:
                    assistant = row.assistant
                    kind = "BASE" if assistant.run is None else "CONSTRUCT"
                    prompt = layout.text(
                        assistant.target_run,
                        assistant.model_basename,
                        kind,
                        "READY",
                        selected=option.id == selected_key,
                    )
                else:
                    prompt = layout.text(
                        row.label,
                        "",
                        "—",
                        "FAULT",
                        failed=True,
                    )
            elif isinstance(row, SavedChat):
                prompt = layout.text(
                    row.assistant.target_run,
                    row.assistant.model_basename,
                    "TRACE",
                    "READY",
                    selected=option.id == selected_key,
                    trace_name=(
                        None
                        if not self._trace_details_expanded
                        and row.id != managed_trace_id
                        else row.title
                    ),
                    trace_active=row.id == managed_trace_id,
                    trace_visible=(
                        self._trace_blink_visible
                        if row.id == managed_trace_id
                        else True
                    ),
                )
            else:
                continue
            chooser.replace_option_prompt(option.id, prompt)
        self._update_catalog_hints()

    def _update_catalog_hints(self) -> None:
        """Show actions available for the current registry row."""
        if (
            self._active_dialog is not None
            and self._active_dialog.kind is DialogKind.CHROMA
        ):
            self.query_one("#catalog-hints", Static).update(
                "←→ MOVE    ENTER EQUIP    ESC CANCEL"
            )
            return
        if self._active_dialog is not None and self._active_dialog.trace_id is not None:
            self.query_one("#catalog-hints", Static).update(
                "ENTER NAME    ESC RETAIN"
                if self._active_dialog.kind is DialogKind.TRACE_RENAME
                else "←→ MOVE    ENTER SELECT    ESC RETAIN"
            )
            return
        has_traces = any(isinstance(row, SavedChat) for row in self._rows.values())
        selected_trace = isinstance(
            self._rows.get(self._selected_catalog_key or ""),
            SavedChat,
        )
        fields = ["↑↓ MOVE", "ENTER CONNECT", "S SPECS", "C CHROMA"]
        if has_traces:
            fields.append("SPACE DETAILS")
        if selected_trace:
            fields.append("BACKSPACE MANAGE")
        fields.append("CTRL+C TERMINATE")
        available = max(20, self.size.width - 4)
        gap = "    " if available >= 60 else "  "
        for removable in (
            "CTRL+C TERMINATE",
            "C CHROMA",
            "SPACE DETAILS",
            "↑↓ MOVE",
        ):
            if len(gap.join(fields)) <= available:
                break
            if removable in fields:
                fields.remove(removable)
        self.query_one("#catalog-hints", Static).update(gap.join(fields))

    def _session(self) -> ChatSession:
        if self.runtime.session is None:
            raise ChatError("Select an assistant first")
        return self.runtime.session
