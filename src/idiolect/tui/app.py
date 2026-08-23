"""Render the local Idiolect chat terminal interface."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
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
from idiolect.chat.state import ChatSession, TurnTelemetry
from idiolect.chat.storage import (
    ChatStorageError,
    ChatStore,
    SavedChat,
    default_chat_title,
)
from idiolect.chat.worker import WorkerError
from idiolect.config import ChatConfig, GenerationConfig
from idiolect.tui.catalog import CatalogLayout
from idiolect.tui.commands import CommandError, completions, parse_command
from idiolect.tui.markdown import is_web_link
from idiolect.tui.specs import HalfCellScrollBarRender, render_specs
from idiolect.tui.widgets import (
    CommandMenu,
    Composer,
    ConfirmModal,
    KeyboardOptionList,
    LoadingStatus,
    SpecsScroll,
    TraceMenuModal,
    TraceNameModal,
    Transcript,
)

_WATERMARK_SOURCE = """     ╭─╮
  ╭──╯ ╰──╮
  │  · ·  │    IDIOLECT
  ╰──╮ ╭──╯    Someone, reconstructed.
     ╰─╯"""
_TAGLINE = "Someone, reconstructed."
_ACCENT_THEMES = (
    ("green", "green"),
    ("yellow", "yellow"),
    ("blue", "blue"),
    ("purple", "magenta"),
    ("cyan", "cyan"),
)


def _watermark(color: str = "green") -> Text:
    """Return the product watermark in one ANSI accent color."""
    value = Text(_WATERMARK_SOURCE, style=Style(color=color, bold=True))
    tagline_start = _WATERMARK_SOURCE.index(_TAGLINE)
    value.stylize(
        Style(color=color, bold=False, dim=True),
        tagline_start,
        tagline_start + len(_TAGLINE),
    )
    return value


def _accent_theme_css() -> str:
    """Return screen-class overrides for each selectable ANSI accent."""
    color_selectors = (
        "OptionList > .option-list--option-highlighted",
        "OptionList:focus > .option-list--option-highlighted",
        "OptionList > .option-list--option-hover",
        "#specs-identity",
        ".brand-eyes",
        "#identity",
        "#composer-prompt",
        ".command-action.-selected .command-name",
        ".command-action.-selected .command-description",
        "#confirm-actions Button:focus",
        "#confirm-actions Button.-active",
        "#trace-actions Button:focus",
        "#trace-actions Button.-active",
    )
    button_selectors = (
        "Button:hover",
        "Button:focus",
        "Button.-active",
    )
    rules = []
    for name, color in _ACCENT_THEMES:
        prefix = f".-accent-{name} "
        colored = ", ".join(prefix + selector for selector in color_selectors)
        buttons = ", ".join(prefix + selector for selector in button_selectors)
        rules.append(f"{colored} {{ color: ansi_{color}; }}")
        rules.append(
            f"{prefix}#composer-bar:focus-within "
            f"{{ border: solid ansi_{color}; }}"
        )
        rules.append(f"{buttons} {{ border: tall ansi_{color}; }}")
    return "\n".join(rules)


WATERMARK = _watermark()

_FOOTER_GAP = "    "


def _telemetry_footer(
    telemetry: TurnTelemetry,
    max_prompt_tokens: int,
    available_width: int,
) -> str:
    """Format the measured values that fit in the footer."""
    pressure = 100 * telemetry.prompt_tokens / max_prompt_tokens
    context = (
        "CTX "
        f"{telemetry.prompt_tokens:,}/{max_prompt_tokens:,} "
        f"({pressure:.0f}%)"
    )
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
    $accent: ansi_green;
    $metadata: ansi_bright_black;
    $failure: ansi_red;
    Screen { background: $terminal; color: $terminal; }
    #landing { align: left top; padding: 0; }
    #landing-box { width: 100%; max-width: 100%; height: 100%; background: $terminal; }
    #watermark { color: $accent; height: 5; padding: 0 2; text-align: left; }
    #catalog-heading { height: 1; margin-top: 1; padding: 0 2; }
    #catalog-title { text-style: bold; }
    #catalog-subtitle { height: 1; padding: 0 2; color: $metadata; }
    #catalog-description { width: 1fr; }
    #catalog-rule { height: 1; margin: 0; padding: 0 2; color: $metadata; }
    #catalog-columns { height: 1; padding: 0 2; color: ansi_white; text-style: bold; }
    #load-status { display: none; height: 1; padding: 0 2; color: $metadata; }
    #chooser { height: 1fr; padding: 0 2; border: none; color: $terminal; background: $terminal; background-tint: transparent; scrollbar-color: $metadata; scrollbar-background: $terminal; }
    OptionList > .option-list--option { padding: 0; color: $terminal; background: $terminal; }
    OptionList > .option-list--option-highlighted, OptionList:focus > .option-list--option-highlighted { color: $accent; background: $terminal; text-style: bold; }
    OptionList > .option-list--option-disabled { color: $metadata; text-style: dim; }
    OptionList > .option-list--option-hover { color: $accent; background: $terminal; text-style: bold; }
    #catalog-hints { height: 1; color: $metadata; background: $terminal; padding: 0 2; }
    #catalog-alert, #chat-alert { display: none; height: 1; color: $metadata; background: $terminal; padding: 0 2; text-align: right; }
    #catalog-alert.-error, #chat-alert.-error { color: $failure; }
    #footer { height: 1; color: $metadata; background: $terminal; padding: 0 2; }
    #specs { display: none; background: $terminal; }
    #specs-heading { height: auto; min-height: 1; margin-top: 1; padding: 0 2; }
    #specs-identity { width: 1fr; height: auto; min-height: 1; color: $accent; text-style: bold; }
    .brand-eyes { width: 4; height: 1; padding-right: 1; color: $accent; text-align: right; text-style: bold; }
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
    #composer-bar { height: auto; min-height: 3; max-height: 10; border: solid $metadata; margin: 0 1; padding: 0 1; background: $terminal; scrollbar-size: 0 0; }
    #composer-bar:focus-within { border: solid $accent; }
    #composer-prompt { width: 1; height: 1; color: $accent; }
    #composer { width: 1fr; min-width: 0; height: auto; min-height: 1; max-height: 10; border: none; margin: 0; padding: 0 1; background: $terminal; scrollbar-size: 0 0; }
    #composer .text-area--cursor, #composer .text-area--selection { color: $terminal; background: $terminal; text-style: reverse; }
    #composer .text-area--cursor-line, #composer .text-area--matching-bracket { background: $terminal; }
    #composer .text-area--gutter, #composer .text-area--suggestion, #composer .text-area--placeholder { color: $metadata; background: $terminal; }
    #command-menu { display: none; height: auto; max-height: 4; margin: 0 1; padding: 0 1; background: $terminal; }
    #command-message { height: 1; color: ansi_white; text-style: bold; }
    #command-actions { height: auto; max-height: 3; }
    .command-action { height: 1; }
    .command-name { width: 12; height: 1; padding: 0 1; color: $terminal; }
    .command-description { width: 1fr; height: 1; color: $metadata; }
    .command-action.-selected .command-name { color: $accent; text-style: bold; }
    .command-action.-selected .command-description { color: $accent; text-style: dim; }
    .command-action.-disabled .command-name, .command-action.-disabled .command-description { color: $failure; text-style: dim; }
    .command-action.-save-disabled .command-name, .command-action.-save-disabled .command-description { color: $metadata; text-style: dim; }
    #status { display: none; height: 1; color: $metadata; background: $terminal; padding: 0 2; }
    #confirm-dialog { width: 100%; height: 2; padding: 0 1; background: $terminal; border: none; }
    #confirm-message { height: 1; color: ansi_white; text-style: bold; }
    #confirm-actions { height: 1; }
    #confirm-actions Button { width: auto; min-width: 0; height: 1; padding: 0 1; border: none; background: $terminal; color: $metadata; text-style: none; }
    #confirm-actions Button:hover { border: none; background: $terminal; color: $metadata; text-style: none; }
    #confirm-actions Button:focus, #confirm-actions Button.-active { border: none; background: $terminal; color: $accent; text-style: bold; }
    #trace-name-dialog { width: 100%; height: 2; padding: 0 1; background: $terminal; border: none; }
    #trace-name-message { height: 1; color: ansi_white; text-style: bold; }
    #trace-name { height: 1; border: none; padding: 0 1; background: $terminal; color: $terminal; }
    #trace-name:focus { border: none; }
    #trace-name > .input--placeholder { color: $metadata; }
    #trace-name > .input--cursor { color: $terminal; background: $terminal; text-style: reverse; }
    #trace-dialog { width: 100%; height: 2; padding: 0 1; background: $terminal; border: none; }
    #trace-message { height: 1; color: ansi_white; text-style: bold; }
    #trace-actions { height: 1; }
    #trace-actions Button { width: auto; min-width: 0; height: 1; padding: 0 1; border: none; background: $terminal; color: $metadata; text-style: none; }
    #trace-actions Button:hover { border: none; background: $terminal; color: $metadata; text-style: none; }
    #trace-actions Button:focus, #trace-actions Button.-active { border: none; background: $terminal; color: $accent; text-style: bold; }
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
        self._loading = False
        self._streaming_text = ""
        self._command_matches: tuple[str, ...] = ()
        self._command_index = 0
        self._dismissed_command_text: str | None = None
        self._confirmation_open = False
        self._trace_name_open = False
        self._trace_menu_id: str | None = None
        self._trace_rename_open = False
        self._trace_blink_visible = True
        self._trace_blink_timer: Timer | None = None
        self._load_status_text: str | None = None
        self._status_text: str | None = None
        self._footer_text: str | None = None
        self._catalog_width: int | None = None
        self._loading_timer: Timer | None = None
        self._alert_timer: Timer | None = None
        self._selected_catalog_key: str | None = None
        self._specs_key: str | None = None
        self._specs_from_chat = False
        self._active_trace: SavedChat | None = None
        self._accent_theme_index = 0

    def compose(self) -> ComposeResult:
        """Create the landing and chat screen widgets."""
        with Container(id="landing"), Vertical(id="landing-box"):
            yield Static(WATERMARK, markup=False, id="watermark")
            with Horizontal(id="catalog-heading"):
                yield Static("REGISTRY", markup=False, id="catalog-title")
            with Horizontal(id="catalog-subtitle"):
                yield Static(
                    "Connect to a BASE, CONSTRUCT, or TRACE.",
                    markup=False,
                    id="catalog-description",
                )
            yield Rule(line_style="solid", id="catalog-rule")
            yield Static("", markup=False, id="catalog-columns")
            yield LoadingStatus(id="load-status")
            yield KeyboardOptionList(id="chooser")
            yield LoadingStatus(id="catalog-alert")
            yield Static(
                "↑↓ MOVE    ENTER CONNECT    CTRL+C QUIT",
                markup=False,
                id="catalog-hints",
            )
        with Container(id="specs"):
            with Horizontal(id="specs-heading"):
                yield Static("", markup=False, id="specs-identity")
                yield Static("· ·", markup=False, id="specs-eyes", classes="brand-eyes")
            yield Rule(line_style="solid", id="specs-rule")
            with SpecsScroll(id="specs-scroll"):
                yield Static("", markup=False, id="specs-body")
            yield Static(
                "↑↓ SCROLL    ←→ MODEL    ESC REGISTRY    CTRL+C QUIT",
                markup=False,
                id="specs-hints",
            )
        with Container(id="chat"):
            with Horizontal(id="chat-heading"):
                yield Static("", markup=False, id="identity")
                yield Static("· ·", markup=False, id="chat-eyes", classes="brand-eyes")
            yield Rule(line_style="solid", id="identity-rule")
            with VerticalScroll(id="transcript-scroll"):
                yield Transcript(id="transcript")
            yield CommandMenu(id="command-menu")
            yield LoadingStatus(id="status")
            yield LoadingStatus(id="chat-alert")
            with Horizontal(id="composer-bar"):
                yield Static(">", markup=False, id="composer-prompt")
                yield Composer(id="composer", language=None)
            yield Static("", markup=False, id="footer")

    def on_mount(self) -> None:
        """Populate the chooser or open a direct selection."""
        specs_scroll = self.query_one("#specs-scroll", VerticalScroll)
        cast(Any, specs_scroll.vertical_scrollbar).renderer = HalfCellScrollBarRender
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
        if self._alert_timer is not None:
            self._alert_timer.stop()
            self._alert_timer = None
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
        elif self.query_one("#specs").display:
            self.call_after_refresh(self._render_specs)

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
            self._trace_menu_id = row.id
            self._trace_rename_open = False
            self._start_trace_blink()
            self._refresh_catalog_prompts(event.key)
            self.push_screen(
                TraceMenuModal(row.title),
                lambda choice: self._after_trace_action(row, choice),
            )

    def on_keyboard_option_list_specs_requested(
        self,
        event: KeyboardOptionList.SpecsRequested,
    ) -> None:
        """Open model details for one available registry row."""
        if self._loading or self._trace_menu_id is not None:
            return
        row = self._rows.get(event.key)
        if isinstance(row, DiscoveryItem):
            if row.available and row.assistant is not None:
                self._show_specs(event.key)
        elif isinstance(row, SavedChat):
            self._show_specs(event.key)

    def on_keyboard_option_list_theme_requested(
        self,
        event: KeyboardOptionList.ThemeRequested,
    ) -> None:
        """Cycle the registry through the configured ANSI accent themes."""
        del event
        self._accent_theme_index = (self._accent_theme_index + 1) % len(
            _ACCENT_THEMES
        )
        name, color = _ACCENT_THEMES[self._accent_theme_index]
        for theme_name, _ in _ACCENT_THEMES:
            self.remove_class(f"-accent-{theme_name}")
        self.add_class(f"-accent-{name}")
        self.query_one("#watermark", Static).update(_watermark(color))
        self.query_one("#transcript", Transcript).set_accent(color)
        if self._selected_catalog_key is not None:
            self._refresh_catalog_prompts(self._selected_catalog_key)

    def on_specs_scroll_cycle_requested(
        self,
        event: SpecsScroll.CycleRequested,
    ) -> None:
        """Show the adjacent available registry entry in SPECS."""
        self._cycle_specs(event.offset)

    def _after_trace_action(self, trace: SavedChat, choice: str | None) -> None:
        if choice == "rename":
            self._trace_rename_open = True
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
            self._show_error("Chat output is not configured")
            return
        try:
            self.store.erase(trace.id)
        except ChatStorageError as error:
            self._show_error(str(error))
            return
        self._fill_chooser()

    def _after_trace_rename(self, trace: SavedChat, title: str | None) -> None:
        self._close_trace_menu()
        if title is None:
            return
        if self.store is None:
            self._show_error("Chat output is not configured")
            return
        try:
            self.store.rename(
                trace.id,
                title if title.strip() else trace.title,
            )
        except ChatStorageError as error:
            self._show_error(str(error))
            return
        self._fill_chooser()

    def _close_trace_menu(self) -> None:
        """Close TRACE controls and stop subject emphasis."""
        selected_key = self._selected_catalog_key
        if self._trace_blink_timer is not None:
            self._trace_blink_timer.stop()
            self._trace_blink_timer = None
        self._trace_blink_visible = True
        self._trace_menu_id = None
        self._trace_rename_open = False
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
        if self._trace_menu_id is None or self._selected_catalog_key is None:
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
        """Show matching commands for one slash prefix."""
        value = event.text_area.text
        if value != self._dismissed_command_text:
            self._dismissed_command_text = None
        self._command_matches = (
            () if value == self._dismissed_command_text else completions(value)
        )
        self._command_index = 0
        self._update_command_menu()

    def on_composer_command_moved(self, event: Composer.CommandMoved) -> None:
        """Move the command highlight with wrapping navigation."""
        enabled = self._enabled_command_indexes()
        if not enabled:
            return
        current = (
            enabled.index(self._command_index)
            if self._command_index in enabled
            else 0
        )
        self._command_index = enabled[(current + event.offset) % len(enabled)]
        self._update_command_menu()

    def on_composer_command_dismissed(self, event: Composer.CommandDismissed) -> None:
        """Hide the command menu until the composer value changes."""
        self._dismissed_command_text = self.query_one(Composer).text
        self._command_matches = ()
        self._update_command_menu()

    def on_composer_command_completed(self, event: Composer.CommandCompleted) -> None:
        """Complete the highlighted command with trailing space."""
        selected = self._selected_command()
        if selected is None:
            return
        composer = self.query_one(Composer)
        composer.clear()
        composer.insert(f"{selected} ")

    def on_composer_submitted(self, event: Composer.Submitted) -> None:
        """Run one command or start one user turn."""
        value = event.value
        if not value.strip():
            return
        if self._loading:
            self._show_error("CONNECTION is not ready")
            return
        composer = self.query_one(Composer)
        try:
            command = parse_command(self._selected_command() or value)
            if command is not None:
                composer.clear()
                self._command(command.name)
                return
            if self._generating:
                return
            session = self._session()
            session.add_user(value)
            composer.clear()
            self._render_transcript()
            self._start_generation(0)
        except (
            ChatError,
            ChatStorageError,
            CommandError,
            ValueError,
            WorkerError,
        ) as error:
            self._show_error(str(error))

    def action_stop(self) -> None:
        """Stop active generation at a token boundary."""
        if self.query_one("#specs").display:
            if self._specs_from_chat:
                self._restore_chat_from_specs()
            else:
                self._show_registry()
        elif self._generating:
            self.runtime.cancel()

    def action_interrupt(self) -> None:
        """Stop active work or open the idle quit confirmation."""
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

    def _start_generation(self, attempt: int) -> None:
        self._generating = True
        self._streaming_text = ""
        self._set_status("generating")
        self._update_command_menu()
        self._generate_thread(attempt)

    @work(thread=True, exclusive=True, group="generation")
    def _generate_thread(self, attempt: int) -> None:
        last_render = 0.0
        try:
            for delta in self.runtime.generate(attempt, self._report_prefill):
                if not self._streaming_text:
                    self.call_from_thread(self._set_status, "generating")
                self._streaming_text += delta
                now = time.monotonic()
                if now - last_render >= 1 / 12:
                    self.call_from_thread(self._render_transcript, True)
                    last_render = now
            self.call_from_thread(self._generation_done)
        except Exception as error:  # noqa: BLE001
            self.call_from_thread(self._generation_failed, str(error))

    def _generation_done(self) -> None:
        self._generating = False
        self._streaming_text = ""
        self._render_transcript()
        self._update_status()
        self._update_footer()
        self._update_command_menu()

    def _generation_failed(self, message: str) -> None:
        self._generating = False
        self._set_status(None)
        self._update_command_menu()
        self._show_error(f"{message}. Return to REGISTRY to start again.")

    def _report_prefill(self, current: int, total: int) -> None:
        self.call_from_thread(self._set_status, f"prefill {current}/{total}")

    def _show_chat(self) -> None:
        session = self._session()
        self.chat_policy = session.chat
        self.generation = session.generation
        self.query_one("#landing").display = False
        self.query_one("#chat").display = True
        self.query_one("#identity", Static).update(session.assistant.name)
        self._render_transcript()
        self._update_status()
        self._update_footer()
        self.query_one(Composer).focus()

    def _render_transcript(self, partial: bool = False) -> None:
        transcript = self.query_one("#transcript", Transcript)
        scroller = self.query_one("#transcript-scroll", VerticalScroll)
        follow_latest = scroller.scroll_y >= scroller.max_scroll_y - 1
        session = self._session()
        turns = []
        for turn in session.turns:
            name = "USER" if turn.role == "user" else self._chat_name(session)
            turns.append((name, turn.content))
        if partial and self._generating:
            turns.append((self._chat_name(session), self._streaming_text or "…"))
        transcript.set_turns(turns)
        if follow_latest:
            self.call_after_refresh(self._scroll_transcript_end)

    def _scroll_transcript_end(self) -> None:
        self.query_one("#transcript-scroll", VerticalScroll).scroll_end(animate=False)

    @staticmethod
    def _chat_name(session: ChatSession) -> str:
        return session.assistant.target_name.upper()

    def _update_footer(self) -> None:
        if self._confirmation_open:
            self._set_footer(
                "ENTER SAVE    ESC RESUME"
                if self._trace_name_open
                else "←→ MOVE    ENTER SELECT    ESC RESUME"
            )
            return
        if self._command_matches:
            self._set_footer(
                "↑↓ MOVE    TAB COMPLETE    ENTER SELECT    ESC CLOSE"
            )
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
        self._set_status(self.runtime.state.value)

    def _set_status(self, value: str | None) -> None:
        normalized = (
            "" if value is None or value.casefold() == "ready" else value.upper()
        )
        if normalized != self._status_text:
            self.query_one("#status", LoadingStatus).set_state(
                normalized,
                animated=normalized not in {"CANCELLED", "FAILED"},
            )
            self._status_text = normalized

    def _command(self, name: str) -> None:
        if name == "exit":
            self.action_interrupt()
        elif name == "registry":
            if self._generating:
                self._show_error("Stop the active reply before returning to REGISTRY")
            else:
                self._return_to_landing()
        elif name == "specs":
            self._show_chat_specs()
        elif name == "save":
            session = self._session()
            if not session.dirty:
                self._show_error("The TRACE has no new data to save")
            elif self.store is None:
                self._show_error("Chat output is not configured")
            elif self._generating:
                self._show_error("Stop the active reply before saving")
            else:
                self._confirmation_open = True
                self._trace_name_open = True
                self._update_confirmation_spacing()
                self._update_footer()
                self.push_screen(
                    TraceNameModal(default_chat_title(session)),
                    self._after_checkpoint_name,
                )

    def _after_checkpoint_name(self, title: str | None) -> None:
        self._close_confirmation()
        if title is not None:
            self._save_from_confirmation(title)

    def _save_enabled(self) -> bool:
        session = self.runtime.session
        return (
            self.store is not None
            and session is not None
            and session.dirty
            and not self._generating
        )

    def _enabled_command_indexes(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, command in enumerate(self._command_matches)
            if (command != "/registry" or not self._generating)
            and (command != "/save" or self._save_enabled())
        )

    def _selected_command(self) -> str | None:
        if (
            not self._command_matches
            or self._command_index not in self._enabled_command_indexes()
        ):
            return None
        return self._command_matches[self._command_index]

    def _update_command_menu(self) -> None:
        enabled = self._enabled_command_indexes()
        if enabled and self._command_index not in enabled:
            self._command_index = enabled[0]
        selected = self._selected_command()
        menu = self.query_one("#command-menu", CommandMenu)
        was_visible = menu.display
        menu.set_commands(
            self._command_matches,
            selected,
            registry_enabled=not self._generating,
            save_enabled=self._save_enabled(),
        )
        composer = self.query_one(Composer)
        composer.command_menu_active = bool(self._command_matches)
        composer.command_menu_escape_enabled = not self._generating
        self._update_footer()
        if was_visible or self._command_matches:
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
        self._confirmation_open = True
        self._trace_name_open = False
        self._update_confirmation_spacing()
        self._update_footer()
        self.push_screen(ConfirmModal(), callback)

    def _push_trace_name(self, callback: Callable[[str | None], None]) -> None:
        self._trace_name_open = True
        self._update_footer()
        self.push_screen(TraceNameModal(default_chat_title(self._session())), callback)

    def _close_confirmation(self) -> None:
        self._confirmation_open = False
        self._trace_name_open = False
        self._update_confirmation_spacing()
        self._update_footer()

    def _update_confirmation_spacing(self) -> None:
        scroller = self.query_one("#transcript-scroll", VerticalScroll)
        bottom = 3 if self._confirmation_open else 1
        scroller.styles.padding = (1, 2, bottom, 2)
        self.call_after_refresh(self._scroll_transcript_end)

    def _save_from_confirmation(self, title: str) -> bool:
        if self.store is None:
            self._show_error("Chat output is not configured")
            return False
        try:
            saved = self.store.save(
                self._session(),
                title if title.strip() else None,
                self.runtime.backend_versions,
            )
        except ChatStorageError as error:
            self._show_error(str(error))
            return False
        self._active_trace = saved
        self._show_alert(f"Saved {saved.id[:8]} — {saved.title}")
        return True

    def _begin_select(self, assistant: Assistant) -> None:
        if not self._prepare_load():
            return
        self._active_trace = None
        self._set_loading(True)
        self._select_thread(assistant)

    def _begin_attach(
        self,
        session: ChatSession,
        *,
        trace: SavedChat | None = None,
    ) -> None:
        if not self._prepare_load():
            return
        self._active_trace = trace
        self.runtime.session = session
        self._set_loading(True)
        self._show_chat()
        self._attach_thread(session)

    def _prepare_load(self) -> bool:
        try:
            self.runtime.ensure_worker()
        except Exception as error:  # noqa: BLE001
            self._load_failed(str(error))
            return False
        return True

    @work(thread=True, exclusive=True, group="model-load")
    def _select_thread(self, assistant: Assistant) -> None:
        try:
            self.runtime.select(assistant)
        except Exception as error:  # noqa: BLE001
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
            self.call_from_thread(self._load_failed, str(error))
            return
        self.call_from_thread(self._load_done)

    def _load_done(self) -> None:
        self._set_loading(False)
        self._show_chat()

    def _load_failed(self, message: str) -> None:
        self._set_loading(False)
        if self.runtime.session is not None:
            self._show_chat()
        self._set_status(None)
        self._show_error(message)

    def _show_error(self, message: str) -> None:
        """Show one right-aligned failure beside the active controls."""
        self._show_alert(message, error=True)

    def _show_alert(self, message: str, *, error: bool = False) -> None:
        """Show one transient message beside the active controls."""
        if self._alert_timer is not None:
            self._alert_timer.stop()
        identifier = (
            "#chat-alert" if self.query_one("#chat").display else "#catalog-alert"
        )
        other = "#catalog-alert" if identifier == "#chat-alert" else "#chat-alert"
        other_alert = self.query_one(other, LoadingStatus)
        other_alert.set_state("")
        other_alert.remove_class("-error")
        value = message.strip()
        if value and not value.endswith((".", "!", "?")):
            value += "."
        alert = self.query_one(identifier, LoadingStatus)
        alert.set_class(error, "-error")
        alert.set_state(value, animated=False)
        self._alert_timer = self.set_timer(5, self._clear_alert)

    def _clear_alert(self) -> None:
        """Clear the transient message line."""
        for identifier in ("#chat-alert", "#catalog-alert"):
            alert = self.query_one(identifier, LoadingStatus)
            alert.set_state("")
            alert.remove_class("-error")
        self._alert_timer = None

    def _set_loading(self, value: bool) -> None:
        self._loading = value
        self.query_one("#chooser", OptionList).disabled = value
        self._refresh_loading_state()

    def _refresh_loading_state(self) -> None:
        if not self.is_mounted or len(self.query("#load-status")) == 0:
            return
        state = self.runtime.state.value
        status = state.upper() if self._loading else ""
        if status != self._load_status_text:
            self.query_one("#load-status", LoadingStatus).set_state(status)
            self._load_status_text = status
        if self._loading and self.query_one("#chat").display:
            self._set_status(state)

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
            layout.line("MODEL", "TYPE", "ENTRY")
        )
        for index, row in enumerate(self.assistants):
            if row.available and row.assistant is not None:
                assistant = row.assistant
                if assistant.run is None:
                    kind = "BASE"
                else:
                    kind = "CONSTRUCT"
                text = layout.text(
                    row.label,
                    kind,
                    "READY",
                )
            else:
                text = layout.text(
                    row.label,
                    "—",
                    "FAULT",
                    failed=True,
                )
            key = f"assistant-{index}"
            self._rows[key] = row
            options.append(Option(text, id=key, disabled=not row.available))
        for saved in saved_chats:
            text = layout.text(
                saved.assistant.name,
                "TRACE",
                "READY",
                trace_name=(
                    saved.title if self._trace_details_expanded else None
                ),
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
        self._specs_from_chat = False
        self._specs_key = key
        self.query_one("#landing").display = False
        self.query_one("#chat").display = False
        self.query_one("#specs").display = True
        self.query_one("#specs-hints", Static).update(
            "↑↓ SCROLL    ←→ MODEL    ESC REGISTRY    CTRL+C QUIT"
        )
        self._render_specs()
        self.query_one("#specs-scroll", VerticalScroll).focus()

    def _show_chat_specs(self) -> None:
        """Show model details as a temporary view over the active chat."""
        self._specs_from_chat = True
        self._specs_key = None
        self.query_one("#landing").display = False
        self.query_one("#chat").display = False
        self.query_one("#specs").display = True
        self.query_one("#specs-hints", Static).update(
            "↑↓ SCROLL    ESC CHAT    CTRL+C QUIT"
        )
        self._render_specs()
        specs_scroll = self.query_one("#specs-scroll", SpecsScroll)
        specs_scroll.scroll_to(y=0, animate=False)
        specs_scroll.focus()

    def _restore_chat_from_specs(self) -> None:
        """Return from temporary model details without changing chat state."""
        self.query_one("#specs").display = False
        self.query_one("#chat").display = True
        self._specs_from_chat = False
        self.query_one(Composer).focus()

    def _show_registry(self) -> None:
        """Return from model details to the unchanged registry selection."""
        self.query_one("#specs").display = False
        self.query_one("#landing").display = True
        self._specs_from_chat = False
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
        self._render_specs()
        self.query_one("#specs-scroll", SpecsScroll).scroll_to(y=0, animate=False)

    def _render_specs(self) -> None:
        """Render the selected model details at the current terminal width."""
        if self._specs_from_chat:
            session = self._session()
            assistant = session.assistant
            generation = session.generation
            trace = self._active_trace
            kind = (
                "TRACE"
                if trace is not None
                else "BASE"
                if assistant.run is None
                else "CONSTRUCT"
            )
        else:
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
                return
        self.query_one("#specs-identity", Static).update(assistant.name)
        body = render_specs(
            assistant,
            generation,
            kind,
            max(24, self.size.width - 4),
            trace,
        )
        self.query_one("#specs-body", Static).update(body)

    def _refresh_catalog_prompts(self, selected_key: str) -> None:
        self._selected_catalog_key = selected_key
        chooser = self.query_one("#chooser", OptionList)
        layout = CatalogLayout.for_terminal(self.size.width)
        for option in chooser.options:
            if option.id is None:
                continue
            row = self._rows.get(option.id)
            if isinstance(row, DiscoveryItem):
                if row.available and row.assistant is not None:
                    assistant = row.assistant
                    kind = "BASE" if assistant.run is None else "CONSTRUCT"
                    prompt = layout.text(
                        row.label,
                        kind,
                        "READY",
                        selected=option.id == selected_key,
                    )
                else:
                    prompt = layout.text(row.label, "—", "FAULT", failed=True)
            elif isinstance(row, SavedChat):
                prompt = layout.text(
                    row.assistant.name,
                    "TRACE",
                    "READY",
                    selected=option.id == selected_key,
                    trace_name=(
                        None
                        if not self._trace_details_expanded
                        and row.id != self._trace_menu_id
                        else row.title
                    ),
                    trace_active=row.id == self._trace_menu_id,
                    trace_visible=(
                        self._trace_blink_visible
                        if row.id == self._trace_menu_id
                        else True
                    ),
                )
            else:
                continue
            chooser.replace_option_prompt(option.id, prompt)
        self._update_catalog_hints()

    def _update_catalog_hints(self) -> None:
        """Show actions available for the current registry row."""
        if self._trace_menu_id is not None:
            self.query_one("#catalog-hints", Static).update(
                "ENTER NAME    ESC RETAIN"
                if self._trace_rename_open
                else "←→ MOVE    ENTER SELECT    ESC RETAIN"
            )
            return
        has_traces = any(isinstance(row, SavedChat) for row in self._rows.values())
        selected_trace = isinstance(
            self._rows.get(self._selected_catalog_key or ""),
            SavedChat,
        )
        fields = ["↑↓ MOVE", "ENTER CONNECT", "S SPECS"]
        if has_traces:
            fields.append("SPACE DETAILS")
        if selected_trace:
            fields.append("BACKSPACE MANAGE")
        fields.append("CTRL+C QUIT")
        available = max(20, self.size.width - 4)
        gap = "    " if available >= 60 else "  "
        for removable in ("CTRL+C QUIT", "SPACE DETAILS", "↑↓ MOVE"):
            if len(gap.join(fields)) <= available:
                break
            if removable in fields:
                fields.remove(removable)
        self.query_one("#catalog-hints", Static).update(gap.join(fields))

    def _session(self) -> ChatSession:
        if self.runtime.session is None:
            raise ChatError("Select an assistant first")
        return self.runtime.session
