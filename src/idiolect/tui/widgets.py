"""Define widgets for the local chat interface."""

import time
from collections.abc import Callable, Sequence
from typing import ClassVar, cast

from rich.console import Group
from rich.padding import Padding
from rich.segment import Segment
from rich.spinner import Spinner
from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import ComposeResult, RenderResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.geometry import Region
from textual.message import Message
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Input, OptionList, Static, TextArea

from idiolect.tui.commands import COMMAND_DESCRIPTIONS, COMMANDS
from idiolect.tui.markdown import ChatMarkdown
from idiolect.tui.menus import (
    HorizontalMenuModal,
    MenuButton,
    MenuCursor,
    MenuItem,
    VerticalMenu,
)
from idiolect.tui.sheets import SheetScroll

SpecsScroll = SheetScroll


class KeyboardOptionList(OptionList):
    """Select options with arrow keys and Enter."""

    selection_changed: Callable[[str], None] | None = None

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "select", "Select", show=False),
    ]

    class DetailsToggled(Message):
        """Request a registry-wide details toggle."""

        def __init__(self, key: str) -> None:
            """Set the highlighted option key."""
            self.key = key
            super().__init__()

    class EraseRequested(Message):
        """Request erasure for one highlighted option."""

        def __init__(self, key: str) -> None:
            """Set the highlighted option key."""
            self.key = key
            super().__init__()

    class SpecsRequested(Message):
        """Request model details for one highlighted option."""

        def __init__(self, key: str) -> None:
            """Set the highlighted option key."""
            self.key = key
            super().__init__()

    class ChromaRequested(Message):
        """Request the interface accent theme menu."""

    async def _on_key(self, event: events.Key) -> None:
        if event.key.lower() == "c":
            event.prevent_default()
            event.stop()
            self.post_message(self.ChromaRequested())
            return
        if event.key.lower() == "s" and self.highlighted is not None:
            option = self.get_option_at_index(self.highlighted)
            if option.id is not None and not option.disabled:
                event.prevent_default()
                event.stop()
                self.post_message(self.SpecsRequested(option.id))
                return
        if event.key == "backspace" and self.highlighted is not None:
            option = self.get_option_at_index(self.highlighted)
            if option.id is not None:
                event.prevent_default()
                event.stop()
                self.post_message(self.EraseRequested(option.id))
                return
        if event.key == "space" and self.highlighted is not None:
            option = self.get_option_at_index(self.highlighted)
            if option.id is not None:
                event.prevent_default()
                event.stop()
                self.post_message(self.DetailsToggled(option.id))
                return
        await super()._on_key(event)

    def watch_highlighted(self, highlighted: int | None) -> None:
        """Refresh selection-dependent prompts before the next screen render."""
        if highlighted is not None and highlighted < len(self.options):
            option = self.get_option_at_index(highlighted)
            if (
                not option.disabled
                and option.id is not None
                and self.selection_changed is not None
            ):
                self.selection_changed(option.id)
        super().watch_highlighted(highlighted)

    async def _on_click(self, event: events.Click) -> None:
        event.prevent_default()
        event.stop()

    def _on_mouse_move(self, event: events.MouseMove) -> None:
        event.stop()


KeyboardButton = MenuButton


class Composer(TextArea):
    """Submit text, insert a line break, or recall submitted text."""

    command_menu_active = False
    command_menu_escape_enabled = False
    reference_menu_active = False
    reference_menu_escape_enabled = False
    reference_selected = False
    command_selected = False
    control_active = False
    _reported_height: int | None = None
    _history: tuple[str, ...] = ()
    _history_index: int | None = None

    def set_history(self, values: Sequence[str]) -> None:
        """Replace history with values from the active chat."""
        self._history = tuple(values)
        self._history_index = None

    def record_submission(self, value: str) -> None:
        """Add one accepted value to the composer history."""
        self._history = (*self._history, value)
        self._history_index = None

    def _navigate_history(self, offset: int) -> bool:
        """Recall one history value without wrapping at either end."""
        if not self._history or (self._history_index is None and self.text):
            return False
        if offset < 0:
            if self._history_index is None:
                self._history_index = len(self._history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
        else:
            if self._history_index is None:
                return False
            if self._history_index < len(self._history) - 1:
                self._history_index += 1
            else:
                self._history_index = None
        value = "" if self._history_index is None else self._history[self._history_index]
        self.load_text(value)
        lines = value.split("\n")
        self.move_cursor((len(lines) - 1, len(lines[-1])))
        return True

    class Resized(Message):
        """Report a composer height change."""

        def __init__(self, height_delta: int) -> None:
            """Set the signed height difference."""
            self.height_delta = height_delta
            super().__init__()

    class Submitted(Message):
        """Report one submitted composer value."""

        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    def on_resize(self, event: events.Resize) -> None:
        """Report composer height changes after layout."""
        previous = self._reported_height
        self._reported_height = event.size.height
        if previous is not None and previous != event.size.height:
            self.post_message(self.Resized(event.size.height - previous))

    class SelectorMoved(Message):
        """Report generic selector movement."""

        def __init__(self, selector: str, offset: int) -> None:
            self.selector = selector
            self.offset = offset
            super().__init__()

    class SelectorAccepted(Message):
        """Report generic selector acceptance."""

        def __init__(self, selector: str) -> None:
            self.selector = selector
            super().__init__()

    class SelectorDismissed(Message):
        """Report generic selector dismissal."""

        def __init__(self, selector: str) -> None:
            self.selector = selector
            super().__init__()

    class AccessoryEscaped(Message):
        """Report removal of one selected composer accessory."""

        def __init__(self, selector: str) -> None:
            self.selector = selector
            super().__init__()

    class ControlRequested(Message):
        """Request the composer CONTROL sheet."""

    class ControlDismissed(Message):
        """Request dismissal of the composer CONTROL sheet."""

        def __init__(self, *, cancel_generation: bool = False) -> None:
            """Set whether dismissal also cancels active generation."""
            self.cancel_generation = cancel_generation
            super().__init__()

    class CommandMoved(Message):
        """Request a new highlighted command."""

        def __init__(self, offset: int) -> None:
            """Set the command selection offset."""
            self.offset = offset
            super().__init__()

    class CommandDismissed(Message):
        """Request that the visible command menu closes."""

    class CommandEscaped(Message):
        """Request removal of the selected command."""

    class ReferenceMoved(Message):
        """Request a new highlighted reference."""

        def __init__(self, offset: int) -> None:
            """Set the reference selection offset."""
            self.offset = offset
            super().__init__()

    class ReferenceDismissed(Message):
        """Request that the visible reference menu closes."""

    class ReferenceSelected(Message):
        """Request selection of the highlighted reference."""

    class ReferenceEscaped(Message):
        """Request removal of the selected reference."""

    def on_key(self, event: events.Key) -> None:
        """Handle mode dismissal when TextArea key bindings run first."""
        if (
            (self.command_selected or self.reference_selected)
            and not self.command_menu_active
            and not self.reference_menu_active
            and (
                event.key == "escape"
                or (event.key == "backspace" and not self.text)
            )
        ):
            event.prevent_default()
            event.stop()
            self.post_message(
                self.CommandEscaped()
                if self.command_selected
                else self.ReferenceEscaped()
            )
            self.post_message(
                self.AccessoryEscaped(
                    "command" if self.command_selected else "reference"
                )
            )

    async def _on_key(self, event: events.Key) -> None:
        if (
            not self.control_active
            and event.character == "?"
            and not self.text
            and self._history_index is None
            and not self.command_menu_active
            and not self.reference_menu_active
            and not self.command_selected
            and not self.reference_selected
        ):
            event.prevent_default()
            event.stop()
            self.control_active = True
            self.post_message(self.ControlRequested())
            return
        if self.control_active and event.key in {"escape", "backspace"}:
            event.prevent_default()
            event.stop()
            self.control_active = False
            self.post_message(self.ControlDismissed(cancel_generation=True))
            return
        if self.control_active and event.key in {"shift+enter", "alt+enter"}:
            event.prevent_default()
            event.stop()
            self.control_active = False
            self.post_message(self.ControlDismissed())
            self.insert("\n")
            return
        if self.control_active and event.character is not None:
            self.control_active = False
            self.post_message(self.ControlDismissed())
        if self.command_menu_active and event.key in {"up", "down"}:
            event.prevent_default()
            event.stop()
            offset = -1 if event.key == "up" else 1
            self.post_message(self.CommandMoved(offset))
            self.post_message(self.SelectorMoved("command", offset))
            return
        if self.command_menu_active and event.key == "tab":
            event.prevent_default()
            event.stop()
            return
        if (
            self.command_menu_active
            and self.command_menu_escape_enabled
            and event.key == "escape"
        ):
            event.prevent_default()
            event.stop()
            self.post_message(self.CommandDismissed())
            self.post_message(self.SelectorDismissed("command"))
            return
        if self.reference_menu_active and event.key in {"up", "down"}:
            event.prevent_default()
            event.stop()
            offset = -1 if event.key == "up" else 1
            self.post_message(self.ReferenceMoved(offset))
            self.post_message(self.SelectorMoved("reference", offset))
            return
        if (
            self.reference_menu_active
            and self.reference_menu_escape_enabled
            and event.key == "escape"
        ):
            event.prevent_default()
            event.stop()
            self.post_message(self.ReferenceDismissed())
            self.post_message(self.SelectorDismissed("reference"))
            return
        if self.reference_menu_active and event.key == "enter":
            event.prevent_default()
            event.stop()
            self.post_message(self.ReferenceSelected())
            self.post_message(self.SelectorAccepted("reference"))
            return
        if (
            (self.command_selected or self.reference_selected)
            and not self.command_menu_active
            and not self.reference_menu_active
            and (
                event.key == "escape"
                or (event.key == "backspace" and not self.text)
            )
        ):
            event.prevent_default()
            event.stop()
            self.post_message(
                self.CommandEscaped()
                if self.command_selected
                else self.ReferenceEscaped()
            )
            self.post_message(
                self.AccessoryEscaped(
                    "command" if self.command_selected else "reference"
                )
            )
            return
        if event.key in {"up", "down"}:
            line, _column = self.cursor_location
            boundary = (
                line == 0
                if event.key == "up"
                else line == self.document.line_count - 1
            )
            if boundary and self._navigate_history(-1 if event.key == "up" else 1):
                event.prevent_default()
                event.stop()
                if self.control_active and self.text:
                    self.control_active = False
                    self.post_message(self.ControlDismissed())
                return
        elif self._history_index is not None:
            self._history_index = None
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            if self.command_menu_active:
                self.post_message(self.SelectorAccepted("command"))
            self.post_message(self.Submitted(self.text))
            return
        if event.key in {"shift+enter", "alt+enter"}:
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        await super()._on_key(event)


class ControlSheet(Static):
    """Show the static composer interaction grammar."""

    ROWS: ClassVar[tuple[tuple[str, str] | None, ...]] = (
        ("/", "COMMAND"),
        ("@", "REFERENCE"),
        ("?", "CONTROL"),
        None,
        ("↩", "TRANSMIT"),
        ("⇧↩", "NEWLINE"),
        ("↑↓", "HISTORY"),
        ("⎋", "CANCEL"),
    )

    def compose(self) -> ComposeResult:
        """Create the heading and all CONTROL rows."""
        yield Static("CONTROL", markup=False, classes="menu-heading")
        with Vertical(classes="control-actions"):
            for row in self.ROWS:
                if row is None:
                    yield Static("", markup=False, classes="control-gap")
                    continue
                key, description = row
                with Horizontal(classes="control-action"):
                    yield Static(key, markup=False, classes="control-key")
                    yield Static(
                        description,
                        markup=False,
                        classes="control-description",
                    )


class CommandMenu(VerticalMenu):
    """Show keyboard-controlled commands above the composer."""

    def __init__(self, *, id: str | None = None) -> None:
        """Create the declarative command menu."""
        super().__init__("COMMAND", id=id)

    def compose(self) -> ComposeResult:
        """Create the command heading and actions."""
        yield Static(
            "COMMAND",
            markup=False,
            id="command-message",
            classes="menu-heading",
        )
        with Vertical(id="command-actions", classes="menu-actions"):
            for command in COMMANDS:
                with Horizontal(
                    id=f"command-{command[1:]}",
                    classes="command-action menu-action",
                ):
                    yield Static(
                        command,
                        markup=False,
                        classes="command-name menu-name",
                    )
                    yield Static(
                        COMMAND_DESCRIPTIONS[command],
                        markup=False,
                        classes="command-description menu-description",
                    )

    def set_commands(
        self,
        commands: tuple[str, ...],
        selected: str | None,
        *,
        registry_enabled: bool,
        retry_enabled: bool,
        trace_enabled: bool,
    ) -> None:
        """Set visible commands and the highlighted command."""
        items = tuple(
            MenuItem(
                command,
                command,
                COMMAND_DESCRIPTIONS[command],
                disabled=(
                    (command == "/disconnect" and not registry_enabled)
                    or (command == "/retry" and not retry_enabled)
                    or (command == "/trace" and not trace_enabled)
                ),
            )
            for command in commands
        )
        cursor = MenuCursor(items, selected)
        preview = tuple(item.identity for item in cursor.viewport())
        self.display = bool(commands)
        for command in COMMANDS:
            action = self.query_one(f"#command-{command[1:]}", Horizontal)
            action.display = command in preview
            disabled = (
                (command == "/disconnect" and not registry_enabled)
                or (command == "/retry" and not retry_enabled)
                or (command == "/trace" and not trace_enabled)
            )
            action.set_class(command == selected and not disabled, "-selected")
            action.set_class(disabled, "-disabled")


class SelectionBar(Static):
    """Own accent state and border rendering for a selection bar."""

    _accent = "green"

    def set_accent(self, color: str) -> None:
        """Set the accent used by the dimmed command border."""
        if color != self._accent:
            self._accent = color
            self.refresh_content()

    def refresh_content(self) -> None:
        """Refresh content after the accent changes."""
        self.refresh()

    def render_lines(self, crop: Region) -> list[Strip]:
        """Dim the command border while keeping its roles explicit."""
        strips = super().render_lines(crop)
        if not strips:
            return strips
        dim = Style(color=self._accent, dim=True)

        def border_style(strip: Strip) -> Strip:
            """Apply the dimmed accent over Textual's default border style."""
            return Strip(
                [Segment(segment.text, dim, segment.control) for segment in strip],
                strip.cell_length,
            )

        height = self.region.height
        rendered: list[Strip] = []
        for offset, strip in enumerate(strips):
            row = crop.y + offset
            if row in {0, height - 1} or strip.cell_length <= 2:
                rendered.append(border_style(strip))
                continue
            left, middle, right = strip.divide(
                [1, strip.cell_length - 1, strip.cell_length]
            )
            rendered.append(
                Strip.join((border_style(left), middle, border_style(right)))
            )
        return rendered


class CommandBar(SelectionBar):
    """Render one selected argument command above the composer."""

    _command: tuple[str, str] | None = None

    def refresh_content(self) -> None:
        """Refresh the selected command with the current accent."""
        if self._command is None:
            self.refresh()
        else:
            self.set_command(*self._command)

    def set_command(self, name: str, description: str) -> None:
        """Show one selected command and its description."""
        self._command = (name, description)
        self.update(
            Text.assemble(
                (name.upper(), f"dim {self._accent}"),
                (f" {description}", "bright_black"),
            )
        )


class ReferenceMenu(VerticalMenu):
    """Show keyboard-controlled chat bubble references."""

    def __init__(self, *, id: str | None = None) -> None:
        """Create the declarative reference menu."""
        super().__init__("REF", id=id)

    def compose(self) -> ComposeResult:
        """Create the reference heading and up to three rows."""
        yield Static(
            "REF",
            markup=False,
            id="reference-message",
            classes="menu-heading",
        )
        with Vertical(id="reference-actions", classes="menu-actions"):
            for index in range(3):
                with Horizontal(
                    id=f"reference-{index}",
                    classes="reference-action menu-action",
                ):
                    yield Static(
                        "",
                        markup=False,
                        classes="reference-name menu-name",
                    )
                    yield Static(
                        "",
                        markup=False,
                        classes="reference-preview menu-description",
                    )

    def set_references(
        self,
        rows: tuple[tuple[str, str], ...],
        selected: int | None,
    ) -> None:
        """Set the visible reference rows and highlighted row."""
        self.display = bool(rows)
        for index in range(3):
            action = self.query_one(f"#reference-{index}", Horizontal)
            visible = index < len(rows)
            action.display = visible
            action.set_class(visible and index == selected, "-selected")
            if not visible:
                continue
            name, preview = rows[index]
            action.query_one(".reference-name", Static).update(name)
            action.query_one(".reference-preview", Static).update(preview)


class ReferenceBar(SelectionBar):
    """Render the selected bubble above the composer."""

    _reference: tuple[str, int, str] | None = None

    def refresh_content(self) -> None:
        """Refresh the selected reference with the current accent."""
        if self._reference is None:
            self.refresh()
        else:
            self.set_reference(*self._reference)

    def set_reference(self, name: str, index: int, preview: str) -> None:
        """Show one selected reference with its fixed ANSI roles."""
        self._reference = (name, index, preview)
        value = Text.assemble(
            (f"{name}:{index:02d}", f"dim {self._accent}"),
            (f" {preview}", "bright_black"),
        )
        self.update(value)


class Transcript(Static):
    """Render transcript labels and focused Markdown message blocks."""

    plain = ""
    _cached_turns: tuple[tuple[str, str, bool], ...] = ()
    _cached_messages: tuple[ChatMarkdown, ...] = ()
    _accent = "green"

    def set_accent(self, color: str) -> None:
        """Set the accent used by transcript speaker labels."""
        if color != self._accent:
            self._accent = color
            self.set_turns(self._cached_turns)

    def set_turns(
        self,
        turns: Sequence[tuple[str, str] | tuple[str, str, bool]],
    ) -> None:
        """Set labeled turns without changing their stored message text."""
        current = tuple(
            (turn[0], turn[1], turn[2] if len(turn) == 3 else False) for turn in turns
        )
        messages = tuple(
            self._cached_messages[index]
            if index < len(self._cached_turns) and turn == self._cached_turns[index]
            else ChatMarkdown(turn[1])
            for index, turn in enumerate(current)
        )
        renderables: list[Text | Padding] = []
        plain_blocks = []
        for index, ((name, message, dimmed), rendered) in enumerate(
            zip(current, messages)
        ):
            if index:
                renderables.append(Text(""))
            renderables.append(self._speaker_label(name, dimmed=dimmed))
            body = _Dimmed(rendered) if dimmed else rendered
            renderables.append(Padding(body, (0, 0, 0, 1)))
            displayed = message.replace("\n", "\n ")
            plain_blocks.append(self._plain_block(name, displayed))
        self._cached_turns = current
        self._cached_messages = messages
        self.plain = "\n\n".join(plain_blocks)
        self.update(Group(*renderables))

    @staticmethod
    def _plain_block(name: str, displayed: str) -> str:
        """Return one transcript block in its plain-text layout."""
        marker = name.find(" [@")
        if marker < 0:
            return f"{name}:\n {displayed}"
        speaker = name[:marker]
        reference = name[marker + 2 : -1]
        return f"{speaker}:\n REF {reference}\n {displayed}"

    def _speaker_label(self, name: str, *, dimmed: bool = False) -> Text:
        """Render a speaker label with a dim reference annotation."""
        marker = name.find(" [@")
        if marker < 0:
            style = Style(color=self._accent, dim=True) if dimmed else self._accent
            return Text(f"{name}:", style=style)
        speaker = name[:marker]
        reference = name[marker + 2 : -1]
        return Text.assemble(
            (f"{speaker}:", self._accent),
            (f"\n REF {reference}", f"dim {self._accent}"),
        )


class _Dimmed:
    """Apply the footer color to one environment renderable."""

    def __init__(self, renderable: object) -> None:
        """Keep the original renderable for delegated rendering."""
        self.renderable = renderable

    def __rich_console__(self, console, options):
        """Render child segments with the metadata style."""
        metadata = Style(color="bright_black")
        for segment in console.render(self.renderable, options):
            style = metadata if segment.style is None else segment.style + metadata
            yield Segment(segment.text, style, segment.control)


class StatusLine(Widget):
    """Show activity or one static notice on a status line."""

    def __init__(self, *, id: str | None = None) -> None:
        """Create a hidden loading state."""
        super().__init__(id=id)
        self.state = ""
        self._content: Text | None = None
        self._animated = True
        self._spinner = Spinner("dots")

    def on_mount(self) -> None:
        """Set the initial spinner refresh state."""
        self.auto_refresh = 1 / 12 if self.state and self._animated else None

    def set_state(self, value: str, *, animated: bool = True) -> None:
        """Set the visible status and its animation state."""
        self.state = value
        self._content = None
        self._animated = animated
        self.display = bool(value)
        self.auto_refresh = 1 / 12 if value and animated else None
        self.refresh()

    def set_content(self, content: Text, state: str) -> None:
        """Set one styled, non-animated status message."""
        self.state = state
        self._content = content
        self._animated = False
        self.display = bool(state)
        self.auto_refresh = None
        self.refresh()

    def render(self) -> RenderResult:
        """Render the spinner before the loading state."""
        if self._content is not None:
            return self._content
        if not self._animated:
            return Text(self.state)
        frame = (
            Text("·")
            if self.app.animation_level == "none"
            else cast(Text, self._spinner.render(time.monotonic()))
        )
        return Text.assemble(frame, " ", self.state)


class ConfirmModal(HorizontalMenuModal):
    """Ask how to handle unsaved transcript changes."""

    def __init__(self) -> None:
        """Declare dirty-link actions with the safe Escape result."""
        super().__init__(
            "LINK DIRTY",
            (
                MenuItem("discard", "DISCONNECT"),
                MenuItem("save", "TRACE"),
                MenuItem("cancel", "RESUME"),
            ),
            "discard",
            "cancel",
            anchor="chat",
            dialog_id="confirm-dialog",
            message_id="confirm-message",
            actions_id="confirm-actions",
        )


class TraceNameModal(ModalScreen[str | None]):
    """Request an optional name before one trace is recorded."""

    def __init__(self, default_name: str, *, registry: bool = False) -> None:
        """Set the generated trace name shown in the empty field."""
        super().__init__()
        self.default_name = default_name
        self.registry = registry

    def compose(self) -> ComposeResult:
        """Create the trace name input."""
        with Vertical(id="trace-name-dialog", classes="-unplaced"):
            yield Static("TRACE NAME", markup=False, id="trace-name-message")
            yield Input(placeholder=self.default_name, id="trace-name")

    def on_mount(self) -> None:
        """Place the hidden trace name dialog before it receives focus."""
        self.call_after_refresh(self._show_dialog)

    def _show_dialog(self) -> None:
        """Place the trace name field before it becomes visible."""
        self._place_dialog()
        self.query_one("#trace-name-dialog", Vertical).remove_class("-unplaced")
        self.query_one(Input).focus()

    def on_resize(self) -> None:
        """Place the dialog next to the composer."""
        self.call_after_refresh(self._place_dialog)

    def _place_dialog(self) -> None:
        if self.registry:
            anchor = self.app.query_one("#catalog-hints")
        else:
            composer_bar = self.app.query_one("#composer-bar", Horizontal)
            command_bar = self.app.query_one("#command-bar", CommandBar)
            reference_bar = self.app.query_one("#reference-bar", ReferenceBar)
            anchor = (
                command_bar
                if command_bar.display
                else reference_bar
                if reference_bar.display
                else composer_bar
            )
        dialog = self.query_one("#trace-name-dialog", Vertical)
        inset = 1 if self.registry else 0
        height = dialog.region.height or 2
        dialog.styles.width = anchor.region.width - (2 * inset)
        dialog.styles.offset = (
            anchor.region.x + inset,
            anchor.region.y - height,
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Return the entered name, including an empty default request."""
        self.dismiss(event.value)

    def on_key(self, event: events.Key) -> None:
        """Resume the chat when naming is cancelled."""
        if event.key == "escape":
            self.dismiss(None)
            event.prevent_default()
            event.stop()


class TraceMenuModal(HorizontalMenuModal):
    """Show actions for one saved trace."""

    def __init__(self) -> None:
        """Declare TRACE actions with RETAIN as the safe selection."""
        super().__init__(
            "TRACE MANAGE",
            (
                MenuItem("erase", "ERASE"),
                MenuItem("rename", "RENAME"),
                MenuItem("retain", "RETAIN"),
            ),
            "retain",
            "retain",
            anchor="registry",
            dialog_id="trace-dialog",
            message_id="trace-message",
            actions_id="trace-actions",
            hidden_until_placed=True,
        )


class ChromaMenuModal(HorizontalMenuModal):
    """Preview and select one interface accent theme."""

    def __init__(
        self,
        themes: Sequence[tuple[str, str]],
        current: str,
        on_change: Callable[[str], None],
    ) -> None:
        """Set the theme names and the live preview callback."""
        super().__init__(
            "CHROMA",
            tuple(MenuItem(name, label) for name, label in themes),
            current,
            None,
            anchor="active",
            on_highlight=on_change,
            dialog_id="chroma-dialog",
            message_id="chroma-message",
            actions_id="chroma-actions",
            button_prefix="chroma-",
            hidden_until_placed=True,
            flush_items=True,
        )
