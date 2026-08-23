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
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.geometry import Region
from textual.message import Message
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Button, Input, OptionList, Static, TextArea

from idiolect.tui.commands import COMMAND_DESCRIPTIONS, COMMANDS
from idiolect.tui.markdown import ChatMarkdown


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

    class ThemeRequested(Message):
        """Request the next interface accent theme."""

    async def _on_key(self, event: events.Key) -> None:
        if event.key.lower() == "t":
            event.prevent_default()
            event.stop()
            self.post_message(self.ThemeRequested())
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


class SpecsScroll(VerticalScroll):
    """Scroll specifications and request adjacent registry entries."""

    class CycleRequested(Message):
        """Request the next or previous specification sheet."""

        def __init__(self, offset: int) -> None:
            """Set the registry movement direction."""
            self.offset = offset
            super().__init__()

    async def _on_key(self, event: events.Key) -> None:
        if event.key in {"left", "right"}:
            event.prevent_default()
            event.stop()
            self.post_message(
                self.CycleRequested(-1 if event.key == "left" else 1)
            )
            return
        await super()._on_key(event)


class KeyboardButton(Button):
    """Activate a modal action with the keyboard."""

    async def _on_click(self, event: events.Click) -> None:
        event.prevent_default()
        event.stop()


class Composer(TextArea):
    """Submit text or insert an explicit line break."""

    command_menu_active = False
    command_menu_escape_enabled = False
    reference_menu_active = False
    reference_menu_escape_enabled = False
    reference_selected = False
    command_selected = False

    class Submitted(Message):
        """Report one submitted composer value."""

        def __init__(self, value: str) -> None:
            self.value = value
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
        """Handle Escape when TextArea key bindings run first."""
        if (
            (self.command_selected or self.reference_selected)
            and not self.command_menu_active
            and not self.reference_menu_active
            and event.key == "escape"
        ):
            event.prevent_default()
            event.stop()
            self.post_message(
                self.CommandEscaped()
                if self.command_selected
                else self.ReferenceEscaped()
            )

    async def _on_key(self, event: events.Key) -> None:
        if self.command_menu_active and event.key in {"up", "down"}:
            event.prevent_default()
            event.stop()
            offset = -1 if event.key == "up" else 1
            self.post_message(self.CommandMoved(offset))
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
            return
        if self.reference_menu_active and event.key in {"up", "down"}:
            event.prevent_default()
            event.stop()
            offset = -1 if event.key == "up" else 1
            self.post_message(self.ReferenceMoved(offset))
            return
        if (
            self.reference_menu_active
            and self.reference_menu_escape_enabled
            and event.key == "escape"
        ):
            event.prevent_default()
            event.stop()
            self.post_message(self.ReferenceDismissed())
            return
        if self.reference_menu_active and event.key == "enter":
            event.prevent_default()
            event.stop()
            self.post_message(self.ReferenceSelected())
            return
        if (
            (self.command_selected or self.reference_selected)
            and not self.command_menu_active
            and not self.reference_menu_active
            and event.key == "escape"
        ):
            event.prevent_default()
            event.stop()
            self.post_message(
                self.CommandEscaped()
                if self.command_selected
                else self.ReferenceEscaped()
            )
            return
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.post_message(self.Submitted(self.text))
            return
        if event.key in {"shift+enter", "alt+enter"}:
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        await super()._on_key(event)


class CommandMenu(Widget):
    """Show keyboard-controlled commands above the composer."""

    def compose(self) -> ComposeResult:
        """Create the command heading and actions."""
        yield Static("COMMAND", markup=False, id="command-message")
        with Vertical(id="command-actions"):
            for command in COMMANDS:
                with Horizontal(
                    id=f"command-{command[1:]}",
                    classes="command-action",
                ):
                    yield Static(
                        command,
                        markup=False,
                        classes="command-name",
                    )
                    yield Static(
                        COMMAND_DESCRIPTIONS[command],
                        markup=False,
                        classes="command-description",
                    )

    def set_commands(
        self,
        commands: tuple[str, ...],
        selected: str | None,
        *,
        registry_enabled: bool,
        save_enabled: bool,
    ) -> None:
        """Set visible commands and the highlighted command."""
        self.display = bool(commands)
        selected_index = commands.index(selected) if selected in commands else 0
        start = min(max(selected_index - 2, 0), max(len(commands) - 3, 0))
        preview = commands[start : start + 3]
        for command in COMMANDS:
            action = self.query_one(f"#command-{command[1:]}", Horizontal)
            action.display = command in preview
            disabled = (
                (command == "/registry" and not registry_enabled)
                or (command == "/save" and not save_enabled)
            )
            action.set_class(command == selected and not disabled, "-selected")
            action.set_class(disabled, "-disabled")
            action.set_class(
                command == "/save" and not save_enabled,
                "-save-disabled",
            )


class CommandBar(Static):
    """Render one selected argument command above the composer."""

    _accent = "green"
    _command: tuple[str, str] | None = None

    def set_accent(self, color: str) -> None:
        """Set the accent used by the dimmed command border."""
        if color != self._accent:
            self._accent = color
            if self._command is None:
                self.refresh()
            else:
                self.set_command(*self._command)

    def set_command(self, name: str, description: str) -> None:
        """Show one selected command and its description."""
        self._command = (name, description)
        value = Text.assemble(
            (f"/ {name.upper()}", f"dim {self._accent}"),
            (f" {description}", "bright_black"),
        )
        self.update(value)

    def render_lines(self, crop: Region) -> list[Strip]:
        """Dim the command border while keeping its roles explicit."""
        strips = super().render_lines(crop)
        if not strips:
            return strips
        dim = Style(color=self._accent, dim=True)

        def border_style(strip: Strip) -> Strip:
            """Apply the dimmed accent over Textual's default border style."""
            return Strip(
                [
                    Segment(segment.text, dim, segment.control)
                    for segment in strip
                ],
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


class ReferenceMenu(Widget):
    """Show keyboard-controlled chat bubble references."""

    def compose(self) -> ComposeResult:
        """Create the reference heading and up to three rows."""
        yield Static("REF", markup=False, id="reference-message")
        with Vertical(id="reference-actions"):
            for index in range(3):
                with Horizontal(
                    id=f"reference-{index}",
                    classes="reference-action",
                ):
                    yield Static("", markup=False, classes="reference-name")
                    yield Static("", markup=False, classes="reference-preview")

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


class ReferenceBar(Static):
    """Render the selected bubble above the composer."""

    _accent = "green"
    _reference: tuple[str, int, str] | None = None

    def set_accent(self, color: str) -> None:
        """Set the accent used by the dimmed reference border."""
        if color != self._accent:
            self._accent = color
            if self._reference is None:
                self.refresh()
            else:
                self.set_reference(*self._reference)

    def set_reference(self, name: str, index: int, preview: str) -> None:
        """Show one selected reference with its fixed ANSI roles."""
        self._reference = (name, index, preview)
        value = Text.assemble(
            ("@ ", f"dim {self._accent}"),
            (f"{name}:{index:02d}", f"dim {self._accent}"),
            (f" {preview}", "bright_black"),
        )
        self.update(value)

    def render_lines(self, crop: Region) -> list[Strip]:
        """Dim the accent border while keeping the reference content bright."""
        strips = super().render_lines(crop)
        if not strips:
            return strips
        dim = Style(color=self._accent, dim=True)

        def border_style(strip: Strip) -> Strip:
            """Apply the border style over Textual's default foreground."""
            return Strip(
                [
                    Segment(segment.text, dim, segment.control)
                    for segment in strip
                ],
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
                Strip.join(
                    (border_style(left), middle, border_style(right))
                )
            )
        return rendered


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
            (turn[0], turn[1], turn[2] if len(turn) == 3 else False)
            for turn in turns
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
            style = (
                Style(color=self._accent, dim=True)
                if dimmed
                else self._accent
            )
            return Text(f"{name}:", style=style)
        speaker = name[:marker]
        reference = name[marker + 2 : -1]
        return Text.assemble(
            (f"{speaker}:", self._accent),
            (f"\n REF {reference}", f"dim {self._accent}"),
        )


class _Dimmed:
    """Apply metadata color and dim styling to one Rich renderable."""

    def __init__(self, renderable: object) -> None:
        """Keep the original renderable for delegated rendering."""
        self.renderable = renderable

    def __rich_console__(self, console, options):
        """Render child segments with the metadata style."""
        metadata = Style(color="bright_black", dim=True)
        for segment in console.render(self.renderable, options):
            style = metadata if segment.style is None else segment.style + metadata
            yield Segment(segment.text, style, segment.control)


class LoadingStatus(Widget):
    """Show a status with an optional activity spinner."""

    def __init__(self, *, id: str | None = None) -> None:
        """Create a hidden loading state."""
        super().__init__(id=id)
        self.state = ""
        self._animated = True
        self._spinner = Spinner("dots")

    def on_mount(self) -> None:
        """Set the initial spinner refresh state."""
        self.auto_refresh = 1 / 12 if self.state and self._animated else None

    def set_state(self, value: str, *, animated: bool = True) -> None:
        """Set the visible status and its animation state."""
        self.state = value
        self._animated = animated
        self.display = bool(value)
        self.auto_refresh = 1 / 12 if value and animated else None
        self.refresh()

    def render(self) -> RenderResult:
        """Render the spinner before the loading state."""
        if not self._animated:
            return Text(self.state)
        frame = (
            Text("·")
            if self.app.animation_level == "none"
            else cast(Text, self._spinner.render(time.monotonic()))
        )
        return Text.assemble(frame, " ", self.state)


class ConfirmModal(ModalScreen[str]):
    """Ask how to handle unsaved transcript changes."""

    def compose(self) -> ComposeResult:
        """Create the confirmation dialog."""
        with Vertical(id="confirm-dialog"):
            yield Static("CONNECTION", markup=False, id="confirm-message")
            with Horizontal(id="confirm-actions"):
                yield KeyboardButton("DISCONNECT", id="discard")
                yield KeyboardButton("SAVE", id="save")
                yield KeyboardButton("RESUME", id="cancel")

    def on_mount(self) -> None:
        """Focus the save action."""
        self.query_one(KeyboardButton).focus()
        self.call_after_refresh(self._place_dialog)

    def on_resize(self) -> None:
        """Place the dialog next to the composer."""
        self.call_after_refresh(self._place_dialog)

    def _place_dialog(self) -> None:
        composer_bar = self.app.query_one("#composer-bar", Horizontal)
        command_bar = self.app.query_one("#command-bar", CommandBar)
        reference_bar = self.app.query_one("#reference-bar", ReferenceBar)
        anchor = command_bar if command_bar.display else reference_bar
        dialog = self.query_one("#confirm-dialog", Vertical)
        dialog.styles.width = composer_bar.region.width
        dialog.styles.offset = (
            composer_bar.region.x,
            (anchor.region.y if anchor.display else composer_bar.region.y)
            - dialog.region.height,
        )

    def on_key(self, event: events.Key) -> None:
        """Move focus between confirmation actions."""
        if event.key == "escape":
            self.dismiss("cancel")
            event.prevent_default()
            event.stop()
            return
        if event.key not in {"left", "right", "up", "down"}:
            return
        buttons = list(self.query(KeyboardButton))
        focused = self.focused
        index = buttons.index(focused) if isinstance(focused, KeyboardButton) else 0
        step = -1 if event.key in {"left", "up"} else 1
        buttons[(index + step) % len(buttons)].focus()
        event.prevent_default()
        event.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Return the selected confirmation action."""
        self.dismiss(event.button.id or "cancel")


class TraceNameModal(ModalScreen[str | None]):
    """Request an optional name before one trace is recorded."""

    def __init__(self, default_name: str, *, registry: bool = False) -> None:
        """Set the generated trace name shown in the empty field."""
        super().__init__()
        self.default_name = default_name
        self.registry = registry

    def compose(self) -> ComposeResult:
        """Create the trace name input."""
        with Vertical(id="trace-name-dialog"):
            yield Static("TRACE NAME", markup=False, id="trace-name-message")
            yield Input(placeholder=self.default_name, id="trace-name")

    def on_mount(self) -> None:
        """Focus the trace name and place the dialog."""
        self.query_one(Input).focus()
        self.call_after_refresh(self._place_dialog)

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
        dialog.styles.width = anchor.region.width - (2 * inset)
        dialog.styles.offset = (
            anchor.region.x + inset,
            anchor.region.y - dialog.region.height,
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


class TraceMenuModal(ModalScreen[str]):
    """Show actions for one saved trace."""

    def __init__(self, trace_name: str) -> None:
        """Set the trace name shown in the menu heading."""
        super().__init__()
        self.trace_name = trace_name

    def compose(self) -> ComposeResult:
        """Create the trace actions."""
        with Vertical(id="trace-dialog"):
            heading = Text("TRACE", style="bold white")
            heading.append(f" {self.trace_name}", style="bright_black")
            yield Static(heading, markup=False, id="trace-message")
            with Horizontal(id="trace-actions"):
                yield KeyboardButton("ERASE", id="erase")
                yield KeyboardButton("RENAME", id="rename")
                yield KeyboardButton("RETAIN", id="retain")

    def on_mount(self) -> None:
        """Focus the safe action and place the dialog."""
        self.query_one("#retain", KeyboardButton).focus()
        self.call_after_refresh(self._place_dialog)

    def on_resize(self) -> None:
        """Place the dialog above the registry hints."""
        self.call_after_refresh(self._place_dialog)

    def _place_dialog(self) -> None:
        hints = self.app.query_one("#catalog-hints", Static)
        dialog = self.query_one("#trace-dialog", Vertical)
        dialog.styles.width = hints.region.width - 2
        dialog.styles.offset = (
            hints.region.x + 1,
            hints.region.y - dialog.region.height,
        )

    def on_key(self, event: events.Key) -> None:
        """Move focus or keep the trace when Escape is pressed."""
        if event.key == "escape":
            self.dismiss("retain")
            event.prevent_default()
            event.stop()
            return
        if event.key not in {"left", "right", "up", "down"}:
            return
        buttons = list(self.query(KeyboardButton))
        focused = self.focused
        index = buttons.index(focused) if isinstance(focused, KeyboardButton) else 1
        step = -1 if event.key in {"left", "up"} else 1
        buttons[(index + step) % len(buttons)].focus()
        event.prevent_default()
        event.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Return the selected trace action."""
        self.dismiss(event.button.id or "retain")
