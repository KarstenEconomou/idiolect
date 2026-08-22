"""Define widgets for the local chat interface."""

import time
from collections.abc import Callable, Sequence
from typing import ClassVar, cast

from rich.console import Group
from rich.padding import Padding
from rich.spinner import Spinner
from rich.text import Text
from textual import events
from textual.app import ComposeResult, RenderResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
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
        """Request a details toggle for one highlighted option."""

        def __init__(self, key: str) -> None:
            """Set the highlighted option key."""
            self.key = key
            super().__init__()

    async def _on_key(self, event: events.Key) -> None:
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


class KeyboardButton(Button):
    """Activate a modal action with the keyboard."""

    async def _on_click(self, event: events.Click) -> None:
        event.prevent_default()
        event.stop()


class Composer(TextArea):
    """Submit text or insert an explicit line break."""

    command_menu_active = False
    command_menu_escape_enabled = False

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

    class CommandCompleted(Message):
        """Request completion of the highlighted command."""

    async def _on_key(self, event: events.Key) -> None:
        if self.command_menu_active and event.key in {"left", "right", "up", "down"}:
            event.prevent_default()
            event.stop()
            offset = -1 if event.key in {"left", "up"} else 1
            self.post_message(self.CommandMoved(offset))
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
        if self.command_menu_active and event.key == "tab":
            event.prevent_default()
            event.stop()
            self.post_message(self.CommandCompleted())
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
    ) -> None:
        """Set visible commands and the highlighted command."""
        self.display = bool(commands)
        selected_index = commands.index(selected) if selected in commands else 0
        start = min(max(selected_index - 2, 0), max(len(commands) - 3, 0))
        preview = commands[start : start + 3]
        for command in COMMANDS:
            action = self.query_one(f"#command-{command[1:]}", Horizontal)
            action.display = command in preview
            action.set_class(command == selected, "-selected")
            action.set_class(
                command == "/registry" and not registry_enabled,
                "-disabled",
            )


class Transcript(Static):
    """Render transcript labels and focused Markdown message blocks."""

    plain = ""
    _cached_turns: tuple[tuple[str, str], ...] = ()
    _cached_messages: tuple[ChatMarkdown, ...] = ()

    def set_turns(self, turns: Sequence[tuple[str, str]]) -> None:
        """Set labeled turns without changing their stored message text."""
        current = tuple(turns)
        messages = tuple(
            self._cached_messages[index]
            if index < len(self._cached_turns) and turn == self._cached_turns[index]
            else ChatMarkdown(turn[1])
            for index, turn in enumerate(current)
        )
        renderables: list[Text | Padding] = []
        plain_blocks = []
        for index, ((name, message), rendered) in enumerate(zip(current, messages)):
            if index:
                renderables.append(Text(""))
            renderables.append(Text(f"{name}:", style="blue"))
            renderables.append(Padding(rendered, (0, 0, 0, 1)))
            displayed = message.replace("\n", "\n ")
            plain_blocks.append(f"{name}:\n {displayed}")
        self._cached_turns = current
        self._cached_messages = messages
        self.plain = "\n\n".join(plain_blocks)
        self.update(Group(*renderables))


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
                yield KeyboardButton("RECORD", id="save")
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
        dialog = self.query_one("#confirm-dialog", Vertical)
        dialog.styles.width = composer_bar.region.width
        dialog.styles.offset = (
            composer_bar.region.x,
            composer_bar.region.y - dialog.region.height,
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

    def compose(self) -> ComposeResult:
        """Create the trace name input."""
        with Vertical(id="trace-name-dialog"):
            yield Static("TRACE NAME", markup=False, id="trace-name-message")
            yield Input(placeholder="Blank uses default", id="trace-name")

    def on_mount(self) -> None:
        """Focus the trace name and place the dialog."""
        self.query_one(Input).focus()
        self.call_after_refresh(self._place_dialog)

    def on_resize(self) -> None:
        """Place the dialog next to the composer."""
        self.call_after_refresh(self._place_dialog)

    def _place_dialog(self) -> None:
        composer_bar = self.app.query_one("#composer-bar", Horizontal)
        dialog = self.query_one("#trace-name-dialog", Vertical)
        dialog.styles.width = composer_bar.region.width
        dialog.styles.offset = (
            composer_bar.region.x,
            composer_bar.region.y - dialog.region.height,
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
