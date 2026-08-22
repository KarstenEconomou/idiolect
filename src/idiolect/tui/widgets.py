"""Define widgets for the local chat interface."""

from typing import ClassVar

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Label, OptionList, Static, TextArea


class KeyboardOptionList(OptionList):
    """Select options with arrow keys and Enter."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "select", "Select", show=False),
    ]

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

    class Submitted(Message):
        """Report one submitted composer value."""

        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    async def _on_key(self, event: events.Key) -> None:
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


class InfoModal(ModalScreen[None]):
    """Show command help or chat statistics."""

    def __init__(self, title: str, body: str) -> None:
        """Set the dialog title and body."""
        super().__init__()
        self.title_value = title
        self.body = body

    def compose(self) -> ComposeResult:
        """Create the information dialog."""
        with Vertical(id="info-dialog"):
            yield Label(self.title_value, id="info-title")
            yield Static(self.body, markup=False, id="info-body")
            yield KeyboardButton("Close", id="close")

    def on_mount(self) -> None:
        """Focus the close action."""
        self.query_one(KeyboardButton).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Close the information dialog."""
        self.dismiss()


class ConfirmModal(ModalScreen[str]):
    """Ask how to handle unsaved transcript changes."""

    def compose(self) -> ComposeResult:
        """Create the confirmation dialog."""
        with Vertical(id="confirm-dialog"):
            yield Static("unsaved changes", markup=False, id="confirm-message")
            with Horizontal(id="confirm-actions"):
                yield KeyboardButton("save", id="save")
                yield KeyboardButton("discard", id="discard")
                yield KeyboardButton("cancel", id="cancel")

    def on_mount(self) -> None:
        """Focus the save action."""
        self.query_one(KeyboardButton).focus()
        self.call_after_refresh(self._place_dialog)

    def on_resize(self) -> None:
        """Place the dialog next to the composer."""
        self.call_after_refresh(self._place_dialog)

    def _place_dialog(self) -> None:
        composer = self.app.query_one(Composer)
        dialog = self.query_one("#confirm-dialog", Vertical)
        dialog.styles.width = composer.region.width
        dialog.styles.offset = (
            composer.region.x,
            composer.region.y - dialog.region.height,
        )

    def on_key(self, event: events.Key) -> None:
        """Move focus between confirmation actions."""
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
