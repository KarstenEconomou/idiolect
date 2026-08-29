"""Define reusable menu models for the terminal interface."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

from rich.cells import cell_len
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, HorizontalScroll, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

MenuVariant = Literal["default", "muted"]
MenuAnchor = Literal["active", "chat", "registry"]


@dataclass(frozen=True, slots=True)
class MenuItem:
    """Declare one menu item."""

    identity: str
    label: str
    description: str = ""
    disabled: bool = False
    variant: MenuVariant = "default"


class MenuCursor:
    """Keep menu selection and a three-row viewport."""

    def __init__(
        self,
        items: Sequence[MenuItem],
        selected: str | None = None,
        *,
        viewport_size: int = 3,
    ) -> None:
        """Select the requested enabled item or the first enabled item."""
        self.items = tuple(items)
        self.viewport_size = viewport_size
        enabled = self.enabled_indexes
        requested = next(
            (
                index
                for index, item in enumerate(self.items)
                if item.identity == selected and not item.disabled
            ),
            None,
        )
        self.index = (
            requested if requested is not None else (enabled[0] if enabled else None)
        )

    @property
    def enabled_indexes(self) -> tuple[int, ...]:
        """Return indexes for all enabled items."""
        return tuple(
            index for index, item in enumerate(self.items) if not item.disabled
        )

    @property
    def selected(self) -> MenuItem | None:
        """Return the selected item."""
        return None if self.index is None else self.items[self.index]

    def move(self, offset: int) -> MenuItem | None:
        """Move with wrapping and skip disabled items."""
        enabled = self.enabled_indexes
        if not enabled:
            self.index = None
            return None
        current = enabled.index(self.index) if self.index in enabled else 0
        self.index = enabled[(current + offset) % len(enabled)]
        return self.selected

    def retain(self, items: Sequence[MenuItem]) -> MenuItem | None:
        """Replace items and retain the selected identity when possible."""
        identity = None if self.selected is None else self.selected.identity
        replacement = MenuCursor(
            items,
            identity,
            viewport_size=self.viewport_size,
        )
        self.items = replacement.items
        self.index = replacement.index
        return self.selected

    def viewport(self) -> tuple[MenuItem, ...]:
        """Return the selected three-row viewport."""
        if not self.items:
            return ()
        selected = 0 if self.index is None else self.index
        start = min(
            max(selected - self.viewport_size + 1, 0),
            max(len(self.items) - self.viewport_size, 0),
        )
        return self.items[start : start + self.viewport_size]


class VerticalMenu(Static):
    """Render a heading and reusable two-column menu rows."""

    def __init__(
        self,
        heading: str,
        *,
        id: str | None = None,
        row_prefix: str = "menu",
        name_class: str = "menu-name",
        description_class: str = "menu-description",
    ) -> None:
        """Set semantic row names and the menu heading."""
        super().__init__(id=id)
        self.heading = heading
        self.row_prefix = row_prefix
        self.name_class = name_class
        self.description_class = description_class
        self._items: tuple[MenuItem, ...] = ()
        self._selected: str | None = None

    def compose(self) -> ComposeResult:
        """Create the heading and three reusable rows."""
        yield Static(self.heading, markup=False, classes="menu-heading")
        with Vertical(classes="menu-actions"):
            for index in range(3):
                with Horizontal(
                    id=f"{self.row_prefix}-{index}",
                    classes="menu-action",
                ):
                    yield Static(
                        "", markup=False, classes=f"menu-name {self.name_class}"
                    )
                    yield Static(
                        "",
                        markup=False,
                        classes=f"menu-description {self.description_class}",
                    )

    def set_items(
        self,
        items: Sequence[MenuItem],
        selected: str | None,
    ) -> None:
        """Render up to three declared items."""
        self._items = tuple(items)
        self._selected = selected
        self.display = bool(items)
        for index in range(3):
            row = self.query_one(f"#{self.row_prefix}-{index}", Horizontal)
            visible = index < len(items)
            row.display = visible
            row.set_class(visible and items[index].identity == selected, "-selected")
            if not visible:
                continue
            item = items[index]
            row.set_class(item.disabled, "-disabled")
            row.set_class(item.variant == "muted", "-muted")
            row.query_one(".menu-name", Static).update(item.label)
            row.query_one(".menu-description", Static).update(item.description)


class MenuButton(Button):
    """Block pointer activation for one modal menu action."""

    FOCUS_ON_CLICK: ClassVar[bool] = False

    def on_mount(self) -> None:
        """Apply optional flush label alignment after default button styles."""
        if self.has_class("-flush-label"):
            self.styles.line_pad = 0
            self.styles.text_align = "left"
            self.styles.content_align = ("left", "middle")
        if self.has_class("-flush-end"):
            self.styles.width = cell_len(str(self.label))

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        event.prevent_default()
        event.stop()

    async def _on_mouse_up(self, event: events.MouseUp) -> None:
        event.prevent_default()
        event.stop()

    async def _on_click(self, event: events.Click) -> None:
        event.prevent_default()
        event.stop()


class KeyboardHorizontalScroll(HorizontalScroll):
    """Allow horizontal menu scrolling only through keyboard focus."""

    def on_resize(self, _event: events.Resize) -> None:
        """Keep the focused action visible when the viewport changes."""
        self.call_after_refresh(self._reveal_focused)

    def _reveal_focused(self) -> None:
        """Reveal the focused menu action without animation."""
        focused = next(
            (button for button in self.query(MenuButton) if button.has_focus),
            None,
        )
        if focused is not None:
            self.scroll_to_widget(focused, animate=False, immediate=True)

    def _block_pointer_scroll(self, event: events.MouseEvent) -> None:
        event.prevent_default()
        event.stop()

    _on_mouse_scroll_down = _block_pointer_scroll
    _on_mouse_scroll_up = _block_pointer_scroll
    _on_mouse_scroll_right = _block_pointer_scroll
    _on_mouse_scroll_left = _block_pointer_scroll


class HorizontalMenuModal(ModalScreen[str | None]):
    """Show a keyboard-only horizontal menu at a declared anchor."""

    def __init__(
        self,
        title: str,
        items: Sequence[MenuItem],
        initial: str,
        escape_result: str | None,
        *,
        anchor: MenuAnchor,
        on_highlight: Callable[[str], None] | None = None,
        dialog_id: str = "horizontal-menu-dialog",
        message_id: str = "horizontal-menu-message",
        actions_id: str = "horizontal-menu-actions",
        button_prefix: str = "",
        hidden_until_placed: bool = False,
        flush_items: bool = False,
    ) -> None:
        """Keep the menu declaration and placement policy."""
        super().__init__()
        self.menu_title = title
        self.items = tuple(items)
        self.cursor = MenuCursor(self.items, initial, viewport_size=len(self.items))
        self.escape_result = escape_result
        self.menu_anchor = anchor
        self.on_highlight = on_highlight
        self.dialog_id = dialog_id
        self.message_id = message_id
        self.actions_id = actions_id
        self.button_prefix = button_prefix
        self.hidden_until_placed = hidden_until_placed
        self.flush_items = flush_items

    def compose(self) -> ComposeResult:
        """Create the heading and horizontal actions."""
        classes = (
            "horizontal-menu -unplaced"
            if self.hidden_until_placed
            else "horizontal-menu"
        )
        with Vertical(id=self.dialog_id, classes=classes):
            yield Static(
                self.menu_title,
                markup=False,
                id=self.message_id,
                classes="menu-heading",
            )
            with KeyboardHorizontalScroll(
                id=self.actions_id,
                classes="horizontal-menu-actions",
            ):
                for index, item in enumerate(self.items):
                    button = MenuButton(
                        item.label,
                        id=f"{self.button_prefix}{item.identity}",
                    )
                    if self.flush_items:
                        button.add_class("-flush-label")
                        if index == len(self.items) - 1:
                            button.add_class("-flush-end")
                    yield button

    def on_mount(self) -> None:
        """Place the menu and focus its initial action."""
        self.call_after_refresh(self._show_dialog)

    def _show_dialog(self) -> None:
        self._place_dialog()
        self.query_one(f"#{self.dialog_id}", Vertical).remove_class("-unplaced")
        self._focus_selected()

    def _focus_selected(self) -> None:
        """Focus and reveal the selected action without animation."""
        selected = self.cursor.selected
        if selected is not None:
            button = self.query_one(
                f"#{self.button_prefix}{selected.identity}",
                MenuButton,
            )
            button.focus()
            self.query_one(
                f"#{self.actions_id}",
                KeyboardHorizontalScroll,
            ).scroll_to_widget(
                button,
                animate=False,
                immediate=True,
            )

    def on_resize(self, _event: events.Resize) -> None:
        """Place the menu again after a resize."""
        self.call_after_refresh(self._resize_dialog)

    def _resize_dialog(self) -> None:
        """Place the resized menu and reveal its selected action."""
        self._place_dialog()
        self.call_after_refresh(self._focus_selected)

    def _place_dialog(self) -> None:
        """Place the menu above chat controls or registry hints."""
        dialog = self.query_one(f"#{self.dialog_id}", Vertical)
        height = dialog.region.height or 2
        if self.menu_anchor == "registry" or (
            self.menu_anchor == "active" and self.app.query_one("#landing").display
        ):
            hints = self.app.query_one("#catalog-hints", Static)
            dialog.styles.width = hints.region.width - 2
            dialog.styles.offset = (hints.region.x + 1, hints.region.y - height)
            return
        composer = self.app.query_one("#composer-bar", Horizontal)
        command = self.app.query_one("#command-bar", Static)
        reference = self.app.query_one("#reference-bar", Static)
        active = command if command.display else reference
        dialog.styles.width = composer.region.width
        dialog.styles.offset = (
            composer.region.x,
            (active.region.y if active.display else composer.region.y) - height,
        )

    def on_key(self, event: events.Key) -> None:
        """Move focus, accept an action, or return the Escape result."""
        if event.key == "escape":
            self.dismiss(self.escape_result)
            event.prevent_default()
            event.stop()
            return
        if event.key in {"up", "down"}:
            event.prevent_default()
            event.stop()
            return
        if event.key not in {"left", "right"}:
            return
        selected = self.cursor.move(-1 if event.key == "left" else 1)
        if selected is not None:
            self._focus_selected()
            if self.on_highlight is not None:
                self.on_highlight(selected.identity)
        event.prevent_default()
        event.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Return the selected item identity."""
        identity = (event.button.id or "").removeprefix(self.button_prefix)
        self.dismiss(identity)
