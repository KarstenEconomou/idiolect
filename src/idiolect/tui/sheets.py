"""Define reusable information sheets for the terminal interface."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Rule, Static

from idiolect.tui.specs import SpecsDocument

SheetOrigin = Literal["chat", "registry"]
SheetRenderer = Callable[[], SpecsDocument]


@dataclass(frozen=True, slots=True)
class SheetPage:
    """Declare one information sheet page."""

    title: str
    renderer: SheetRenderer
    origin: SheetOrigin
    link: str | None
    hints: str
    cycle: bool = False
    connect: bool = False


class SheetScroll(VerticalScroll):
    """Scroll a sheet and report declared page actions."""

    class CycleRequested(Message):
        """Request the next or previous sheet page."""

        def __init__(self, offset: int) -> None:
            self.offset = offset
            super().__init__()

    class ConnectRequested(Message):
        """Request the current sheet connection."""

    async def _on_key(self, event: events.Key) -> None:
        page = self.app.query_one(InfoSheet).page
        if event.key == "enter" and page is not None and page.connect:
            event.prevent_default()
            event.stop()
            self.post_message(self.ConnectRequested())
            return
        if event.key in {"left", "right"} and page is not None and page.cycle:
            event.prevent_default()
            event.stop()
            self.post_message(self.CycleRequested(-1 if event.key == "left" else 1))
            return
        await super()._on_key(event)


class InfoSheet(Widget):
    """Own the shared information sheet structure and behavior."""

    page: SheetPage | None = None

    def compose(self) -> ComposeResult:
        """Create the heading, divider, scrolling body, and footer."""
        with Horizontal(id="specs-heading", classes="sheet-heading"):
            yield Static("", markup=False, id="specs-identity", classes="sheet-title")
            yield Static(
                "", markup=False, id="specs-link", classes="brand-link sheet-link"
            )
        yield Rule(line_style="solid", id="specs-rule", classes="sheet-rule")
        with SheetScroll(id="specs-scroll", classes="sheet-scroll"):
            yield Static("", markup=False, id="specs-body", classes="sheet-body")
        yield Static("", markup=False, id="specs-hints", classes="sheet-hints")

    def open(self, page: SheetPage) -> None:
        """Open, render, reset, and focus one declared page."""
        self.page = page
        self.display = True
        self.query_one("#specs-identity", Static).update(page.title)
        link = self.query_one("#specs-link", Static)
        link.update(page.link or "")
        link.display = page.link is not None
        self.query_one("#specs-hints", Static).update(page.hints)
        self.refresh_page(reset=True)
        self.query_one(SheetScroll).focus()

    def refresh_page(self, *, reset: bool = False) -> None:
        """Render the current page and optionally reset its scroll position."""
        if self.page is None:
            return
        self.query_one("#specs-identity", Static).update(self.page.title)
        self.query_one("#specs-body", Static).update(self.page.renderer())
        if reset:
            self.query_one(SheetScroll).scroll_to(y=0, animate=False)

    def close(self) -> SheetOrigin | None:
        """Close the page and return its origin."""
        origin = None if self.page is None else self.page.origin
        self.page = None
        self.display = False
        return origin
