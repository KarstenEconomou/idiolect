"""Test reusable TUI sheet and menu primitives."""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.containers import Horizontal, HorizontalScroll
from textual.widgets import Static

from idiolect.tui.menus import (
    HorizontalMenuModal,
    MenuButton,
    MenuCursor,
    MenuItem,
    VerticalMenu,
)
from idiolect.tui.sheets import InfoSheet, SheetPage, SheetScroll
from idiolect.tui.specs import SheetDocument


def test_menu_cursor_wraps_skips_disabled_and_retains_selection() -> None:
    """Check shared cursor movement, retention, and viewport placement."""
    items = (
        MenuItem("a", "A"),
        MenuItem("b", "B", disabled=True),
        MenuItem("c", "C"),
        MenuItem("d", "D"),
        MenuItem("e", "E"),
    )
    cursor = MenuCursor(items, "c")

    assert cursor.move(-1) == items[0]
    assert cursor.move(-1) == items[4]
    assert cursor.viewport() == items[2:]
    assert cursor.retain(items[:-1]) == items[0]


class _VerticalMenuApp(App[None]):
    def compose(self) -> ComposeResult:
        yield VerticalMenu("COMMAND", id="menu")


def test_vertical_menu_renders_rows_and_variants() -> None:
    """Check declarative rows keep labels, descriptions, and variants."""
    async def verify() -> None:
        async with _VerticalMenuApp().run_test() as pilot:
            menu = pilot.app.query_one(VerticalMenu)
            menu.set_items(
                (
                    MenuItem("one", "ONE", "First."),
                    MenuItem("two", "TWO", "Second.", True, "muted"),
                ),
                "one",
            )
            await pilot.pause()

            first = menu.query_one("#menu-0", Horizontal)
            second = menu.query_one("#menu-1", Horizontal)
            assert str(first.query_one(".menu-name", Static).content) == "ONE"
            assert str(first.query_one(".menu-description", Static).content) == "First."
            assert first.has_class("-selected")
            assert second.has_class("-disabled")
            assert second.has_class("-muted")

    asyncio.run(verify())


class _ModalApp(App[None]):
    CSS = """
    #landing, #catalog-hints, #command-bar, #reference-bar { height: 1; }
    #composer-bar { height: 3; }
    .horizontal-menu { height: 2; }
    .menu-heading, .horizontal-menu-actions { height: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="landing")
        yield Static("", id="catalog-hints")
        yield Static("", id="command-bar")
        yield Static("", id="reference-bar")
        with Horizontal(id="composer-bar"):
            yield Static("")


def test_horizontal_menu_focus_wrap_escape_and_highlight() -> None:
    """Check shared horizontal keyboard behavior and safe cancellation."""
    async def verify() -> None:
        highlights: list[str] = []
        result: list[str | None] = []
        app = _ModalApp()
        async with app.run_test(size=(80, 20)) as pilot:
            app.query_one("#landing").display = False
            app.query_one("#command-bar").display = False
            app.query_one("#reference-bar").display = False
            app.push_screen(
                HorizontalMenuModal(
                    "ACTIONS",
                    (
                        MenuItem("a", "A"),
                        MenuItem("b", "B"),
                        MenuItem("c", "C"),
                    ),
                    "b",
                    "c",
                    anchor="chat",
                    on_highlight=highlights.append,
                ),
                result.append,
            )
            await pilot.pause()
            assert app.screen.query_one("#b", MenuButton).has_focus
            dialog = app.screen.query_one("#horizontal-menu-dialog")
            composer = app.query_one("#composer-bar")
            assert dialog.region.x == composer.region.x
            assert dialog.styles.offset.y.value == composer.region.y - 2
            assert dialog.styles.width is not None
            assert dialog.styles.width.value == composer.region.width

            await pilot.press("right", "right")
            assert app.screen.query_one("#a", MenuButton).has_focus
            assert highlights == ["c", "a"]
            await pilot.press("escape")
            assert result == ["c"]

    asyncio.run(verify())


def test_horizontal_menu_reveals_selection_at_narrow_widths() -> None:
    """Check focus-following scroll survives narrow widths and resize."""
    async def verify() -> None:
        items = tuple(
            MenuItem(identity, label)
            for identity, label in (
                ("a", "LOCKSMITH"),
                ("b", "LOOKOUT"),
                ("c", "PICKPOCKET"),
                ("d", "CLEANER"),
                ("e", "MOLE"),
                ("f", "GENTLEMAN"),
                ("g", "HACKER"),
                ("h", "REDHEAD"),
            )
        )
        app = _ModalApp()
        async with app.run_test(size=(80, 20)) as pilot:
            app.query_one("#landing").display = False
            app.query_one("#command-bar").display = False
            app.query_one("#reference-bar").display = False
            app.push_screen(
                HorizontalMenuModal(
                    "CHROMA",
                    items,
                    "g",
                    "g",
                    anchor="chat",
                )
            )
            await pilot.pause()

            actions = app.screen.query_one(
                "#horizontal-menu-actions",
                HorizontalScroll,
            )

            def selected_is_visible(identity: str) -> bool:
                button = app.screen.query_one(f"#{identity}", MenuButton)
                return (
                    button.region.x >= actions.content_region.x
                    and button.region.right <= actions.content_region.right
                )

            assert selected_is_visible("g")
            await pilot.resize_terminal(48, 20)
            await pilot.pause()
            assert selected_is_visible("g")
            await pilot.resize_terminal(24, 20)
            await pilot.pause()
            assert selected_is_visible("g")

            await pilot.press("right")
            assert selected_is_visible("h")
            await pilot.press("right")
            assert selected_is_visible("a")

    asyncio.run(verify())


def _document(value: str) -> SheetDocument:
    document = SheetDocument()
    document.section("DATA")
    document.field("VALUE", value)
    document.note("Complete.")
    return document


class _SheetApp(App[None]):
    def compose(self) -> ComposeResult:
        yield InfoSheet(id="specs")


def test_info_sheet_opens_refreshes_resets_and_focuses() -> None:
    """Check one page owns shared sheet rendering and interaction capabilities."""
    async def verify() -> None:
        rendered = ["ONE"]
        async with _SheetApp().run_test(size=(60, 12)) as pilot:
            sheet = pilot.app.query_one(InfoSheet)
            page = SheetPage(
                "SPECS",
                lambda: _document(rendered[0]),
                "registry",
                None,
                "ESC REGISTRY",
                cycle=True,
                connect=True,
            )
            sheet.open(page)
            await pilot.pause()

            assert sheet.query_one(SheetScroll).has_focus
            body = sheet.query_one("#specs-body", Static).content
            assert isinstance(body, SheetDocument)
            assert "VALUE\n ONE\n" in body.plain
            assert sheet.page is not None and sheet.page.cycle and sheet.page.connect
            rendered[0] = "TWO"
            sheet.refresh_page(reset=True)
            refreshed = sheet.query_one("#specs-body", Static).content
            assert isinstance(refreshed, SheetDocument)
            assert "VALUE\n TWO\n" in refreshed.plain
            assert sheet.close() == "registry"
            assert not sheet.display

    asyncio.run(verify())
