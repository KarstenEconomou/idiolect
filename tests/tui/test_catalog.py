"""Test assistant registry row formatting."""

from rich.cells import cell_len

from idiolect.tui.catalog import CatalogLayout


def test_catalog_columns_follow_terminal_width() -> None:
    """Check the fields that fit at each supported width."""
    narrow = CatalogLayout.for_terminal(45).line("MODEL", "DATA", "WINDOW", "STATUS")
    medium = CatalogLayout.for_terminal(62).line("MODEL", "DATA", "WINDOW", "STATUS")
    wide = CatalogLayout.for_terminal(100).line("MODEL", "DATA", "WINDOW", "STATUS")

    assert "DATA" not in narrow
    assert "WINDOW" not in narrow
    assert "WINDOW" in medium
    assert "DATA" in wide
    assert all("STATUS" in line for line in (narrow, medium, wide))


def test_catalog_row_has_a_stable_cell_width_for_unicode_names() -> None:
    """Check terminal-cell alignment for a non-ASCII assistant name."""
    layout = CatalogLayout.for_terminal(80)

    row = layout.text(
        "IDIOLECT // 模型@BASE [LOCAL]", "PERSONA", "32", "AVAILABLE"
    )

    assert cell_len(row.plain) == sum(
        (layout.model, layout.data, layout.window, layout.status)
    ) + 3
    assert row.plain.endswith("AVAILABLE")


def test_catalog_status_ends_at_the_right_edge_of_the_line() -> None:
    """Check status labels align with the registry divider edge."""
    layout = CatalogLayout.for_terminal(80)

    header = layout.line("MODEL", "DATA", "WINDOW", "STATUS")
    available = layout.text("MODEL", "DATA", "WINDOW", "AVAILABLE")
    unavailable = layout.text("MODEL", "DATA", "WINDOW", "UNAVAILABLE")

    assert header.endswith("STATUS")
    assert available.plain.endswith("AVAILABLE")
    assert unavailable.plain.endswith("UNAVAILABLE")


def test_catalog_metadata_columns_leave_room_for_values() -> None:
    """Check registry metadata columns have deliberate breathing room."""
    layout = CatalogLayout.for_terminal(80)

    assert (layout.data, layout.window, layout.status) == (10, 9, 13)
