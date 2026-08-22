"""Test assistant registry row formatting."""

from rich.cells import cell_len

from idiolect.tui.catalog import CatalogLayout


def test_catalog_columns_follow_terminal_width() -> None:
    """Check the fields that fit at each supported width."""
    narrow = CatalogLayout.for_terminal(45).line("MODEL", "DATA", "WINDOW", "ENTRY")
    medium = CatalogLayout.for_terminal(62).line("MODEL", "DATA", "WINDOW", "ENTRY")
    wide = CatalogLayout.for_terminal(100).line("MODEL", "DATA", "WINDOW", "ENTRY")

    assert "DATA" not in narrow
    assert "WINDOW" not in narrow
    assert "WINDOW" in medium
    assert "DATA" in wide
    assert all("ENTRY" in line for line in (narrow, medium, wide))


def test_catalog_row_has_a_stable_cell_width_for_unicode_names() -> None:
    """Check terminal-cell alignment for a non-ASCII assistant name."""
    layout = CatalogLayout.for_terminal(80)

    row = layout.text(
        "IDIOLECT // 模型@BASE [LOCAL]", "BASE", "32", "READY"
    )

    assert cell_len(row.plain) == sum(
        (layout.model, layout.data, layout.window, layout.status)
    ) + 3
    assert row.plain.rstrip().endswith("READY")


def test_catalog_status_labels_fill_the_right_edge() -> None:
    """Check each fixed-width status reaches the registry divider edge."""
    layout = CatalogLayout.for_terminal(80)

    rows = {
        status: layout.text("MODEL", "DATA", "WINDOW", status).plain
        for status in ("READY", "FAULT")
    }

    assert all(row.endswith(status) for status, row in rows.items())
    assert len({row.index(status) for status, row in rows.items()}) == 1
    assert layout.status == 5


def test_catalog_metadata_columns_leave_room_for_values() -> None:
    """Check registry metadata columns have deliberate breathing room."""
    layout = CatalogLayout.for_terminal(80)

    assert (layout.data, layout.window, layout.status) == (10, 9, 5)


def test_catalog_data_labels_identify_entry_lineage() -> None:
    """Check base, construct, and trace labels fit the data column."""
    layout = CatalogLayout.for_terminal(80)

    rows = {
        data: layout.text("MODEL", data, "32", "READY").plain
        for data in ("BASE", "CONSTRUCT", "TRACE")
    }

    assert all(data in row for data, row in rows.items())
