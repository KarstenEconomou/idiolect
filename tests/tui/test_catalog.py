"""Test assistant registry row formatting."""

from rich.cells import cell_len
from rich.console import Console

from idiolect.tui.catalog import CatalogLayout


def test_catalog_columns_follow_terminal_width() -> None:
    """Check the fields that fit at each supported width."""
    narrow = CatalogLayout.for_terminal(45).line("MODEL", "TYPE", "ENTRY")
    medium = CatalogLayout.for_terminal(62).line("MODEL", "TYPE", "ENTRY")
    wide = CatalogLayout.for_terminal(100).line("MODEL", "TYPE", "ENTRY")

    assert "TYPE" not in narrow
    assert "TYPE" not in medium
    assert "TYPE" in wide
    assert all("ENTRY" in line for line in (narrow, medium, wide))


def test_catalog_row_has_a_stable_cell_width_for_unicode_names() -> None:
    """Check terminal-cell alignment for a non-ASCII assistant name."""
    layout = CatalogLayout.for_terminal(80)

    row = layout.text("IDIOLECT // 模型@BASE [LOCAL]", "BASE", "READY")

    assert cell_len(row.plain) == sum(
        (layout.model, layout.kind, layout.status)
    ) + 2
    assert row.plain.rstrip().endswith("READY")


def test_catalog_status_labels_fill_the_right_edge() -> None:
    """Check each fixed-width status reaches the registry divider edge."""
    layout = CatalogLayout.for_terminal(80)

    rows = {
        status: layout.text("MODEL", "TYPE", status).plain
        for status in ("READY", "FAULT")
    }

    assert all(row.endswith(status) for status, row in rows.items())
    assert all(cell_len(row) == 76 for row in rows.values())
    assert len({row.index(status) for status, row in rows.items()}) == 1
    assert layout.status == 5


def test_catalog_metadata_columns_leave_room_for_values() -> None:
    """Check registry metadata columns have deliberate breathing room."""
    layout = CatalogLayout.for_terminal(80)

    assert (layout.model, layout.kind, layout.status) == (59, 10, 5)

    line = layout.line("MODEL", "CONSTRUCT", "ENTRY")
    kind_gap = line.index("ENTRY") - (line.index("CONSTRUCT") + len("CONSTRUCT"))
    assert kind_gap == 2


def test_catalog_type_labels_identify_entry_lineage() -> None:
    """Check base, construct, and trace labels fit the type column."""
    layout = CatalogLayout.for_terminal(80)

    rows = {
        kind: layout.text("MODEL", kind, "READY").plain
        for kind in ("BASE", "CONSTRUCT", "TRACE")
    }

    assert all(kind in row for kind, row in rows.items())


def test_catalog_type_and_entry_follow_description_selection_style() -> None:
    """Check TYPE and ENTRY use the same slash-description styling."""
    layout = CatalogLayout.for_terminal(80)
    unselected = layout.text("MODEL", "BASE", "READY")
    selected = layout.text("MODEL", "BASE", "READY", selected=True)
    console = Console()

    type_style = unselected.get_style_at_offset(
        console, unselected.plain.index("BASE")
    )
    entry_style = unselected.get_style_at_offset(
        console, unselected.plain.index("READY")
    )
    selected_type_style = selected.get_style_at_offset(
        console, selected.plain.index("BASE")
    )
    selected_entry_style = selected.get_style_at_offset(
        console, selected.plain.index("READY")
    )

    assert type_style.color is not None and type_style.color.number == 8
    assert entry_style.color is not None and entry_style.color.number == 8
    assert not type_style.dim and type_style.bold is False
    assert not entry_style.dim and entry_style.bold is False
    assert selected_type_style.color is None and selected_type_style.dim
    assert selected_entry_style.color is None and selected_entry_style.dim
    assert selected_type_style.bold is False
    assert selected_entry_style.bold is False


def test_catalog_fault_matches_an_unavailable_slash_command() -> None:
    """Check every unavailable row field is muted and dimmed."""
    row = CatalogLayout.for_terminal(80).text(
        "Unavailable model",
        "—",
        "FAULT",
        failed=True,
    )
    console = Console()

    for text in ("Unavailable model", "—", "FAULT"):
        style = row.get_style_at_offset(console, row.plain.index(text))
        assert style.color is not None and style.color.number == 8
        assert style.dim
        assert style.bold is False


def test_catalog_trace_places_metadata_name_after_model() -> None:
    """Check inline trace metadata and its selected description style."""
    layout = CatalogLayout.for_terminal(80)
    row = layout.text(
        "IDIOLECT // DIXIE@BASE [M]",
        "TRACE",
        "READY",
        trace_name="Night session",
    )
    trace_style = row.get_style_at_offset(
        Console(), row.plain.index("Night session")
    )

    assert "\n" not in row.plain
    assert row.plain.startswith("IDIOLECT // DIXIE@BASE [M] Night session")
    assert row.plain.index("Night session") < row.plain.index("TRACE")
    assert trace_style.color is not None
    assert trace_style.color.number == 8

    selected = layout.text(
        "IDIOLECT // DIXIE@BASE [M]",
        "TRACE",
        "READY",
        selected=True,
        trace_name="Night session",
    )
    selected_trace_style = selected.get_style_at_offset(
        Console(), selected.plain.index("Night session")
    )
    assert selected_trace_style.color is None
    assert selected_trace_style.dim


def test_catalog_ellipsizes_only_the_inline_trace_name() -> None:
    """Check a long trace name yields before the canonical model identity."""
    layout = CatalogLayout.for_terminal(80)
    model = "IDIOLECT // DIXIE@BASE [M]"
    row = layout.text(
        model,
        "TRACE",
        "READY",
        trace_name="A trace name that is much too long for the remaining model column",
    )

    model_cell = row.plain[: layout.model]
    assert model_cell.startswith(f"{model} ")
    assert model_cell.endswith("…")
    assert "much too long" not in model_cell
    assert cell_len(row.plain) == 76
