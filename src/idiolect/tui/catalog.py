"""Format rows for the assistant registry."""

from dataclasses import dataclass

from rich.cells import set_cell_size
from rich.style import Style
from rich.text import Text

_METADATA = Style(dim=True, bold=False)
_TRACE_NAME = Style(color="bright_black", bold=False)
_READY_SELECTED = Style(dim=False, bold=False)
_FAILED = Style(color="red", dim=True, bold=False)


@dataclass(frozen=True, slots=True)
class CatalogLayout:
    """Keep the visible registry column widths."""

    model: int
    kind: int
    status: int

    @classmethod
    def for_terminal(cls, terminal_width: int) -> CatalogLayout:
        """Return the registry layout for one terminal width."""
        content_width = max(24, terminal_width - 4)
        status_width = 5
        kind_width = (16 if content_width >= 100 else 10) if content_width >= 70 else 0
        visible_separators = sum(width > 0 for width in (kind_width, status_width))
        model_width = max(
            8,
            content_width
            - kind_width
            - status_width
            - visible_separators,
        )
        return cls(model_width, kind_width, status_width)

    def line(self, model: str, kind: str, status: str) -> str:
        """Return one plain registry line."""
        values = [set_cell_size(model, self.model)]
        if self.kind:
            values.append(set_cell_size(kind, self.kind))
        values.append(set_cell_size(status, self.status))
        return " ".join(values)

    def text(
        self,
        model: str,
        kind: str,
        status: str,
        *,
        failed: bool = False,
        selected: bool = False,
        trace_name: str | None = None,
        trace_active: bool = False,
        trace_visible: bool = True,
    ) -> Text:
        """Return one styled registry entry and optional trace name."""
        value = Text(set_cell_size(model, self.model))
        if self.kind:
            value.append(" ", style=_METADATA)
            value.append(set_cell_size(kind, self.kind), style=_METADATA)
        value.append(" ", style=_METADATA)
        status_style = (
            _FAILED if failed else _READY_SELECTED if selected else _METADATA
        )
        value.append(set_cell_size(status, self.status), style=status_style)
        if trace_name is not None:
            trace_style = _METADATA if selected or trace_active else _TRACE_NAME
            value.append("\n")
            value.append(" ", style=trace_style)
            value.append(
                set_cell_size(
                    trace_name if trace_visible else "",
                    max(self.model - 1, 1),
                ),
                trace_style,
            )
        return value
