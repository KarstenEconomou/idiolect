"""Format rows for the assistant registry."""

from dataclasses import dataclass

from rich.cells import cell_len, set_cell_size
from rich.style import Style
from rich.text import Text

_DESCRIPTION = Style(color="bright_black", bold=False)
_SELECTED_DESCRIPTION = Style(dim=True, bold=False)
_UNAVAILABLE = Style(color="bright_black", dim=True, bold=False)


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
        """Return one styled registry entry and optional inline trace name."""
        description_style = (
            _UNAVAILABLE
            if failed
            else _SELECTED_DESCRIPTION
            if selected
            else _DESCRIPTION
        )
        trace_style = (
            _SELECTED_DESCRIPTION if selected or trace_active else _DESCRIPTION
        )
        value = self._model_text(
            model,
            trace_name,
            trace_visible=trace_visible,
            model_style=_UNAVAILABLE if failed else None,
            trace_style=trace_style,
        )
        if self.kind:
            value.append(" ", style=description_style)
            value.append(
                set_cell_size(kind, self.kind),
                style=description_style,
            )
        value.append(" ", style=description_style)
        value.append(
            set_cell_size(status, self.status),
            style=description_style,
        )
        return value

    def _model_text(
        self,
        model: str,
        trace_name: str | None,
        *,
        trace_visible: bool,
        model_style: Style | None,
        trace_style: Style,
    ) -> Text:
        """Return the fixed-width MODEL cell with optional TRACE metadata."""
        model_width = cell_len(model)
        if trace_name is None or model_width >= self.model - 1:
            text = set_cell_size(model, self.model)
            return (
                Text(text)
                if model_style is None
                else Text(text, style=model_style)
            )
        trace_width = self.model - model_width - 1
        value = (
            Text(model)
            if model_style is None
            else Text(model, style=model_style)
        )
        value.append(" ", style=trace_style)
        visible_name = trace_name if trace_visible else ""
        value.append(_ellipsize(visible_name, trace_width), style=trace_style)
        return value


def _ellipsize(value: str, width: int) -> str:
    """Fit one trace name to a fixed cell width with a final ellipsis."""
    if not value:
        return " " * width
    if cell_len(value) <= width:
        return set_cell_size(value, width)
    if width == 1:
        return "…"
    return f"{set_cell_size(value, width - 1)}…"
