"""Format rows for the assistant registry."""

from dataclasses import dataclass

from rich.cells import set_cell_size
from rich.text import Text


@dataclass(frozen=True, slots=True)
class CatalogLayout:
    """Keep the visible registry column widths."""

    model: int
    data: int
    window: int
    status: int

    @classmethod
    def for_terminal(cls, terminal_width: int) -> CatalogLayout:
        """Return the registry layout for one terminal width."""
        content_width = max(24, min(116, terminal_width - 8))
        status_width = 13
        data_width = (16 if content_width >= 100 else 10) if content_width >= 70 else 0
        window_width = 9 if content_width >= 54 else 0
        visible_separators = sum(
            width > 0 for width in (data_width, window_width, status_width)
        )
        model_width = max(
            8,
            content_width
            - data_width
            - window_width
            - status_width
            - visible_separators,
        )
        return cls(model_width, data_width, window_width, status_width)

    def line(self, model: str, data: str, window: str, status: str) -> str:
        """Return one plain registry line."""
        values = [set_cell_size(model, self.model)]
        if self.data:
            values.append(set_cell_size(data, self.data))
        if self.window:
            values.append(set_cell_size(window, self.window))
        values.append(set_cell_size(status, self.status))
        return " ".join(values)

    def text(
        self,
        model: str,
        data: str,
        window: str,
        status: str,
        *,
        failed: bool = False,
    ) -> Text:
        """Return one registry line with status emphasis."""
        value = Text(set_cell_size(model, self.model))
        if self.data:
            value.append(" ", style="dim")
            value.append(set_cell_size(data, self.data), style="dim")
        if self.window:
            value.append(" ", style="dim")
            value.append(set_cell_size(window, self.window), style="dim")
        value.append(" ", style="dim")
        status_style = "red" if failed else "bold"
        value.append(set_cell_size(status, self.status), style=status_style)
        return value
