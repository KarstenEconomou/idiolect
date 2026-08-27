"""Render full model details for the terminal interface."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from rich.color import Color
from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.padding import Padding
from rich.segment import Segment, Segments
from rich.style import Style
from rich.text import Text
from textual.scrollbar import ScrollBarRender

from idiolect.chat.discovery import Assistant
from idiolect.chat.state import ChatBubble, ChatSession, PreparedPrompt
from idiolect.chat.storage import SavedChat
from idiolect.chat.worker import LoadProbe, RuntimeProbe
from idiolect.config import GenerationConfig
from idiolect.types import Split

_ABBREVIATIONS = {
    "CONTEXT": "CTX",
    "EVALUATION": "EVAL",
    "GENERATION": "GEN",
    "REPETITION": "REP",
    "TOKENS": "TOK",
}
_PROMPT_BLOCK_FIELDS = frozenset(
    {
        "system_prompt",
        "prompt_prefix",
        "prompt_suffix",
        "completion_prefix",
        "completion_suffix",
    }
)
_PROBE_LABELS = {
    "device_name": "name",
    "max_recommended_working_set_size": "working_set_limit",
    "memory_size": "memory",
}
_DEFAULT_SCROLL_BACK = Color.parse("#555555")
_DEFAULT_SCROLL_BAR = Color.parse("bright_magenta")
_FIELD_NAME = Style(bold=False)
_DESCRIPTION = Style(color="bright_black", bold=False)
_MUTED_DESCRIPTION = Style(color="bright_black", dim=True, bold=False)


class SpecsDocument:
    """Render specification lines with independently padded value blocks."""

    def __init__(self) -> None:
        """Create an empty specification document."""
        self._renderables: list[RenderableType] = []
        self._text = Text()

    def __bool__(self) -> bool:
        """Return true when the document contains a rendered line."""
        return bool(self._renderables)

    @property
    def plain(self) -> str:
        """Return the logical text used by tests and accessibility tools."""
        return self._text.plain

    def get_style_at_offset(self, console: Console, offset: int) -> Style:
        """Return the resolved logical-text style at one offset."""
        return self._text.get_style_at_offset(console, offset)

    def append_line(self, line: Text | None = None) -> None:
        """Append one unpadded visual line."""
        rendered = line or Text()
        self._renderables.append(rendered)
        self._text.append_text(rendered)
        self._text.append("\n")

    def append_inset(self, label: Text, value: Text) -> None:
        """Append one label and a value with a one-cell render inset."""
        self.append_line(label)
        self._renderables.append(Padding(value, (0, 0, 0, 1)))
        for line in value.split("\n", allow_blank=True):
            self._text.append(" ")
            self._text.append_text(line)
            self._text.append("\n")

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        """Yield each specification block to Rich."""
        del console, options
        yield from self._renderables


class HalfCellScrollBarRender(ScrollBarRender):
    """Render a vertical scrollbar with one half-cell glyph."""

    @classmethod
    def render_bar(
        cls,
        size: int = 25,
        virtual_size: float = 50,
        window_size: float = 20,
        position: float = 0,
        thickness: int = 1,
        vertical: bool = True,
        back_color: Color = _DEFAULT_SCROLL_BACK,
        bar_color: Color = _DEFAULT_SCROLL_BAR,
    ) -> Segments:
        """Return the standard bar with a half-cell vertical thumb."""
        rendered = super().render_bar(
            size,
            virtual_size,
            window_size,
            position,
            thickness,
            vertical,
            back_color,
            bar_color,
        )
        if not vertical:
            return rendered
        segments = []
        for segment in rendered.segments:
            meta = None if segment.style is None else segment.style.meta
            if meta is not None and meta.get("@mouse.down") == "grab":
                segment = Segment(
                    "▐" * thickness,
                    Style(color=bar_color, bgcolor=back_color, meta=meta),
                )
            segments.append(segment)
        return Segments(segments, new_lines=rendered.new_lines)


def render_specs(
    assistant: Assistant,
    generation: GenerationConfig,
    kind: str,
    trace: SavedChat | None = None,
) -> SpecsDocument:
    """Return one responsive model specification document."""
    document = SpecsDocument()
    _section(document, "IDENTITY")
    _field(document, "TYPE", kind)
    _field(document, "CONSTRUCT", assistant.target_run)
    _field(document, "BASE", assistant.model_basename)
    _field(document, "ENTRY", "READY")

    _section(document, "MODEL")
    _field(document, "NAME", assistant.model.name)
    _field(document, "SOURCE", assistant.model.source.upper())
    _field(document, "REVISION", assistant.model.revision)
    _field(document, "CACHE", assistant.model.cache)
    _field(document, "MODEL DIGEST", _upper_hex(assistant.model_digest))
    _field(
        document,
        "TRUST REMOTE CODE",
        "YES" if assistant.model.trust_remote_code else "NO",
    )

    if assistant.run is not None:
        _section(document, "LINEAGE")
        _field(document, "RUN ID", _upper_hex(assistant.run_id))
        _field(document, "RUN PATH", assistant.run.ref.path)
        _field(document, "DATASET ID", _upper_hex(assistant.dataset_id))
        if assistant.dataset is not None:
            _field(document, "DATASET PATH", assistant.dataset.dataset.path)
        _field(document, "ADAPTER PATH", assistant.run.adapter_path)
        _field(document, "ADAPTER DIGEST", _upper_hex(assistant.adapter_digest))
        _field(document, "TRAINING SEED", assistant.training_seed)
        _field(document, "MAX SEQUENCE", assistant.run.max_seq_length)
        _field(document, "DATASET SPLITS", _split_counts(assistant))

        _section(document, "TRAINING")
        for key, value in _flatten(assistant.run.policy):
            _field(document, key, value)

    if trace is not None:
        _section(document, "TRACE")
        _field(document, "NAME", trace.title)
        _field(document, "TRACE ID", trace.id.upper())
        _field(document, "TRACE PATH", trace.path)
        _field(document, "PARENT ID", _upper_hex(trace.parent_id))
        _field(document, "CREATED", trace.created_at.isoformat())
        _field(document, "TURNS", len(trace.turns))

    _section(document, "CONVERSATION POLICY")
    _field(document, "CONTEXT MESSAGES", assistant.context_messages)
    for key, value in asdict(assistant.data).items():
        _field(document, key, value)

    _section(document, "GENERATION")
    for key, value in asdict(generation).items():
        _field(document, key, value)

    _section(document, "FIDELITY")
    _field(document, "STATUS", "NOT EVALUATED")
    _note(document, "No recorded evaluation was supplied to this registry.")
    return document


def render_probe(
    runtime: RuntimeProbe | None,
    load: LoadProbe | None,
) -> SpecsDocument:
    """Return one hardware and model-load probe document."""
    document = SpecsDocument()
    _section(document, "STACK")
    _field(document, "MLX VERSION", None if runtime is None else runtime.mlx_version)
    _field(
        document,
        "MLX-LM VERSION",
        None if runtime is None else runtime.mlx_lm_version,
    )
    _field(
        document,
        "DEVICE",
        None if runtime is None else _device_type(runtime.device),
    )
    _field(
        document,
        "MACHINE ARCHITECTURE",
        None if runtime is None else runtime.architecture,
    )

    _section(document, "DEVICE")
    if runtime is None or not runtime.device_properties:
        _note(document, "No Metal device properties were reported.")
    else:
        for name, value in runtime.device_properties:
            displayed = _format_bytes(value) if _byte_property(name, value) else value
            _field(document, _PROBE_LABELS.get(name, name), displayed)

    _section(document, "PAYLOAD")
    _field(
        document,
        "MODEL DIGEST",
        None if load is None else _upper_hex(load.model_digest),
    )
    _field(
        document,
        "MODEL SIZE",
        None if load is None else _format_bytes(load.model_size),
    )
    _field(
        document,
        "ADAPTER SIZE",
        None
        if load is None or load.adapter_size is None
        else _format_bytes(load.adapter_size),
    )
    _field(
        document,
        "LOAD TIME",
        None if load is None else f"{load.load_duration:.3f} S",
    )
    return document


def _device_type(value: str) -> str:
    """Return the uppercase type from one MLX device representation."""
    normalized = value.strip()
    if normalized.casefold().startswith("device(") and normalized.endswith(")"):
        normalized = normalized[7:-1].split(",", 1)[0].strip()
    return normalized.upper()


def render_buffer(
    session: ChatSession,
    prepared: PreparedPrompt | None,
) -> SpecsDocument:
    """Return context use and resident reference details."""
    document = SpecsDocument()
    _section(document, "CONTEXT")
    _field(
        document,
        "TURNS",
        f"— / {session.assistant.context_messages:,}"
        if prepared is None
        else f"{prepared.active_turns:,} / {session.assistant.context_messages:,}",
    )
    _field(
        document,
        "TOKENS",
        f"— / {session.generation.max_prompt_tokens:,}"
        if prepared is None
        else (
            f"{prepared.prompt_tokens:,} / {session.generation.max_prompt_tokens:,} "
            f"({100 * prepared.prompt_tokens / session.generation.max_prompt_tokens:.1f}%)"
        ),
        abbreviate=False,
    )
    _field(
        document,
        "EVICTED",
        None if prepared is None else prepared.dropped_messages,
    )
    _field(
        document,
        "STATE DIGEST",
        None if prepared is None else _upper_hex(prepared.prompt_digest),
    )
    _field(
        document,
        "HEAD",
        None
        if prepared is None or not prepared.active_references
        else _buffer_reference_name(session, prepared.active_references[-1]),
    )

    _section(document, "RESIDENT")
    if prepared is None:
        _note(document, "No prompt state is resident.")
    elif not prepared.active_references:
        _note(document, "No references are resident.")
    else:
        for reference in prepared.active_references:
            document.append_line(
                Text(_buffer_reference_name(session, reference), style=_FIELD_NAME)
            )
    return document


def _buffer_reference_name(session: ChatSession, reference: ChatBubble) -> str:
    """Return one stable BUFFER reference identity."""
    name = "OP" if reference.role == "user" else session.assistant.target_name
    return f"@{name.upper()}:{reference.index:02d}"


def _byte_property(name: str, value: object) -> bool:
    """Return true when one integer device property reports bytes."""
    normalized = name.casefold()
    return isinstance(value, int) and not isinstance(value, bool) and (
        normalized.endswith(("_size", "_memory"))
        or "working_set_size" in normalized
    )


def _format_bytes(value: object) -> str:
    """Return one byte measurement with a compact IEC unit."""
    if not isinstance(value, int) or isinstance(value, bool):
        return _display(value)
    if value < 1024:
        return f"{value:,} B"
    amount = float(value)
    unit = "B"
    for candidate in ("KiB", "MiB", "GiB", "TiB", "PiB"):
        amount /= 1024
        unit = candidate
        if amount < 1024 or candidate == "PiB":
            break
    return f"{amount:.2f} {unit}"


def _upper_hex(value: str | None) -> str | None:
    """Return one displayed hexadecimal identity in uppercase."""
    return None if value is None else value.upper()


def _section(document: SpecsDocument, name: str) -> None:
    """Append one specification section heading."""
    if document:
        document.append_line()
    document.append_line(Text(name, style="bold white"))


def _field(
    document: SpecsDocument,
    name: str,
    value: object,
    *,
    value_style: str | Style = _DESCRIPTION,
    abbreviate: bool = True,
) -> None:
    """Append one aligned field and preserve multiline values."""
    label = _label(name) if abbreviate else name.upper()
    field_name = name.replace(" ", "_").casefold()
    if field_name in _PROMPT_BLOCK_FIELDS:
        _prompt_block_field(
            document,
            label,
            value if isinstance(value, str) else _display(value),
            value_style,
            system_prompt=field_name == "system_prompt",
        )
        return
    displayed = _display(value)
    document.append_inset(
        Text(label, style=_FIELD_NAME),
        Text(displayed, style=value_style),
    )


def _prompt_block_field(
    document: SpecsDocument,
    label: str,
    value: str,
    value_style: str | Style,
    *,
    system_prompt: bool,
) -> None:
    """Append one prompt-format value with one-cell wrapped indentation."""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    if system_prompt:
        normalized = normalized.rstrip("\n")
        displayed = normalized or "—"
    else:
        visible = json.dumps(normalized, ensure_ascii=False)[1:-1]
        displayed = visible or "—"
    document.append_inset(
        Text(label, style=_FIELD_NAME),
        Text(displayed, style=value_style),
    )


def _label(value: str) -> str:
    """Return one uppercase interface label with standard abbreviations."""
    words = value.replace("_", " ").upper().split()
    return " ".join(_ABBREVIATIONS.get(word, word) for word in words)


def _note(document: SpecsDocument, value: str) -> None:
    """Append one metadata note."""
    document.append_line(Text(value, style=_DESCRIPTION))


def _display(value: object) -> str:
    """Return a stable display value for one specification field."""
    if value is None or value == "" or value == ():
        return "—"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (tuple, list)):
        return ", ".join(_display(item) for item in value)
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _flatten(
    values: Mapping[str, Any],
    prefix: str = "",
) -> tuple[tuple[str, object], ...]:
    """Flatten a recorded policy into stable display fields."""
    result: list[tuple[str, object]] = []
    for key in sorted(values):
        value = values[key]
        name = f"{prefix} {key}".strip()
        if isinstance(value, Mapping):
            result.extend(_flatten(value, name))
        else:
            result.append((name, value))
    return tuple(result)


def _split_counts(assistant: Assistant) -> str:
    """Return verified dataset counts in pipeline order."""
    return "    ".join(
        f"{split.value.upper()} {assistant.counts.get(split, 0):,}"
        for split in (Split.TRAIN, Split.VALID, Split.TEST)
    )
