"""Render full model details for the terminal interface."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from rich.color import Color
from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.padding import Padding
from rich.segment import Segment, Segments
from rich.style import Style
from rich.text import Text
from textual.scrollbar import ScrollBarRender

from idiolect.chat.discovery import Assistant
from idiolect.chat.state import ChatBubble, ChatSession, PreparedPrompt, TurnTelemetry
from idiolect.chat.storage import SavedChat
from idiolect.chat.worker import LoadProbe, RuntimeProbe, WorkerState
from idiolect.config import ChatConfig, GenerationConfig
from idiolect.types import Split

_ABBREVIATIONS = {
    "CONTEXT": "CTX",
    "EVALUATION": "EVAL",
    "GENERATION": "GEN",
    "REPETITION": "REP",
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
    "architecture": "architecture",
    "device_name": "name",
    "memory_size": "memory",
    "max_recommended_working_set_size": "rec_working_set",
    "max_buffer_length": "max_buffer_size",
}
_PROBE_PROPERTY_ORDER = (
    "device_name",
    "architecture",
    "memory_size",
    "active_memory",
    "cache_memory",
    "max_recommended_working_set_size",
    "max_buffer_size",
    "max_buffer_length",
)
_DEFAULT_SCROLL_BACK = Color.parse("#555555")
_DEFAULT_SCROLL_BAR = Color.parse("bright_magenta")
_FIELD_NAME = Style(bold=False)
_DESCRIPTION = Style(color="bright_black", bold=False)
_MUTED_DESCRIPTION = Style(color="bright_black", dim=True, bold=False)


class SheetDocument:
    """Build sheet lines with independently padded value blocks."""

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

    def section(self, name: str) -> None:
        """Append one sheet section heading."""
        _section(self, name)

    def field(
        self,
        name: str,
        value: object,
        *,
        value_style: str | Style = _DESCRIPTION,
        abbreviate: bool = True,
    ) -> None:
        """Append one labeled sheet field."""
        _field(
            self,
            name,
            value,
            value_style=value_style,
            abbreviate=abbreviate,
        )

    def note(self, value: str) -> None:
        """Append one metadata-colored sheet note."""
        _note(self, value)

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


SpecsDocument = SheetDocument


def render_specs(
    assistant: Assistant,
    generation: GenerationConfig,
    kind: str,
    trace: SavedChat | None = None,
    chat: ChatConfig | None = None,
) -> SheetDocument:
    """Return one responsive model specification document."""
    document = SheetDocument()
    _section(document, "IDENTITY")
    _field(document, "CONSTRUCT", assistant.target_run)
    _field(document, "BASE", assistant.model_basename)
    _field(document, "TYPE", kind)

    _section(document, "MODEL")
    _field(document, "NAME", assistant.model.name)
    _field(document, "SOURCE", assistant.model.source.upper())
    _field(document, "REVISION", assistant.model.revision)
    _field(document, "DIGEST", _upper_hex(assistant.model_digest))
    _field(
        document,
        "TRUST REMOTE CODE",
        "YES" if assistant.model.trust_remote_code else "NO",
    )

    if assistant.run is not None:
        _section(document, "RUN")
        _field(document, "ID", _upper_hex(assistant.run_id))
        _field(document, "PATH", assistant.run.ref.path)

        _section(document, "DATASET")
        _field(document, "ID", str(assistant.run.ref.dataset_id).upper())
        _field(
            document,
            "PATH",
            None if assistant.dataset is None else assistant.dataset.dataset.path,
        )
        _field(document, "SPLITS", _split_counts(assistant))

        _section(document, "ADAPTER")
        _field(document, "PATH", assistant.run.adapter_path)
        _field(document, "DIGEST", _upper_hex(assistant.adapter_digest))

        _section(document, "TRAINING")
        _field(document, "SEED", assistant.training_seed)
        _field(document, "MAX SEQUENCE TOKENS", assistant.run.max_seq_length)
        for key, value in _flatten_training_policy(assistant.run.policy):
            _field(document, key, value)

    chat = trace.chat if trace is not None else chat
    _render_conversation_policy(document, assistant, generation, chat)
    _section(document, "GENERATION")
    _field(document, "BACKEND", generation.backend.upper())
    _field(document, "SEED", None if chat is None else chat.seed)
    _render_sampling_policy(document, generation)

    if trace is not None:
        _section(document, "TRACE")
        _field(document, "NAME", trace.title)
        _field(document, "ID", trace.id.upper())
        _field(document, "PARENT", _upper_hex(trace.parent_id))
        _field(document, "CREATED", trace.created_at.isoformat())
        _field(
            document,
            "TURNS",
            sum(turn.role != "env" for turn in trace.turns),
        )

    _section(document, "FIDELITY")
    _field(document, "STATUS", "NOT EVALUATED")
    return document


def _render_conversation_policy(
    document: SheetDocument,
    assistant: Assistant,
    generation: GenerationConfig,
    chat: ChatConfig | None,
) -> None:
    """Append conversation, prompt, and completion policy fields."""
    data = assistant.data
    chat_format = data.format != "completion"

    _section(document, "CONVERSATION")
    _field(document, "FORMAT", data.format)
    _field(document, "TURN CAPACITY", assistant.context_messages)
    _field(document, "PARTICIPANT", None if chat is None else chat.participant_name)
    if chat_format:
        _prompt_block_field(
            document,
            "SYSTEM",
            data.system_prompt,
            _DESCRIPTION,
            system_prompt=True,
        )

    _section(document, "PROMPT")
    if chat_format:
        _field(document, "ROLE", data.prompt_role)
    _prompt_block_field(
        document,
        "PREFIX",
        data.prompt_prefix,
        _DESCRIPTION,
        system_prompt=False,
    )
    _prompt_block_field(
        document,
        "SUFFIX",
        data.prompt_suffix,
        _DESCRIPTION,
        system_prompt=False,
    )
    _field(document, "LIMIT", f"{generation.max_prompt_tokens:,} TOK")

    _section(document, "COMPLETION")
    if chat_format:
        _field(document, "ROLE", data.completion_role)
    _prompt_block_field(
        document,
        "PREFIX",
        data.completion_prefix,
        _DESCRIPTION,
        system_prompt=False,
    )
    _prompt_block_field(
        document,
        "SUFFIX",
        data.completion_suffix,
        _DESCRIPTION,
        system_prompt=False,
    )
    _field(document, "LIMIT", f"{generation.max_tokens:,} TOK")


def _render_sampling_policy(
    document: SheetDocument,
    generation: GenerationConfig,
) -> None:
    """Append contextual sampling and repetition fields."""
    _section(document, "SAMPLING")
    _field(document, "TEMPERATURE", generation.temperature)
    _field(document, "TOP-P", generation.top_p)
    _field(document, "TOP-K", generation.top_k)
    _field(document, "MIN-P", generation.min_p)
    _field(
        document,
        "MIN-P FLOOR",
        f"{generation.min_tokens_to_keep:,} TOK",
    )

    _section(document, "REPETITION")
    _field(document, "PENALTY", generation.repetition_penalty)
    _field(
        document,
        "WINDOW",
        f"{generation.repetition_context_size:,} TOK",
    )


def render_probe(
    runtime: RuntimeProbe | None,
    load: LoadProbe | None,
    telemetry: TurnTelemetry | None,
    state: WorkerState | str | None = None,
) -> SheetDocument:
    """Return one runtime, model-load, and generation probe document."""
    document = SheetDocument()
    _section(document, "RUNTIME")
    _field(document, "STATE", _probe_state(state))
    _field(document, "MLX", None if runtime is None else runtime.mlx_version)
    _field(
        document,
        "MLX-LM",
        None if runtime is None else runtime.mlx_lm_version,
    )
    _field(
        document,
        "HOST ARCHITECTURE",
        None if runtime is None else runtime.architecture,
    )

    if runtime is not None:
        _section(document, "DEVICE")
        _field(document, "TYPE", _device_type(runtime.device))
        for name, value in _ordered_device_properties(runtime.device_properties):
            displayed = _format_device_property(name, value)
            _field(document, _PROBE_LABELS.get(name, name), displayed)

    _section(document, "MODEL")
    _field(
        document,
        "SIZE",
        None if load is None else _format_bytes(load.model_size),
    )
    if load is None or load.adapter_size is not None:
        _field(
            document,
            "ADAPTER SIZE",
            None if load is None else _format_bytes(load.adapter_size),
        )
    _field(
        document,
        "LOAD TIME",
        None if load is None else f"{load.load_duration:.3f} S",
    )

    _section(document, "TELEMETRY")
    _field(
        document,
        "OUTPUT",
        None if telemetry is None else f"{telemetry.generated_tokens:,} TOK",
    )
    _field(
        document,
        "PREFILL THROUGHPUT",
        None
        if telemetry is None or telemetry.prompt_throughput is None
        else f"{telemetry.prompt_throughput:.1f} TOK/S",
    )
    _field(
        document,
        "DECODE THROUGHPUT",
        None
        if telemetry is None or telemetry.generation_throughput is None
        else f"{telemetry.generation_throughput:.1f} TOK/S",
    )
    _field(
        document,
        "TIME TO FIRST TOKEN",
        None
        if telemetry is None or telemetry.time_to_first_token is None
        else f"{telemetry.time_to_first_token:.3f} S",
    )
    _field(
        document,
        "INFERENCE LATENCY",
        None
        if telemetry is None or telemetry.generation_time is None
        else f"{telemetry.generation_time:.3f} S",
        abbreviate=False,
    )
    _field(
        document,
        "PEAK MEMORY",
        None
        if telemetry is None or telemetry.peak_memory is None
        else f"{telemetry.peak_memory:.2f} GB",
    )
    return document


def _device_type(value: str) -> str:
    """Return the uppercase type from one MLX device representation."""
    normalized = value.strip()
    if normalized.casefold().startswith("device(") and normalized.endswith(")"):
        normalized = normalized[7:-1].split(",", 1)[0].strip()
    return normalized.upper()


def _probe_state(value: WorkerState | str | None) -> str | None:
    """Return one uppercase worker state for the probe sheet."""
    if value is None:
        return None
    return value.value.upper() if isinstance(value, WorkerState) else value.upper()


def render_buffer(
    session: ChatSession,
    prepared: PreparedPrompt | None,
) -> SheetDocument:
    """Return context capacity, composition, and history details."""
    document = SheetDocument()
    _section(document, "CAPACITY")
    _field(
        document,
        "CONTEXT WINDOW",
        f"{session.generation.max_prompt_tokens:,} TOK",
        abbreviate=False,
    )
    _field(
        document,
        "USED",
        None
        if prepared is None
        else (
            f"{prepared.prompt_tokens:,} TOK\n"
            f"{100 * prepared.prompt_tokens / session.generation.max_prompt_tokens:.1f}%"
        ),
    )

    _section(document, "COMPOSITION")
    _field(
        document,
        "SYSTEM",
        None if prepared is None else f"{prepared.system_tokens:,} TOK",
    )
    _field(
        document,
        "HISTORY",
        None if prepared is None else f"{prepared.history_tokens:,} TOK",
    )
    _field(
        document,
        "INPUT",
        None if prepared is None else f"{prepared.input_tokens:,} TOK",
    )

    _section(document, "HISTORY")
    _field(
        document,
        "TURNS",
        (
            f"— / {session.assistant.context_messages:,}"
            if prepared is None
            else f"{prepared.active_turns:,} / {session.assistant.context_messages:,}"
        ),
    )
    _field(
        document,
        "EVICTED",
        None
        if prepared is None
        else (
            f"{prepared.dropped_messages:,} TURNS\n"
            f"{prepared.evicted_tokens:,} TOK"
        ),
    )
    references = () if prepared is None else prepared.active_references
    if references and references[-1].role == "user":
        references = references[:-1]
    _field(
        document,
        "ACTIVE REF RANGE",
        _buffer_active_range(session, references),
    )
    return document


def _buffer_active_range(
    session: ChatSession,
    references: tuple[ChatBubble, ...],
) -> str | None:
    """Return the first and last active BUFFER references."""
    if not references:
        return None
    first = _buffer_reference_name(session, references[0])
    last = _buffer_reference_name(session, references[-1])
    return first if len(references) == 1 else f"{first}\n{last}"


def _buffer_reference_name(session: ChatSession, reference: ChatBubble) -> str:
    """Return one stable BUFFER reference identity."""
    name = "OP" if reference.role == "user" else session.assistant.target_name
    return f"@{name.upper()}:{reference.index:02d}"


def _byte_property(name: str, value: object) -> bool:
    """Return true when one integer device property reports bytes."""
    normalized = name.casefold()
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (
            normalized.endswith(("_length", "_size", "_memory"))
            or "working_set_size" in normalized
        )
    )


def _ordered_device_properties(
    properties: tuple[tuple[str, object], ...],
) -> tuple[tuple[str, object], ...]:
    """Return device properties in the stable sheet order."""
    rank = {name: index for index, name in enumerate(_PROBE_PROPERTY_ORDER)}
    return tuple(
        sorted(
            (item for item in properties if item[0].casefold() != "resource_limit"),
            key=lambda item: (rank.get(item[0], len(rank)), item[0]),
        )
    )


def _format_device_property(name: str, value: object) -> object:
    """Return one device property with its applicable unit."""
    if _byte_property(name, value):
        return _format_bytes(value)
    if (
        name.casefold() == "resource_limit"
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
        return f"{value:,} BUFFERS"
    return value


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


def _section(document: SheetDocument, name: str) -> None:
    """Append one specification section heading."""
    if document:
        document.append_line()
    document.append_line(Text(name, style="bold white"))


def _field(
    document: SheetDocument,
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
    if (
        ("token" in field_name.split("_") or "tokens" in field_name.split("_"))
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
        displayed = f"{value:,} TOK"
    document.append_inset(
        Text(label, style=_FIELD_NAME),
        Text(displayed, style=value_style),
    )


def _prompt_block_field(
    document: SheetDocument,
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


def _note(document: SheetDocument, value: str) -> None:
    """Append one metadata note."""
    document.append_line(Text(value, style=_DESCRIPTION))


def _display(value: object) -> str:
    """Return a stable display value for one specification field."""
    if value is None or value == "" or value == ():
        return "—"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (tuple, list)):
        return ", ".join(_display(item) for item in value)
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _flatten_training_policy(
    values: Mapping[str, Any],
    prefix: str = "",
) -> tuple[tuple[str, object], ...]:
    """Flatten a recorded policy into stable display fields."""
    if not prefix:
        priority = {"fine_tune_type": 0, "optimizer": 1, "lora": 3}
        keys = sorted(values, key=lambda key: (priority.get(key, 2), key))
    elif prefix == "lora":
        priority = {"rank": 0, "scale": 1}
        keys = sorted(values, key=lambda key: (priority.get(key, 2), key))
    else:
        keys = sorted(values)
    result: list[tuple[str, object]] = []
    for key in keys:
        value = values[key]
        name = (
            "fine-tune type"
            if not prefix and key == "fine_tune_type"
            else f"{prefix} {key}".strip()
        )
        if isinstance(value, Mapping):
            result.extend(_flatten_training_policy(value, name))
        else:
            result.append((name, value))
    return tuple(result)


def _split_counts(assistant: Assistant) -> str | None:
    """Return verified dataset counts in pipeline order."""
    if assistant.dataset is None:
        return None
    return "    ".join(
        f"{split.value.upper()} {assistant.counts.get(split, 0):,}"
        for split in (Split.TRAIN, Split.VALID, Split.TEST)
    )
