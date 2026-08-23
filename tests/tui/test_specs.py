"""Test model specification rendering."""

from datetime import UTC, datetime
from pathlib import Path

from rich.color import Color
from rich.console import Console

from idiolect.chat.discovery import Assistant
from idiolect.chat.storage import SavedChat
from idiolect.config import ChatConfig, GenerationConfig, TrainDataConfig
from idiolect.data.local import BuildResult
from idiolect.model import ModelSpec
from idiolect.train.base import LoadedRun
from idiolect.tui.specs import HalfCellScrollBarRender, render_specs
from idiolect.types import DatasetId, DatasetRef, PersonId, RunId, RunRef, Split

_NOW = datetime(2026, 8, 22, 12, tzinfo=UTC)
_RUN_ID = "a" * 64
_DATASET_ID = "d" * 64
_ADAPTER_DIGEST = "b" * 64
_TRACE_PARENT_ID = "p" * 64


def test_construct_specs_show_verified_lineage_and_no_invented_evaluation(
    tmp_path: Path,
) -> None:
    """Check complete construct metadata without private artifact discovery."""
    assistant = _construct(tmp_path)

    document = render_specs(
        assistant,
        GenerationConfig(backend="mlx-lm", max_prompt_tokens=1920),
        "CONSTRUCT",
    )

    assert _RUN_ID in document.plain
    assert _DATASET_ID in document.plain
    assert _ADAPTER_DIGEST in document.plain
    assert "CONSTRUCT" in document.plain
    assert "DIXIE::aaaaaaaa" in document.plain
    assert "CONSTRUCT\n DIXIE::aaaaaaaa" in document.plain
    assert "BASE" in document.plain
    assert "TRAIN 90    VALID 10    TEST 5" in document.plain
    assert "LORA RANK" in document.plain
    assert "NOT EVALUATED" in document.plain
    assert "SYNTHETIC // UI FIXTURE" not in document.plain


def test_trace_specs_add_snapshot_lineage_to_the_underlying_model(
    tmp_path: Path,
) -> None:
    """Check that TRACE details include the immutable snapshot identity."""
    assistant = _construct(tmp_path)
    trace = SavedChat(
        "c" * 64,
        tmp_path / ("c" * 64),
        _NOW,
        "Night session",
        _TRACE_PARENT_ID,
        assistant,
        ChatConfig(output=tmp_path),
        GenerationConfig(backend="mlx-lm", temperature=0.3),
        (),
    )

    document = render_specs(
        assistant,
        trace.generation,
        "TRACE",
        trace,
    )

    assert "Night session" in document.plain
    assert trace.id in document.plain
    assert _TRACE_PARENT_ID in document.plain
    assert _NOW.isoformat() in document.plain
    assert "NOT EVALUATED" in document.plain
    console = Console(width=36, color_system=None)
    with console.capture() as capture:
        console.print(document)
    rendered = capture.get().splitlines()
    for label, next_label, value in (
        ("MODEL DIGEST", "TRUST REMOTE CODE", assistant.model_digest),
        ("TRACE ID", "TRACE PATH", trace.id),
        ("TRACE PATH", "PARENT ID", trace.path),
    ):
        start = rendered.index(label) + 1
        end = next(
            index
            for index in range(start, len(rendered))
            if rendered[index].startswith(next_label)
        )
        value_lines = [line.rstrip() for line in rendered[start:end]]
        assert len(value_lines) > 1
        assert all(line.startswith(" ") for line in value_lines)
        assert "".join(line[1:] for line in value_lines) == str(value)


def test_specs_abbreviate_telemetry_and_align_prompt_format_blocks(
    tmp_path: Path,
) -> None:
    """Check compact labels and one fixed prompt-format value offset."""
    document = render_specs(
        _base(tmp_path),
        GenerationConfig(
            max_prompt_tokens=1920,
            repetition_penalty=1.1,
            repetition_context_size=20,
        ),
        "BASE",
    )

    assert "CTX MESSAGES" in document.plain
    assert "GEN POLICY" in document.plain
    assert "MAX PROMPT TOK" in document.plain
    assert "REP PENALTY" in document.plain
    assert "REP CTX SIZE" in document.plain
    assert "EVAL\n" in document.plain
    offset = " "
    assert (
        f"SYSTEM PROMPT\n{offset}First line.\n{offset}\n{offset}Second line."
        in document.plain
    )
    assert f"PROMPT PREFIX\n{offset}—" in document.plain
    assert f"PROMPT SUFFIX\n{offset}\\n" in document.plain
    assert f"COMPLETION PREFIX\n{offset}<assistant>\\n" in document.plain
    assert f"COMPLETION SUFFIX\n{offset}—" in document.plain
    assert "↵" not in document.plain
    assert "CONTEXT" not in document.plain
    assert "TOKENS" not in document.plain
    assert "REPETITION" not in document.plain
    assert "EVALUATION" not in document.plain
    assert "GENERATION" not in document.plain


def test_prompt_blocks_drop_trailing_system_blank(
    tmp_path: Path,
) -> None:
    """Check logical prompt lines preserve inset without a trailing blank."""
    assistant = _base(
        tmp_path,
        system_prompt="First line.\n\nSecond line.\n",
    )

    document = render_specs(assistant, GenerationConfig(), "BASE")
    system_block = document.plain.split("SYSTEM PROMPT\n", 1)[1].split(
        "PROMPT ROLE",
        1,
    )[0]
    assert system_block == " First line.\n \n Second line.\n"


def test_specs_scrollbar_uses_a_half_cell_thumb() -> None:
    """Check the narrow scrollbar renderer keeps one clickable cell."""
    rendered = HalfCellScrollBarRender.render_bar(
        size=12,
        virtual_size=48,
        window_size=12,
        bar_color=Color.parse("white"),
        back_color=Color.parse("black"),
    )

    thumb = [segment for segment in rendered.segments if segment.text == "▐"]
    assert thumb
    assert all(
        segment.style is not None
        and segment.style.meta.get("@mouse.down") == "grab"
        for segment in thumb
    )


def _base(
    tmp_path: Path,
    *,
    system_prompt: str = "First line.\n\nSecond line.",
) -> Assistant:
    return Assistant(
        "IDIOLECT // DIXIE::BASE [M]",
        "DIXIE",
        "M",
        None,
        None,
        32,
        ModelSpec("example/M", "hub", "revision-1", tmp_path / "cache", False),
        TrainDataConfig(
            format="chat-template",
            system_prompt=system_prompt,
            prompt_role="user",
            prompt_suffix="\n",
            completion_role="assistant",
            completion_prefix="<assistant>\n",
        ),
    )


def _construct(tmp_path: Path) -> Assistant:
    model = ModelSpec("example/M", "hub", "revision-1", tmp_path / "cache", False)
    data = TrainDataConfig(
        format="chat-template",
        system_prompt="Speak with terse technical precision.",
        prompt_role="user",
        completion_role="assistant",
    )
    dataset_ref = DatasetRef(
        DatasetId(_DATASET_ID),
        PersonId("person-1"),
        tmp_path / _DATASET_ID,
        _NOW,
    )
    run_ref = RunRef(
        RunId(_RUN_ID),
        dataset_ref.id,
        tmp_path / _RUN_ID,
        _NOW,
    )
    run = LoadedRun(
        run_ref,
        model,
        "m" * 64,
        data,
        tmp_path / "adapter.safetensors",
        _ADAPTER_DIGEST,
        {"optimizer": "adamw", "lora": {"rank": 8, "scale": 16}},
        7,
        2048,
    )
    dataset = BuildResult(
        dataset_ref,
        {Split.TRAIN: 90, Split.VALID: 10, Split.TEST: 5},
    )
    return Assistant(
        "IDIOLECT // DIXIE::aaaaaaaa [M]",
        "DIXIE",
        "M",
        run,
        dataset,
        32,
    )
