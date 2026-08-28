"""Test model specification rendering."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from rich.color import Color
from rich.console import Console

from idiolect.chat.discovery import Assistant
from idiolect.chat.state import ChatSession, ChatTurn, TurnTelemetry, prepare_prompt
from idiolect.chat.storage import SavedChat
from idiolect.chat.worker import LoadProbe, RuntimeProbe
from idiolect.config import ChatConfig, GenerationConfig, TrainDataConfig
from idiolect.data.local import BuildResult
from idiolect.model import ModelSpec
from idiolect.train.base import LoadedRun
from idiolect.tui.specs import (
    HalfCellScrollBarRender,
    render_buffer,
    render_probe,
    render_specs,
)
from idiolect.types import DatasetId, DatasetRef, PersonId, RunId, RunRef, Split

_NOW = datetime(2026, 8, 22, 12, tzinfo=UTC)
_RUN_ID = "a" * 64
_DATASET_ID = "d" * 64
_ADAPTER_DIGEST = "b" * 64
_TRACE_PARENT_ID = "p" * 64


def test_probe_shows_only_live_hardware_runtime_and_load_details() -> None:
    """Check probe values without model policy or lineage content."""
    document = render_probe(
        RuntimeProbe(
            "0.32.1",
            "0.31.3",
            "Device(gpu, 0)",
            "arm64",
            (
                ("architecture", "applegpu_g16g"),
                ("device_name", "Apple M4 Max"),
                ("max_buffer_length", 5 * 1024**3),
                ("max_recommended_working_set_size", 4 * 1024**3),
                ("memory_size", 16 * 1024**3),
                ("resource_limit", 499_000),
                ("unified_memory", True),
            ),
        ),
        LoadProbe("a" * 64, 8 * 1024**3, 64 * 1024**2, 2.3456),
    )

    assert "STACK\n" in document.plain
    assert "MLX VERSION\n 0.32.1\n" in document.plain
    assert "MLX-LM VERSION\n 0.31.3\n" in document.plain
    assert "DEVICE TYPE\n GPU\n" in document.plain
    assert document.plain.count("DEVICE\n") == 1
    assert "DEVICE\nNAME\n Apple M4 Max\nARCHITECTURE\n" in document.plain
    assert "HOST ARCHITECTURE\n arm64\n" in document.plain
    assert "ARCHITECTURE\n applegpu_g16g\n" in document.plain
    assert "HOST\n" not in document.plain
    assert "MAX BUFFER LENGTH\n 5.00 GiB\n" in document.plain
    assert "WORKING SET LIMIT\n 4.00 GiB\n" in document.plain
    assert "MEMORY\n 16.00 GiB\n" in document.plain
    assert "RESOURCE LIMIT\n 499,000 BUFFERS\n" in document.plain
    assert "PAYLOAD\n" in document.plain
    assert "MODEL SIZE\n 8.00 GiB\n" in document.plain
    assert "ADAPTER SIZE\n 64.00 MiB\n" in document.plain
    assert "LOAD TIME\n 2.346 S\n" in document.plain
    assert ("a" * 64).upper() in document.plain
    assert "IDENTITY\n" not in document.plain
    assert "GENERATION\n" not in document.plain
    assert "LINEAGE\n" not in document.plain
    assert "FIDELITY\n" not in document.plain


def test_probe_omits_structurally_absent_device_and_base_adapter() -> None:
    """Check that a base load omits sections and fields that do not apply."""
    document = render_probe(None, LoadProbe("a" * 64, 512, None, 0.5))

    assert "MODEL SIZE\n 512 B\n" in document.plain
    assert "ADAPTER SIZE\n" not in document.plain
    assert "DEVICE\n" not in document.plain


def test_buffer_shows_context_measurements_and_active_references(tmp_path: Path) -> None:
    """Check BUFFER reports fitted context and complete active bubble text."""
    assistant = _base(tmp_path)
    assert assistant.base_data is not None
    assistant = Assistant(
        assistant.name,
        assistant.target_name,
        assistant.model_basename,
        assistant.run,
        assistant.dataset,
        assistant.context_messages,
        assistant.base_model,
        TrainDataConfig(
            format="chat",
            system_prompt=assistant.base_data.system_prompt,
            prompt_role=assistant.base_data.prompt_role,
            completion_role=assistant.base_data.completion_role,
        ),
        assistant.base_model_digest,
    )
    session = ChatSession(
        assistant,
        ChatConfig(
            context_policy="recorded-window-drop-oldest",
            participant_name="person_01",
        ),
        GenerationConfig(max_prompt_tokens=100),
        (
            ChatTurn("user", "old"),
            ChatTurn("assistant", "first\n[new message]\nsecond", telemetry=TurnTelemetry(2, 2)),
            ChatTurn("user", "active question"),
        ),
    )
    prepared = prepare_prompt(session, lambda _value: 25, 0)

    document = render_buffer(session, prepared)

    assert "PROMPT\n" in document.plain
    assert "POLICY\n" not in document.plain
    assert "PROMPT\nDIGEST\n" in document.plain
    assert "TOKENS\n" in document.plain
    assert "TOKENS\nPROMPT\n 25 TOK\n" in document.plain
    assert "LIMIT\n 100 TOK\n" in document.plain
    assert "UTILIZATION\n 25.0%\n" in document.plain
    assert "TURNS\n" in document.plain
    assert "TURNS\nACTIVE\n 3\n" in document.plain
    assert "CAPACITY\n 32\n" in document.plain
    assert "EVICTED\n 0\n" in document.plain
    assert "RESIDENT\n" in document.plain
    assert "HEAD\n @OP:03\n" in document.plain
    assert "REFS\n @OP:00\n @DIXIE:01\n @DIXIE:02\n @OP:03\n" in document.plain
    assert "@OP:00\n" in document.plain
    assert "@DIXIE:01\n" in document.plain
    assert "@DIXIE:02\n" in document.plain
    assert "@OP:03\n" in document.plain
    assert "active question" not in document.plain

    empty = render_buffer(session, None)
    assert "TOKENS\nPROMPT\n —\nLIMIT\n 100 TOK\nUTILIZATION\n —\n" in empty.plain
    assert "TURNS\nACTIVE\n —\nCAPACITY\n 32\nEVICTED\n —\n" in empty.plain
    assert "RESIDENT\nHEAD\n —\nREFS\n —\n" in empty.plain
    assert "No references are resident." not in empty.plain


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

    assert _RUN_ID.upper() in document.plain
    assert _DATASET_ID.upper() in document.plain
    assert _ADAPTER_DIGEST.upper() in document.plain
    assert "CONSTRUCT" in document.plain
    assert "DIXIE::AAAAAAAA" in document.plain
    assert "CONSTRUCT\n DIXIE::AAAAAAAA" in document.plain
    assert "BASE" in document.plain
    assert document.plain.index("CONSTRUCT\n") < document.plain.index("BASE\n")
    assert document.plain.index("BASE\n") < document.plain.index("TYPE\n")
    assert "ENTRY" not in document.plain
    assert "MODEL\nNAME\n example/M\n" in document.plain
    assert "RUN\nID\n" in document.plain
    assert "DATASET\nID\n" in document.plain
    assert "ADAPTER\nPATH\n" in document.plain
    assert "TRAIN 90    VALID 10    TEST 5" in document.plain
    assert "TRAINING\nSEED\n 7\nMAX SEQUENCE TOKENS\n 2,048 TOK\n" in document.plain
    assert "TRAINING POLICY" not in document.plain
    assert "RUN ID\n" not in document.plain
    assert "DATASET ID\n" not in document.plain
    assert "ADAPTER PATH\n" not in document.plain
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
        (
            ChatTurn("env", "local"),
            ChatTurn("user", "hello"),
            ChatTurn("assistant", "hi"),
        ),
    )

    document = render_specs(
        assistant,
        trace.generation,
        "TRACE",
        trace,
    )

    assert "Night session" in document.plain
    assert "TRACE\nNAME\n Night session\n" in document.plain
    assert trace.id.upper() in document.plain
    assert _TRACE_PARENT_ID.upper() in document.plain
    assert _NOW.isoformat() in document.plain
    assert "NOT EVALUATED" in document.plain
    assert "TURNS\n 2\n" in document.plain
    console = Console(width=36, color_system=None)
    model_digest = assistant.model_digest
    assert model_digest is not None
    with console.capture() as capture:
        console.print(document)
    rendered = capture.get().splitlines()
    trace_start = rendered.index("TRACE")
    for label, next_label, value, search_start in (
        ("DIGEST", "TRUST REMOTE CODE", model_digest.upper(), 0),
        ("ID", "PATH", trace.id.upper(), trace_start),
        ("PATH", "PARENT ID", trace.path, trace_start),
    ):
        start = rendered.index(label, search_start) + 1
        end = next(
            index
            for index in range(start, len(rendered))
            if rendered[index].startswith(next_label)
        )
        value_lines = [line.rstrip() for line in rendered[start:end]]
        assert len(value_lines) > 1
        assert all(line.startswith(" ") for line in value_lines)
        assert "".join(line[1:] for line in value_lines) == str(value)


def test_specs_group_generation_policy_and_align_prompt_format_blocks(
    tmp_path: Path,
) -> None:
    """Check contextual policy labels and one prompt-format value offset."""
    document = render_specs(
        _base(tmp_path),
        GenerationConfig(
            max_prompt_tokens=1920,
            repetition_penalty=1.1,
            repetition_context_size=20,
        ),
        "BASE",
    )

    assert "CONVERSATION\nFORMAT\n chat-template\nTURN CAPACITY\n 32\n" in document.plain
    assert "DIGEST\n —\n" in document.plain
    assert "LINEAGE\n" not in document.plain
    assert "TRACE\n" not in document.plain
    assert "GENERATION\n" not in document.plain
    assert "BACKEND\n" not in document.plain
    assert "SYSTEM\n First line.\n \n Second line.\n" in document.plain
    assert "PROMPT\nROLE\n user\nPREFIX\n —\nSUFFIX\n \\n\nLIMIT\n 1,920 TOK\n" in document.plain
    assert "COMPLETION\nROLE\n assistant\n" in document.plain
    assert "PREFIX\n <assistant>\\n\nSUFFIX\n —\nLIMIT\n 128 TOK\n" in document.plain
    assert "SAMPLING\nTEMPERATURE\n 0.7\nTOP-P\n 0.8\n" in document.plain
    assert "TOP-K\n 20\nMIN-P\n 0\nMIN-P FLOOR\n 1 TOK\n" in document.plain
    assert "REPETITION\nPENALTY\n 1.1\nWINDOW\n 20 TOK\n" in document.plain
    assert "FIDELITY\n" in document.plain
    offset = " "
    assert (
        f"SYSTEM\n{offset}First line.\n{offset}\n{offset}Second line."
        in document.plain
    )
    assert "↵" not in document.plain
    assert "CONTEXT TURNS" not in document.plain
    assert "EVALUATION" not in document.plain
    assert "GENERATION POLICY" not in document.plain
    assert "GENERATION\n" not in document.plain
    assert "FIDELITY\n" in document.plain


def test_prompt_blocks_drop_trailing_system_blank(
    tmp_path: Path,
) -> None:
    """Check logical prompt lines preserve inset without a trailing blank."""
    assistant = _base(
        tmp_path,
        system_prompt="First line.\n\nSecond line.\n",
    )

    document = render_specs(assistant, GenerationConfig(), "BASE")
    system_block = document.plain.split("SYSTEM\n", 1)[1].split(
        "\nPROMPT\n",
        1,
    )[0]
    assert system_block == " First line.\n \n Second line.\n"


def test_completion_format_omits_structurally_invalid_system_and_roles(
    tmp_path: Path,
) -> None:
    """Check completion policies show only fields that can apply."""
    assistant = _base(tmp_path)
    assert assistant.base_data is not None
    assistant = replace(
        assistant,
        base_data=replace(
            assistant.base_data,
            format="completion",
            system_prompt="",
            prompt_role="",
            completion_role="",
        ),
    )

    document = render_specs(assistant, GenerationConfig(), "BASE")
    policy = document.plain.split("CONVERSATION\n", 1)[1].split("\nSAMPLING\n", 1)[0]

    assert "SYSTEM\n" not in policy
    assert "ROLE\n" not in policy
    assert "PROMPT\nPREFIX\n —\nSUFFIX\n \\n\nLIMIT\n 0 TOK\n" in policy
    assert "COMPLETION\nPREFIX\n <assistant>\\n\nSUFFIX\n —\nLIMIT\n 128 TOK\n" in policy


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
