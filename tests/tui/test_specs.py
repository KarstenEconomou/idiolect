"""Test model specification rendering."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

from idiolect.chat.discovery import Assistant
from idiolect.chat.state import ChatSession, ChatTurn, TurnTelemetry, prepare_prompt
from idiolect.chat.storage import SavedChat
from idiolect.chat.worker import LoadProbe, RuntimeProbe, WorkerState
from idiolect.config import ChatConfig, GenerationConfig, TrainDataConfig
from idiolect.data.local import BuildResult
from idiolect.model import ModelSpec
from idiolect.train.base import LoadedRun
from idiolect.tui.specs import render_buffer, render_probe, render_specs
from idiolect.types import DatasetId, DatasetRef, PersonId, RunId, RunRef, Split

_NOW = datetime(2026, 8, 22, 12, tzinfo=UTC)
_RUN_ID = "a" * 64
_DATASET_ID = "d" * 64
_ADAPTER_DIGEST = "b" * 64
_TRACE_PARENT_ID = "p" * 64


def test_probe_shows_runtime_load_and_latest_generation_telemetry() -> None:
    """Check probe measurements without context, model policy, or lineage."""
    document = render_probe(
        RuntimeProbe(
            "0.32.1",
            "0.31.3",
            "Device(gpu, 0)",
            "arm64",
            (
                ("architecture", "applegpu_g16g"),
                ("device_name", "Apple M4 Max"),
                ("max_buffer_size", 5 * 1024**3),
                ("max_recommended_working_set_size", 4 * 1024**3),
                ("memory_size", 16 * 1024**3),
                ("active_memory", 3 * 1024**3),
                ("cache_memory", 1 * 1024**3),
                ("resource_limit", 499_000),
                ("unified_memory", True),
            ),
        ),
        LoadProbe("a" * 64, 8 * 1024**3, 64 * 1024**2, 2.3456),
        TurnTelemetry(
            512,
            64,
            prompt_throughput=120.5,
            generation_throughput=12.3,
            time_to_first_token=0.42,
            generation_time=5.25,
            peak_memory=3.75,
        ),
        WorkerState.READY,
    )

    assert "RUNTIME\nSTATE\n READY\nMLX\n 0.32.1\nMLX-LM\n 0.31.3\n" in document.plain
    assert "STACK\n" not in document.plain
    assert "MLX VERSION\n" not in document.plain
    assert "MLX-LM VERSION\n" not in document.plain
    assert document.plain.count("DEVICE\n") == 1
    assert "DEVICE\nTYPE\n GPU\nNAME\n Apple M4 Max\nARCHITECTURE\n" in document.plain
    assert "HOST ARCHITECTURE\n arm64\n" in document.plain
    assert "ARCHITECTURE\n applegpu_g16g\n" in document.plain
    assert "HOST\n" not in document.plain
    assert "MAX BUFFER SIZE\n 5.00 GiB\n" in document.plain
    assert "MAX BUFFER LENGTH\n" not in document.plain
    assert "REC WORKING SET\n 4.00 GiB\n" in document.plain
    assert "WORKING SET LIMIT\n" not in document.plain
    assert "MEMORY\n 16.00 GiB\n" in document.plain
    assert "ACTIVE MEMORY\n 3.00 GiB\n" in document.plain
    assert "CACHE MEMORY\n 1.00 GiB\n" in document.plain
    assert "RESOURCE LIMIT\n" not in document.plain
    assert "MODEL\n" in document.plain
    assert "MODEL\nSIZE\n 8.00 GiB\nADAPTER SIZE\n 64.00 MiB\n" in document.plain
    assert "DIGEST\n" not in document.plain
    assert "MODEL SIZE\n" not in document.plain
    assert "LOAD TIME\n 2.346 S\n" in document.plain
    assert "TELEMETRY\n" in document.plain
    assert "OUTPUT\n 64 TOK\n" in document.plain
    assert "PREFILL THROUGHPUT\n 120.5 TOK/S\n" in document.plain
    assert "DECODE THROUGHPUT\n 12.3 TOK/S\n" in document.plain
    assert "TIME TO FIRST TOKEN\n 0.420 S\n" in document.plain
    assert "INFERENCE LATENCY\n 5.250 S\n" in document.plain
    assert "PEAK MEMORY\n 3.75 GB\n" in document.plain
    assert "PROMPT TOKENS\n" not in document.plain
    assert "UTILIZATION\n" not in document.plain
    assert "RESIDENT\n" not in document.plain
    assert "IDENTITY\n" not in document.plain
    assert "GENERATION\n" not in document.plain
    assert "LINEAGE\n" not in document.plain
    assert "FIDELITY\n" not in document.plain


def test_probe_omits_structurally_absent_device_and_base_adapter() -> None:
    """Check that a base load omits sections and fields that do not apply."""
    document = render_probe(None, LoadProbe("a" * 64, 512, None, 0.5), None)

    assert "SIZE\n 512 B\n" in document.plain
    assert "ADAPTER SIZE\n" not in document.plain
    assert "DEVICE\n" not in document.plain
    assert "TELEMETRY\nOUTPUT\n —\n" in document.plain


def test_buffer_shows_capacity_composition_and_active_history_range(
    tmp_path: Path,
) -> None:
    """Check BUFFER reports token use, composition, eviction, and active range."""
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
            ChatTurn(
                "assistant",
                "first\n[new message]\nsecond",
                telemetry=TurnTelemetry(2, 2),
            ),
            ChatTurn("user", "active question"),
        ),
    )
    prepared = replace(
        prepare_prompt(session, lambda _value: 25, 0),
        dropped_messages=4,
        system_tokens=5,
        history_tokens=17,
        input_tokens=3,
        evicted_tokens=927,
    )

    document = render_buffer(session, prepared)

    assert document.plain.startswith(
        "CAPACITY\n"
        "CONTEXT WINDOW\n 100 TOK\n"
        "USED\n 25 TOK\n 25.0%\n\n"
        "COMPOSITION\n"
        "SYSTEM\n 5 TOK\n"
        "HISTORY\n 17 TOK\n"
        "INPUT\n 3 TOK\n\n"
        "HISTORY\n"
        "TURNS\n 3 / 32\n"
        "EVICTED\n 4 TURNS\n 927 TOK\n"
        "ACTIVE REF RANGE\n @OP:00\n @DIXIE:02\n"
    )
    assert "DIGEST\n" not in document.plain
    assert "RESIDENT\n" not in document.plain
    assert "@DIXIE:01\n" not in document.plain
    assert "@OP:03\n" not in document.plain
    assert "active question" not in document.plain

    empty = render_buffer(session, None)
    assert "CAPACITY\nCONTEXT WINDOW\n 100 TOK\nUSED\n —\n" in empty.plain
    assert "COMPOSITION\nSYSTEM\n —\nHISTORY\n —\nINPUT\n —\n" in empty.plain
    assert (
        "HISTORY\nTURNS\n — / 32\nEVICTED\n —\nACTIVE REF RANGE\n —\n"
        in empty.plain
    )


def test_construct_specs_show_verified_lineage_and_no_invented_evaluation(
    tmp_path: Path,
) -> None:
    """Check complete construct metadata without private artifact discovery."""
    assistant = _construct(tmp_path)

    document = render_specs(
        assistant,
        GenerationConfig(backend="mlx-lm", max_prompt_tokens=1920),
        "CONSTRUCT",
        chat=ChatConfig(seed=101, participant_name="person_01"),
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
    assert "CACHE\n" not in document.plain
    assert "RUN\nID\n" in document.plain
    assert "DATASET\nID\n" in document.plain
    assert "ADAPTER\nPATH\n" in document.plain
    assert "TRAIN 90    VALID 10    TEST 5" in document.plain
    assert "TRAINING\nSEED\n 7\nMAX SEQUENCE TOKENS\n 2,048 TOK\n" in document.plain
    assert "FINE-TUNE TYPE\n lora\nOPTIMIZER\n adamw\n" in document.plain
    assert document.plain.index("OPTIMIZER\n") < document.plain.index("LORA RANK\n")
    assert document.plain.index("LORA RANK\n") < document.plain.index(
        "LORA SCALE\n"
    )
    assert "TRAINING POLICY" not in document.plain
    assert "RUN ID\n" not in document.plain
    assert "DATASET ID\n" not in document.plain
    assert "ADAPTER PATH\n" not in document.plain
    assert "LORA RANK" in document.plain
    assert "PARTICIPANT\n person_01\n" in document.plain
    assert "GENERATION\nBACKEND\n MLX-LM\nSEED\n 101\n" in document.plain
    section_positions = [
        document.plain.index(f"{section}\n")
        for section in (
            "IDENTITY",
            "MODEL",
            "RUN",
            "DATASET",
            "ADAPTER",
            "TRAINING",
            "CONVERSATION",
            "PROMPT",
            "COMPLETION",
            "GENERATION",
            "SAMPLING",
            "REPETITION",
            "FIDELITY",
        )
    ]
    assert section_positions == sorted(section_positions)
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
    assert "PATH\n" not in document.plain.split("\nTRACE\n", 1)[1]
    assert "PARENT\n" in document.plain
    assert "PARENT ID\n" not in document.plain
    trace_section = document.plain.index("\nTRACE\n")
    assert document.plain.index("REPETITION\n") < trace_section
    assert trace_section < document.plain.index("FIDELITY\n")
    console = Console(width=36, color_system=None)
    model_digest = assistant.model_digest
    assert model_digest is not None
    with console.capture() as capture:
        console.print(document)
    rendered = capture.get().splitlines()
    trace_start = rendered.index("TRACE")
    for label, next_label, value, search_start in (
        ("DIGEST", "TRUST REMOTE CODE", model_digest.upper(), 0),
        ("ID", "PARENT", trace.id.upper(), trace_start),
        ("PARENT", "CREATED", _TRACE_PARENT_ID.upper(), trace_start),
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
            backend="mlx-lm",
            max_prompt_tokens=1920,
            repetition_penalty=1.1,
            repetition_context_size=20,
        ),
        "BASE",
        chat=ChatConfig(seed=101, participant_name="person_01"),
    )

    assert (
        "CONVERSATION\nFORMAT\n chat-template\nTURN CAPACITY\n 32\n"
        "PARTICIPANT\n person_01\n" in document.plain
    )
    assert "DIGEST\n —\n" in document.plain
    assert "LINEAGE\n" not in document.plain
    assert "TRACE\n" not in document.plain
    assert "GENERATION\nBACKEND\n MLX-LM\nSEED\n 101\n" in document.plain
    assert "SYSTEM\n First line.\n \n Second line.\n" in document.plain
    assert (
        "PROMPT\nROLE\n user\nPREFIX\n —\nSUFFIX\n \\n\nLIMIT\n 1,920 TOK\n"
        in document.plain
    )
    assert "COMPLETION\nROLE\n assistant\n" in document.plain
    assert "PREFIX\n <assistant>\\n\nSUFFIX\n —\nLIMIT\n 128 TOK\n" in document.plain
    assert "SAMPLING\nTEMPERATURE\n 0.7\nTOP-P\n 0.8\n" in document.plain
    assert "TOP-K\n 20\nMIN-P\n 0\nMIN-P FLOOR\n 1 TOK\n" in document.plain
    assert "REPETITION\nPENALTY\n 1.1\nWINDOW\n 20 TOK\n" in document.plain
    assert "FIDELITY\n" in document.plain
    offset = " "
    assert (
        f"SYSTEM\n{offset}First line.\n{offset}\n{offset}Second line." in document.plain
    )
    assert "↵" not in document.plain
    assert "CONTEXT TURNS" not in document.plain
    assert "EVALUATION" not in document.plain
    assert "GENERATION POLICY" not in document.plain
    assert document.plain.index("COMPLETION\n") < document.plain.index("GENERATION\n")
    assert document.plain.index("GENERATION\n") < document.plain.index("SAMPLING\n")
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
    assert (
        "COMPLETION\nPREFIX\n <assistant>\\n\nSUFFIX\n —\nLIMIT\n 128 TOK\n" in policy
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
        {
            "fine_tune_type": "lora",
            "optimizer": "adamw",
            "lora": {"rank": 8, "scale": 16},
        },
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
