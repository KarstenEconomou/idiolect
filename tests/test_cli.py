"""Test the Signal command-line workflow."""

import io
import json
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import idiolect.chat.cli
import idiolect.cli
import idiolect.eval.cli
import idiolect.inference.cli
import idiolect.ingest.cli
import idiolect.train.cli
from idiolect.cli import main
from idiolect.inference.base import Prediction
from idiolect.types import EvaluationId, EvaluationRef, RunId, RunRef, TrainResult


def test_root_help_lists_only_canonical_product_commands(capsys) -> None:
    """Check that removed root commands cannot appear as supported syntax."""
    with pytest.raises(SystemExit, match="0"):
        main(("--help",))
    output = capsys.readouterr().out
    assert "{signal,data,train,infer,eval,chat,config}" in output
    assert "inference" not in output


@pytest.mark.parametrize(
    "arguments",
    (
        ("inference",),
        ("data", "build", "NAME", "--self"),
        ("chat", "run", "RUN", "DATASET"),
        ("infer", "RUN", "--base-of", "OTHER"),
    ),
)
def test_removed_syntax_fails(arguments: tuple[str, ...]) -> None:
    """Check that legacy product syntax has no compatibility dispatch."""
    with pytest.raises(SystemExit, match="2"):
        main(arguments)


def test_incomplete_modes_fail_before_configuration(capsys) -> None:
    """Check coupled chat and inference options at the CLI boundary."""
    assert main(("chat", "--dataset", "DATASET")) == 2
    assert main(("infer", "RUN", "--data", "DATASET")) == 2
    assert main(("eval", "policy", "dataset", "run")) == 2
    errors = capsys.readouterr().err
    assert "--run and --dataset" in errors
    assert "--data and --split" in errors
    assert "policy is not a valid artifact reference" in errors


@pytest.mark.parametrize(
    ("action", "expected"),
    (
        ("status", ("launchctl", "print", "gui/501/com.idiolect.collect")),
        ("stop", ("launchctl", "bootout", "gui/501/com.idiolect.collect")),
    ),
)
def test_collector_lifecycle_uses_launchctl_without_configuration(
    action: str, expected: tuple[str, ...], monkeypatch
) -> None:
    """Check the installed collector lifecycle subprocess boundary."""
    seen = []
    monkeypatch.setattr(idiolect.ingest.cli.sys, "platform", "darwin")
    monkeypatch.setattr(idiolect.ingest.cli.os, "getuid", lambda: 501)
    monkeypatch.setattr(
        idiolect.ingest.cli.subprocess,
        "run",
        lambda command, check: seen.append((command, check)),
    )

    assert main(("-c", "missing", "signal", action)) == 0
    assert seen == [(expected, True)]


def test_import_and_stats_use_configured_store(
    local_config: Path,
    signal_events: Path,
    configured_signal_environment,
    capsys,
) -> None:
    """Check the local file harvest workflow."""
    import_code = main(
        ("--config", str(local_config), "signal", "import", str(signal_events))
    )
    import_output = capsys.readouterr()
    reindex_code = main(("--config", str(local_config), "signal", "reindex"))
    reindex_output = capsys.readouterr()
    people_code = main(("--config", str(local_config), "data", "people"))
    people_output = capsys.readouterr()
    build_code = main(
        (
            "--config",
            str(local_config),
            "data",
            "build",
            "DIXIE",
        )
    )
    build_output = capsys.readouterr()
    stats_code = main(("--config", str(local_config), "signal", "stats"))
    stats_output = capsys.readouterr()

    assert import_code == 0
    assert (
        "received=6 stored=4 messages=3 reactions=1 skipped=2 duplicates=0"
        in import_output.out
    )
    assert import_output.err == ""
    assert reindex_code == 0
    assert "scanned=4 updated=4 messages=3 reactions=1 skipped=0" in reindex_output.out
    assert reindex_output.err == ""
    assert people_code == 0
    assert "\tself\t" in people_output.out
    assert people_output.err == ""
    assert build_code == 0
    assert "train=1 valid=0 test=0" in build_output.out
    assert build_output.err == ""
    assert stats_code == 0
    assert "events=4 messages=2 reactions=1" in stats_output.out
    assert stats_output.err == ""


def test_missing_config_has_actionable_error(tmp_path: Path, capsys) -> None:
    """Check the error for a missing settings file."""
    code = main(("--config", str(tmp_path / "missing.toml"), "signal", "stats"))

    output = capsys.readouterr()
    assert code == 2
    assert "Configuration file does not exist" in output.err


def test_message_processing_requires_a_chat_whitelist(
    tmp_path: Path,
    signal_events: Path,
    monkeypatch,
    capsys,
) -> None:
    """Check that message input cannot run without a whitelist."""
    for name in (
        "IDIOLECT_SIGNAL_ACCOUNT",
        "IDIOLECT_SIGNAL_BIN",
        "IDIOLECT_SIGNAL_DATA_DIR",
        "IDIOLECT_SIGNAL_CHATS",
    ):
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "empty.toml"
    config.write_text("[signal]\n", encoding="utf-8")

    code = main(("--config", str(config), "signal", "import", str(signal_events)))

    output = capsys.readouterr()
    assert code == 2
    assert "Set IDIOLECT_SIGNAL_CHATS" in output.err


def test_train_command_uses_fixed_dataset_and_config(
    local_config: Path,
    signal_events: Path,
    configured_signal_environment,
    monkeypatch,
    capsys,
) -> None:
    """Check that the CLI passes one verified dataset to the trainer."""
    assert (
        main(("--config", str(local_config), "signal", "import", str(signal_events)))
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            (
                "--config",
                str(local_config),
                "data",
                "build",
                    "DIXIE",
            )
        )
        == 0
    )
    capsys.readouterr()
    dataset_path = next((local_config.parent / "data").iterdir())
    seen = []

    class FakeTrainer:
        """Return one fixed run without model work."""

        def train(self, dataset, config) -> TrainResult:
            """Record the verified dataset and settings."""
            seen.append((dataset, config))
            run = RunRef(
                RunId("run-id"),
                dataset.id,
                local_config.parent / "runs" / "run-id",
                dataset.created_at,
            )
            return TrainResult((run,))

    monkeypatch.setattr(idiolect.train.cli, "MlxTrainer", FakeTrainer)
    monkeypatch.setattr(idiolect.train.cli, "keep_awake", nullcontext)

    code = main(("--config", str(local_config), "train", str(dataset_path)))

    output = capsys.readouterr()
    assert code == 0
    assert len(seen) == 1
    assert seen[0][0].path == dataset_path
    assert "run=run-id" in output.out
    assert output.err == ""


def test_inference_text_reads_stdin_and_writes_json_lines(
    local_config: Path,
    configured_signal_environment,
    monkeypatch,
    capsys,
) -> None:
    """Check that one-shot inference keeps prompt text out of arguments."""
    seen = []

    class FakeInferencer:
        """Return one prediction without model work."""

        def __init__(self, backend) -> None:
            """Accept the configured backend boundary."""

        def validate(self, config) -> None:
            """Accept the complete inference policy."""

        def text(self, target, prompt, config) -> tuple[Prediction, ...]:
            """Record the prompt and return one fixed prediction."""
            seen.append((target, prompt, config))
            return (Prediction("example", 0, 101, 202, "reply", "stop", 8, 2),)

    monkeypatch.setattr(idiolect.inference.cli, "configured_target", lambda config: "base")
    monkeypatch.setattr(idiolect.inference.cli, "MlxBackend", object)
    monkeypatch.setattr(idiolect.inference.cli, "LocalInferencer", FakeInferencer)
    monkeypatch.setattr(idiolect.inference.cli.sys, "stdin", io.StringIO("private prompt"))

    code = main(("--config", str(local_config), "infer", "--base"))

    output = capsys.readouterr()
    assert code == 0
    assert seen[0][1] == "private prompt"
    assert json.loads(output.out) == {
        "example_id": "example",
        "index": 0,
        "seed": 101,
        "rng_seed": 202,
        "text": "reply",
        "finish_reason": "stop",
        "prompt_tokens": 8,
        "generated_tokens": 2,
    }
    assert output.err == ""


def test_inference_text_reports_invalid_utf8_before_model_access(
    local_config: Path,
    configured_signal_environment,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Check that an invalid prompt file returns one controlled error."""
    prompt = tmp_path / "invalid.txt"
    prompt.write_bytes(b"\xff")
    called = False

    def target(config):
        nonlocal called
        called = True
        return "unused"

    class FakeInferencer:
        """Validate configuration without model packages."""

        def __init__(self, backend) -> None:
            """Accept one fake backend."""

        def validate(self, config) -> None:
            """Accept one complete inference policy."""

    monkeypatch.setattr(idiolect.inference.cli, "configured_target", target)
    monkeypatch.setattr(idiolect.inference.cli, "MlxBackend", object)
    monkeypatch.setattr(idiolect.inference.cli, "LocalInferencer", FakeInferencer)

    code = main(
        ("--config", str(local_config), "infer", "--base", str(prompt))
    )

    output = capsys.readouterr()
    assert code == 2
    assert "Cannot read inference prompt" in output.err
    assert called is False


def test_eval_policy_uses_every_supplied_run(
    local_config: Path,
    configured_signal_environment,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Check that the CLI passes the complete run set to evaluation."""
    seen = []
    evaluation_path = tmp_path / "eval" / "evaluation-id"
    current_policy = idiolect.eval.cli.training_policy(
        idiolect.eval.cli.load_config(local_config).train
    )
    recorded_policy = json.loads(json.dumps(current_policy))

    class FakeEvaluator:
        """Return one fixed evaluation without model work."""

        def __init__(self, scorer, inferencer) -> None:
            """Accept local backend boundaries."""

        def evaluate(self, runs, dataset, config, inference) -> EvaluationRef:
            """Record the policy inputs and return one result."""
            seen.append((runs, dataset, config, inference))
            return EvaluationRef(
                EvaluationId("evaluation-id"),
                evaluation_path,
                datetime(2026, 8, 20, tzinfo=UTC),
                True,
            )

    monkeypatch.setattr(
        idiolect.eval.cli,
        "load_dataset",
        lambda path: SimpleNamespace(dataset="fixed-dataset"),
    )
    monkeypatch.setattr(
        idiolect.eval.cli,
        "load_run",
        lambda path: SimpleNamespace(name=path.name, policy=recorded_policy),
    )
    monkeypatch.setattr(idiolect.eval.cli, "MlxScoreBackend", object)
    monkeypatch.setattr(idiolect.eval.cli, "MlxBackend", object)
    monkeypatch.setattr(idiolect.eval.cli, "LocalInferencer", lambda backend: "inferencer")
    monkeypatch.setattr(idiolect.eval.cli, "LocalEvaluator", FakeEvaluator)
    monkeypatch.setattr(idiolect.eval.cli, "keep_awake", nullcontext)

    code = main(
        (
            "--config",
            str(local_config),
            "eval",
            "dataset",
            "run-17",
            "run-42",
        )
    )

    output = capsys.readouterr()
    assert code == 0
    assert tuple(value.name for value in seen[0][0]) == ("run-17", "run-42")
    assert seen[0][1] == "fixed-dataset"
    assert "state=eligible" in output.out
    assert output.err == ""


def test_chat_opens_chooser_without_model_or_private_data(
    tmp_path: Path,
    configured_signal_environment,
    monkeypatch,
    capsys,
) -> None:
    """Check that landing discovery does not load MLX, Signal, or DuckDB."""
    config_path = tmp_path / "chat.toml"
    config_path.write_text(
        f"""
[data]
output = "{(tmp_path / "data").as_posix()}"

[train]
base_model = "mlx-community/Qwen3-8B-4bit"
model_source = "hub"
model_revision = "fixed"
model_cache = "{(tmp_path / "models").as_posix()}"
output = "{(tmp_path / "runs").as_posix()}"

[train.data]
format = "chat"
system_prompt = ""
prompt_role = "user"
completion_role = "assistant"
prompt_prefix = ""
prompt_suffix = ""
completion_prefix = ""
completion_suffix = ""

[inference]
backend = "mlx-lm"
max_prompt_tokens = 1920
max_tokens = 128
temperature = 0.7
top_p = 0.8
top_k = 20
min_p = 0.0
min_tokens_to_keep = 1
repetition_penalty = 1.0
repetition_context_size = 20

[chat]
output = "{(tmp_path / "chat").as_posix()}"
seed = 101
participant_name = "person_01"
context_policy = "recorded-window-drop-oldest"
history = "explicit-save"
default_model = "train-base"
default_name = "DIXIE"
default_context_messages = 32
default_system_prompt = "Speak with terse technical precision."
""",
        encoding="utf-8",
    )
    seen = []
    monkeypatch.setattr(
        idiolect.chat.cli,
        "run_chat_app",
        lambda *args, **kwargs: seen.append((args, kwargs)),
    )

    code = main(("--config", str(config_path), "chat"))

    output = capsys.readouterr()
    assert code == 0
    assert len(seen) == 1
    assistants = seen[0][1]["assistants"]
    assert len(assistants) == 1
    assert assistants[0].label == "IDIOLECT // DIXIE::BASE [Qwen3-8B-4bit]"
    assert assistants[0].assistant.run is None
    assert output.out == ""
    assert output.err == ""
