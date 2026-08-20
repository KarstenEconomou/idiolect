"""Test the Signal command-line workflow."""

from pathlib import Path

import idiolect.cli
from idiolect.cli import main
from idiolect.types import RunId, RunRef, TrainResult


def test_import_and_stats_use_configured_store(
    local_config: Path,
    signal_events: Path,
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
            "--self",
            "--name",
            "Karsten",
        )
    )
    build_output = capsys.readouterr()
    stats_code = main(("--config", str(local_config), "signal", "stats"))
    stats_output = capsys.readouterr()

    assert import_code == 0
    assert "received=6 stored=4 messages=3 reactions=1 skipped=2 duplicates=0" in import_output.out
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


def test_message_processing_requires_a_chat_allowlist(
    tmp_path: Path,
    signal_events: Path,
    capsys,
) -> None:
    """Check that message input cannot run without an allowlist."""
    config = tmp_path / "empty.toml"
    config.write_text("[signal]\n", encoding="utf-8")

    code = main(("--config", str(config), "signal", "import", str(signal_events)))

    output = capsys.readouterr()
    assert code == 2
    assert "Set IDIOLECT_SIGNAL_CHATS" in output.err


def test_train_command_uses_fixed_dataset_and_config(
    local_config: Path,
    signal_events: Path,
    monkeypatch,
    capsys,
) -> None:
    """Check that the CLI passes one verified dataset to the trainer."""
    assert main(
        ("--config", str(local_config), "signal", "import", str(signal_events))
    ) == 0
    capsys.readouterr()
    assert main(
        (
            "--config",
            str(local_config),
            "data",
            "build",
            "--self",
            "--name",
            "Karsten",
        )
    ) == 0
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

    monkeypatch.setattr(idiolect.cli, "MlxTrainer", FakeTrainer)

    code = main(("--config", str(local_config), "train", str(dataset_path)))

    output = capsys.readouterr()
    assert code == 0
    assert len(seen) == 1
    assert seen[0][0].path == dataset_path
    assert "run=run-id" in output.out
    assert output.err == ""
