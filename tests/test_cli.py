"""Test the Signal command-line workflow."""

from pathlib import Path

from idiolect.cli import main


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
