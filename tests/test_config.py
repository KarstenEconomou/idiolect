"""Test local configuration behavior."""

from pathlib import Path

import pytest

from idiolect.config import ConfigError, load_config


def test_environment_replaces_sensitive_signal_values(local_config: Path) -> None:
    """Check that the environment can replace account settings."""
    config = load_config(
        local_config,
        {
            "IDIOLECT_SIGNAL_ACCOUNT": "+19999999999",
            "IDIOLECT_SIGNAL_BIN": "/safe/signal-cli",
            "IDIOLECT_SIGNAL_DATA_DIR": "/safe/data",
        },
    )

    assert config.signal.account == "+19999999999"
    assert config.signal.binary == "/safe/signal-cli"
    assert config.signal.data_dir == Path("/safe/data")
    assert config.store.database_path == local_config.parent / "test.duckdb"


def test_unknown_setting_is_rejected(tmp_path: Path) -> None:
    """Check that a setting typo cannot change collection silently."""
    path = tmp_path / "invalid.toml"
    path.write_text('[signal]\nchat = ["wrong-key"]\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown values: chat"):
        load_config(path, {})
