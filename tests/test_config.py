"""Test local configuration behavior."""

from pathlib import Path

import pytest

from idiolect.chat.runtime import ChatError, validate_chat_policy
from idiolect.config import ConfigError, load_config
from idiolect.train.mlx import training_policy


def test_environment_replaces_sensitive_signal_values(local_config: Path) -> None:
    """Check that the environment can replace account settings."""
    config = load_config(
        local_config,
        {
            "IDIOLECT_SIGNAL_ACCOUNT": "+19999999999",
            "IDIOLECT_SIGNAL_BIN": "/safe/signal-cli",
            "IDIOLECT_SIGNAL_DATA_DIR": "/safe/data",
            "IDIOLECT_SIGNAL_CHATS": '["env-group-one", "env-group-two"]',
        },
    )

    assert config.signal.account == "+19999999999"
    assert config.signal.binary == "/safe/signal-cli"
    assert config.signal.data_dir == Path("/safe/data")
    assert config.signal.chats == ("env-group-one", "env-group-two")
    assert config.store.database_path == local_config.parent / "test.duckdb"


@pytest.mark.parametrize(
    "value",
    (
        "not-json",
        '{"group": "not-a-list"}',
        '["valid", ""]',
        '["valid", 4]',
    ),
)
def test_signal_chat_environment_rejects_invalid_lists(
    local_config: Path,
    value: str,
) -> None:
    """Check that an invalid private whitelist stops configuration."""
    with pytest.raises(ConfigError, match="must be a JSON list"):
        load_config(local_config, {"IDIOLECT_SIGNAL_CHATS": value})


def test_signal_chat_environment_rejects_duplicates(local_config: Path) -> None:
    """Check that one chat cannot occur twice in the whitelist."""
    with pytest.raises(ConfigError, match="must not contain duplicate"):
        load_config(
            local_config,
            {"IDIOLECT_SIGNAL_CHATS": '["same-group", "same-group"]'},
        )


@pytest.mark.parametrize(
    ("name", "value"),
    (("account", '"+10000000000"'), ("chats", '["group-private"]')),
)
def test_private_signal_identifiers_are_rejected_in_toml(
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    """Check that private Signal identifiers cannot load from TOML."""
    path = tmp_path / "invalid.toml"
    path.write_text(f"[signal]\n{name} = {value}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match=f"unknown values: {name}"):
        load_config(path, {})


def test_training_policy_preserves_explicit_nested_values(tmp_path: Path) -> None:
    """Check nested TOML values become the recorded training policy."""
    path = tmp_path / "train.toml"
    path.write_text(
        """
[train]
base_model = "safe/model"
model_source = "hub"
model_revision = "fixed"
seeds = [17, 42]
optimizer = "adamw"
epochs = 2
optimizer_options = { betas = [0.9, 0.98] }

[train.data]
format = "chat"
prompt_role = "user"
completion_role = "assistant"
prompt_suffix = "\\n/no_think"

[train.lora]
keys = ["self_attn.q_proj", "self_attn.v_proj"]
rank = 8
""",
        encoding="utf-8",
    )

    config = load_config(path, {})

    assert config.train.model_revision == "fixed"
    assert config.train.seeds == (17, 42)
    assert dict(config.train.optimizer_options)["betas"] == (0.9, 0.98)
    assert config.train.data.prompt_suffix == "\n/no_think"
    assert config.train.lora.rank == 8

    policy = training_policy(config.train)
    assert policy["optimizer_options"] == {"betas": [0.9, 0.98]}
    assert policy["data"]["prompt_suffix"] == "\n/no_think"
    assert policy["lora"]["keys"] == [
        "self_attn.q_proj",
        "self_attn.v_proj",
    ]


def test_training_rejects_two_run_limits(tmp_path: Path) -> None:
    """Check that epochs and iterations cannot conflict."""
    path = tmp_path / "invalid.toml"
    path.write_text("[train]\nepochs = 2\niterations = 100\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="epochs or iterations"):
        load_config(path, {})


def test_inference_rejects_duplicate_seeds(tmp_path: Path) -> None:
    """Check that one generation seed cannot occur twice."""
    path = tmp_path / "invalid.toml"
    path.write_text("[inference]\nseeds = [101, 101]\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Inference seeds must be unique"):
        load_config(path, {})


def test_unknown_chat_value_is_rejected_only_at_chat_boundary(tmp_path: Path) -> None:
    """Check that private older settings still load for unrelated stages."""
    path = tmp_path / "chat.toml"
    path.write_text("[chat]\nfuture_value = true\n", encoding="utf-8")

    config = load_config(path, {})

    with pytest.raises(ChatError, match="unknown values: future_value"):
        validate_chat_policy(config.chat, config.inference, config.train)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("temperature", "nan"),
        ("repetition_penalty", "+inf"),
    ),
)
def test_inference_rejects_non_finite_sampling_values(
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    """Check that non-finite sampling values stop configuration."""
    path = tmp_path / "invalid.toml"
    path.write_text(f"[inference]\n{name} = {value}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="must be finite"):
        load_config(path, {})
