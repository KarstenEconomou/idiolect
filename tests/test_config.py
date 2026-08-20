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


def test_training_policy_loads_without_backend_defaults(tmp_path: Path) -> None:
    """Check nested training values and exclusive run limits."""
    path = tmp_path / "train.toml"
    path.write_text(
        """
[train]
base_model = "safe/model"
model_source = "hub"
model_revision = "fixed"
model_cache = "var/models"
output = "var/runs"
adapter_file = "adapters.safetensors"
command = ["mlx_lm.lora"]
seeds = [17, 42]
fine_tune_type = "lora"
optimizer = "adamw"
batch_size = 1
epochs = 2
learning_rate = 0.00001
num_layers = 16
val_batches = -1
test = false
test_batches = -1
max_seq_length = 2048
grad_checkpoint = false
grad_accumulation_steps = 8
clear_cache_threshold = 0
steps_per_report = 10
steps_per_eval = 200
save_every = 100
mask_prompt = true
trust_remote_code = false
schedule = "constant"
report_to = ""
project_name = ""

[train.optimizer_options]
betas = [0.9, 0.98]

[train.data]
format = "chat"
prompt_role = "user"
completion_role = "assistant"
prompt_suffix = "\\n/no_think"
completion_prefix = "<think>\\n\\n</think>\\n\\n"

[train.lora]
keys = ["self_attn.q_proj", "self_attn.v_proj"]
rank = 8
scale = 20.0
dropout = 0.05
""",
        encoding="utf-8",
    )

    config = load_config(path, {})

    assert config.train.model_revision == "fixed"
    assert config.train.seeds == (17, 42)
    assert dict(config.train.optimizer_options)["betas"] == (0.9, 0.98)
    assert config.train.data.prompt_suffix == "\n/no_think"
    assert config.train.lora.rank == 8


def test_training_rejects_two_run_limits(tmp_path: Path) -> None:
    """Check that epochs and iterations cannot conflict."""
    path = tmp_path / "invalid.toml"
    path.write_text("[train]\nepochs = 2\niterations = 100\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="epochs or iterations"):
        load_config(path, {})
