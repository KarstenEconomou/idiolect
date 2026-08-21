"""Test local configuration behavior."""

import json
from pathlib import Path

import pytest

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
    """Check that an invalid private allowlist stops configuration."""
    with pytest.raises(ConfigError, match="must be a JSON list"):
        load_config(local_config, {"IDIOLECT_SIGNAL_CHATS": value})


def test_signal_chat_environment_rejects_duplicates(local_config: Path) -> None:
    """Check that one chat cannot occur twice in the allowlist."""
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

[eval]
output = "var/eval"
backend = "mlx-lm"
suite = "fidelity"
split = "valid"
max_examples = 100
bootstrap_seed = 7
bootstrap_samples = 1000
confidence_level = 0.95
long_match_chars = 50
max_empty_rate = 0.0
max_format_violation_rate = 0.0
max_truncation_rate = 0.05
max_memorization_rate_delta = 0.0
ballot_seed = 11
ballots_per_rater = 40
control_fraction = 0.2
min_panel_raters = 3
min_primary_comparisons = 60

[infer]
output = "var/infer"
backend = "mlx-lm"
seeds = [101, 202]
max_examples = 100
max_prompt_tokens = 1920
max_tokens = 128
temperature = 0.7
top_p = 0.8
top_k = 20
min_p = 0.0
min_tokens_to_keep = 1
repetition_penalty = 1.0
repetition_context_size = 20
""",
        encoding="utf-8",
    )

    config = load_config(path, {})

    assert config.train.model_revision == "fixed"
    assert config.train.seeds == (17, 42)
    assert dict(config.train.optimizer_options)["betas"] == (0.9, 0.98)
    assert config.train.data.prompt_suffix == "\n/no_think"
    assert config.train.lora.rank == 8
    assert config.eval.suite == "fidelity"
    assert config.eval.split == "valid"
    assert config.eval.bootstrap_samples == 1000
    assert config.eval.ballots_per_rater == 40
    assert config.infer.seeds == (101, 202)
    assert config.infer.max_prompt_tokens == 1920

    policy = training_policy(config.train)
    assert policy == json.loads(json.dumps(policy))


def test_training_rejects_two_run_limits(tmp_path: Path) -> None:
    """Check that epochs and iterations cannot conflict."""
    path = tmp_path / "invalid.toml"
    path.write_text("[train]\nepochs = 2\niterations = 100\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="epochs or iterations"):
        load_config(path, {})


def test_inference_rejects_duplicate_seeds(tmp_path: Path) -> None:
    """Check that one generation seed cannot occur twice."""
    path = tmp_path / "invalid.toml"
    path.write_text("[infer]\nseeds = [101, 101]\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Inference seeds must be unique"):
        load_config(path, {})


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("temperature", "nan"),
        ("temperature", "+inf"),
        ("repetition_penalty", "nan"),
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
    path.write_text(f"[infer]\n{name} = {value}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="must be finite"):
        load_config(path, {})
