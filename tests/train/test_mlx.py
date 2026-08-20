"""Test local MLX-LM training orchestration."""

import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from idiolect.config import LoraConfig, TrainConfig, TrainDataConfig
from idiolect.train.mlx import MlxTrainer, TrainError, load_run
from idiolect.types import DatasetId, DatasetRef, PersonId

_NOW = datetime(2026, 8, 19, tzinfo=UTC)


class FakeRunner:
    """Record backend requests and create synthetic adapters."""

    def __init__(self) -> None:
        """Create an empty command record."""
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: Sequence[str], log_path: Path) -> int:
        """Create the configured adapter without model work."""
        self.commands.append(tuple(command))
        request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        adapter = Path(request["adapter_path"])
        (adapter / "adapters.safetensors").write_bytes(b"synthetic-adapter")
        (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        log_path.write_text("synthetic training\n", encoding="utf-8")
        return 0


def test_trainer_builds_fixed_qwen_runs_without_changing_source_data(
    tmp_path: Path,
) -> None:
    """Check formatting, backend settings, seeds, and immutable reuse."""
    dataset = _dataset(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    runner = FakeRunner()
    config = _config(tmp_path)
    trainer = MlxTrainer(
        runner=runner,
        resolver=lambda _config: model,
        clock=lambda: _NOW,
    )

    first = trainer.train(dataset, config)
    second = trainer.train(dataset, config)

    assert first == second
    assert len(first.runs) == 2
    assert len(runner.commands) == 2
    assert (dataset.path / "train.jsonl").read_text(encoding="utf-8") == (
        '{"prompt":"Context","completion":"Reply"}\n'
    )
    first_run = first.runs[0]
    train_row = json.loads(
        (first_run.path / "data" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert train_row == {
        "messages": [
            {"role": "user", "content": "Context\n/no_think"},
            {
                "role": "assistant",
                "content": "<think>\n\n</think>\n\nReply",
            },
        ]
    }
    request = json.loads(
        (first_run.path / "request.json").read_text(encoding="utf-8")
    )
    assert request["model"] == str(model)
    assert request["iters"] == 2
    assert request["mask_prompt"] is True
    assert request["optimizer_config"] == {
        "adamw": {
            "betas": [0.9, 0.98],
            "bias_correction": True,
            "eps": 1e-8,
            "weight_decay": 0.0,
        }
    }
    assert request["lora_parameters"] == {
        "dropout": 0.05,
        "keys": ["self_attn.q_proj", "self_attn.v_proj"],
        "rank": 8,
        "scale": 20.0,
    }
    assert request["report_to"] is None

    loaded = load_run(first_run.path)
    assert loaded.ref == first_run
    assert loaded.model.name == "safe/model"
    assert loaded.data.prompt_suffix == "\n/no_think"
    assert loaded.adapter_path == first_run.path / "adapter"

    (first_run.path / "adapter" / "adapters.safetensors").write_bytes(b"changed")
    with pytest.raises(TrainError, match="does not match its manifest"):
        trainer.train(dataset, config)


def test_trainer_rejects_incomplete_policy_before_model_work(tmp_path: Path) -> None:
    """Check that code does not supply an omitted experiment choice."""
    dataset = _dataset(tmp_path)
    resolved = False

    def resolver(_config: TrainConfig) -> Path:
        nonlocal resolved
        resolved = True
        return tmp_path / "model"

    with pytest.raises(TrainError, match="configuration is incomplete"):
        MlxTrainer(resolver=resolver).train(dataset, TrainConfig())

    assert resolved is False


def test_trainer_does_not_fill_an_omitted_toml_choice(tmp_path: Path) -> None:
    """Check that valid fallback values do not become training policy."""
    dataset = _dataset(tmp_path)
    config = replace(_config(tmp_path), specified=frozenset({"base_model"}))

    with pytest.raises(TrainError, match="mask_prompt"):
        MlxTrainer(resolver=lambda value: tmp_path / "model").train(dataset, config)


def test_trainer_rejects_unaudited_model_code(tmp_path: Path) -> None:
    """Check that a model code loader cannot bypass the local policy."""
    dataset = _dataset(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        '{"auto_map":{"AutoModel":"model.CustomModel"}}\n', encoding="utf-8"
    )
    runner = FakeRunner()

    with pytest.raises(TrainError, match="requires remote code"):
        MlxTrainer(runner=runner, resolver=lambda config: model).train(
            dataset, _config(tmp_path)
        )

    assert runner.commands == []


def _dataset(tmp_path: Path) -> DatasetRef:
    """Create one small canonical completion dataset."""
    path = tmp_path / ("a" * 64)
    path.mkdir()
    (path / "train.jsonl").write_text(
        '{"prompt":"Context","completion":"Reply"}\n', encoding="utf-8"
    )
    (path / "valid.jsonl").write_text(
        '{"prompt":"Later","completion":"Answer"}\n', encoding="utf-8"
    )
    return DatasetRef(DatasetId(path.name), PersonId("target"), path, _NOW)


def _config(tmp_path: Path) -> TrainConfig:
    """Return one complete synthetic training policy."""
    return TrainConfig(
        base_model="safe/model",
        model_source="hub",
        model_revision="fixed-revision",
        model_cache=tmp_path / "models",
        output=tmp_path / "runs",
        adapter_file="adapters.safetensors",
        command=("mlx_lm.lora",),
        seeds=(17, 42),
        fine_tune_type="lora",
        optimizer="adamw",
        optimizer_options=(
            ("betas", (0.9, 0.98)),
            ("bias_correction", True),
            ("eps", 1e-8),
            ("weight_decay", 0.0),
        ),
        batch_size=1,
        epochs=2,
        learning_rate=1e-5,
        num_layers=16,
        val_batches=-1,
        test=False,
        test_batches=-1,
        max_seq_length=2048,
        grad_checkpoint=False,
        grad_accumulation_steps=8,
        clear_cache_threshold=0,
        steps_per_report=10,
        steps_per_eval=200,
        save_every=100,
        mask_prompt=True,
        trust_remote_code=False,
        schedule="constant",
        report_to="",
        project_name="",
        data=TrainDataConfig(
            format="chat",
            system_prompt="",
            prompt_role="user",
            completion_role="assistant",
            prompt_prefix="",
            prompt_suffix="\n/no_think",
            completion_prefix="<think>\n\n</think>\n\n",
            completion_suffix="",
        ),
        lora=LoraConfig(
            keys=("self_attn.q_proj", "self_attn.v_proj"),
            rank=8,
            scale=20.0,
            dropout=0.05,
        ),
    )
