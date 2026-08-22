"""Test the MLX-LM inference boundary."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from idiolect.config import InferConfig, TrainDataConfig
from idiolect.infer.base import ModelTarget, TargetMode
from idiolect.infer.mlx import MlxBackend
from idiolect.prompt import format_prompt


class FakeTokenizer:
    """Record model chat template options."""

    has_chat_template = True

    def __init__(self) -> None:
        """Create an empty option record."""
        self.options = []

    def apply_chat_template(self, turns, **options):
        """Record the turns and return synthetic tokens."""
        self.options.append((turns, options))
        return [4, 5, 6]


def test_mlx_backend_applies_adapter_format_seed_and_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """Check the complete translation to the MLX-LM boundary."""
    tokenizer = FakeTokenizer()
    seen = {}
    mlx_lm = ModuleType("mlx_lm")
    sample_utils = ModuleType("mlx_lm.sample_utils")
    mlx = ModuleType("mlx")
    mlx_core = ModuleType("mlx.core")
    random = SimpleNamespace(seed=lambda value: seen.update(seed=value))
    mlx_core.__dict__["random"] = random
    mlx_core.__dict__["clear_cache"] = lambda: seen.update(cache_cleared=True)

    def load(path, **options):
        print("load diagnostic")
        seen["load"] = (path, options)
        return object(), tokenizer

    def sampler(*values, **options):
        seen["sampler"] = (values, options)
        return "sampler"

    def processors(**options):
        seen["processors"] = options
        return ["processor"]

    def stream_generate(model, current_tokenizer, tokens, **options):
        print("generate diagnostic")
        seen["generate"] = (model, current_tokenizer, tokens, options)
        yield SimpleNamespace(
            text="reply",
            finish_reason="stop",
            prompt_tokens=3,
            generation_tokens=1,
            prompt_tps=120.5,
            generation_tps=42.25,
            peak_memory=3.75,
        )

    mlx_lm.__dict__["load"] = load
    mlx_lm.__dict__["stream_generate"] = stream_generate
    sample_utils.__dict__["make_sampler"] = sampler
    sample_utils.__dict__["make_logits_processors"] = processors
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    adapter = tmp_path / "adapter"
    target = ModelTarget(
        "target",
        TargetMode.RUN_ADAPTER,
        tmp_path / "model",
        "digest",
        TrainDataConfig(
            format="chat",
            prompt_role="user",
            completion_role="assistant",
            completion_prefix="prefill",
        ),
        trust_remote_code=True,
        adapter_path=adapter,
    )
    config = InferConfig(
        max_tokens=12,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        min_tokens_to_keep=1,
        repetition_penalty=1.1,
        repetition_context_size=32,
    )

    session = MlxBackend().load(target)
    model_input = format_prompt("context", target.data)
    result = session.generate(model_input, 123, config)
    session.close()

    output = capsys.readouterr()
    assert output.out == ""
    assert "load diagnostic" in output.err
    assert "generate diagnostic" in output.err
    assert seen["load"] == (
        str(target.model_path),
        {
            "adapter_path": str(adapter),
            "tokenizer_config": {"trust_remote_code": True},
        },
    )
    assert tokenizer.options == [
        (
            [
                {"role": "user", "content": "context"},
                {"role": "assistant", "content": "prefill"},
            ],
            {
                "tokenize": True,
                "return_dict": False,
                "continue_final_message": True,
            },
        )
    ]
    assert seen["seed"] == 123
    assert seen["sampler"] == ((0.7, 0.8, 0.0, 1), {"top_k": 20})
    assert seen["processors"] == {
        "repetition_penalty": 1.1,
        "repetition_context_size": 32,
    }
    assert seen["generate"][2:] == (
        [4, 5, 6],
        {
            "max_tokens": 12,
            "sampler": "sampler",
            "logits_processors": ["processor"],
        },
    )
    assert result.text == "reply"
    assert result.prompt_throughput == 120.5
    assert result.generation_throughput == 42.25
    assert result.peak_memory == 3.75
    assert seen["cache_cleared"] is True
