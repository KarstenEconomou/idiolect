"""Test the MLX-LM completion-scoring boundary."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from idiolect.config import TrainDataConfig
from idiolect.eval.mlx import EvalBackendError, MlxScoreBackend
from idiolect.inference.base import ModelTarget, TargetMode
from idiolect.prompt import ModelExample, ModelInput, Turn


class FakeArray:
    """Provide the array operations used by completion scoring."""

    def __getitem__(self, key: object) -> FakeArray:
        """Return one view without changing the synthetic data."""
        return self


class FakeLogits:
    """Provide the precision conversion used by completion scoring."""

    def astype(self, value: object) -> FakeLogits:
        """Return logits in the requested synthetic precision."""
        return self


class FakeLosses:
    """Keep the synthetic loss slice selected by the scorer."""

    def __init__(self, values: list[float]) -> None:
        """Set all next-token losses."""
        self.values = values

    def __getitem__(self, key: tuple[int, slice]) -> FakeLosses:
        """Return the requested completion loss slice."""
        return FakeLosses(self.values[key[1]])

    @property
    def size(self) -> int:
        """Return the selected token count."""
        return len(self.values)

    def sum(self) -> SimpleNamespace:
        """Return the selected negative log-likelihood."""
        return SimpleNamespace(item=lambda: sum(self.values))


class FakeTokenizer:
    """Return fixed prompt and complete chat token sequences."""

    has_chat_template = True

    def __init__(self, changed_boundary: bool = False) -> None:
        """Set whether the complete sequence changes the prompt prefix."""
        self.changed_boundary = changed_boundary

    def apply_chat_template(
        self, turns: list[dict[str, str]], **options: object
    ) -> list[int]:
        """Return tokens for the prompt or completed conversation."""
        if options.get("add_generation_prompt") or options.get(
            "continue_final_message"
        ):
            return [10, 11, 12]
        prefix = [10, 99, 12] if self.changed_boundary else [10, 11, 12]
        return [*prefix, 20, 21]


def test_mlx_score_counts_only_completion_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check the token mask at the prompt and completion boundary."""
    session = _session(tmp_path, monkeypatch, FakeTokenizer())

    score = session.score(
        ModelExample(ModelInput((Turn("user", "prompt"),), False), "reply")
    )

    assert score.prompt_tokens == 3
    assert score.tokens == 2
    assert score.negative_log_likelihood == 4.0


def test_mlx_score_rejects_a_changed_token_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check that retokenized prompt tokens cannot change the score mask."""
    session = _session(tmp_path, monkeypatch, FakeTokenizer(changed_boundary=True))

    with pytest.raises(EvalBackendError, match="completion boundary"):
        session.score(
            ModelExample(ModelInput((Turn("user", "prompt"),), False), "reply")
        )


def test_mlx_score_excludes_prefill_tokens_from_completion_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check that static assistant prefill stays inside the prompt mask."""
    session = _session(tmp_path, monkeypatch, FakeTokenizer())

    score = session.score(
        ModelExample(
            ModelInput(
                (Turn("user", "prompt"), Turn("assistant", "prefill")),
                True,
            ),
            "reply",
        )
    )

    assert score.prompt_tokens == 3
    assert score.tokens == 2
    assert score.negative_log_likelihood == 4.0


def _session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: FakeTokenizer,
):
    mlx_lm = ModuleType("mlx_lm")
    mlx = ModuleType("mlx")
    mlx_core = ModuleType("mlx.core")
    nn = SimpleNamespace(
        losses=SimpleNamespace(
            cross_entropy=lambda logits, tokens, reduction: FakeLosses(
                [0.25, 0.5, 1.5, 2.5]
            )
        )
    )
    mlx_lm.__dict__["load"] = lambda *args, **kwargs: (
        lambda tokens: FakeLogits(),
        tokenizer,
    )
    mlx.__dict__["nn"] = nn
    mlx_core.__dict__.update(
        array=lambda values: FakeArray(),
        eval=lambda values: None,
        float32=object(),
        clear_cache=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    target = ModelTarget(
        "test-target",
        TargetMode.RUN_BASE,
        tmp_path / "model",
        "model-digest",
        TrainDataConfig(format="chat"),
    )
    return MlxScoreBackend().load(target)
