"""Score fixed completions with MLX-LM."""

import contextlib
import sys
from importlib.metadata import PackageNotFoundError
from typing import Any

from idiolect.eval.base import CompletionScore, ScoreSession
from idiolect.inference.base import ModelTarget
from idiolect.model import mlx_runtime_fingerprint
from idiolect.prompt import ModelInput, Turn


class EvalBackendError(RuntimeError):
    """Report one MLX-LM evaluation backend failure."""


class MlxScoreBackend:
    """Load MLX-LM completion scoring sessions."""

    @property
    def version(self) -> str:
        """Return the installed backend versions."""
        try:
            return mlx_runtime_fingerprint()
        except PackageNotFoundError as error:
            raise EvalBackendError(
                "Evaluation packages are not installed. Run: uv sync --extra train"
            ) from error

    def load(self, target: ModelTarget) -> ScoreSession:
        """Load one verified target and return its scoring session."""
        try:
            from mlx_lm import load
        except ImportError as error:
            raise EvalBackendError(
                "Evaluation packages are not installed. Run: uv sync --extra train"
            ) from error
        try:
            with contextlib.redirect_stdout(sys.stderr):
                loaded = load(
                    str(target.model_path),
                    adapter_path=(
                        str(target.adapter_path)
                        if target.adapter_path is not None
                        else None
                    ),
                    tokenizer_config={"trust_remote_code": target.trust_remote_code},
                )
            model = loaded[0]
            tokenizer = loaded[1]
        except Exception as error:
            raise EvalBackendError(
                f"Cannot load evaluation target: {target.id}"
            ) from error
        if not tokenizer.has_chat_template:
            raise EvalBackendError("Evaluation tokenizer does not have a chat template")
        return _MlxScoreSession(model, tokenizer)


class _MlxScoreSession:
    def __init__(self, model: Any, tokenizer: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer

    def score(self, prompt: ModelInput, completion: str) -> CompletionScore:
        """Return the completion token likelihood cost."""
        try:
            import mlx.core as mx
            from mlx import nn

            prompt_tokens = self._prompt_tokens(prompt)
            full_tokens = self._full_tokens(prompt, completion)
            if full_tokens[: len(prompt_tokens)] != prompt_tokens:
                raise EvalBackendError(
                    "Evaluation tokenizer changed tokens at the completion boundary"
                )
            if len(full_tokens) <= len(prompt_tokens):
                raise EvalBackendError("Evaluation completion does not contain tokens")
            tokens = mx.array(full_tokens)
            logits = self._model(tokens[:-1][None]).astype(mx.float32)
            losses = nn.losses.cross_entropy(
                logits,
                tokens[1:][None],
                reduction="none",
            )[0, len(prompt_tokens) - 1 :]
            mx.eval(losses)
            return CompletionScore(
                len(prompt_tokens),
                int(losses.size),
                float(losses.sum().item().real),
            )
        except EvalBackendError:
            raise
        except Exception as error:
            raise EvalBackendError("MLX-LM completion scoring failed") from error

    def close(self) -> None:
        """Release the loaded model and its cache."""
        try:
            import mlx.core as mx

            del self._model
            del self._tokenizer
            mx.clear_cache()
        except (AttributeError, ImportError):
            return

    def _prompt_tokens(self, value: ModelInput) -> list[int]:
        turns = [_turn_value(turn) for turn in value.turns]
        options: dict[str, Any] = {"tokenize": True, "return_dict": False}
        if value.has_prefill:
            options["continue_final_message"] = True
        elif value.completion_role != "assistant":
            turns.append({"role": value.completion_role, "content": ""})
            options["continue_final_message"] = True
        else:
            options["add_generation_prompt"] = True
        return self._tokens(turns, options)

    def _full_tokens(self, value: ModelInput, completion: str) -> list[int]:
        turns = list(value.turns)
        if value.has_prefill:
            final = turns[-1]
            turns[-1] = Turn(final.role, f"{final.content}{completion}")
        else:
            turns.append(Turn(value.completion_role, completion))
        return self._tokens([_turn_value(turn) for turn in turns], {})

    def _tokens(
        self,
        turns: list[dict[str, str]],
        options: dict[str, Any],
    ) -> list[int]:
        value = self._tokenizer.apply_chat_template(turns, **options)
        if not isinstance(value, list) or not all(
            isinstance(token, int) for token in value
        ):
            raise EvalBackendError("Evaluation tokenizer returned invalid tokens")
        return value


def _turn_value(turn: Turn) -> dict[str, str]:
    return {"role": turn.role, "content": turn.content}
