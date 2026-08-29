"""Score renderer-verified completion tokens with MLX-LM."""

import contextlib
import sys
from importlib.metadata import PackageNotFoundError
from typing import Any

from idiolect.eval.base import CompletionScore, ScoreSession
from idiolect.inference.base import ModelTarget
from idiolect.model import mlx_runtime_fingerprint
from idiolect.prompt import ModelExample
from idiolect.render import ChatTemplateRenderer, ModelRenderer, RenderError


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
        return _MlxScoreSession(model, tokenizer, ChatTemplateRenderer(tokenizer))


class _MlxScoreSession:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        renderer: ModelRenderer,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._renderer = renderer

    def score(self, example: ModelExample) -> CompletionScore:
        """Return the completion token likelihood cost."""
        try:
            import mlx.core as mx
            from mlx import nn

            rendered = self._renderer.render_example(example)
            tokens = mx.array(rendered.token_ids)
            logits = self._model(tokens[:-1][None]).astype(mx.float32)
            losses = nn.losses.cross_entropy(
                logits,
                tokens[1:][None],
                reduction="none",
            )[0, rendered.prompt_tokens - 1 :]
            mx.eval(losses)
            return CompletionScore(
                rendered.prompt_tokens,
                int(losses.size),
                float(losses.sum().item().real),
            )
        except RenderError as error:
            raise EvalBackendError(str(error)) from error
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
