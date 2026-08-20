"""Generate local text with MLX-LM."""

import contextlib
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from idiolect.config import InferConfig
from idiolect.infer.base import (
    BackendResult,
    ModelTarget,
    Session,
)
from idiolect.infer.local import InferenceError
from idiolect.prompt import ModelInput


class MlxBackend:
    """Load MLX-LM generation sessions."""

    @property
    def version(self) -> str:
        """Return the installed MLX-LM version."""
        try:
            return f"mlx-lm={version('mlx-lm')};mlx={version('mlx')}"
        except PackageNotFoundError as error:
            raise InferenceError(
                "Inference packages are not installed. Run: uv sync --extra train"
            ) from error

    def load(self, target: ModelTarget) -> Session:
        """Load one base model and its optional adapter."""
        try:
            from mlx_lm import load
        except ImportError as error:
            raise InferenceError(
                "Inference packages are not installed. Run: uv sync --extra train"
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
                    tokenizer_config={
                        "trust_remote_code": target.trust_remote_code
                    },
                )
            model = loaded[0]
            tokenizer = loaded[1]
        except Exception as error:
            raise InferenceError(f"Cannot load inference target: {target.id}") from error
        if not tokenizer.has_chat_template:
            raise InferenceError("Inference tokenizer does not have a chat template")
        return _MlxSession(model, tokenizer)


class _MlxSession:
    def __init__(self, model: Any, tokenizer: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer

    def count_tokens(self, value: ModelInput) -> int:
        """Return the formatted prompt token count."""
        return len(self._tokens(value))

    def generate(
        self,
        value: ModelInput,
        seed: int,
        config: InferConfig,
    ) -> BackendResult:
        """Generate one result with MLX-LM."""
        try:
            import mlx.core as mx
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_logits_processors, make_sampler

            tokens = self._tokens(value)
            mx.random.seed(seed)
            sampler = make_sampler(
                config.temperature,
                config.top_p,
                config.min_p,
                config.min_tokens_to_keep,
                top_k=config.top_k,
            )
            processors = make_logits_processors(
                repetition_penalty=config.repetition_penalty,
                repetition_context_size=config.repetition_context_size,
            )
            text = []
            final = None
            with contextlib.redirect_stdout(sys.stderr):
                for response in stream_generate(
                    self._model,
                    self._tokenizer,
                    tokens,
                    max_tokens=config.max_tokens,
                    sampler=sampler,
                    logits_processors=processors,
                ):
                    text.append(response.text)
                    final = response
            if final is None or final.finish_reason is None:
                raise InferenceError("MLX-LM did not return a complete result")
            return BackendResult(
                "".join(text),
                final.finish_reason,
                final.prompt_tokens,
                final.generation_tokens,
            )
        except InferenceError:
            raise
        except Exception as error:
            raise InferenceError("MLX-LM generation failed") from error

    def close(self) -> None:
        """Release the loaded model and its cache."""
        try:
            import mlx.core as mx

            del self._model
            del self._tokenizer
            mx.clear_cache()
        except (AttributeError, ImportError):
            return

    def _tokens(self, value: ModelInput) -> list[int]:
        turns = [
            {"role": turn.role, "content": turn.content} for turn in value.turns
        ]
        options = {"tokenize": True, "return_dict": False}
        if value.has_prefill:
            options["continue_final_message"] = True
        else:
            options["add_generation_prompt"] = True
        tokens = self._tokenizer.apply_chat_template(turns, **options)
        if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
            raise InferenceError("Inference tokenizer returned invalid prompt tokens")
        return tokens
