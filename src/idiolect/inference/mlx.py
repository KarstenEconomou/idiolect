"""Execute renderer-produced prompt tokens with MLX-LM."""

import contextlib
import sys
from collections.abc import Callable, Iterator
from importlib.metadata import PackageNotFoundError
from typing import Any

from idiolect.config import GenerationConfig, InferenceConfig
from idiolect.inference.base import (
    BackendResult,
    Cancellation,
    GenerationEvent,
    ModelTarget,
    StreamingSession,
)
from idiolect.inference.local import InferenceError
from idiolect.model import mlx_runtime_fingerprint
from idiolect.prompt import ModelInput
from idiolect.render import ChatTemplateRenderer, ModelRenderer, RenderError


class MlxBackend:
    """Load MLX-LM generation sessions."""

    @property
    def version(self) -> str:
        """Return the installed MLX-LM version."""
        try:
            return mlx_runtime_fingerprint()
        except PackageNotFoundError as error:
            raise InferenceError(
                "Inference packages are not installed. Run: uv sync --extra train"
            ) from error

    def load(self, target: ModelTarget) -> StreamingSession:
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
                    tokenizer_config={"trust_remote_code": target.trust_remote_code},
                )
            model = loaded[0]
            tokenizer = loaded[1]
        except Exception as error:
            raise InferenceError(
                f"Cannot load inference target: {target.id}"
            ) from error
        if not tokenizer.has_chat_template:
            raise InferenceError("Inference tokenizer does not have a chat template")
        return _MlxSession(model, tokenizer, ChatTemplateRenderer(tokenizer))


class _MlxSession:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        renderer: ModelRenderer,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._renderer = renderer

    def count_tokens(self, value: ModelInput) -> int:
        """Return the formatted prompt token count."""
        try:
            return len(self._renderer.render_prompt(value).token_ids)
        except RenderError as error:
            raise InferenceError(str(error)) from error

    def generate(
        self,
        value: ModelInput,
        seed: int,
        config: InferenceConfig,
    ) -> BackendResult:
        """Generate one result with MLX-LM."""
        text = []
        result = None
        for event in self.stream(value, seed, config):
            text.append(event.text)
            if event.result is not None:
                result = event.result
        if result is None:
            raise InferenceError("MLX-LM did not return a complete result")
        return BackendResult(
            "".join(text),
            result.finish_reason,
            result.prompt_tokens,
            result.generated_tokens,
            result.prompt_throughput,
            result.generation_throughput,
            result.peak_memory,
        )

    def stream(
        self,
        value: ModelInput,
        seed: int,
        config: GenerationConfig,
        cancel: Cancellation | None = None,
        prompt_progress: Callable[[int, int], None] | None = None,
    ) -> Iterator[GenerationEvent]:
        """Yield one MLX-LM generation as text deltas."""
        try:
            import mlx.core as mx
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_logits_processors, make_sampler

            tokens = list(self._renderer.render_prompt(value).token_ids)
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
            final = None
            cancelled = False
            generation_options: dict[str, Any] = {
                "max_tokens": config.max_tokens,
                "sampler": sampler,
                "logits_processors": processors,
            }
            if prompt_progress is not None:
                generation_options["prompt_progress_callback"] = prompt_progress
            with contextlib.redirect_stdout(sys.stderr):
                for response in stream_generate(
                    self._model,
                    self._tokenizer,
                    tokens,
                    **generation_options,
                ):
                    if cancel is not None and cancel.is_set():
                        cancelled = True
                        final = response
                        break
                    yield GenerationEvent(text=response.text)
                    final = response
            if final is None:
                raise InferenceError("MLX-LM did not return a complete result")
            finish_reason = "cancelled" if cancelled else final.finish_reason
            if finish_reason is None:
                raise InferenceError("MLX-LM did not return a complete result")
            result = BackendResult(
                "",
                finish_reason,
                final.prompt_tokens,
                final.generation_tokens,
                _metric(final, "prompt_tps", "prompt_throughput"),
                _metric(final, "generation_tps", "generation_throughput"),
                _metric(final, "peak_memory"),
            )
            yield GenerationEvent(result=result)
        except (InferenceError, RenderError) as error:
            if isinstance(error, RenderError):
                raise InferenceError(str(error)) from error
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
        except AttributeError, ImportError:
            return


def _metric(value: Any, *names: str) -> float | None:
    for name in names:
        metric = getattr(value, name, None)
        if isinstance(metric, (int, float)) and not isinstance(metric, bool):
            return float(metric)
    return None
