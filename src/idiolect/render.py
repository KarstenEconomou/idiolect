"""Render stage-neutral model conversations as exact model tokens."""

from dataclasses import dataclass
from typing import Any, Protocol

from idiolect.prompt import ModelExample, ModelInput, Turn, completed_turns


class RenderError(ValueError):
    """Report an invalid or unstable model token rendering."""


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """Keep the exact token identifiers for one model prompt."""

    token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RenderedExample:
    """Keep one complete token sequence and its prompt boundary."""

    token_ids: tuple[int, ...]
    prompt_tokens: int

    @property
    def completion_tokens(self) -> tuple[int, ...]:
        """Return the supervised completion token identifiers."""
        return self.token_ids[self.prompt_tokens :]


class ModelRenderer(Protocol):
    """Render model conversations as token identifiers."""

    def render_prompt(self, value: ModelInput) -> RenderedPrompt:
        """Render one prompt for generation."""
        ...

    def render_example(self, value: ModelExample) -> RenderedExample:
        """Render one completed example with its prompt boundary."""
        ...


class Tokenizer(Protocol):
    """Apply one model chat template."""

    has_chat_template: bool

    def apply_chat_template(self, messages: Any, **options: Any) -> Any:
        """Return tokens for one sequence of chat messages."""
        ...


class ChatTemplateRenderer:
    """Render conversations with one tokenizer chat template."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        """Set the tokenizer that owns the model token grammar."""
        self._tokenizer = tokenizer

    def render_prompt(self, value: ModelInput) -> RenderedPrompt:
        """Render one prompt for generation."""
        options: dict[str, Any] = {"tokenize": True, "return_dict": False}
        if value.has_prefill:
            options["continue_final_message"] = True
        else:
            options["add_generation_prompt"] = True
        tokens = self._render(value.turns, options, "prompt")
        return RenderedPrompt(tokens)

    def render_example(self, value: ModelExample) -> RenderedExample:
        """Render one completed example with a stable prompt boundary."""
        prompt = self.render_prompt(value.prompt).token_ids
        full = self._render(
            completed_turns(value),
            {"tokenize": True, "return_dict": False},
            "completed example",
        )
        if len(full) <= len(prompt):
            raise RenderError("Model completion does not contain supervised tokens")
        if full[: len(prompt)] != prompt:
            raise RenderError(
                "Model tokenizer changed tokens at the completion boundary"
            )
        return RenderedExample(full, len(prompt))

    def _render(
        self,
        turns: tuple[Turn, ...],
        options: dict[str, Any],
        name: str,
    ) -> tuple[int, ...]:
        messages = [{"role": turn.role, "content": turn.content} for turn in turns]
        try:
            value = self._tokenizer.apply_chat_template(messages, **options)
        except Exception as error:
            raise RenderError(f"Cannot render model {name}") from error
        if (
            not isinstance(value, (list, tuple))
            or not value
            or not all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in value
            )
        ):
            raise RenderError(f"Model tokenizer returned invalid {name} tokens")
        return tuple(value)
