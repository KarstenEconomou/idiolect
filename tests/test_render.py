"""Test the stage-neutral model token rendering boundary."""

import pytest

from idiolect.prompt import ModelExample, ModelInput, Turn
from idiolect.render import ChatTemplateRenderer, RenderError


class FakeTokenizer:
    """Return deterministic tokens and record chat-template calls."""

    has_chat_template = True

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    def apply_chat_template(self, messages, **options):
        self.calls.append((messages, options))
        if len(messages) == 1:
            return [10, 11, 12]
        return [10, 11, 12, 20, 21]


def test_prompt_selects_generation_or_prefill_template_mode() -> None:
    """Check that prompt meaning selects the exact tokenizer option."""
    tokenizer = FakeTokenizer()
    renderer = ChatTemplateRenderer(tokenizer)

    ordinary = renderer.render_prompt(ModelInput((Turn("user", "ask"),), False))
    prefill = renderer.render_prompt(
        ModelInput(
            (Turn("user", "ask"), Turn("assistant", "start")),
            True,
        )
    )

    assert ordinary.token_ids == (10, 11, 12)
    assert prefill.token_ids == (10, 11, 12, 20, 21)
    assert tokenizer.calls[0][1] == {
        "tokenize": True,
        "return_dict": False,
        "add_generation_prompt": True,
    }
    assert tokenizer.calls[1][1] == {
        "tokenize": True,
        "return_dict": False,
        "continue_final_message": True,
    }


def test_example_exposes_stable_completion_slice() -> None:
    """Check that rendering keeps the exact supervised token boundary."""
    rendered = ChatTemplateRenderer(FakeTokenizer()).render_example(
        ModelExample(ModelInput((Turn("user", "ask"),), False), "reply")
    )

    assert rendered.token_ids == (10, 11, 12, 20, 21)
    assert rendered.prompt_tokens == 3
    assert rendered.completion_tokens == (20, 21)


@pytest.mark.parametrize("value", ([], [1, True], [1, "two"], None))
def test_rejects_empty_or_invalid_token_output(value: object) -> None:
    """Check that every renderer result is a nonempty integer sequence."""

    class InvalidTokenizer:
        has_chat_template = True

        def apply_chat_template(self, messages, **options):
            return value

    with pytest.raises(RenderError, match="invalid prompt tokens"):
        ChatTemplateRenderer(InvalidTokenizer()).render_prompt(
            ModelInput((Turn("user", "ask"),), False)
        )


def test_rejects_missing_supervised_tokens_and_unstable_boundary() -> None:
    """Check that an example must extend its exact rendered prompt."""

    class PathologicalTokenizer:
        has_chat_template = True

        def __init__(self, full: list[int]) -> None:
            self.full = full

        def apply_chat_template(self, messages, **options):
            return [1, 2] if len(messages) == 1 else self.full

    example = ModelExample(ModelInput((Turn("user", "ask"),), False), "reply")
    with pytest.raises(RenderError, match="supervised tokens"):
        ChatTemplateRenderer(PathologicalTokenizer([1, 2])).render_example(example)
    with pytest.raises(RenderError, match="completion boundary"):
        ChatTemplateRenderer(PathologicalTokenizer([1, 9, 3])).render_example(example)
