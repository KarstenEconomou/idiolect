"""Test fixed model text formatting."""

import pytest

from idiolect.config import TrainDataConfig
from idiolect.prompt import (
    MESSAGE_BOUNDARY,
    PromptError,
    completed_turns,
    format_example,
    format_prompt,
    join_bubbles,
    split_bubbles,
)


@pytest.mark.parametrize(
    "config",
    (
        TrainDataConfig(format="unknown"),
        TrainDataConfig(
            format="chat",
            prompt_role="invalid",
            completion_role="assistant",
        ),
        TrainDataConfig(format="completion", system_prompt="ignored value"),
    ),
)
def test_prompt_rejects_invalid_or_ignored_format_values(
    config: TrainDataConfig,
) -> None:
    """Check that invalid text policies stop before tokenization."""
    with pytest.raises(PromptError):
        format_prompt("context", config)


def test_bubble_serialization_round_trips_exact_text() -> None:
    """Check that joining and splitting preserves every bubble exactly."""
    bubbles = ("first bubble", "", "multi\nline\nbubble", "last")

    serialized = join_bubbles(bubbles)

    assert split_bubbles(serialized) == bubbles


def test_bubble_serialization_rejects_ambiguous_boundary_lines() -> None:
    """Check that source text cannot forge a message boundary."""
    for text in (
        MESSAGE_BOUNDARY,
        f"one\n{MESSAGE_BOUNDARY}\ntwo",
        f" {MESSAGE_BOUNDARY} ",
    ):
        with pytest.raises(PromptError, match="reserved boundary"):
            join_bubbles((text,))

    assert join_bubbles(("one two",)) == "one two"
    assert split_bubbles("one two") == ("one two",)


def test_example_keeps_prefill_in_prompt_and_suffix_in_target() -> None:
    """Check that the generation boundary excludes only the target text."""
    example = format_example(
        "context",
        "reply",
        TrainDataConfig(
            format="chat",
            prompt_role="user",
            completion_role="assistant",
            prompt_prefix="before ",
            prompt_suffix=" after",
            completion_prefix="prefill ",
            completion_suffix=" end",
        ),
    )

    assert example.prompt.turns[-1].content == "prefill "
    assert example.completion == "reply end"
    assert [turn.content for turn in completed_turns(example)] == [
        "before context after",
        "prefill reply end",
    ]
