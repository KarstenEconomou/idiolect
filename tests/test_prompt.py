"""Test fixed model text formatting."""

import pytest

from idiolect.config import TrainDataConfig
from idiolect.prompt import PromptError, format_prompt


@pytest.mark.parametrize(
    "config",
    (
        TrainDataConfig(format="unknown"),
        TrainDataConfig(
            format="chat",
            prompt_role="invalid",
            completion_role="assistant",
        ),
        TrainDataConfig(
            format="chat",
            prompt_role="user",
            completion_role="invalid",
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
