"""Apply one fixed model text format."""

from dataclasses import dataclass
from typing import Any

from idiolect.config import TrainDataConfig


@dataclass(frozen=True, slots=True)
class Turn:
    """Keep one model chat turn."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelInput:
    """Keep one formatted model input."""

    turns: tuple[Turn, ...]
    has_prefill: bool


def format_prompt(prompt: str, config: TrainDataConfig) -> ModelInput:
    """Format one prompt with the training data policy."""
    content = f"{config.prompt_prefix}{prompt}{config.prompt_suffix}"
    turns = []
    if config.format == "completion":
        turns.append(Turn("user", content))
        completion_role = "assistant"
    else:
        if config.system_prompt:
            turns.append(Turn("system", config.system_prompt))
        turns.append(Turn(config.prompt_role, content))
        completion_role = config.completion_role
    has_prefill = bool(config.completion_prefix)
    if has_prefill:
        turns.append(Turn(completion_role, config.completion_prefix))
    return ModelInput(tuple(turns), has_prefill)


def format_row(
    prompt: str,
    completion: str,
    config: TrainDataConfig,
) -> dict[str, Any]:
    """Format one complete training row."""
    prompt = f"{config.prompt_prefix}{prompt}{config.prompt_suffix}"
    completion = (
        f"{config.completion_prefix}{completion}{config.completion_suffix}"
    )
    if config.format == "completion":
        return {"prompt": prompt, "completion": completion}
    messages = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.extend(
        (
            {"role": config.prompt_role, "content": prompt},
            {"role": config.completion_role, "content": completion},
        )
    )
    return {"messages": messages}
