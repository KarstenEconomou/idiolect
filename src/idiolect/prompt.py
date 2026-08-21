"""Apply one fixed model text format."""

from dataclasses import dataclass
from typing import Any

from idiolect.config import TrainDataConfig


class PromptError(ValueError):
    """Report an invalid model text format."""


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
    completion_role: str = "assistant"


def format_prompt(prompt: str, config: TrainDataConfig) -> ModelInput:
    """Format one prompt with the training data policy."""
    validate_prompt_config(config)
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
    return ModelInput(tuple(turns), has_prefill, completion_role)


def format_row(
    prompt: str,
    completion: str,
    config: TrainDataConfig,
) -> dict[str, Any]:
    """Format one complete training row."""
    validate_prompt_config(config)
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


def validate_prompt_config(config: TrainDataConfig) -> None:
    """Verify one model text format."""
    if config.format not in {"chat", "completion"}:
        raise PromptError("Model text format must be chat or completion")
    if config.format == "chat":
        roles = {"system", "user", "assistant"}
        if config.prompt_role not in roles:
            raise PromptError("Model prompt role is not valid")
        if config.completion_role not in roles:
            raise PromptError("Model completion role is not valid")
        return
    if config.system_prompt or config.prompt_role or config.completion_role:
        raise PromptError(
            "Completion text format cannot contain chat system or role values"
        )
