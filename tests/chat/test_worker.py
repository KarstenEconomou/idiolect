"""Test typed worker protocol records without model work."""

from idiolect.chat.worker import (
    CancelCommand,
    CountCommand,
    GenerateCommand,
    LoadCommand,
    ProbeCommand,
    ShutdownCommand,
    UnloadCommand,
    command_from_value,
    command_value,
)
from idiolect.config import GenerationConfig
from idiolect.prompt import ModelInput, Turn


def test_worker_commands_round_trip_as_plain_values() -> None:
    """Check every process command serialization contract."""
    prompt = ModelInput((Turn("user", "literal [text]"),), False)
    commands = (
        ProbeCommand(),
        LoadCommand("run"),
        CountCommand(prompt),
        GenerateCommand(prompt, 17, GenerationConfig(max_prompt_tokens=10)),
        CancelCommand(),
        UnloadCommand(),
        ShutdownCommand(),
    )

    assert (
        tuple(command_from_value(command_value(value)) for value in commands)
        == commands
    )
