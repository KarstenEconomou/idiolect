"""Test typed worker protocol records without model work."""

import importlib.metadata
import sys
from types import ModuleType

import pytest

from idiolect.chat import worker
from idiolect.chat.worker import (
    CancelCommand,
    CountCommand,
    GenerateCommand,
    LoadBaseCommand,
    LoadCommand,
    ProbeCommand,
    ShutdownCommand,
    UnloadCommand,
    command_from_value,
    command_value,
)
from idiolect.config import GenerationConfig, TrainDataConfig
from idiolect.model import ModelSpec
from idiolect.prompt import ModelInput, Turn


class _MlxCore(ModuleType):
    """Provide synthetic MLX runtime measurements."""

    @staticmethod
    def default_device() -> str:
        return "Device(gpu, 0)"

    @staticmethod
    def device_info() -> dict[str, object]:
        return {"device_name": "Synthetic GPU"}

    @staticmethod
    def get_active_memory() -> int:
        return 3 * 1024**3

    @staticmethod
    def get_cache_memory() -> int:
        return 1024**3


class _Mlx(ModuleType):
    """Provide one synthetic MLX package namespace."""

    core: _MlxCore


def test_probe_records_live_mlx_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that runtime probing adds live MLX memory measurements."""
    mlx = _Mlx("mlx")
    core = _MlxCore("mlx.core")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    versions = {"mlx": "0.32.1", "mlx-lm": "0.31.3"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)

    probe = worker._probe()

    assert dict(probe.device_properties) == {
        "active_memory": 3 * 1024**3,
        "cache_memory": 1024**3,
        "device_name": "Synthetic GPU",
    }


def test_worker_commands_round_trip_as_plain_values() -> None:
    """Check every process command serialization contract."""
    prompt = ModelInput((Turn("user", "literal [text]"),), False)
    commands = (
        ProbeCommand(),
        LoadCommand("run"),
        LoadBaseCommand(
            ModelSpec("org/model", "hub", "fixed", None, False),
            TrainDataConfig(
                format="chat",
                system_prompt="Be terse.",
                prompt_role="user",
                completion_role="assistant",
            ),
            "a" * 64,
        ),
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
