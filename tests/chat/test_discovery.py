"""Test local assistant names and chooser collision safety."""

import json
from types import SimpleNamespace
from typing import cast

from idiolect.chat.discovery import (
    Assistant,
    canonical_name,
    default_assistant,
    discover_assistants,
    model_basename,
)
from idiolect.config import ChatConfig, TrainConfig, TrainDataConfig


def test_assistant_name_uses_requested_identity_and_final_model_component() -> None:
    """Check uppercase identity and final model values."""
    assert model_basename("mlx-community/Qwen3-14B-4bit") == "Qwen3-14B-4bit"
    assert model_basename("/models/local-model/") == "local-model"
    assert canonical_name("target", "7f3a91c2" + "0" * 56, "org/Qwen") == (
        "IDIOLECT // TARGET@7f3a91c2 [Qwen]"
    )


def test_default_assistant_uses_configured_base_and_system_persona() -> None:
    """Check that DIXIE is available without one run or dataset artifact."""
    assistant = default_assistant(
        TrainConfig(
            base_model="mlx-community/Qwen3-14B-4bit",
            model_source="hub",
            model_revision="fixed",
            data=TrainDataConfig(
                format="chat",
                prompt_role="user",
                completion_role="assistant",
            ),
        ),
        ChatConfig(
            default_name="Dixie",
            default_context_messages=32,
            default_system_prompt="Use terse technical language.",
        ),
    )

    assert assistant.name == "IDIOLECT // DIXIE@BASE [Qwen3-14B-4bit]"
    assert assistant.run is None
    assert assistant.dataset is None
    assert assistant.data.system_prompt == "Use terse technical language."


def test_discovery_disables_every_colliding_short_run_id(tmp_path, monkeypatch) -> None:
    """Check that no ambiguous eight-character assistant can be selected."""
    runs = tmp_path / "runs"
    data = tmp_path / "data"
    runs.mkdir()
    data.mkdir()
    ids = ("deadbeef" + "1" * 56, "deadbeef" + "2" * 56)
    dataset_id = "a" * 64
    for run_id in ids:
        path = runs / run_id
        path.mkdir()
        (path / "manifest.json").write_text(
            json.dumps({"dataset_id": dataset_id}), encoding="utf-8"
        )

    def fake_load(run_path, _dataset_path):
        assistant = SimpleNamespace(
            name=f"IDIOLECT // K@{run_path.name[:8]} [M]",
            run_id=run_path.name,
            dataset_id=dataset_id,
            run=SimpleNamespace(ref=SimpleNamespace(id=run_path.name)),
            dataset=SimpleNamespace(dataset=SimpleNamespace(id=dataset_id)),
        )
        return cast(Assistant, assistant)

    monkeypatch.setattr("idiolect.chat.discovery.load_assistant", fake_load)

    rows = discover_assistants(runs, data)

    assert len(rows) == 2
    assert all(not row.available for row in rows)
    assert all(all(run_id in (row.error or "") for run_id in ids) for row in rows)
