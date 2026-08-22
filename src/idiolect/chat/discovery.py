"""Discover verified local chat assistants."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from idiolect.config import ChatConfig, TrainConfig, TrainDataConfig
from idiolect.data.local import (
    BuildResult,
    DataError,
    load_dataset_metadata,
)
from idiolect.inference.base import TargetMode
from idiolect.model import ModelSpec
from idiolect.train.base import LoadedRun
from idiolect.train.mlx import TrainError, load_run
from idiolect.types import Split


class ChatDiscoveryError(ValueError):
    """Report an invalid chat assistant selection."""


@dataclass(frozen=True, slots=True)
class Assistant:
    """Keep one fixed local model assistant for chat."""

    name: str
    target_name: str
    model_basename: str
    run: LoadedRun | None
    dataset: BuildResult | None
    context_messages: int
    base_model: ModelSpec | None = None
    base_data: TrainDataConfig | None = None
    base_model_digest: str | None = None

    @property
    def mode(self) -> TargetMode:
        """Return the model target mode for this assistant."""
        return (
            TargetMode.CONFIG_BASE
            if self.base_model is not None
            else TargetMode.RUN_ADAPTER
        )

    @property
    def data(self) -> TrainDataConfig:
        """Return the model prompt policy for this assistant."""
        if self.run is not None:
            return self.run.data
        if self.base_data is not None:
            return self.base_data
        raise ChatDiscoveryError("The assistant prompt policy is not available")

    @property
    def model(self) -> ModelSpec:
        """Return the fixed model specification for this assistant."""
        if self.run is not None:
            return self.run.model
        if self.base_model is not None:
            return self.base_model
        raise ChatDiscoveryError("The assistant model is not available")

    @property
    def model_digest(self) -> str | None:
        """Return the verified model digest when it is available."""
        return self.run.model_digest if self.run is not None else self.base_model_digest

    @property
    def adapter_digest(self) -> str | None:
        """Return the adapter digest when this assistant uses an adapter."""
        return self.run.adapter_digest if self.run is not None else None

    @property
    def training_seed(self) -> int | None:
        """Return the training seed when this assistant uses an adapter."""
        return self.run.seed if self.run is not None else None

    @property
    def run_id(self) -> str | None:
        """Return the run ID when this assistant uses an adapter."""
        return str(self.run.ref.id) if self.run is not None else None

    @property
    def dataset_id(self) -> str | None:
        """Return the dataset ID when this assistant uses an adapter."""
        return str(self.dataset.dataset.id) if self.dataset is not None else None

    @property
    def counts(self) -> Mapping[Split, int]:
        """Return the verified dataset split counts."""
        return {} if self.dataset is None else self.dataset.counts


@dataclass(frozen=True, slots=True)
class DiscoveryItem:
    """Keep one landing chooser row."""

    label: str
    run_id: str
    dataset_id: str | None
    assistant: Assistant | None
    error: str | None = None

    @property
    def available(self) -> bool:
        """Return true when this row can be selected."""
        return self.assistant is not None and self.error is None


def model_basename(value: str) -> str:
    """Return the final hub repository or local path component."""
    name = value.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]
    if not name:
        raise ChatDiscoveryError("The recorded model name does not have a basename")
    return name


def canonical_name(target_name: str, run_id: str, model_name: str) -> str:
    """Return one canonical local assistant name."""
    return (
        f"IDIOLECT // {target_name.upper()}@{run_id[:8]} [{model_basename(model_name)}]"
    )


def default_assistant(train: TrainConfig, chat: ChatConfig) -> Assistant:
    """Create the configured base-model assistant without model resolution."""
    model = ModelSpec(
        train.base_model,
        train.model_source,
        train.model_revision,
        train.model_cache,
        train.trust_remote_code,
    )
    data = replace(train.data, system_prompt=chat.default_system_prompt)
    basename = model_basename(model.name)
    return Assistant(
        f"IDIOLECT // {chat.default_name.upper()}@BASE [{basename}]",
        chat.default_name.upper(),
        basename,
        None,
        None,
        chat.default_context_messages,
        model,
        data,
    )


def load_assistant(run_path: Path, dataset_path: Path) -> Assistant:
    """Load and verify one matching run and dataset pair."""
    try:
        run = load_run(run_path)
        metadata = load_dataset_metadata(dataset_path)
        dataset = BuildResult(metadata.dataset, metadata.counts)
        target_name = metadata.target_name
        context = metadata.context_messages
        if run.ref.dataset_id != dataset.dataset.id:
            raise ChatDiscoveryError(
                "The selected run does not record the selected dataset ID"
            )
        basename = model_basename(run.model.name)
        return Assistant(
            canonical_name(target_name, str(run.ref.id), basename),
            target_name,
            basename,
            run,
            dataset,
            context,
        )
    except (DataError, TrainError) as error:
        raise ChatDiscoveryError(str(error)) from error
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, ChatDiscoveryError):
            raise
        raise ChatDiscoveryError(
            f"Cannot read chat metadata for run: {run_path}"
        ) from error


def discover_assistants(
    run_output: Path | None,
    data_output: Path | None,
) -> tuple[DiscoveryItem, ...]:
    """Return chooser rows for local run directories."""
    if run_output is None or data_output is None:
        return ()
    if not run_output.is_dir():
        return ()
    rows: list[DiscoveryItem] = []
    for path in sorted(run_output.iterdir()):
        if not path.is_dir() or not _digest(path.name):
            continue
        dataset_id = _recorded_dataset_id(path)
        if dataset_id is None:
            rows.append(
                DiscoveryItem(path.name, path.name, None, None, "Run is corrupt")
            )
            continue
        try:
            assistant = load_assistant(path, data_output / dataset_id)
        except ChatDiscoveryError as error:
            rows.append(
                DiscoveryItem(path.name, path.name, dataset_id, None, str(error))
            )
        else:
            rows.append(
                DiscoveryItem(
                    assistant.name,
                    assistant.run_id or path.name,
                    assistant.dataset_id,
                    assistant,
                )
            )
    prefixes: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        prefixes.setdefault(row.run_id[:8], []).append(index)
    for prefix, indexes in prefixes.items():
        if len(indexes) < 2:
            continue
        ids = ", ".join(rows[index].run_id for index in indexes)
        message = f"Run prefix {prefix} is not unique: {ids}"
        for index in indexes:
            rows[index] = replace(rows[index], assistant=None, error=message)
    return tuple(rows)


def _read_manifest(path: Path) -> Mapping[str, Any]:
    value = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError
    return value


def _recorded_dataset_id(path: Path) -> str | None:
    try:
        value = _read_manifest(path).get("dataset_id")
    except OSError, TypeError, ValueError, json.JSONDecodeError:
        return None
    return value if isinstance(value, str) and _digest(value) else None


def _digest(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
