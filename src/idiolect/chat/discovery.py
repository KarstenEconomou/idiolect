"""Discover verified local chat assistants."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from idiolect.data.local import (
    BuildResult,
    DataError,
    load_dataset_metadata,
)
from idiolect.train.base import LoadedRun
from idiolect.train.mlx import TrainError, load_run
from idiolect.types import Split


class ChatDiscoveryError(ValueError):
    """Report an invalid chat assistant selection."""


@dataclass(frozen=True, slots=True)
class Assistant:
    """Keep one verified run and dataset pair for chat."""

    name: str
    target_name: str
    model_basename: str
    run: LoadedRun
    dataset: BuildResult
    context_messages: int

    @property
    def counts(self) -> Mapping[Split, int]:
        """Return the verified dataset split counts."""
        return self.dataset.counts


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
        f"IDIOLECT // {target_name.upper()}@{run_id[:8]} "
        f"[{model_basename(model_name)}]"
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
                    str(assistant.run.ref.id),
                    str(assistant.dataset.dataset.id),
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
