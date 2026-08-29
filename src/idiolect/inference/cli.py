"""Register local inference command-line operations."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from idiolect.command import artifact_path, keep_awake
from idiolect.config import ConfigError, load_config, resolve_config_path
from idiolect.data.local import DataError, load_dataset
from idiolect.inference.local import (
    InferenceError,
    LocalInferencer,
    configured_target,
    recorded_target,
)
from idiolect.inference.mlx import MlxBackend
from idiolect.train.mlx import TrainError
from idiolect.types import Split

ERRORS = (ConfigError, DataError, InferenceError, TrainError)


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the inference command."""
    parser = commands.add_parser("infer", help="generate local model text")
    parser.add_argument("run", nargs="?", help="run content ID or directory")
    parser.add_argument("input", nargs="?", help="UTF-8 prompt file or - for stdin")
    parser.add_argument("--base", action="store_true", help="use a base model")
    parser.add_argument("--data", dest="dataset", help="dataset content ID or directory")
    parser.add_argument("--split", type=Split, choices=tuple(Split), help="dataset split")
    parser.set_defaults(handler=_infer)


def _infer(arguments: argparse.Namespace) -> int:
    _normalize_configured_base_input(arguments)
    if (arguments.dataset is None) != (arguments.split is None):
        raise InferenceError("--data and --split must be used together")
    if arguments.dataset is not None and arguments.input is not None:
        raise InferenceError("Prompt input cannot be used with --data")
    if arguments.run is None and not arguments.base:
        raise InferenceError("Set RUN or --base for inference")
    prompt = None
    if arguments.dataset is None:
        prompt = _read_prompt(Path(arguments.input or "-"))
    config = load_config(resolve_config_path(arguments.config))
    inferencer = LocalInferencer(MlxBackend())
    inferencer.validate(config.inference)
    if arguments.run is None:
        target = configured_target(config.train)
    else:
        run_path = artifact_path(arguments.run, config.train.output)
        target = recorded_target(run_path, adapter=not arguments.base)
    if arguments.dataset is None:
        assert prompt is not None
        for prediction in inferencer.text(target, prompt, config.inference):
            print(json.dumps(asdict(prediction), ensure_ascii=False, separators=(",", ":")))
        return 0
    dataset_path = artifact_path(arguments.dataset, config.data.output)
    dataset = load_dataset(dataset_path).dataset
    with keep_awake():
        result = inferencer.dataset(target, dataset, arguments.split, config.inference)
    print(f"inference={result.id} predictions={result.predictions} path={result.path}")
    return 0


def _normalize_configured_base_input(arguments: argparse.Namespace) -> None:
    """Distinguish a configured-base input placed after the base option."""
    if not arguments.base or arguments.run is None or arguments.input is not None:
        return
    raw = arguments.raw_arguments
    base_index = raw.index("--base")
    try:
        run_index = raw.index(arguments.run)
    except ValueError:
        return
    if run_index > base_index:
        arguments.input = arguments.run
        arguments.run = None


def _read_prompt(path: Path) -> str:
    try:
        return sys.stdin.read() if path == Path("-") else path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise InferenceError(f"Cannot read inference prompt: {path}") from error
