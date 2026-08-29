"""Register training command-line operations."""

import argparse

from idiolect.command import artifact_path, keep_awake
from idiolect.config import ConfigError, load_config, resolve_config_path
from idiolect.data.local import DataError, load_dataset
from idiolect.train.mlx import MlxTrainer, TrainError

ERRORS = (ConfigError, DataError, TrainError)


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the training command."""
    parser = commands.add_parser("train", help="train configured local adapters")
    parser.add_argument("dataset", help="dataset content ID or directory")
    parser.set_defaults(handler=_train)


def _train(arguments: argparse.Namespace) -> int:
    config = load_config(resolve_config_path(arguments.config))
    path = artifact_path(arguments.dataset, config.data.output)
    dataset = load_dataset(path).dataset
    with keep_awake():
        result = MlxTrainer().train(dataset, config.train)
    for run in result.runs:
        print(f"run={run.id} dataset={run.dataset_id} path={run.path}")
    return 0
