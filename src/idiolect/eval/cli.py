"""Register evaluation command-line operations."""

import argparse

from idiolect.command import artifact_path, keep_awake
from idiolect.config import ConfigError, load_config, resolve_config_path
from idiolect.data.local import DataError, load_dataset
from idiolect.eval.local import EvaluationError, LocalEvaluator
from idiolect.eval.mlx import EvalBackendError, MlxScoreBackend
from idiolect.eval.panel import collect_judgments, create_panel
from idiolect.inference.local import LocalInferencer
from idiolect.inference.mlx import MlxBackend
from idiolect.train.mlx import TrainError, load_run, training_policy

ERRORS = (ConfigError, DataError, EvalBackendError, EvaluationError, TrainError)


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register evaluation commands."""
    parser = commands.add_parser("eval", help="evaluate model fidelity")
    parser.add_argument("references", nargs="+", help="evaluation mode or artifacts")
    parser.add_argument("--rater", help="pseudonymous rater ID")
    parser.set_defaults(handler=_evaluate)


def _evaluate(arguments: argparse.Namespace) -> int:
    mode = arguments.references[0]
    if mode == "policy":
        raise EvaluationError("policy is not a valid artifact reference")
    config = load_config(resolve_config_path(arguments.config))
    if mode == "rate":
        if len(arguments.references) != 2 or arguments.rater is None:
            raise EvaluationError("Use eval rate EVAL --rater ID")
        evaluation = artifact_path(arguments.references[1], config.eval.output)
        result = collect_judgments(evaluation, arguments.rater, config.eval)
        print(f"judgment={result.id} judgments={result.judgments} path={result.path}")
        return 0
    if mode == "panel":
        if len(arguments.references) < 3 or arguments.rater is not None:
            raise EvaluationError("Use eval panel EVAL JUDGMENTS...")
        evaluation = artifact_path(arguments.references[1], config.eval.output)
        judgments = tuple(
            artifact_path(value, config.eval.output, "judgments")
            for value in arguments.references[2:]
        )
        result = create_panel(evaluation, judgments, config.eval)
        state = "complete" if result.complete else "incomplete"
        print(f"panel={result.id} state={state} path={result.path}")
        return 0
    if arguments.rater is not None or len(arguments.references) < 2:
        raise EvaluationError("Use eval DATASET RUN...")
    dataset_path = artifact_path(mode, config.data.output)
    dataset = load_dataset(dataset_path).dataset
    runs = tuple(
        load_run(artifact_path(value, config.train.output))
        for value in arguments.references[1:]
    )
    if any(run.policy != training_policy(config.train) for run in runs):
        raise EvaluationError("Selected configuration does not match the training runs")
    with keep_awake():
        result = LocalEvaluator(
            MlxScoreBackend(), LocalInferencer(MlxBackend())
        ).evaluate(runs, dataset, config.eval, config.inference)
    state = "eligible" if result.eligible else "ineligible"
    print(f"evaluation={result.id} state={state} path={result.path}")
    return 0
