"""Run the Idiolect command-line interface."""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from idiolect.chat.discovery import (
    ChatDiscoveryError,
    discover_assistants,
    load_assistant,
)
from idiolect.chat.runtime import ChatError, validate_chat_policy
from idiolect.chat.storage import ChatStorageError, ChatStore
from idiolect.config import ConfigError, TrainConfig, load_config
from idiolect.data.local import (
    DataError,
    LocalBuilder,
    load_dataset,
    resolve_self,
    summarize_people,
)
from idiolect.eval.local import EvaluationError, LocalEvaluator
from idiolect.eval.mlx import EvalBackendError, MlxScoreBackend
from idiolect.eval.panel import collect_judgments, create_panel
from idiolect.infer.base import ModelTarget
from idiolect.infer.local import (
    InferenceError,
    LocalInferencer,
    configured_target,
    recorded_target,
)
from idiolect.infer.mlx import MlxBackend
from idiolect.ingest import harvest
from idiolect.ingest.harvest import reindex
from idiolect.ingest.signal import (
    SignalError,
    SignalFileSource,
    SignalParser,
    SignalSource,
)
from idiolect.store.duck import DuckRepository, StoreError
from idiolect.train.mlx import MlxTrainer, TrainError, load_run, training_policy
from idiolect.tui import ChatTuiError, run_chat_app
from idiolect.types import PersonId, Split


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config)
        if arguments.command == "chat":
            validate_chat_policy(config.chat, config.infer)
            if config.chat.output is None:
                raise ChatError("Chat output is not configured")
            store = ChatStore(config.chat.output)
            initial_assistant = None
            initial_chat = None
            rows = discover_assistants(config.train.output, config.data.output)
            if arguments.chat_command == "run":
                run_path = _artifact_path(arguments.run, config.train.output)
                dataset_path = _artifact_path(arguments.dataset, config.data.output)
                initial_assistant = load_assistant(run_path, dataset_path)
                selected = next(
                    (
                        row
                        for row in rows
                        if row.run_id == str(initial_assistant.run.ref.id)
                    ),
                    None,
                )
                if selected is not None and not selected.available:
                    raise ChatDiscoveryError(
                        selected.error or "Assistant is unavailable"
                    )
            elif arguments.chat_command == "resume":
                initial_chat = store.load(arguments.saved_chat)
            run_chat_app(
                config.chat,
                config.infer.generation,
                assistants=rows,
                store=store,
                initial_assistant=initial_assistant,
                initial_chat=initial_chat,
            )
            return 0
        if arguments.command == "data":
            repository = DuckRepository(config.store.database_path)
            people = summarize_people(repository.messages())
            if arguments.data_command == "people":
                for person in people:
                    state = "self" if person.is_self else "member"
                    name = person.name or "(unknown)"
                    print(f"{person.id}\t{state}\t{person.messages}\t{name}")
                return 0
            person_id = (
                resolve_self(people)
                if arguments.self_person
                else PersonId(arguments.person)
            )
            if config.data.output is None:
                raise ConfigError("Set data.output before dataset construction")
            result = LocalBuilder(repository, config.data.output).build(
                person_id,
                arguments.name,
                config.data,
            )
            counts = result.counts
            print(
                f"dataset={result.dataset.id} train={counts.get(Split.TRAIN, 0)} "
                f"valid={counts.get(Split.VALID, 0)} test={counts.get(Split.TEST, 0)} "
                f"path={result.dataset.path}"
            )
            return 0
        if arguments.command == "train":
            dataset = load_dataset(arguments.dataset).dataset
            result = MlxTrainer().train(dataset, config.train)
            for run in result.runs:
                print(f"run={run.id} dataset={run.dataset_id} path={run.path}")
            return 0
        if arguments.command == "infer":
            inferencer = LocalInferencer(MlxBackend())
            inferencer.validate(config.infer)
            if arguments.infer_command == "text":
                prompt = _read_prompt(arguments.input)
                target = _inference_target(arguments, config.train)
                for prediction in inferencer.text(target, prompt, config.infer):
                    print(
                        json.dumps(
                            asdict(prediction),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                return 0
            dataset = load_dataset(arguments.dataset).dataset
            target = _inference_target(arguments, config.train)
            result = inferencer.dataset(
                target,
                dataset,
                arguments.split,
                config.infer,
            )
            print(
                f"inference={result.id} predictions={result.predictions} "
                f"path={result.path}"
            )
            return 0
        if arguments.command == "eval":
            if arguments.eval_command == "policy":
                dataset = load_dataset(arguments.dataset).dataset
                runs = tuple(load_run(path) for path in arguments.runs)
                if any(run.policy != training_policy(config.train) for run in runs):
                    raise EvaluationError(
                        "Selected configuration does not match the training runs"
                    )
                result = LocalEvaluator(
                    MlxScoreBackend(),
                    LocalInferencer(MlxBackend()),
                ).evaluate(runs, dataset, config.eval, config.infer)
                state = "eligible" if result.eligible else "ineligible"
                print(f"evaluation={result.id} state={state} path={result.path}")
                return 0
            if arguments.eval_command == "rate":
                result = collect_judgments(
                    arguments.evaluation,
                    arguments.rater,
                    config.eval,
                )
                print(
                    f"judgment={result.id} judgments={result.judgments} "
                    f"path={result.path}"
                )
                return 0
            result = create_panel(
                arguments.evaluation,
                arguments.judgments,
                config.eval,
            )
            state = "complete" if result.complete else "incomplete"
            print(f"panel={result.id} state={state} path={result.path}")
            return 0
        if arguments.signal_command == "groups":
            source = SignalSource(config.signal)
            for group in source.groups():
                state = "active" if group.active else "inactive"
                print(f"{group.id}\t{state}\t{group.name}")
            return 0
        if arguments.signal_command != "stats" and not config.signal.chats:
            raise ConfigError(
                "Set IDIOLECT_SIGNAL_CHATS before Signal message processing"
            )
        repository = DuckRepository(config.store.database_path)
        if arguments.signal_command == "stats":
            stats = repository.stats()
            print(
                f"events={stats.events} messages={stats.messages} "
                f"reactions={stats.reactions} database={repository.path}"
            )
            return 0

        parser_adapter = SignalParser(config.signal.chats)
        if arguments.signal_command == "reindex":
            result = reindex(parser_adapter, repository)
            print(
                f"scanned={result.scanned} updated={result.updated} "
                f"messages={result.messages} reactions={result.reactions} "
                f"skipped={result.skipped}"
            )
            return 0
        if arguments.signal_command == "import":
            source = SignalFileSource(arguments.path)
        else:
            timeout = -1 if arguments.follow else arguments.timeout
            max_messages = (
                arguments.max_messages
                if arguments.max_messages is not None
                else None
                if arguments.follow
                else config.signal.max_messages
            )
            signal = replace(
                config.signal,
                timeout=config.signal.timeout if timeout is None else timeout,
                max_messages=max_messages,
            )
            source = SignalSource(signal)
        result = harvest(source, parser_adapter, repository)
        print(
            f"received={result.received} stored={result.stored} "
            f"messages={result.messages} reactions={result.reactions} "
            f"skipped={result.skipped} duplicates={result.duplicates}"
        )
        return 0
    except (
        ConfigError,
        ChatDiscoveryError,
        ChatError,
        ChatStorageError,
        ChatTuiError,
        DataError,
        EvalBackendError,
        EvaluationError,
        InferenceError,
        SignalError,
        StoreError,
        TrainError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Operation stopped.", file=sys.stderr)
        return 130


def _parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(prog="idiolect")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("IDIOLECT_CONFIG", "conf/idiolect.toml")),
        help="TOML configuration path",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    signal = commands.add_parser("signal", help="collect Signal group messages")
    signal_commands = signal.add_subparsers(dest="signal_command", required=True)
    signal_commands.add_parser("groups", help="list known Signal groups")
    collect = signal_commands.add_parser(
        "collect", help="collect queued Signal messages"
    )
    wait = collect.add_mutually_exclusive_group()
    wait.add_argument("--timeout", type=int, help="receive timeout in seconds")
    collect.add_argument("--max-messages", type=int, help="maximum event count")
    wait.add_argument(
        "--follow",
        action="store_true",
        help="wait until collection is stopped",
    )
    import_command = signal_commands.add_parser(
        "import",
        help="import signal-cli JSON lines",
    )
    import_command.add_argument("path", type=Path, help="JSON lines path")
    signal_commands.add_parser("stats", help="show stored record counts")
    signal_commands.add_parser(
        "reindex",
        help="refresh normalized records from stored events",
    )
    data = commands.add_parser("data", help="build model datasets")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    data_commands.add_parser("people", help="list normalized message authors")
    build = data_commands.add_parser("build", help="build one target dataset")
    target = build.add_mutually_exclusive_group(required=True)
    target.add_argument("--self", dest="self_person", action="store_true")
    target.add_argument("--person", help="normalized target person ID")
    build.add_argument("--name", required=True, help="target name in model text")
    train = commands.add_parser("train", help="train configured local adapters")
    train.add_argument("dataset", type=Path, help="immutable dataset directory")
    chat = commands.add_parser("chat", help="chat with verified local adapters")
    chat_commands = chat.add_subparsers(dest="chat_command")
    chat_run = chat_commands.add_parser("run", help="open one run and dataset pair")
    chat_run.add_argument("run", type=Path, help="run ID or immutable run directory")
    chat_run.add_argument(
        "dataset", type=Path, help="dataset ID or immutable dataset directory"
    )
    chat_resume = chat_commands.add_parser("resume", help="resume one saved chat")
    chat_resume.add_argument("saved_chat", help="saved chat ID")
    infer = commands.add_parser("infer", help="generate local model text")
    infer_commands = infer.add_subparsers(dest="infer_command", required=True)
    infer_text = infer_commands.add_parser("text", help="generate one private prompt")
    _inference_target_options(infer_text)
    infer_text.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("-"),
        help="UTF-8 prompt file or - for standard input",
    )
    infer_data = infer_commands.add_parser(
        "data",
        help="generate one immutable dataset split",
    )
    _inference_target_options(infer_data)
    infer_data.add_argument("dataset", type=Path, help="immutable dataset directory")
    infer_data.add_argument(
        "--split",
        type=Split,
        choices=tuple(Split),
        required=True,
        help="dataset split",
    )
    evaluate = commands.add_parser("eval", help="evaluate model fidelity")
    eval_commands = evaluate.add_subparsers(dest="eval_command", required=True)
    eval_policy = eval_commands.add_parser(
        "policy",
        help="compare one complete training policy with its base",
    )
    eval_policy.add_argument("dataset", type=Path, help="immutable dataset directory")
    eval_policy.add_argument(
        "runs",
        type=Path,
        nargs="+",
        help="complete same-policy training run set",
    )
    eval_rate = eval_commands.add_parser(
        "rate",
        help="complete one private familiar-rater session",
    )
    eval_rate.add_argument("evaluation", type=Path, help="evaluation directory")
    eval_rate.add_argument("--rater", required=True, help="pseudonymous rater ID")
    eval_panel = eval_commands.add_parser(
        "panel",
        help="summarize familiar-rater judgments",
    )
    eval_panel.add_argument("evaluation", type=Path, help="evaluation directory")
    eval_panel.add_argument(
        "judgments",
        type=Path,
        nargs="+",
        help="judgment artifact directories",
    )
    return parser


def _inference_target_options(parser: argparse.ArgumentParser) -> None:
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--base",
        action="store_true",
        help="use the base model in the selected configuration",
    )
    target.add_argument(
        "--base-of",
        type=Path,
        metavar="RUN",
        help="use the base model recorded by a run",
    )
    target.add_argument(
        "--run",
        type=Path,
        help="use the adapter recorded by a run",
    )


def _inference_target(
    arguments: argparse.Namespace,
    config: TrainConfig,
) -> ModelTarget:
    if arguments.base:
        return configured_target(config)
    if arguments.base_of is not None:
        return recorded_target(arguments.base_of, adapter=False)
    if arguments.run is None:
        raise InferenceError("Inference run is not configured")
    return recorded_target(arguments.run, adapter=True)


def _read_prompt(path: Path) -> str:
    try:
        return (
            sys.stdin.read() if path == Path("-") else path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError) as error:
        raise InferenceError(f"Cannot read inference prompt: {path}") from error


def _artifact_path(value: Path, root: Path | None) -> Path:
    if value.exists() or value.is_absolute() or len(value.parts) != 1:
        return value
    return value if root is None else root / value
