"""Save and verify immutable private chat snapshots."""

import hashlib
import json
import math
import os
import shutil
import tempfile
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeGuard

from idiolect.artifact import canonical_json_bytes, is_digest, write_json
from idiolect.chat.discovery import Assistant, load_assistant, model_basename
from idiolect.chat.state import ChatSession, ChatTurn, TurnTelemetry, enumerate_bubbles
from idiolect.config import (
    ChatConfig,
    ConfigError,
    GenerationConfig,
    TrainDataConfig,
    validate_generation_values,
)
from idiolect.inference.base import TargetMode
from idiolect.model import ModelSpec
from idiolect.prompt import split_bubbles, validate_prompt_config

_SNAPSHOT_VERSION = 2
_ASSISTANT_KEYS = {
    "name",
    "target_name",
    "mode",
    "context_messages",
    "run_id",
    "run_path",
    "dataset_id",
    "dataset_path",
    "model_name",
    "model_source",
    "model_revision",
    "model_cache",
    "trust_remote_code",
    "model_digest",
    "adapter_digest",
    "training_seed",
    "data",
}
_GENERATION_KEYS = {
    "backend",
    "max_prompt_tokens",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "min_tokens_to_keep",
    "repetition_penalty",
    "repetition_context_size",
}


class ChatStorageError(ValueError):
    """Report an invalid saved chat operation."""


@dataclass(frozen=True, slots=True)
class SavedChat:
    """Keep one verified saved chat snapshot."""

    id: str
    path: Path
    created_at: datetime
    title: str
    parent_id: str | None
    assistant: Assistant
    chat: ChatConfig
    generation: GenerationConfig
    turns: tuple[ChatTurn, ...]
    backend_versions: Mapping[str, str | None] | None = None


_MANIFEST_KEYS = {
    "chat_id",
    "created_at",
    "version",
    "assistant",
    "chat_policy",
    "generation_policy",
    "title",
    "parent_chat_id",
    "turn_count",
    "files",
}
_IDENTITY_KEYS = (
    "version",
    "assistant",
    "chat_policy",
    "generation_policy",
    "title",
    "parent_chat_id",
    "turn_count",
    "files",
)


class ChatStore:
    """Manage content-addressed chat snapshots under one private root."""

    def __init__(self, root: Path, clock=None) -> None:
        """Set the output root and optional clock."""
        self.root = root.expanduser().resolve()
        self._clock = (lambda: datetime.now(UTC)) if clock is None else clock

    def save(
        self,
        state: ChatSession,
        title: str | None = None,
        backend_versions: Mapping[str, str | None] | None = None,
    ) -> SavedChat:
        """Save or return one immutable snapshot for the current state."""
        if state.generating:
            raise ChatStorageError("Stop generation before saving")
        if not state.turns:
            raise ChatStorageError("A chat must contain a message before saving")
        if (
            not state.dirty
            and state.saved_chat_id is not None
            and (title is None or _title(state.turns, title) == state.title)
        ):
            # The referenced snapshot may have been erased outside this store.
            try:
                return self.load(state.saved_chat_id)
            except ChatStorageError:
                pass
        chosen_title = (
            default_chat_title(state) if title is None else _title(state.turns, title)
        )
        rows = [_turn_value(turn) for turn in state.turns]
        turns_bytes = _jsonl_bytes(rows)
        versions = dict(backend_versions) if backend_versions is not None else None
        identity = {
            "version": _SNAPSHOT_VERSION,
            "assistant": _assistant_value(state.assistant),
            "chat_policy": _chat_value(state.chat),
            "generation_policy": asdict(state.generation),
            "title": chosen_title,
            "parent_chat_id": state.saved_chat_id,
            "turn_count": len(rows),
            "backend_versions": versions,
            "files": {"turns.jsonl": hashlib.sha256(turns_bytes).hexdigest()},
        }
        chat_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        destination = self.root / chat_id
        if destination.exists():
            saved = self.load(chat_id)
            state.mark_saved(saved.id, saved.title)
            return saved
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        temporary = Path(tempfile.mkdtemp(prefix=".chat-", dir=self.root))
        try:
            turns_path = temporary / "turns.jsonl"
            with turns_path.open("xb") as stream:
                stream.write(turns_bytes)
            os.chmod(turns_path, 0o600)
            manifest = {
                "chat_id": chat_id,
                "created_at": self._clock().isoformat(),
                **identity,
            }
            write_json(temporary / "manifest.json", manifest)
            temporary.rename(destination)
        except (OSError, TypeError, ValueError) as error:
            shutil.rmtree(temporary, ignore_errors=True)
            if destination.exists():
                saved = self.load(chat_id)
                state.mark_saved(saved.id, saved.title)
                return saved
            raise ChatStorageError(f"Cannot save chat: {destination}") from error
        saved = self.load(chat_id)
        state.mark_saved(saved.id, saved.title)
        return saved

    def load(self, value: str | Path) -> SavedChat:
        """Load and verify one saved chat snapshot."""
        path = value if isinstance(value, Path) else self.root / value
        try:
            if not is_digest(path.name):
                raise ChatStorageError(f"Chat path does not contain an ID: {path}")
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or manifest.get("chat_id") != path.name:
                raise ChatStorageError(f"Chat manifest does not match its path: {path}")
            version = manifest.get("version")
            if version != _SNAPSHOT_VERSION:
                raise ChatStorageError(
                    f"Chat snapshot version is not supported: {path}"
                )
            identity_keys = (*_IDENTITY_KEYS, "backend_versions")
            if set(manifest) != set(identity_keys) | {"chat_id", "created_at"}:
                raise ChatStorageError(f"Chat manifest does not match its path: {path}")
            identity = {key: manifest[key] for key in identity_keys}
            if hashlib.sha256(canonical_json_bytes(identity)).hexdigest() != path.name:
                raise ChatStorageError(f"Chat identity does not match its ID: {path}")
            if {item.name for item in path.iterdir() if item.is_file()} != {
                "manifest.json",
                "turns.jsonl",
            }:
                raise ChatStorageError(f"Chat files do not match its manifest: {path}")
            turns_path = path / "turns.jsonl"
            files = manifest["files"]
            if not isinstance(files, dict) or set(files) != {"turns.jsonl"}:
                raise TypeError
            expected = files["turns.jsonl"]
            if not is_digest(expected):
                raise TypeError
            if hashlib.sha256(turns_path.read_bytes()).hexdigest() != expected:
                raise ChatStorageError(
                    f"Chat file does not match its manifest: {turns_path}"
                )
            turns = _read_turns(turns_path)
            if (
                not isinstance(manifest["turn_count"], int)
                or isinstance(manifest["turn_count"], bool)
                or len(turns) != manifest["turn_count"]
            ):
                raise ChatStorageError(f"Chat turn count does not match: {path}")
            assistant_value = manifest["assistant"]
            if (
                not isinstance(assistant_value, dict)
                or set(assistant_value) != _ASSISTANT_KEYS
            ):
                raise TypeError
            mode = _required_text(assistant_value, "mode")
            if mode == TargetMode.RUN_ADAPTER.value:
                assistant = load_assistant(
                    Path(_required_text(assistant_value, "run_path")),
                    Path(_required_text(assistant_value, "dataset_path")),
                )
            elif mode == TargetMode.CONFIG_BASE.value:
                assistant = _base_assistant(assistant_value)
            else:
                raise TypeError
            if canonical_json_bytes(
                _assistant_value(assistant)
            ) != canonical_json_bytes(assistant_value):
                raise ChatStorageError(
                    "Saved chat assistant does not match local artifacts"
                )
            chat = _chat_config(manifest["chat_policy"])
            generation = _generation_config(manifest["generation_policy"])
            try:
                validate_generation_values(generation)
            except ConfigError as error:
                raise ChatStorageError(
                    f"Saved chat generation policy is not valid: {error}"
                ) from error
            backend_versions = _backend_versions(manifest["backend_versions"])
            created_at = datetime.fromisoformat(manifest["created_at"])
            if created_at.utcoffset() is None:
                raise TypeError
            title = _required_text(manifest, "title")
            parent = manifest["parent_chat_id"]
            if parent is not None and (
                not isinstance(parent, str) or not is_digest(parent)
            ):
                raise TypeError
        except ChatStorageError:
            raise
        except Exception as error:
            raise ChatStorageError(f"Cannot read saved chat: {path}") from error
        return SavedChat(
            path.name,
            path,
            created_at,
            title,
            parent,
            assistant,
            chat,
            generation,
            turns,
            backend_versions,
        )

    def leaves(self) -> tuple[SavedChat, ...]:
        """Return verified snapshots that have no verified child."""
        chats = self._verified_chats()
        parents = {chat.parent_id for chat in chats if chat.parent_id is not None}
        leaves = [chat for chat in chats if chat.id not in parents]
        return tuple(
            sorted(leaves, key=lambda chat: (chat.created_at, chat.id), reverse=True)
        )

    def resume(self, value: str | Path) -> ChatSession:
        """Return an in-memory session from one verified snapshot."""
        saved = self.load(value)
        return ChatSession(
            saved.assistant,
            saved.chat,
            saved.generation,
            saved.turns,
            saved.id,
            saved.title,
        )

    def erase(self, chat_id: str) -> None:
        """Erase one verified leaf snapshot."""
        saved = self.load(chat_id)
        if any(chat.parent_id == saved.id for chat in self._verified_chats()):
            raise ChatStorageError("Cannot erase a TRACE that has a child")
        try:
            shutil.rmtree(saved.path)
        except OSError as error:
            raise ChatStorageError(f"Cannot erase chat: {saved.path}") from error

    def rename(self, chat_id: str, title: str) -> SavedChat:
        """Replace one verified leaf snapshot with a renamed snapshot."""
        saved = self.load(chat_id)
        if any(chat.parent_id == saved.id for chat in self._verified_chats()):
            raise ChatStorageError("Cannot rename a TRACE that has a child")
        chosen_title = _title(list(saved.turns), title)
        if chosen_title == saved.title:
            return saved
        state = ChatSession(
            saved.assistant,
            saved.chat,
            saved.generation,
            saved.turns,
            saved.parent_id,
            saved.title,
        )
        renamed = self.save(state, chosen_title, saved.backend_versions)
        self.erase(saved.id)
        return renamed

    def _verified_chats(self) -> tuple[SavedChat, ...]:
        """Return all verified snapshots in the private root."""
        if not self.root.is_dir():
            return ()
        chats = []
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or not is_digest(path.name):
                continue
            try:
                chats.append(self.load(path))
            except ChatStorageError:
                continue
        return tuple(chats)


def default_chat_title(state: ChatSession) -> str:
    """Return the generated title for one chat state."""
    return _title(state.turns, None)


def _assistant_value(assistant: Assistant) -> dict[str, Any]:
    model = assistant.model
    run = assistant.run
    dataset = assistant.dataset
    model_digest = assistant.model_digest
    if model_digest is None:
        raise ChatStorageError("Load the base model before saving the chat")
    return {
        "name": assistant.name,
        "target_name": assistant.target_name,
        "mode": assistant.mode.value,
        "context_messages": assistant.context_messages,
        "run_id": str(run.ref.id) if run is not None else None,
        "run_path": str(run.ref.path.resolve()) if run is not None else None,
        "dataset_id": str(dataset.dataset.id) if dataset is not None else None,
        "dataset_path": (
            str(dataset.dataset.path.resolve()) if dataset is not None else None
        ),
        "model_name": model.name,
        "model_source": model.source,
        "model_revision": model.revision,
        "model_cache": str(model.cache) if model.cache is not None else None,
        "trust_remote_code": model.trust_remote_code,
        "model_digest": model_digest,
        "adapter_digest": assistant.adapter_digest,
        "training_seed": assistant.training_seed,
        "data": asdict(assistant.data),
    }


def _base_assistant(value: dict[str, Any]) -> Assistant:
    if any(
        value[name] is not None
        for name in (
            "run_id",
            "run_path",
            "dataset_id",
            "dataset_path",
            "adapter_digest",
            "training_seed",
        )
    ):
        raise TypeError
    cache = value["model_cache"]
    if cache is not None and not isinstance(cache, str):
        raise TypeError
    trust = value["trust_remote_code"]
    context = value["context_messages"]
    if not isinstance(trust, bool) or not _nonnegative_int(context) or context < 1:
        raise TypeError
    digest = _required_text(value, "model_digest")
    if not is_digest(digest):
        raise TypeError
    model = ModelSpec(
        _required_text(value, "model_name"),
        _required_text(value, "model_source"),
        _text(value, "model_revision"),
        Path(cache) if cache is not None else None,
        trust,
    )
    data = _train_data_config(value["data"])
    validate_prompt_config(data)
    assistant = Assistant(
        _required_text(value, "name"),
        _required_text(value, "target_name"),
        model_basename(model.name),
        None,
        None,
        context,
        model,
        data,
        digest,
    )
    return assistant


def _train_data_config(value: Any) -> TrainDataConfig:
    fields = {
        "format",
        "system_prompt",
        "prompt_role",
        "completion_role",
        "prompt_prefix",
        "prompt_suffix",
        "completion_prefix",
        "completion_suffix",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise TypeError
    if not all(isinstance(value[name], str) for name in fields):
        raise TypeError
    return TrainDataConfig(**value)


def _chat_value(config: ChatConfig) -> dict[str, Any]:
    value = asdict(config)
    value.pop("specified")
    value.pop("unknown")
    value["output"] = str(config.output) if config.output is not None else None
    return value


def _chat_config(value: Any) -> ChatConfig:
    if not isinstance(value, dict) or set(value) != {
        "output",
        "seed",
        "participant_name",
        "context_policy",
        "history",
        "default_model",
        "default_name",
        "default_context_messages",
        "default_system_prompt",
    }:
        raise TypeError
    output = value["output"]
    if output is not None and not isinstance(output, str):
        raise TypeError
    if (
        not isinstance(value["seed"], int)
        or isinstance(value["seed"], bool)
        or not all(
            isinstance(value[name], str)
            for name in (
                "participant_name",
                "context_policy",
                "history",
                "default_model",
                "default_name",
                "default_system_prompt",
            )
        )
        or not isinstance(value["default_context_messages"], int)
        or isinstance(value["default_context_messages"], bool)
    ):
        raise TypeError
    return ChatConfig(
        output=Path(output) if isinstance(output, str) else None,
        seed=value["seed"],
        participant_name=value["participant_name"],
        context_policy=value["context_policy"],
        history=value["history"],
        default_model=value["default_model"],
        default_name=value["default_name"],
        default_context_messages=value["default_context_messages"],
        default_system_prompt=value["default_system_prompt"],
    )


def _generation_config(value: Any) -> GenerationConfig:
    if not isinstance(value, dict) or set(value) != _GENERATION_KEYS:
        raise TypeError
    return GenerationConfig(**value)


def _backend_versions(value: object) -> Mapping[str, str | None]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and (item is None or isinstance(item, str))
        for name, item in value.items()
    ):
        raise TypeError
    return dict(value)


def _turn_value(turn: ChatTurn) -> dict[str, Any]:
    return asdict(turn)


def _read_turns(path: Path) -> tuple[ChatTurn, ...]:
    turns = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
            if not isinstance(value, dict) or set(value) != {
                "role",
                "content",
                "attempt",
                "finish_reason",
                "seed",
                "telemetry",
                "reference",
            }:
                raise TypeError
            telemetry_value = value["telemetry"]
            telemetry = (
                TurnTelemetry(**telemetry_value)
                if isinstance(telemetry_value, dict)
                else None
            )
            turn = ChatTurn(
                value["role"],
                value["content"],
                value["attempt"],
                value["finish_reason"],
                value["seed"],
                telemetry,
                value["reference"],
            )
        except Exception as error:
            raise ChatStorageError(
                f"Chat turn is not valid: {path}:{number}"
            ) from error
        if turn.role not in {"user", "assistant", "env"} or not isinstance(
            turn.content, str
        ):
            raise ChatStorageError(f"Chat turn is not valid: {path}:{number}")
        if turn.role in {"user", "env"} and (
            turn.attempt != 0
            or turn.finish_reason is not None
            or turn.seed is not None
            or turn.telemetry is not None
        ):
            label = "user" if turn.role == "user" else "ENV"
            raise ChatStorageError(f"Chat {label} turn is not valid: {path}:{number}")
        if turn.reference is not None and (
            not isinstance(turn.reference, int)
            or isinstance(turn.reference, bool)
            or turn.reference < 0
        ):
            raise ChatStorageError(f"Chat turn reference is not valid: {path}:{number}")
        if turn.role != "user" and turn.reference is not None:
            label = "assistant" if turn.role == "assistant" else "ENV"
            raise ChatStorageError(f"Chat {label} turn is not valid: {path}:{number}")
        if turn.role == "assistant" and (
            turn.finish_reason is None
            or turn.finish_reason not in {"stop", "length", "cancelled"}
            or not _valid_seed(turn.seed)
            or not _nonnegative_int(turn.attempt)
            or not _valid_telemetry(turn.telemetry)
        ):
            raise ChatStorageError(f"Chat assistant turn is not valid: {path}:{number}")
        if turn.role == "env":
            previous = next(
                (item for item in reversed(turns) if item.role != "env"),
                None,
            )
            if previous is not None and previous.role == "user":
                raise ChatStorageError(f"Chat ENV turn is not valid: {path}:{number}")
        turns.append(turn)
    model_turns = tuple(turn for turn in turns if turn.role != "env")
    if (model_turns and model_turns[0].role != "user") or any(
        turn.role == model_turns[index - 1].role
        for index, turn in enumerate(model_turns[1:], 1)
    ):
        raise ChatStorageError(f"Chat transcript order is not valid: {path}")
    bubbles = enumerate_bubbles(turns)
    indexes = {bubble.index for bubble in bubbles}
    current = 0
    for turn in turns:
        if turn.reference is not None and (
            turn.reference not in indexes or turn.reference >= current
        ):
            raise ChatStorageError(f"Chat turn reference is not valid: {path}")
        if turn.role == "user":
            current += 1
        elif turn.role == "assistant":
            segments = tuple(
                segment for segment in split_bubbles(turn.content) if segment.strip()
            )
            current += len(segments) or 1
    return tuple(turns)


def _title(turns: list[ChatTurn], explicit: str | None) -> str:
    if explicit is not None:
        value = " ".join(explicit.split())
        if not value:
            raise ChatStorageError("A chat title must contain text")
        return _display_prefix(value, 64)
    first = next((turn.content for turn in turns if turn.role == "user"), "Chat")
    value = " ".join(first.split()) or "Chat"
    return _display_prefix(value, 48)


def _display_prefix(value: str, limit: int) -> str:
    if sum(_character_width(character) for character in value) <= limit:
        return value
    result = []
    width = 0
    for character in value:
        size = _character_width(character)
        if width + size > limit - 1:
            break
        result.append(character)
        width += size
    return "".join(result).rstrip() + "…"


def _character_width(character: str) -> int:
    if unicodedata.combining(character):
        return 0
    return 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1


def _required_text(value: dict[str, Any], name: str) -> str:
    result = _text(value, name)
    if not result:
        raise TypeError
    return result


def _text(value: dict[str, Any], name: str) -> str:
    result = value[name]
    if not isinstance(result, str):
        raise TypeError
    return result


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return (
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows
        )
        + "\n"
    ).encode()


def _nonnegative_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_seed(value: object) -> bool:
    return _nonnegative_int(value) and value <= 0x7FFF_FFFF


def _valid_telemetry(value: TurnTelemetry | None) -> bool:
    if value is None or not _nonnegative_int(value.generated_tokens):
        return False
    if not _nonnegative_int(value.prompt_tokens) or value.prompt_tokens < 1:
        return False
    optional = (
        value.prompt_throughput,
        value.generation_throughput,
        value.time_to_first_token,
        value.generation_time,
        value.peak_memory,
    )
    return all(
        item is None
        or (
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(item)
            and item >= 0
        )
        for item in optional
    )
