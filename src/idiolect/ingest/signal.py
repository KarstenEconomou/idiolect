"""Read and parse Signal messages."""

import hashlib
import json
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from idiolect.config import SignalConfig
from idiolect.types import (
    Attachment,
    ChatId,
    Event,
    EventId,
    Mention,
    Message,
    MessageId,
    PersonId,
    Quote,
    Reaction,
    Record,
)


class SignalError(RuntimeError):
    """Report a Signal input error."""


@dataclass(frozen=True, slots=True)
class SignalGroup:
    """Keep one Signal group entry."""

    id: str
    name: str
    active: bool


class Runner(Protocol):
    """Run commands for a Signal client."""

    def run(self, command: Sequence[str]) -> bytes:
        """Run one command and return its output."""
        ...

    def lines(self, command: Sequence[str]) -> Iterable[bytes]:
        """Run one command and return its output lines."""
        ...


class SubprocessRunner:
    """Run Signal commands as child processes."""

    def run(self, command: Sequence[str]) -> bytes:
        """Run one command and return its output."""
        try:
            result = subprocess.run(command, check=False, capture_output=True)
        except FileNotFoundError as error:
            raise SignalError(f"Signal program does not exist: {command[0]}") from error
        if result.returncode != 0:
            detail = result.stderr.decode(errors="replace").strip()
            raise SignalError(detail or f"Signal command failed with code {result.returncode}")
        return result.stdout

    def lines(self, command: Sequence[str]) -> Iterable[bytes]:
        """Run one command and yield its output lines."""
        try:
            with tempfile.TemporaryFile() as errors:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
                if process.stdout is None:
                    raise SignalError("Signal command has no output stream")
                try:
                    yield from process.stdout
                    return_code = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
                if return_code != 0:
                    errors.seek(0)
                    detail = errors.read().decode(errors="replace").strip()
                    raise SignalError(detail or f"Signal command failed with code {return_code}")
        except FileNotFoundError as error:
            raise SignalError(f"Signal program does not exist: {command[0]}") from error


class SignalSource:
    """Read queued events with signal-cli."""

    def __init__(
        self,
        config: SignalConfig,
        runner: Runner | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Set the Signal client options."""
        if not config.account:
            raise SignalError("Set IDIOLECT_SIGNAL_ACCOUNT before Signal collection")
        if config.timeout < -1:
            raise SignalError("Signal timeout must be -1 or greater")
        if config.max_messages is not None and config.max_messages < 1:
            raise SignalError("Signal max message count must be greater than zero")
        self._config = config
        self._runner = SubprocessRunner() if runner is None else runner
        self._clock = _utc_now if clock is None else clock

    def groups(self) -> tuple[SignalGroup, ...]:
        """Return groups that the Signal account knows."""
        output = self._runner.run((*self._base_command(), "listGroups"))
        value = _json_value(output)
        if not isinstance(value, list):
            raise SignalError("Signal group output must be a list")
        groups: list[SignalGroup] = []
        for item in value:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise SignalError("Signal group output has an invalid group")
            groups.append(
                SignalGroup(
                    id=item["id"],
                    name=item.get("name") if isinstance(item.get("name"), str) else "",
                    active=bool(item.get("isMember", True)) and not bool(item.get("isBlocked", False)),
                )
            )
        return tuple(groups)

    def events(self) -> Iterable[Event]:
        """Return queued Signal events."""
        command = [
            *self._base_command(),
            "receive",
            "--timeout",
            str(self._config.timeout),
            "--ignore-attachments",
            "--ignore-stories",
            "--ignore-avatars",
            "--ignore-stickers",
        ]
        if self._config.max_messages is not None:
            command.extend(("--max-messages", str(self._config.max_messages)))
        return self._events_from_lines(self._runner.lines(command))

    def _base_command(self) -> list[str]:
        command = [self._config.binary, "--output", "json"]
        if self._config.data_dir is not None:
            command.extend(("--data-dir", str(self._config.data_dir)))
        command.extend(("--account", cast(str, self._config.account)))
        return command

    def _events_from_lines(self, lines: Iterable[bytes]) -> Iterator[Event]:
        # One malformed line must not discard the drained events after it.
        for line in lines:
            payload = line.strip()
            if not payload:
                continue
            try:
                value = _json_value(payload)
            except SignalError:
                continue
            if not isinstance(value, dict):
                continue
            digest = hashlib.sha256(payload).hexdigest()
            yield Event(
                id=EventId(f"signal:{digest}"),
                source="signal-cli",
                source_id=_source_id(value, digest),
                received_at=self._clock(),
                payload=payload,
            )


class SignalFileSource:
    """Read Signal JSON lines from one file."""

    def __init__(self, path: Path, clock: Callable[[], datetime] | None = None) -> None:
        """Set the Signal file path."""
        self._path = path
        self._clock = _utc_now if clock is None else clock

    def events(self) -> Iterable[Event]:
        """Return Signal events from the file."""
        try:
            with self._path.open("rb") as stream:
                for line in stream:
                    payload = line.strip()
                    if not payload:
                        continue
                    try:
                        value = _json_value(payload)
                    except SignalError:
                        continue
                    if not isinstance(value, dict):
                        continue
                    digest = hashlib.sha256(payload).hexdigest()
                    yield Event(
                        id=EventId(f"signal:{digest}"),
                        source="signal-cli-file",
                        source_id=_source_id(value, digest),
                        received_at=self._clock(),
                        payload=payload,
                    )
        except OSError as error:
            raise SignalError(f"Cannot read Signal file: {self._path}") from error


class SignalParser:
    """Convert Signal events to normalized records."""

    def __init__(self, chats: Sequence[ChatId]) -> None:
        """Set the group whitelist."""
        if not chats:
            raise SignalError("Add at least one Signal group to the chat whitelist")
        self._chats = frozenset(str(chat) for chat in chats)

    def records(self, event: Event) -> Iterable[Record]:
        """Return allowed records from one Signal event."""
        value = _json_value(event.payload)
        root = _receive_root(value)
        envelope = root.get("envelope")
        if not isinstance(envelope, dict):
            return ()

        message_data, edit_target = _message_data(envelope)
        if message_data is None:
            return ()
        group = message_data.get("groupInfo")
        if not isinstance(group, dict) or not isinstance(group.get("groupId"), str):
            return ()
        raw_chat_id = group["groupId"]
        if raw_chat_id not in self._chats:
            return ()

        raw_author = _author(envelope, root)
        if raw_author is None:
            raise SignalError("Signal group message has no author")
        chat_id = _chat_id(raw_chat_id)
        author_id = _person_id(raw_author)
        author_name = _source_name(envelope)
        is_self = _is_self(envelope)
        event_time = _time(_required_timestamp(envelope, message_data))

        reaction = message_data.get("reaction")
        if isinstance(reaction, dict):
            return _reaction_records(event, reaction, chat_id, author_id, event_time)

        remote_delete = message_data.get("remoteDelete")
        if isinstance(remote_delete, dict):
            target = _integer(remote_delete.get("timestamp"), "remote delete timestamp")
            return (
                Message(
                    id=_message_id(chat_id, author_id, target),
                    event_id=event.id,
                    chat_id=chat_id,
                    author_id=author_id,
                    sent_at=_time(target),
                    author_name=author_name,
                    is_self=is_self,
                    deleted_at=event_time,
                ),
            )

        sent_timestamp = edit_target or _required_timestamp(message_data, envelope)
        raw_text = message_data.get("message")
        if raw_text is not None and not isinstance(raw_text, str):
            raise SignalError("Signal message text must be text or null")
        mentions = _mentions(raw_text, message_data.get("mentions"))
        text = raw_text
        attachments = _attachments(message_data.get("attachments"))
        if text is None and not attachments:
            return ()
        reply_to, quote = _reply(message_data.get("quote"), chat_id)
        return (
            Message(
                id=_message_id(chat_id, author_id, sent_timestamp),
                event_id=event.id,
                chat_id=chat_id,
                author_id=author_id,
                sent_at=_time(sent_timestamp),
                author_name=author_name,
                is_self=is_self,
                text=text,
                reply_to=reply_to,
                edited_at=event_time if edit_target is not None else None,
                attachments=attachments,
                mentions=mentions,
                quote=quote,
            ),
        )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _json_value(payload: bytes) -> Any:
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SignalError("Signal output is not valid JSON") from error


def _source_id(value: dict[str, Any], fallback: str) -> str:
    root = _receive_root(value)
    envelope = root.get("envelope")
    if not isinstance(envelope, dict):
        return fallback
    timestamp = envelope.get("timestamp")
    source = envelope.get("sourceUuid") or envelope.get("sourceNumber") or envelope.get("source")
    return f"{source or 'unknown'}:{timestamp or fallback}"


def _receive_root(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SignalError("Signal event must be a JSON object")
    if value.get("method") != "receive":
        return value
    params = value.get("params")
    if not isinstance(params, dict):
        raise SignalError("Signal receive event has no parameters")
    result = params.get("result")
    return result if isinstance(result, dict) else params


def _message_data(envelope: dict[str, Any]) -> tuple[dict[str, Any] | None, int | None]:
    edit = envelope.get("editMessage")
    if isinstance(edit, dict):
        data = edit.get("dataMessage")
        if isinstance(data, dict):
            target = _integer(edit.get("targetSentTimestamp"), "edit target timestamp")
            return data, target
    data = envelope.get("dataMessage")
    if isinstance(data, dict):
        return data, None
    sync = envelope.get("syncMessage")
    if isinstance(sync, dict):
        sent = sync.get("sentMessage")
        if isinstance(sent, dict):
            return sent, None
    return None, None


def _author(envelope: dict[str, Any], root: dict[str, Any]) -> str | None:
    sync = envelope.get("syncMessage")
    if isinstance(sync, dict) and isinstance(sync.get("sentMessage"), dict):
        value = envelope.get("sourceUuid") or root.get("account")
    else:
        value = envelope.get("sourceUuid") or envelope.get("sourceNumber") or envelope.get("source")
    return value if isinstance(value, str) else None


def _source_name(envelope: dict[str, Any]) -> str | None:
    value = envelope.get("sourceName")
    return value if isinstance(value, str) else None


def _is_self(envelope: dict[str, Any]) -> bool:
    sync = envelope.get("syncMessage")
    return isinstance(sync, dict) and isinstance(sync.get("sentMessage"), dict)


def _required_timestamp(first: dict[str, Any], second: dict[str, Any]) -> int:
    value = first.get("timestamp", second.get("timestamp"))
    return _integer(value, "message timestamp")


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SignalError(f"Signal {name} must be an integer")
    return value


def _time(milliseconds: int) -> datetime:
    try:
        return datetime.fromtimestamp(milliseconds / 1000, UTC)
    except (OSError, OverflowError, ValueError) as error:
        raise SignalError("Signal timestamp is outside the valid range") from error


def _chat_id(value: str) -> ChatId:
    return ChatId(f"signal-chat:{hashlib.sha256(value.encode()).hexdigest()}")


def _person_id(value: str) -> PersonId:
    return PersonId(f"signal-person:{hashlib.sha256(value.encode()).hexdigest()}")


def _message_id(chat_id: ChatId, author_id: PersonId, milliseconds: int) -> MessageId:
    value = f"{chat_id}\0{author_id}\0{milliseconds}".encode()
    return MessageId(f"signal-message:{hashlib.sha256(value).hexdigest()}")


def _reply(value: Any, chat_id: ChatId) -> tuple[MessageId | None, Quote | None]:
    if not isinstance(value, dict):
        return None, None
    author = value.get("authorUuid") or value.get("authorNumber") or value.get("author")
    timestamp = value.get("id")
    if not isinstance(author, str) or not isinstance(timestamp, int):
        return None, None
    raw_text = value.get("text", value.get("message"))
    if raw_text is not None and not isinstance(raw_text, str):
        raise SignalError("Signal quote text must be text or null")
    mentions = _mentions(raw_text, value.get("mentions"))
    return (
        _message_id(chat_id, _person_id(author), timestamp),
        Quote(
            author_id=_person_id(author),
            sent_at=_time(timestamp),
            text=raw_text,
            mentions=mentions,
        ),
    )


def _mentions(text: str | None, value: Any) -> tuple[Mention, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise SignalError("Signal mentions must be a list")
    if not value:
        return ()
    if text is None:
        raise SignalError("Signal mentions require message text")

    mentions: list[Mention] = []
    for item in value:
        if not isinstance(item, dict):
            raise SignalError("Signal mention must be an object")
        start = _integer(item.get("start"), "mention start")
        length = _integer(item.get("length"), "mention length")
        identity = item.get("uuid") or item.get("number") or item.get("recipient")
        if not isinstance(identity, str):
            raise SignalError("Signal mention has no identity")
        name = _mention_name(item.get("name"))
        if start < 0 or length < 1:
            raise SignalError("Signal mention range is not valid")
        _utf16_index(text, start)
        _utf16_index(text, start + length)
        mentions.append(Mention(_person_id(identity), start, length, name))

    cursor = 0
    for mention in sorted(mentions, key=lambda item: item.start_utf16):
        if mention.start_utf16 < cursor:
            raise SignalError("Signal mention ranges overlap")
        cursor = mention.start_utf16 + mention.length_utf16
    return tuple(mentions)


def _mention_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    name = " ".join(value.strip().lstrip("@").split())
    if not name or name == "\ufffc":
        return None
    return name


def _utf16_index(text: str, units: int) -> int:
    used = 0
    for index, character in enumerate(text):
        if used == units:
            return index
        used += 2 if ord(character) > 0xFFFF else 1
        if used > units:
            raise SignalError("Signal mention splits one Unicode character")
    if used == units:
        return len(text)
    raise SignalError("Signal mention range is outside message text")


def _attachments(value: Any) -> tuple[Attachment, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise SignalError("Signal attachments must be a list")
    attachments: list[Attachment] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise SignalError("Signal attachment must be an object")
        source_id = item.get("id")
        if not isinstance(source_id, str):
            source_id = f"{index}:{item.get('filename', '')}"
        attachment_id = hashlib.sha256(source_id.encode()).hexdigest()
        media_type = item.get("contentType")
        name = item.get("filename")
        size = item.get("size")
        attachments.append(
            Attachment(
                id=f"signal-attachment:{attachment_id}",
                media_type=media_type if isinstance(media_type, str) else None,
                name=name if isinstance(name, str) else None,
                size=size if isinstance(size, int) and not isinstance(size, bool) else None,
            )
        )
    return tuple(attachments)


def _reaction_records(
    event: Event,
    value: dict[str, Any],
    chat_id: ChatId,
    author_id: PersonId,
    sent_at: datetime,
) -> tuple[Reaction, ...]:
    target_author = value.get("targetAuthorUuid") or value.get("targetAuthorNumber") or value.get("targetAuthor")
    target_time = value.get("targetSentTimestamp")
    emoji = value.get("emoji")
    if not isinstance(target_author, str) or not isinstance(target_time, int) or not isinstance(emoji, str):
        raise SignalError("Signal reaction has invalid target data")
    return (
        Reaction(
            event_id=event.id,
            message_id=_message_id(chat_id, _person_id(target_author), target_time),
            chat_id=chat_id,
            author_id=author_id,
            value=emoji,
            sent_at=sent_at,
            removed=bool(value.get("isRemove", False)),
        ),
    )
