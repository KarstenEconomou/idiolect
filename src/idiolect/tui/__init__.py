"""Provide the Textual presentation for local chat."""

from collections.abc import Callable, Iterable

from idiolect.chat.discovery import Assistant, DiscoveryItem
from idiolect.chat.runtime import ChatRuntime
from idiolect.chat.storage import ChatStore, SavedChat
from idiolect.config import ChatConfig, GenerationConfig


class ChatTuiError(RuntimeError):
    """Report an unavailable terminal chat interface."""


def run_chat_app(
    chat: ChatConfig,
    generation: GenerationConfig,
    assistants: Iterable[DiscoveryItem] = (),
    store: ChatStore | None = None,
    runtime_factory: Callable[..., ChatRuntime] = ChatRuntime,
    initial_assistant: Assistant | None = None,
    initial_chat: SavedChat | None = None,
) -> None:
    """Import and run the optional Textual application."""
    try:
        from idiolect.tui.app import ChatApp
    except ModuleNotFoundError as error:
        if error.name is None or not (
            error.name == "rich"
            or error.name == "textual"
            or error.name.startswith("rich.")
            or error.name.startswith("textual.")
        ):
            raise
        raise ChatTuiError(
            "Chat packages are not installed. Run: just setup-chat"
        ) from error
    ChatApp(
        chat,
        generation,
        assistants,
        store,
        runtime_factory,
        initial_assistant,
        initial_chat,
    ).run(mouse=True)


__all__ = ["ChatTuiError", "run_chat_app"]
