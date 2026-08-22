"""Provide the Textual presentation for local chat."""


class ChatTuiError(RuntimeError):
    """Report an unavailable terminal chat interface."""


def run_chat_app(*args, **kwargs) -> None:
    """Import and run the optional Textual application."""
    try:
        from idiolect.tui.app import ChatApp
    except ImportError as error:
        raise ChatTuiError(
            "Chat packages are not installed. Run: just setup-chat"
        ) from error
    ChatApp(*args, **kwargs).run()


__all__ = ["ChatTuiError", "run_chat_app"]
