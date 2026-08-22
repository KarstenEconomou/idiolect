"""Render the local Idiolect chat terminal interface."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import asdict
from typing import ClassVar

from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
from textual.widgets import OptionList, Rule, Static, TextArea
from textual.widgets.option_list import Option

from idiolect.chat.discovery import Assistant, DiscoveryItem
from idiolect.chat.runtime import ChatError, ChatRuntime
from idiolect.chat.state import ChatSession
from idiolect.chat.storage import ChatStorageError, ChatStore, SavedChat
from idiolect.chat.worker import WorkerError
from idiolect.config import ChatConfig, GenerationConfig
from idiolect.tui.catalog import CatalogLayout
from idiolect.tui.commands import COMMANDS, CommandError, completions, parse_command
from idiolect.tui.widgets import (
    Composer,
    ConfirmModal,
    InfoModal,
    KeyboardOptionList,
    LoadingStatus,
)

WATERMARK = """     ╭─╮
  ╭──╯ ╰──╮
  │  · ·  │    IDIOLECT
  ╰──╮ ╭──╯    someone, reconstructed.
     ╰─╯"""


class ChatApp(App[None]):
    """Run the assistant registry and local chat screen."""

    CSS = """
    $terminal: ansi_default;
    $accent: ansi_blue;
    $metadata: ansi_bright_black;
    $failure: ansi_red;
    Screen { background: $terminal; color: $terminal; }
    #landing { align: center middle; padding: 1 2; }
    #landing-box { width: 100%; max-width: 120; height: 100%; background: $terminal; }
    #watermark { color: $accent; height: 5; padding: 0 2; text-align: left; text-style: bold; }
    #catalog-heading { height: 1; margin-top: 1; padding: 0 2; }
    #catalog-title { text-style: bold; }
    #catalog-subtitle { height: 1; padding: 0 2; color: $metadata; }
    #catalog-description { width: 1fr; }
    #catalog-summary { width: auto; }
    #catalog-rule { height: 1; margin: 0; padding: 0 2; color: $metadata; }
    #catalog-columns { height: 1; padding: 0 2; color: $metadata; text-style: bold; }
    #load-status { display: none; height: 1; padding: 0 2; color: $accent; text-style: bold; }
    #chooser { height: 1fr; padding: 0 2; border: none; color: $terminal; background: $terminal; background-tint: transparent; scrollbar-color: $metadata; scrollbar-background: $terminal; }
    OptionList > .option-list--option { padding: 0; color: $terminal; background: $terminal; }
    OptionList > .option-list--option-highlighted, OptionList:focus > .option-list--option-highlighted { color: $accent; background: $terminal; text-style: bold; }
    OptionList > .option-list--option-disabled { color: $failure; text-style: dim; }
    OptionList > .option-list--option-hover { color: $accent; background: $terminal; text-style: bold; }
    #catalog-hints { height: 1; padding: 0 2; color: $metadata; }
    #chat { display: none; }
    #identity { height: 2; padding: 0 2; color: $accent; background: $terminal; border-bottom: solid $metadata; text-style: bold; }
    #transcript-scroll { height: 1fr; padding: 1 2; background: $terminal; scrollbar-size: 0 0; }
    #transcript { width: 100%; height: auto; background: $terminal; }
    #composer { height: auto; min-height: 3; max-height: 10; border: solid $metadata; margin: 0 1; padding: 0 1; background: $terminal; scrollbar-size: 0 0; }
    #composer:focus { border: solid $accent; }
    #composer .text-area--cursor, #composer .text-area--selection { color: $terminal; background: $terminal; text-style: reverse; }
    #composer .text-area--cursor-line, #composer .text-area--matching-bracket { background: $terminal; }
    #composer .text-area--gutter, #composer .text-area--suggestion, #composer .text-area--placeholder { color: $metadata; background: $terminal; }
    #completion { height: auto; max-height: 5; color: $accent; background: $terminal; padding: 0 2; }
    #status { display: none; height: 1; color: $accent; background: $terminal; padding: 0 1; text-style: bold; }
    #footer { height: 1; color: $metadata; background: $terminal; padding: 0 1; }
    #info-dialog { width: 80%; max-width: 90; height: auto; padding: 1 2; background: $terminal; border: solid $metadata; }
    #info-title { color: $accent; text-style: bold; border-bottom: solid $metadata; }
    #info-body { max-height: 70vh; overflow-y: auto; }
    #confirm-dialog { width: 100%; height: 2; padding: 0 1; background: $terminal; border: none; }
    #confirm-message { height: 1; color: $metadata; }
    #confirm-actions { height: 1; }
    #confirm-actions Button { width: auto; min-width: 0; height: 1; padding: 0 1; border: none; background: $terminal; color: $metadata; text-style: none; }
    #confirm-actions Button:hover { border: none; background: $terminal; color: $metadata; text-style: none; }
    #confirm-actions Button:focus, #confirm-actions Button.-active { border: none; background: $terminal; color: $accent; text-style: bold; }
    Button { border: tall $metadata; background: $terminal; color: $terminal; }
    Button:hover, Button:focus, Button.-active { border: tall $accent; background: $terminal; color: $terminal; background-tint: transparent; tint: transparent; text-style: reverse bold; }
    ModalScreen { background: transparent; }
    Toast { color: $terminal; background: $terminal; border-left: outer $accent; }
    Toast .toast--title { color: $accent; }
    Toast.-error { border-left: outer $failure; }
    Toast.-error .toast--title { color: $failure; }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "stop", "Stop"),
        Binding("ctrl+c", "interrupt", "Stop or quit"),
        Binding(
            "ctrl+up",
            "scroll_transcript_up",
            "Scroll chat up",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+down",
            "scroll_transcript_down",
            "Scroll chat down",
            show=False,
            priority=True,
        ),
    ]

    def __init__(
        self,
        chat: ChatConfig,
        generation: GenerationConfig,
        assistants: Iterable[DiscoveryItem] = (),
        store: ChatStore | None = None,
        runtime_factory: Callable[..., ChatRuntime] = ChatRuntime,
        initial_assistant: Assistant | None = None,
        initial_chat: SavedChat | None = None,
    ) -> None:
        super().__init__(ansi_color=True)
        self.chat_policy = chat
        self.generation = generation
        self.assistants = tuple(assistants)
        self.store = ChatStore(chat.output) if store is None and chat.output else store
        self.runtime = runtime_factory(chat, generation)
        self.initial_assistant = initial_assistant
        self.initial_chat = initial_chat
        self._rows: dict[str, DiscoveryItem | SavedChat] = {}
        self._generating = False
        self._loading = False
        self._streaming_text = ""
        self._active_attempt = 0
        self._load_status_text: str | None = None
        self._status_text: str | None = None
        self._footer_text: str | None = None
        self._catalog_width: int | None = None
        self._loading_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        """Create the landing and chat screen widgets."""
        with Container(id="landing"), Vertical(id="landing-box"):
            yield Static(WATERMARK, markup=False, id="watermark")
            with Horizontal(id="catalog-heading"):
                yield Static("REGISTRY", markup=False, id="catalog-title")
            with Horizontal(id="catalog-subtitle"):
                yield Static(
                    "Choose a persona, adapter, or saved snapshot.",
                    markup=False,
                    id="catalog-description",
                )
                yield Static("", markup=False, id="catalog-summary")
            yield Rule(line_style="solid", id="catalog-rule")
            yield Static("", markup=False, id="catalog-columns")
            yield LoadingStatus(id="load-status")
            yield KeyboardOptionList(id="chooser")
            yield Static(
                "↑↓ move · Enter select · Esc stop · Ctrl+C quit",
                markup=False,
                id="catalog-hints",
            )
        with Container(id="chat"):
            yield Static("", markup=False, id="identity")
            with VerticalScroll(id="transcript-scroll"):
                yield Static("", markup=False, id="transcript")
            yield Static("", markup=False, id="completion")
            yield LoadingStatus(id="status")
            yield Composer(id="composer", language=None)
            yield Static("", markup=False, id="footer")

    def on_mount(self) -> None:
        """Populate the chooser or open a direct selection."""
        self._fill_chooser()
        self._loading_timer = self.set_interval(0.1, self._refresh_loading_state)
        if self.initial_chat is not None:
            self._begin_attach(self._saved_session(self.initial_chat))
        elif self.initial_assistant is not None:
            self._begin_select(self.initial_assistant)

    def on_unmount(self) -> None:
        """Stop the loading-state timer before the screen is removed."""
        if self._loading_timer is not None:
            self._loading_timer.stop()
            self._loading_timer = None

    def on_resize(self, event: events.Resize) -> None:
        """Update catalog columns when the terminal width changes."""
        if (
            self._catalog_width is not None
            and event.size.width != self._catalog_width
            and self.query_one("#landing").display
        ):
            self.call_after_refresh(self._fill_chooser)
        elif self.query_one("#chat").display:
            self.call_after_refresh(self._update_footer)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Open the selected assistant or saved chat."""
        if self._loading:
            return
        self._open_row(str(event.option.id))

    def _open_row(self, key: str) -> None:
        row = self._rows.get(key)
        if isinstance(row, DiscoveryItem) and row.assistant is not None:
            self._begin_select(row.assistant)
        elif isinstance(row, SavedChat):
            self._begin_attach(self._saved_session(row))

    @staticmethod
    def _saved_session(saved: SavedChat) -> ChatSession:
        return ChatSession(
            saved.assistant,
            saved.chat,
            saved.generation,
            saved.turns,
            saved.id,
            saved.title,
        )

    def on_composer_changed(self, event: TextArea.Changed) -> None:
        """Show command completions for a slash prefix."""
        values = completions(event.text_area.text)
        self.query_one("#completion", Static).update("  ".join(values))

    def on_composer_submitted(self, event: Composer.Submitted) -> None:
        """Run one command or start one user turn."""
        value = event.value
        if not value.strip() or self._generating:
            return
        if self._loading:
            self.notify("Wait for the assistant to finish loading", severity="warning")
            return
        composer = self.query_one(Composer)
        try:
            command = parse_command(value)
            if command is not None:
                self._command(command.name, command.argument)
                composer.clear()
                return
            session = self._session()
            session.add_user(value)
            composer.clear()
            self._render_transcript()
            self._start_generation(0)
        except (
            ChatError,
            ChatStorageError,
            CommandError,
            ValueError,
            WorkerError,
        ) as error:
            self.notify(str(error), severity="error")

    def action_stop(self) -> None:
        """Stop active generation at a token boundary."""
        if self._generating:
            self.runtime.cancel()

    def action_interrupt(self) -> None:
        """Stop active work or open the idle quit confirmation."""
        if self._generating:
            self.runtime.cancel()
        else:
            self._request_quit()

    def action_scroll_transcript_up(self) -> None:
        """Move the chat viewport up without leaving the composer."""
        if self.query_one("#chat").display:
            self.query_one("#transcript-scroll", VerticalScroll).scroll_relative(
                y=-3,
                animate=False,
            )

    def action_scroll_transcript_down(self) -> None:
        """Move the chat viewport down without leaving the composer."""
        if self.query_one("#chat").display:
            self.query_one("#transcript-scroll", VerticalScroll).scroll_relative(
                y=3,
                animate=False,
            )

    def _start_generation(self, attempt: int) -> None:
        self._generating = True
        self._streaming_text = ""
        self._active_attempt = attempt
        self._set_status("generating")
        self._generate_thread(attempt)

    @work(thread=True, exclusive=True, group="generation")
    def _generate_thread(self, attempt: int) -> None:
        last_render = 0.0
        try:
            for delta in self.runtime.generate(attempt, self._report_prefill):
                if not self._streaming_text:
                    self.call_from_thread(self._set_status, "generating")
                self._streaming_text += delta
                now = time.monotonic()
                if now - last_render >= 1 / 12:
                    self.call_from_thread(self._render_transcript, True)
                    last_render = now
            self.call_from_thread(self._generation_done)
        except Exception as error:  # noqa: BLE001
            self.call_from_thread(self._generation_failed, str(error))

    def _generation_done(self) -> None:
        self._generating = False
        self._streaming_text = ""
        self._render_transcript()
        self._update_status()
        self._update_footer()

    def _generation_failed(self, message: str) -> None:
        self._generating = False
        self._set_status("failed")
        self.notify(f"{message}. Use /retry to reload and try again.", severity="error")

    def _report_prefill(self, current: int, total: int) -> None:
        self.call_from_thread(self._set_status, f"prefill {current}/{total}")

    def _show_chat(self) -> None:
        session = self._session()
        self.chat_policy = session.chat
        self.generation = session.generation
        self.query_one("#landing").display = False
        self.query_one("#chat").display = True
        self.query_one("#identity", Static).update(session.assistant.name)
        self._render_transcript()
        self._update_status()
        self._update_footer()
        self.query_one(Composer).focus()

    def _render_transcript(self, partial: bool = False) -> None:
        transcript = self.query_one("#transcript", Static)
        scroller = self.query_one("#transcript-scroll", VerticalScroll)
        follow_latest = scroller.scroll_y >= scroller.max_scroll_y - 1
        session = self._session()
        content = Text()
        for turn in session.turns:
            if content:
                content.append("\n\n")
            name = "USER" if turn.role == "user" else self._chat_name(session)
            content.append(f"{name}:", style="blue")
            content.append("\n")
            content.append(turn.content)
        if partial and self._generating:
            if content:
                content.append("\n\n")
            content.append(f"{self._chat_name(session)}:", style="blue")
            content.append("\n")
            content.append(self._streaming_text or "…")
        transcript.update(content)
        if follow_latest:
            self.call_after_refresh(self._scroll_transcript_end)

    def _scroll_transcript_end(self) -> None:
        self.query_one("#transcript-scroll", VerticalScroll).scroll_end(animate=False)

    @staticmethod
    def _chat_name(session: ChatSession) -> str:
        return session.assistant.target_name.upper()

    def _update_footer(self) -> None:
        session = self._session()
        last = next((turn for turn in reversed(session.turns) if turn.telemetry), None)
        if last is None or last.telemetry is None:
            value = ""
        else:
            telemetry = last.telemetry
            pressure = 100 * telemetry.prompt_tokens / self.generation.max_prompt_tokens
            fields = [
                (
                    "context "
                    f"{telemetry.prompt_tokens}/{self.generation.max_prompt_tokens} "
                    f"({pressure:.0f}%)"
                ),
                f"generated {telemetry.generated_tokens}",
            ]
            if self.size.width >= 80 and telemetry.generation_throughput is not None:
                fields.append(f"{telemetry.generation_throughput:.1f} tok/s")
            if self.size.width >= 105 and telemetry.peak_memory is not None:
                fields.append(f"peak {telemetry.peak_memory:.2f} GB")
            value = "    ".join(fields)
        self._set_footer(value)

    def _set_footer(self, value: str) -> None:
        value = value.upper()
        if value != self._footer_text:
            self.query_one("#footer", Static).update(value)
            self._footer_text = value

    def _update_status(self) -> None:
        self._set_status(self.runtime.state.value)

    def _set_status(self, value: str | None) -> None:
        normalized = (
            "" if value is None or value.casefold() == "ready" else value.upper()
        )
        if normalized != self._status_text:
            self.query_one("#status", LoadingStatus).set_state(
                normalized,
                animated=normalized not in {"CANCELLED", "FAILED"},
            )
            self._status_text = normalized

    def _command(self, name: str, argument: str | None) -> None:
        if name == "save":
            if self.store is None:
                raise ChatStorageError("Chat output is not configured")
            saved = self.store.save(self._session(), argument)
            self.notify(f"Saved {saved.id[:8]} — {saved.title}")
        elif name == "retry":
            session = self._session()
            if session.turns and session.turns[-1].role == "user":
                attempt = self._active_attempt + 1
                self._active_attempt = attempt
                self._render_transcript()
                self._begin_attach(session, generate_attempt=attempt)
                return
            else:
                attempt = session.retry()
            self._render_transcript()
            self._start_generation(attempt)
        elif name == "stats":
            self.push_screen(InfoModal("Chat statistics", self._stats_text()))
        elif name == "help":
            self.push_screen(InfoModal("Chat commands", "\n".join(COMMANDS)))
        elif name == "quit":
            self._request_quit()
        elif name == "new":
            self._request_new()
        elif name in {"assistant", "resume"}:
            self._return_to_landing()

    def _stats_text(self) -> str:
        session = self._session()
        assistant = session.assistant
        values = {
            "assistant": assistant.name,
            "mode": assistant.mode.value,
            "run_id": assistant.run_id,
            "dataset_id": assistant.dataset_id,
            "base_revision": assistant.model.revision,
            "model_digest": assistant.model_digest,
            "adapter_digest": assistant.adapter_digest,
            "training_seed": assistant.training_seed,
            "context_messages": assistant.context_messages,
            "dirty": session.dirty,
            "saved_chat_id": session.saved_chat_id,
            **self.runtime.probe,
            **asdict(self.runtime.stats),
        }
        last = next((turn for turn in reversed(session.turns) if turn.telemetry), None)
        if last is not None and last.telemetry is not None:
            values.update(
                {
                    "prompt_tokens_last": last.telemetry.prompt_tokens,
                    "generated_tokens_last": last.telemetry.generated_tokens,
                    "prompt_throughput": last.telemetry.prompt_throughput,
                    "generation_throughput": last.telemetry.generation_throughput,
                    "time_to_first_token": last.telemetry.time_to_first_token,
                    "generation_time": last.telemetry.generation_time,
                    "peak_memory": last.telemetry.peak_memory,
                    "finish_reason": last.finish_reason,
                    "rng_seed": last.seed,
                    "attempt": last.attempt,
                    "context_pressure": (
                        last.telemetry.prompt_tokens / self.generation.max_prompt_tokens
                    ),
                }
            )
        return "\n".join(
            f"{name}: {value}" for name, value in values.items() if value is not None
        )

    def _return_to_landing(self) -> None:
        if self._session().dirty:
            self.push_screen(ConfirmModal(), self._after_landing_confirm)
            return
        self.query_one("#chat").display = False
        self.query_one("#landing").display = True
        self._fill_chooser()

    def _request_new(self) -> None:
        if self._session().dirty:
            self.push_screen(ConfirmModal(), self._after_new_confirm)
        else:
            self._new_chat()

    def _after_new_confirm(self, choice: str | None) -> None:
        if choice == "save" and not self._save_from_confirmation():
            return
        if choice in {"save", "discard"}:
            self._new_chat()

    def _new_chat(self) -> None:
        current = self._session()
        self.runtime.session = ChatSession(
            current.assistant,
            self.chat_policy,
            self.generation,
        )
        self._render_transcript()
        self._update_status()
        self._update_footer()

    def _after_landing_confirm(self, choice: str | None) -> None:
        if choice == "save" and not self._save_from_confirmation():
            return
        if choice in {"save", "discard"}:
            self.query_one("#chat").display = False
            self.query_one("#landing").display = True
            self._fill_chooser()

    def _request_quit(self) -> None:
        if self.runtime.session is not None and self.runtime.session.dirty:
            self.push_screen(ConfirmModal(), self._after_quit_confirm)
        else:
            self.runtime.close()
            self.exit()

    def _after_quit_confirm(self, choice: str | None) -> None:
        if choice == "save" and not self._save_from_confirmation():
            return
        if choice in {"save", "discard"}:
            self.runtime.close()
            self.exit()

    def _save_from_confirmation(self) -> bool:
        if self.store is None:
            self.notify("Chat output is not configured", severity="error")
            return False
        try:
            saved = self.store.save(self._session())
        except ChatStorageError as error:
            self.notify(str(error), severity="error")
            return False
        self.notify(f"Saved {saved.id[:8]} — {saved.title}")
        return True

    def _begin_select(self, assistant: Assistant) -> None:
        if not self._prepare_load():
            return
        self._set_loading(True)
        self._select_thread(assistant)

    def _begin_attach(
        self,
        session: ChatSession,
        *,
        generate_attempt: int | None = None,
    ) -> None:
        if not self._prepare_load():
            return
        self.runtime.session = session
        self._set_loading(True)
        self._show_chat()
        self._attach_thread(session, generate_attempt)

    def _prepare_load(self) -> bool:
        try:
            self.runtime.ensure_worker()
        except Exception as error:  # noqa: BLE001
            self._load_failed(str(error))
            return False
        return True

    @work(thread=True, exclusive=True, group="model-load")
    def _select_thread(self, assistant: Assistant) -> None:
        try:
            self.runtime.select(assistant)
        except Exception as error:  # noqa: BLE001
            self.call_from_thread(self._load_failed, str(error))
            return
        self.call_from_thread(self._load_done, None)

    @work(thread=True, exclusive=True, group="model-load")
    def _attach_thread(
        self,
        session: ChatSession,
        generate_attempt: int | None,
    ) -> None:
        try:
            self.runtime.attach(session)
        except Exception as error:  # noqa: BLE001
            self.call_from_thread(self._load_failed, str(error))
            return
        self.call_from_thread(self._load_done, generate_attempt)

    def _load_done(self, generate_attempt: int | None) -> None:
        self._set_loading(False)
        self._show_chat()
        if generate_attempt is not None:
            self._start_generation(generate_attempt)

    def _load_failed(self, message: str) -> None:
        self._set_loading(False)
        if self.runtime.session is not None:
            self._show_chat()
        self._set_status("failed")
        self.notify(message, severity="error")

    def _set_loading(self, value: bool) -> None:
        self._loading = value
        self.query_one("#chooser", OptionList).disabled = value
        self._refresh_loading_state()

    def _refresh_loading_state(self) -> None:
        if not self.is_mounted or len(self.query("#load-status")) == 0:
            return
        state = self.runtime.state.value
        status = state.upper() if self._loading else ""
        if status != self._load_status_text:
            self.query_one("#load-status", LoadingStatus).set_state(status)
            self._load_status_text = status
        if self._loading and self.query_one("#chat").display:
            self._set_status(state)

    def _fill_chooser(self) -> None:
        self._catalog_width = self.size.width
        chooser = self.query_one("#chooser", OptionList)
        chooser.clear_options()
        self._rows.clear()
        options = []
        available = 0
        saved_chats = () if self.store is None else self.store.leaves()
        layout = CatalogLayout.for_terminal(self.size.width)
        self.query_one("#catalog-columns", Static).update(
            layout.line("MODEL", "DATA", "WINDOW", "STATUS")
        )
        for index, row in enumerate(self.assistants):
            if row.available and row.assistant is not None:
                assistant = row.assistant
                if assistant.run is None:
                    data = "PERSONA"
                else:
                    counts = assistant.counts
                    data = (
                        f"{counts.get('train', 0)}/"
                        f"{counts.get('valid', 0)}/{counts.get('test', 0)}"
                    )
                text = layout.text(
                    row.label,
                    data,
                    str(assistant.context_messages),
                    "READY",
                )
                available += 1
            else:
                text = layout.text(
                    row.label,
                    "—",
                    "—",
                    "UNAVAILABLE",
                    failed=True,
                )
            key = f"assistant-{index}"
            self._rows[key] = row
            options.append(Option(text, id=key, disabled=not row.available))
        for saved in saved_chats:
            text = layout.text(
                f"{saved.title} · {saved.assistant.name}",
                "SNAPSHOT",
                "—",
                "SAVED",
            )
            key = f"saved-{saved.id}"
            self._rows[key] = saved
            options.append(Option(text, id=key))
        chooser.add_options(options)
        summary = f"{available} available"
        if saved_chats:
            summary += f" · {len(saved_chats)} saved"
        self.query_one("#catalog-summary", Static).update(summary)
        for option_index, option in enumerate(chooser.options):
            if not option.disabled:
                chooser.highlighted = option_index
                chooser.focus()
                break

    def _session(self) -> ChatSession:
        if self.runtime.session is None:
            raise ChatError("Select an assistant first")
        return self.runtime.session
