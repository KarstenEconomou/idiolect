"""Render the local Idiolect chat terminal interface."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict
from typing import ClassVar

from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, RichLog, Static, TextArea
from textual.widgets.option_list import Option

from idiolect.chat.discovery import Assistant, DiscoveryItem
from idiolect.chat.runtime import ChatError, ChatRuntime
from idiolect.chat.state import ChatSession
from idiolect.chat.storage import ChatStorageError, ChatStore, SavedChat
from idiolect.chat.worker import WorkerError
from idiolect.config import ChatConfig, GenerationConfig
from idiolect.tui.commands import COMMANDS, CommandError, completions, parse_command

WATERMARK = """     ╭─╮
  ╭──╯ ╰──╮
  │  · ·  │    IDIOLECT
  ╰──╮ ╭──╯    someone, reconstructed.
     ╰─╯"""


class Composer(TextArea):
    """Submit Enter while keeping explicit multiline key bindings."""

    class Submitted(Message):
        """Report one submitted composer value."""

        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.post_message(self.Submitted(self.text))
            return
        if event.key in {"shift+enter", "alt+enter"}:
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        await super()._on_key(event)


class InfoModal(ModalScreen[None]):
    """Show literal help or measured statistics."""

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self.title_value = title
        self.body = body

    def compose(self) -> ComposeResult:
        """Create the information dialog widgets."""
        with Vertical(id="info-dialog"):
            yield Label(self.title_value, id="info-title")
            yield Static(self.body, markup=False, id="info-body")
            yield Button("Close", id="close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Close the information dialog."""
        self.dismiss()


class ConfirmModal(ModalScreen[str]):
    """Ask how to handle unsaved transcript changes."""

    def compose(self) -> ComposeResult:
        """Create the unsaved-change choice widgets."""
        with Vertical(id="confirm-dialog"):
            yield Static("This chat has unsaved changes.", markup=False)
            with Horizontal():
                yield Button("Save", id="save", variant="primary")
                yield Button("Discard", id="discard", variant="warning")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Return the selected unsaved-change choice."""
        self.dismiss(event.button.id or "cancel")


class ChatApp(App[None]):
    """Run the searchable landing chooser and sparse chat screen."""

    CSS = """
    $background: #090b0d;
    $surface: #111518;
    $amber: #e2a447;
    $cyan: #45c5d4;
    $body: #e8dfcf;
    $muted: #7b8588;
    $failure: #d2645a;
    Screen { background: $background; color: $body; }
    #landing { align: center middle; padding: 1 3; }
    #landing-box { width: 90%; max-width: 100; height: 90%; border: heavy $amber; background: $surface; padding: 1 2; }
    #watermark { color: $amber; height: 7; text-align: center; text-style: bold; }
    #search { border: tall $cyan; margin-bottom: 1; background: $background; }
    #search:focus { border: heavy $amber; }
    #load-status { height: 1; color: $cyan; text-style: bold; }
    #chooser { height: 1fr; border: tall $cyan; background: $background; }
    OptionList > .option-list--option-highlighted { color: $background; background: $amber; text-style: bold; }
    OptionList > .option-list--option-disabled { color: $failure; }
    #chat { display: none; }
    #identity { height: 3; padding: 1 2; color: $amber; background: $surface; border-bottom: heavy $amber; text-style: bold; }
    #transcript { height: 1fr; padding: 1 3; background: $background; scrollbar-color: $cyan; scrollbar-background: $surface; }
    #composer { height: 6; border: tall $cyan; margin: 0 1; background: $surface; }
    #composer:focus { border: heavy $amber; }
    #completion { height: auto; max-height: 5; color: $amber; background: $surface; padding: 0 2; }
    #footer { height: 1; color: $cyan; background: $surface; padding: 0 2; text-style: bold; }
    #info-dialog, #confirm-dialog { width: 80%; max-width: 90; height: auto; padding: 1 2; background: $surface; border: heavy $amber; }
    #info-title { color: $amber; text-style: bold; border-bottom: tall $cyan; }
    #info-body { max-height: 70vh; overflow-y: auto; }
    Button { border: tall $cyan; background: $background; color: $body; }
    Button:focus { border: heavy $amber; color: $amber; }
    """

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "stop", "Stop"),
        ("ctrl+c", "interrupt", "Stop or quit"),
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
        super().__init__()
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

    def compose(self) -> ComposeResult:
        """Create the landing and chat screen widgets."""
        with Container(id="landing"), Vertical(id="landing-box"):
            yield Static(WATERMARK, markup=False, id="watermark")
            yield Input(placeholder="Search assistants and saved chats", id="search")
            yield Static("", markup=False, id="load-status")
            yield OptionList(id="chooser")
        with Container(id="chat"):
            yield Static("", markup=False, id="identity")
            yield RichLog(markup=False, wrap=True, id="transcript")
            yield Static("", markup=False, id="completion")
            yield Composer(id="composer", language=None)
            yield Static("probing", markup=False, id="footer")

    def on_mount(self) -> None:
        """Populate the chooser or open a direct selection."""
        self._fill_chooser("")
        self.set_interval(0.1, self._refresh_loading_state)
        if self.initial_chat is not None:
            self._begin_attach(
                ChatSession(
                    self.initial_chat.assistant,
                    self.initial_chat.chat,
                    self.initial_chat.generation,
                    self.initial_chat.turns,
                    self.initial_chat.id,
                    self.initial_chat.title,
                )
            )
        elif self.initial_assistant is not None:
            self._begin_select(self.initial_assistant)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter chooser rows with the landing search value."""
        if event.input.id == "search":
            self._fill_chooser(event.value)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Open the selected assistant or saved chat."""
        if self._loading:
            return
        row = self._rows.get(str(event.option.id))
        if isinstance(row, DiscoveryItem) and row.assistant is not None:
            self._begin_select(row.assistant)
        elif isinstance(row, SavedChat):
            self._begin_attach(
                ChatSession(
                    row.assistant,
                    row.chat,
                    row.generation,
                    row.turns,
                    row.id,
                    row.title,
                )
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

    def _start_generation(self, attempt: int) -> None:
        self._generating = True
        self._streaming_text = ""
        self._active_attempt = attempt
        self.query_one("#footer", Static).update("generating")
        self._generate_thread(attempt)

    @work(thread=True, exclusive=True, group="generation")
    def _generate_thread(self, attempt: int) -> None:
        try:
            for delta in self.runtime.generate(attempt):
                self._streaming_text += delta
                self.call_from_thread(self._render_transcript, True)
            self.call_from_thread(self._generation_done)
        except Exception as error:  # noqa: BLE001 - keep the alternate screen valid.
            self.call_from_thread(self._generation_failed, str(error))

    def _generation_done(self) -> None:
        self._generating = False
        self._streaming_text = ""
        self._render_transcript()
        self._update_footer()

    def _generation_failed(self, message: str) -> None:
        self._generating = False
        self.query_one("#footer", Static).update("failed")
        self.notify(f"{message}. Use /retry to reload and try again.", severity="error")

    def _show_chat(self) -> None:
        session = self._session()
        self.chat_policy = session.chat
        self.generation = session.generation
        self.query_one("#landing").display = False
        self.query_one("#chat").display = True
        self.query_one("#identity", Static).update(session.assistant.name)
        self._render_transcript()
        self._update_footer()
        self.query_one(Composer).focus()

    def _render_transcript(self, partial: bool = False) -> None:
        log = self.query_one(RichLog)
        log.clear()
        session = self._session()
        for turn in session.turns:
            name = "You" if turn.role == "user" else session.assistant.name
            log.write(Text(name, style="#d99b43"))
            log.write(Text(turn.content))
        if partial and self._generating:
            log.write(Text(session.assistant.name, style="#d99b43"))
            log.write(Text(self._streaming_text or "…"))

    def _update_footer(self) -> None:
        session = self._session()
        last = next((turn for turn in reversed(session.turns) if turn.telemetry), None)
        state = self.runtime.state.value
        if last is None or last.telemetry is None:
            value = state
        else:
            telemetry = last.telemetry
            pressure = 100 * telemetry.prompt_tokens / self.generation.max_prompt_tokens
            value = f"{state}  context {telemetry.prompt_tokens}/{self.generation.max_prompt_tokens} {pressure:.0f}%  generated {telemetry.generated_tokens}"
            if self.size.width >= 80 and telemetry.generation_throughput is not None:
                value += f"  {telemetry.generation_throughput:.1f} tok/s"
            if self.size.width >= 105 and telemetry.peak_memory is not None:
                value += f"  peak {telemetry.peak_memory:.2f} GB"
        self.query_one("#footer", Static).update(value)

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
            "run_id": str(assistant.run.ref.id),
            "dataset_id": str(assistant.dataset.dataset.id),
            "base_revision": assistant.run.model.revision,
            "model_digest": assistant.run.model_digest,
            "adapter_digest": assistant.run.adapter_digest,
            "training_seed": assistant.run.seed,
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
        self._fill_chooser("")

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
        self._update_footer()

    def _after_landing_confirm(self, choice: str | None) -> None:
        if choice == "save" and not self._save_from_confirmation():
            return
        if choice in {"save", "discard"}:
            self.query_one("#chat").display = False
            self.query_one("#landing").display = True
            self._fill_chooser("")

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
        self._set_loading(True)
        self._select_thread(assistant)

    def _begin_attach(
        self,
        session: ChatSession,
        *,
        generate_attempt: int | None = None,
    ) -> None:
        self.runtime.session = session
        self._set_loading(True)
        self._show_chat()
        self._attach_thread(session, generate_attempt)

    @work(thread=True, exclusive=True, group="model-load")
    def _select_thread(self, assistant: Assistant) -> None:
        try:
            self.runtime.select(assistant)
        except Exception as error:  # noqa: BLE001 - keep the event loop active.
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
        except Exception as error:  # noqa: BLE001 - keep the event loop active.
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
        self.query_one("#footer", Static).update("failed")
        self.notify(message, severity="error")

    def _set_loading(self, value: bool) -> None:
        self._loading = value
        self.query_one("#chooser", OptionList).disabled = value
        self._refresh_loading_state()

    def _refresh_loading_state(self) -> None:
        if not self._loading:
            self.query_one("#load-status", Static).update("")
            return
        state = self.runtime.state.value.upper()
        self.query_one("#load-status", Static).update(f"{state} // MODEL SESSION")
        if self.query_one("#chat").display:
            self.query_one("#footer", Static).update(state.lower())

    def _fill_chooser(self, search: str) -> None:
        chooser = self.query_one("#chooser", OptionList)
        chooser.clear_options()
        self._rows.clear()
        query = search.casefold().strip()
        options = []
        for index, row in enumerate(self.assistants):
            if row.available and row.assistant is not None:
                assistant = row.assistant
                counts = assistant.counts
                adapter_size = sum(
                    item.stat().st_size
                    for item in assistant.run.adapter_path.rglob("*")
                    if item.is_file()
                )
                text = (
                    f"{row.label}\n"
                    f"SEED {assistant.run.seed}  ADAPTER {adapter_size / 1_048_576:.1f} MiB  "
                    f"DATA {counts.get('train', 0)}/{counts.get('valid', 0)}/{counts.get('test', 0)}  "
                    f"WINDOW {assistant.context_messages}  PROMPT {self.generation.max_prompt_tokens}"
                )
            else:
                text = f"{row.label} — UNAVAILABLE // {row.error}"
            if query and query not in text.casefold():
                continue
            key = f"assistant-{index}"
            self._rows[key] = row
            options.append(Option(Text(text), id=key, disabled=not row.available))
        if self.store is not None:
            for saved in self.store.leaves():
                text = f"Saved: {saved.title} — {saved.assistant.name}"
                if query and query not in text.casefold():
                    continue
                key = f"saved-{saved.id}"
                self._rows[key] = saved
                options.append(Option(Text(text), id=key))
        chooser.add_options(options)

    def _session(self) -> ChatSession:
        if self.runtime.session is None:
            raise ChatError("Select an assistant first")
        return self.runtime.session
