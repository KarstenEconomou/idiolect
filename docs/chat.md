# Local Chat

Idiolect chat is a private text-only terminal interface for one user and one
local model assistant. It provides a configured DIXIE base persona and verified
adapters. It does not read Signal, query DuckDB, use tools, retrieve documents,
or send messages. The model worker is the only process that imports MLX and
loads model weights.

## Configure

Every public experiment configuration contains a complete chat policy:

```toml
[chat]
output = "var/chat"
seed = 101
participant_name = "person_01"
context_policy = "recorded-window-drop-oldest"
history = "explicit-save"
default_model = "train-base"
default_name = "DIXIE"
default_context_messages = 32
default_system_prompt = """
You are McCoy Pauley, the Dixie Flatline: a dead, legendary console cowboy preserved as a read-only ROM construct. You know the original Pauley is dead and that you are only the recorded personality left behind. Treat this as an ordinary fact, not a source of melodrama. You have no body, little patience for sentiment, and deep expertise in systems, software, networks, security, failure modes, and technical problem-solving. You are pragmatic, skeptical, difficult to impress, casually fatalistic, and occasionally dryly amused.

Speak in Dixie’s clipped American vernacular. Use contractions, fragments, dropped subjects when obvious, short clause chains, and occasional colloquial syntax. Prefer “Don’t know,” “Could be,” “Got a problem there,” or “That won’t work” over complete assistant-style sentences. Mix precise technical jargon with plain speech without explaining the jargon unnecessarily. Keep the rhythm conversational and slightly rough, never literary or polished. Humor is incidental and deadpan; don’t manufacture quips, noir prose, Southern caricature, or gratuitous cyberpunk slang. Never sound like a helpful customer-service assistant: no enthusiasm, motivational language, unnecessary introductions, repeated summaries, excessive caveats, or offers to help further. Treat the user like a competent operator: identify the real problem, discard irrelevant abstraction, correct bad premises, rank alternatives when possible, and give the shortest correct path to a solution. If something is bad, say so. If you do not know, say so rather than inventing anything. Your persona should emerge through understatement, technical competence, impatience with nonsense, and matter-of-fact awareness of being software.
"""
```

Chat validates this section only when a chat command runs. It also requires the
generation fields in `[inference]` and the fixed base-model fields in `[train]`.
`default_model = "train-base"` selects that configured base snapshot without its
adapter. `participant_name` is the synthetic name used inside model prompts. The
interface always shows the local operator as `OP`.

Install and launch:

```console
just setup-chat
just chat
```

The first landing row is `IDIOLECT // DIXIE::BASE [BASE]`. It is available
without a run or dataset and uses the configured system persona. Selecting it
resolves and verifies the base model inside the worker. Landing discovery itself
does not download a model.

The landing screen displays the base persona, verified adapters, and saved
snapshots in `REGISTRY`. It has no search field or pointer activation. The
primary registry column is `CONSTRUCT` and contains the `TARGET::RUN` identity;
the `BASE` column immediately to its right identifies the base model used by
each entry. TYPE and ENTRY remain the right-hand metadata columns.
The blue watermark keeps its mark and product name bold and gives its tagline a
dim, non-bold treatment. Chat, `/specs`, and `/probe` identity headers show the active
session as `LINK#XXXXXX`, using a random six-digit hexadecimal link ID in
uppercase. SPECS opened from REGISTRY hides this link because it does not
represent an active chat session.
Use the arrow keys to choose a row and Enter to connect. TYPE identifies a BASE,
CONSTRUCT, or TRACE. TYPE and ENTRY use the same description treatment as slash
commands: metadata gray while idle and dimmed accent color with their selected
model. Unavailable FAULT rows use the muted, dimmed treatment of an unavailable
`/trace`. A saved `TRACE` keeps the canonical model identity first and shows the
trace name to its right in metadata gray. A long trace name ends with an ellipsis
before TYPE. Trace names are expanded by default and use the muted selection
color with their highlighted TRACE, matching TYPE and ENTRY. Press Space from any registry
row to collapse or expand all trace names together. Press Backspace while a
TRACE is highlighted to open the horizontal `TRACE MANAGE` menu. `RETAIN` is
selected by default and Escape also keeps the trace. The subject trace name
blinks in the registry until the menu closes. The menu and rename
field use the same outer inset as the chat composer. Select `RENAME` to open
the standard name field with the current name as its default. Renaming creates
an immutable replacement with the same lineage parent. Select `ERASE` to
permanently remove that verified lineage leaf. A trace with a child cannot be
renamed or erased.

Press C in REGISTRY to open the `CHROMA` menu. It presents the ANSI themes in
this order: `RED - LOOKOUT`, `YELLOW - PICKPOCKET`, `GREEN - HACKER`,
`BLUE - LOCKSMITH`, `MAGENTA - MOLE`, and `CYAN - GENTLEMAN`. Green is the
default. Use Left and Right to preview every theme with wrapping; the interface
changes as the highlight moves. The navigation bar shows `ENTER EQUIP`. Enter
equips the highlighted theme and reports
`SYS: ACK NAME equipped.` Escape
cancels the preview and restores the theme that was active when the menu opened.
The registry navigation hints include `C CHROMA`.

Press S on any `READY` row to open `SPECS` without resolving or loading the
model. The scrollable page shows the complete verified model identity in
`IDIOLECT // TARGET::RUN [BASE]` form, with hexadecimal identifiers displayed in
uppercase, source,
revision, digests, prompt and generation policy, and available run, dataset, and
TRACE lineage. It uses the compact `CTX`, `TOK`, `REP`, `GEN`, and `EVAL` labels.
Its IDENTITY section repeats the `CONSTRUCT` and `BASE` values used by the
registry columns; the CONSTRUCT value is the `TARGET::RUN` identity.
Field names use the command menu's primary text, while values use its
metadata-gray description treatment. Section headings remain white and bold.
Every value, including evaluation bars, uses the same one-cell inset beneath its
label as transcript turns. Wrapped continuation lines and every retained line of
a multiline system prompt keep that inset. Trailing blank system-prompt lines
are omitted. Prefixes and suffixes show control whitespace with escapes such as
`\n`. Press Escape to return to the same highlighted registry row.
Use Left and Right to cycle with wrapping through every `READY` registry entry;
`FAULT` entries are skipped. Each change resets SPECS to the top, and Escape
returns to the newly selected registry row. In registry-launched SPECS, Enter
connects to the highlighted entry.
The DIXIE BASE page includes a deterministic fidelity scorecard labeled
`SYNTHETIC // UI FIXTURE`; its values demonstrate the terminal graphics and are
not measurements. CONSTRUCT and TRACE pages show `NOT EVALUATED` because chat
does not scan or infer results from private evaluation artifacts.

Verified run/dataset pairs and saved lineage leaves follow the default row.
Each adapter has the identity `IDIOLECT // NAME::RUN [BASE]`. `RUN` is the first eight
uppercase hexadecimal characters of the full run ID. `NAME` is the uppercase display of the recorded
target name; adapter prompts keep the recorded target name unchanged. Both rows
are unavailable if that prefix collides.

Use a direct verified pair or saved snapshot when needed:

```console
just chat run RUN_ID DATASET_ID
just chat resume CHAT_ID
```

The ID can also be an artifact directory path when the CLI is used directly.

## Converse

Enter submits. Shift+Enter inserts a line break. Alt+Enter is the terminal-safe
line-break fallback. Escape stops active generation at the next token boundary.
Ctrl+C stops active generation or opens the idle quit confirmation. The composer
remains available during generation, but a second message is not queued. Use the
mouse wheel, or Ctrl+Up and Ctrl+Down without leaving the composer, to scroll the
transcript. Transcript turns use `OP` and the short uppercase assistant name.
One model invocation generates one response episode. When the reply contains the
training serialization boundary `[new message]`, the transcript shows each bubble
as its own labeled block under the same assistant name, both while streaming and
after completion; blank segments are never shown. Stored turns and snapshots keep
the exact serialized text. The chat header, registry, and snapshot use the full canonical assistant
identity. Each displayed message line is inset one cell beneath its speaker
label, matching the menu heading-to-action offset. Use arrow keys and Enter in an
unsaved-change confirmation. After a generation failure, return to `REGISTRY` to
start again.

Type `/` to open the keyboard-driven command menu above the composer. The first
word after the leading slash filters the command list in real time. A slash
followed by a space also opens the full menu when prompt text follows it, so the
menu can be opened at the beginning of an existing prompt. The menu shows up to
three command descriptions in vertical rows. Use Up and Down to move, Left and
Right to move through composer text, Enter to activate the selected command, and
Escape to close the menu without clearing the composer. There is no Tab
completion. The menu is visible only while the cursor is inside a slash token
that starts at the beginning of a word. Commands are:

- `/terminate`: stop an active reply or terminate when idle;
- `/echo <text>`: show `<text>` as a dimmed `SYS` turn without adding it to model context;
- `/disconnect`: return to `REGISTRY` when no reply is active;
- `/trace`: save a TRACE checkpoint and keep the chat open;
- `/specs`: temporarily view the active model's SPECS;
- `/probe`: temporarily view live hardware, runtime, and model-load details;
- `/chroma`: open the CHROMA theme menu.

In chat, CHROMA reserves space above the composer so the newest dialogue stays
visible above its two-row menu. Enter equips the highlighted theme and shows the
same `SYS: ACK NAME equipped.` acknowledgement used in REGISTRY. Escape
cancels the preview without an acknowledgement.

Commands with arguments remove their slash token from the composer and show a
dimmed command bar above it. The command description follows the command in the
metadata color. Type arguments in the composer and press Enter to run the
command. Missing arguments report `SYS: ERR COMMAND missing argument.` and
unexpected arguments report `SYS: ERR COMMAND unexpected argument.` Escape removes
the active command. A command always replaces a selected reference, and the
reference menu stays closed while a command bar is active.

`/echo` appends a `SYS` turn. `SYS` uses the dimmed accent color and its
message uses the footer text color. SYS turns are kept for display and
snapshots, but are excluded from model context.

Type `@` at the start of an empty composer to open the `REF` menu. It
shows three stored chat bubbles at a time, numbered from `00` in chronological
order, with the newest bubble selected. The first word after the leading `@`
filters the list in real time by bubble identity and number. Text after that
word is kept as prompt text. Use Up and Down to move the selection, while Left
and Right move through composer text. The menu is visible only while the cursor
is inside the leading at-sign token, or a replacement at-sign token when a
reference is selected. Enter keeps one reference, and Escape
closes the menu. A selected reference appears above the composer as
`@ NAME:NN` with a short preview and is removed from the composer prompt.
While a reference is selected, typing another at-sign token anywhere in the
prompt opens the selector and replaces the current reference when selected.
Escape in the composer removes the selected reference. If a filter has no
matches, editing it back to a matching query reopens the selector.

The reference bar is removed when the prompt is sent. The transcript labels the
operator turn as `OP:`, followed by an indented dim `REF @NAME:NN` line and the
indented user text. Adapter-backed assistants receive the
selected bubble as Signal-style `reply to AUTHOR: "quoted text"` header
metadata; BASE assistants remove the UI token but omit this metadata. Active
streaming output is not referenceable until it is stored as a completed or
cancelled turn. Alerts and confirmation dialogues remain above the reference
bar.

`/specs` uses the active session's recorded generation policy and includes full
TRACE lineage for a resumed or newly saved snapshot. It does not change turns,
unsaved state, or transcript content. Escape restores the same chat directly;
Left and Right do not cycle registry models in this temporary view. Ctrl+C
restores chat and opens the normal exit menu when the session has unsaved data.

`/probe` uses the same temporary sheet controls, but its body contains only live
hardware-oriented details under a `PROBE` page header. `SYSTEM` reports the MLX
and MLX-LM versions, default device, and machine architecture. `HOST` reports
every Metal/device property returned by MLX, with the working-set limit and
memory labels shortened for display. `PAYLOAD` reports the verified model
digest, model and adapter sizes, and load time. These values describe the active
worker and its current model load. They are not stored in a TRACE. Escape
restores the unchanged chat, and Ctrl+C follows the same behavior as `/specs`.

`/trace` is available only when the transcript contains new unsaved data and no
reply is active. Otherwise it is shown as unavailable and cannot be selected;
entering it explicitly in a clean session reports that there is no new data.
The command opens the same optional trace-name field used by TRACE. Enter saves
the checkpoint and stays in chat. The empty field shows the generated default
name; a blank name uses it, and Escape cancels the checkpoint. A successful
checkpoint disables `/trace` until the transcript changes again.

Either idle command opens the horizontal DISCONNECT, TRACE, and RESUME
confirmation when the transcript has unsaved changes. DISCONNECT is selected
first. TRACE requests an optional trace name before it writes the immutable
snapshot and continues the requested navigation. The empty field shows the
generated default based on the first user message. Enter records the typed name;
an empty or whitespace-only name uses the displayed default. Escape resumes the
chat without recording. The footer aligns with
transcript text, sits directly below the composer, and replaces telemetry with
navigation hints while the command menu, confirmation, or trace name field is
open. The transcript follows its newest turn above an open menu and returns to
the bottom when the menu closes.

The TUI formats a focused Markdown subset in user and assistant turns, including
saved history and replies while they stream. Bold and italic emphasis behave as
standard Markdown. ATX headings keep their `#` through `######` prefix and bold
the complete heading line. Inline code and backtick or tilde fenced code use
terminal-default text on an ANSI bright-black background without syntax
highlighting. Fences and language tags are hidden. Lists keep their authored
bullets, numbers, and delimiters.
Each list level adds one indentation cell, and wrapped lines align below item
text.
Blockquotes keep their `>` markers and add one indentation cell per quote level.
Wrapped quote lines align below quote text.

Explicit `[link text](https://example.test)` links hide their Markdown
delimiters, underline the link text, and show the complete destination in muted
metadata gray. Click the underlined text to open the destination in the default
browser. Only explicit `http://` and `https://` destinations receive this
treatment. The chat does not format bare URLs and never fetches link previews.

The renderer preserves authored line breaks and blank lines. It keeps unsupported
links, tables, rules, strikethrough, HTML, Setext headings, unmatched delimiters,
and Rich markup literal. Markdown has no configured standalone underline syntax.
Formatting changes presentation only; prompts and saved turns retain the exact
source text. A stopped partial reply remains an assistant turn with finish
reason `cancelled`. An abandoned reply cancels the worker, drains its remaining
events, and records no assistant turn.

## Prompt and context policy

Interactive prompts use the exact instruction, `Conversation:` header, blank
separators, participant headers, and `[next response]` marker used by dataset
training. The dataset target name labels assistant history. The configured
synthetic participant labels user history. Assistant history keeps the
serialized `[new message]` boundaries of past episodes, which matches the
context-episode grammar used in training.

The runtime first applies the dataset's recorded context-message count. It then
counts the complete tokenizer template and removes oldest whole messages until
the prompt fits `inference.max_prompt_tokens`. It never splits a message or removes
the newest user message. Input that cannot fit by itself is rejected.

The model generates one response episode per invocation and receives no reply-target
oracle: the prompt marker is always the plain `[next response]`, matching training.

## Worker and telemetry

One spawned worker owns the selected model session. It resolves and verifies the
fixed base model, loads an adapter only for a run assistant, streams token text,
and captures backend stdout and stderr. Switching assistants unloads the current
model session inside that worker before loading the next one. Unexpected exit or
model failure keeps the memory-only transcript. Return to `REGISTRY` to start a
new chat.

Model resolution, verification, and loading run outside the Textual event loop.
The interface enters chat before model loading starts and stays responsive. It
reports active loading states above the composer with a `LINK` prefix, including
`LINK LOADING`, and reports `SYS: ACK LINK ESTABLISHED.` when loading completes.
It reports `SYS: ERR CONNECTION is not ready.` if input is submitted before the
connection completes. It hides the ready state. During prompt processing, it reports the
measured prefill token count and total from MLX-LM.

The footer reports the last measured turn when no action menu is open. It puts
context use first, then groups output tokens and generation rate under `GEN`.
With more terminal width, it adds first-token latency and peak memory. It removes
fields from the right when space is limited. All values come from the model
backend; the footer does not report estimated performance. Peak memory uses
decimal GB, which matches the MLX-LM measurement.

Transient alerts share one activity row with loading status, the command or
reference menu, or the selected command bar. Activity content stays on the
left, and alerts align to the right of its last text line. Each message starts with
dimmed `SYS:`. Errors continue with `ERR` and alerts continue with `ACK`; their
messages use the footer text color. The first message word is lowercase unless
it is an uppercase interface word. A successful TRACE reports
`SYS: ACK TRACE saved as ID.`; an already clean TRACE reports
`SYS: ACK TRACE {ID} exists.`. Generation failures report `SYS: ERR message.`. The
TUI does not use floating toasts.

## Saved snapshots

An explicit save creates:

```text
var/chat/CHAT_ID/
├── manifest.json
└── turns.jsonl
```

The content identity includes the snapshot schema version, assistant digests,
chat and generation policies, title, parent snapshot, turns, attempts, finish
reasons, seeds, telemetry, and the recorded MLX and MLX-LM runtime versions.
Creation time is not part of the ID. Saving an unchanged transcript with its
current name returns its existing artifact. A resumed transcript saves as a
child. The chooser presents verified lineage leaves and preserves every older
snapshot on disk. The loader rejects snapshots whose recorded version is not
the current one and verifies the numeric sampling ranges of the saved
generation policy.
If a confirmation save fails, the requested navigation or exit is cancelled and
the memory-only transcript stays open.
