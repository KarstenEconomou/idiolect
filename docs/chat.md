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
interface always shows the local user as `USER`.

Install and launch:

```console
just setup-chat
just chat
```

The first landing row is `IDIOLECT // DIXIE@BASE [MODEL]`. It is available
without a run or dataset and uses the configured system persona. Selecting it
resolves and verifies the base model inside the worker. Landing discovery itself
does not download a model.

The landing screen displays the base persona, verified adapters, and saved
snapshots in `REGISTRY`. It has no search field or pointer activation.
The blue watermark keeps its mark and product name bold and gives its tagline a
dim, non-bold treatment. Chat and SPECS identity headers carry the matching
`· ·` brand motif at their right edge with one blank cell completing the face.
Use the arrow keys to choose a row and Enter to connect. TYPE identifies a BASE,
CONSTRUCT, or TRACE. TYPE and ENTRY use the same description treatment as slash
commands: metadata gray while idle and dimmed accent color with their selected
model. Unavailable FAULT rows use the muted, dimmed treatment of an unavailable
`/save`. A saved `TRACE` keeps the canonical model identity first and shows the
trace name to its right in metadata gray. A long trace name ends with an ellipsis
before TYPE. Trace names are expanded by default and use the muted selection
color with their highlighted TRACE, matching TYPE and ENTRY. Press Space from any registry
row to collapse or expand all trace names together. Press Backspace while a
TRACE is highlighted to open the horizontal `TRACE` management menu. `RETAIN` is
selected by default and Escape also keeps the trace. Its heading shows the
current trace name directly after `TRACE` in metadata gray,
and the subject trace name blinks until the menu closes. The menu and rename
field use the same outer inset as the chat composer. Select `RENAME` to open
the standard name field with the current name as its default. Renaming creates
an immutable replacement with the same lineage parent. Select `ERASE` to
permanently remove that verified lineage leaf. A trace with a child cannot be
renamed or erased.

Press T in REGISTRY to cycle the interface accent through ANSI green, yellow,
blue, purple, and cyan. Green is the default; the first press selects yellow,
and the cycle wraps to green after cyan. This branding control is intentionally
omitted from the navigation hints.

Press S on any `READY` row to open `SPECS` without resolving or loading the
model. The scrollable page shows the complete verified model identity, source,
revision, digests, prompt and generation policy, and available run, dataset, and
TRACE lineage. It uses the compact `CTX`, `TOK`, `REP`, `GEN`, and `EVAL` labels.
Field names use the command menu's primary text, while values use its
metadata-gray description treatment. Section headings remain white and bold.
Every value, including evaluation bars, uses the same one-cell inset beneath its
label as transcript turns. Wrapped continuation lines and every retained line of
a multiline system prompt keep that inset. Trailing blank system-prompt lines
are omitted. Prefixes and suffixes show control whitespace with escapes such as
`\n`. Press Escape to return to the same highlighted registry row.
Use Left and Right to cycle with wrapping through every `READY` registry entry;
`FAULT` entries are skipped. Each change resets SPECS to the top, and Escape
returns to the newly selected registry row.
The DIXIE BASE page includes a deterministic fidelity scorecard labeled
`SYNTHETIC // UI FIXTURE`; its values demonstrate the terminal graphics and are
not measurements. CONSTRUCT and TRACE pages show `NOT EVALUATED` because chat
does not scan or infer results from private evaluation artifacts.

Verified run/dataset pairs and saved lineage leaves follow the default row. Each
adapter has the identity `IDIOLECT // NAME@run [MODEL]`. `run` is the first eight
characters of the full run ID. `NAME` is the uppercase display of the recorded
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
transcript. Transcript turns use `USER` and the short uppercase assistant name.
The chat header, registry, and snapshot use the full canonical assistant
identity. Each displayed message line is inset one cell beneath its speaker
label, matching the menu heading-to-action offset. Use arrow keys and Enter in an
unsaved-change confirmation. After a generation failure, return to `REGISTRY` to
start again.

Type `/` to open the keyboard-driven command menu above the composer. The menu
shows up to three command descriptions in vertical rows. Use arrow keys to move,
Tab to complete the selected command with a trailing space, Enter to select, and
Escape to close the menu without clearing the composer. Commands are:

- `/exit`: stop an active reply or exit when idle;
- `/registry`: return to `REGISTRY` when no reply is active;
- `/save`: save a TRACE checkpoint and keep the chat open;
- `/specs`: temporarily view the active model's SPECS.

`/specs` uses the active session's recorded generation policy and includes full
TRACE lineage for a resumed or newly saved snapshot. It does not change turns,
unsaved state, or transcript content. Escape restores the same chat directly;
Left and Right do not cycle registry models in this temporary view.

`/save` is available only when the transcript contains new unsaved data and no
reply is active. Otherwise it is shown as unavailable and cannot be selected;
entering it explicitly in a clean session reports that there is no new data.
The command opens the same optional trace-name field used by SAVE. Enter saves
the checkpoint and stays in chat. The empty field shows the generated default
name; a blank name uses it, and Escape cancels the checkpoint. A successful
checkpoint disables `/save` until the transcript changes again.

Either idle command opens the horizontal DISCONNECT, SAVE, and RESUME
confirmation when the transcript has unsaved changes. DISCONNECT is selected
first. SAVE requests an optional trace name before it writes the immutable
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
synthetic participant labels user history.

The runtime first applies the dataset's recorded context-message count. It then
counts the complete tokenizer template and removes oldest whole messages until
the prompt fits `inference.max_prompt_tokens`. It never splits a message or removes
the newest user message. Input that cannot fit by itself is rejected.

## Worker and telemetry

One spawned worker owns the selected model session. It resolves and verifies the
fixed base model, loads an adapter only for a run assistant, streams token text,
and captures backend stdout and stderr. Switching assistants unloads the current
model session inside that worker before loading the next one. Unexpected exit or
model failure keeps the memory-only transcript. Return to `REGISTRY` to start a
new chat.

Model resolution, verification, and loading run outside the Textual event loop.
The interface stays responsive and reports active worker states above the
composer. It reports `CONNECTION is not ready.` if input is submitted before the
load completes. It hides the ready state. During prompt processing, it reports
the measured prefill token count and total from MLX-LM.

The footer reports the last measured turn when no action menu is open. It puts
context use first, then groups output tokens and generation rate under `GEN`.
With more terminal width, it adds first-token latency and peak memory. It removes
fields from the right when space is limited. All values come from the model
backend; the footer does not report estimated performance. Peak memory uses
decimal GB, which matches the MLX-LM measurement.

Transient alerts appear as right-aligned lines immediately above the active
control bar with the same spacing as loading status. Informational and success
alerts use metadata gray; errors alone use the failure color. The TUI does not
use floating toasts.

## Saved snapshots

An explicit save creates:

```text
var/chat/CHAT_ID/
├── manifest.json
└── turns.jsonl
```

The content identity includes assistant digests, chat and generation policies,
title, parent snapshot, turns, attempts, finish reasons, seeds, telemetry, and
the recorded MLX and MLX-LM runtime versions. Creation time is not part of the
ID. Saving an unchanged transcript with its
current name returns its existing artifact. A resumed transcript saves as a
child. The chooser presents verified lineage leaves and preserves every older
snapshot on disk.
The loader accepts the previous snapshot version without recorded runtime
versions and verifies the numeric sampling ranges of the saved generation
policy.
If a confirmation save fails, the requested navigation or exit is cancelled and
the memory-only transcript stays open.
