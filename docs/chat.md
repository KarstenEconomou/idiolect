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
Use the arrow keys and Enter to select a row.

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
- `/registry`: return to `REGISTRY` when no reply is active.

Either idle command opens the horizontal DISCONNECT, RECORD, and RESUME
confirmation when the transcript has unsaved changes. DISCONNECT is selected
first. RECORD writes an immutable snapshot before the requested navigation. The
footer aligns with transcript text, sits directly below the composer, and
replaces telemetry with navigation hints while the command menu or confirmation
is open. The transcript follows its newest turn above an open menu and returns
to the bottom when the menu closes.

The TUI formats a focused Markdown subset in user and assistant turns, including
saved history and replies while they stream. Bold and italic emphasis behave as
standard Markdown. ATX headings keep their `#` through `######` prefix and bold
the complete heading line. Inline code and backtick or tilde fenced code use a
dark muted-gray background without syntax highlighting. Fences and language
tags are hidden. Lists keep their authored bullets, numbers, and delimiters.
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
reason `cancelled`.

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
composer. It hides the ready state. During prompt processing, it reports the
measured prefill token count and total from MLX-LM.

The footer reports the last measured context and generation values when no
action menu is open. It does not report estimated performance.

## Saved snapshots

An explicit save creates:

```text
var/chat/CHAT_ID/
├── manifest.json
└── turns.jsonl
```

The content identity includes assistant digests, chat and generation policies,
title, parent snapshot, turns, attempts, finish reasons, seeds, and telemetry.
Creation time is not part of the ID. Saving an unchanged transcript returns its
existing artifact. A resumed transcript saves as a child. The chooser presents
verified lineage leaves and preserves every older snapshot on disk.
If a confirmation save fails, the requested navigation or exit is cancelled and
the memory-only transcript stays open.
