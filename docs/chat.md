# Local Chat

## Purpose

Idiolect chat is a private terminal application. It connects one local user to
the configured base model or to a verified adapter. It does not read Signal or
DuckDB. It does not use tools or send messages to another service.

Chat keeps the transcript in memory until the user explicitly saves a snapshot.
It does not autosave.

## Keyboard shortcut notation

Chat uses macOS keyboard symbols as the canonical notation in CONTROL, footers,
catalog hints, dialogs, and SPECS navigation. `↩` means Return, `⇧` means Shift,
`⌃` means Control, `⎋` means Escape, `⌫` means Backspace/Delete, and `⎵` means
Space. Arrow glyphs such as `↑↓` and `←→` retain their usual meaning.

## Install and open

Install the model and terminal packages:

```console
just setup-chat
```

Open the local registry:

```console
idiolect chat
```

The registry lists the configured base model, verified run-and-dataset pairs,
and verified saved snapshots. Registry discovery does not load a model. The
model worker resolves and loads the selected model after the session starts.

Open one run and dataset pair directly:

```console
idiolect chat --run RUN_ID --dataset DATASET_ID
```

Resume one saved snapshot directly:

```console
idiolect chat --resume CHAT_ID
```

The run and dataset arguments can be artifact IDs or paths. Resume uses a saved
chat ID.

## Configuration

The `[chat]` table defines the snapshot output, fixed seed, participant label,
context policy, history policy, and default base persona. Chat also uses the
generation fields from `[inference]` and the base-model fields from `[train]`.

The canonical policy uses these fixed rules:

- `default_model = "train-base"` selects the configured training base without
  an adapter.
- `history = "explicit-save"` disables automatic transcript storage.
- `context_policy = "recorded-window-drop-oldest"` keeps whole messages and
  removes the oldest messages when the token limit requires it.
- `participant_name` supplies the synthetic operator name in model text.
- `default_context_messages` limits the base persona when no dataset supplies a
  recorded limit.

Run targets use the model, text policy, and dataset metadata recorded by the
verified artifacts. The selected TOML file supplies the live generation policy.

## Prompt and context rules

Chat uses the same conversation grammar as dataset construction. One model call
generates one response episode. The terminal renders each non-empty part after
a standalone `[new message]` boundary line as a separate message. It does not
show the boundary. Spaces around the boundary and different line endings do not
change this behavior.

For an adapter, the runtime first applies the dataset's recorded context-message
limit. It then renders the complete model prompt. If the prompt is too long, it
removes the oldest whole messages. It does not divide a message. It does not
remove the newest user message. It rejects a newest message that cannot fit by
itself.

A user can attach reply context to a stored chat message. Each rendered message
has its own stable reference number, including messages from one response
episode. Adapter prompts encode that context with the same reply-header grammar
as dataset prompts. The base persona does not receive adapter-specific reply
metadata.

## Model worker

One spawned worker owns the loaded model. It verifies the model and adapter
before generation. It unloads the current session before it loads another
assistant. The Textual process does not import MLX or hold model weights.

The worker streams text, prompt progress, token counts, throughput, and peak
memory to the terminal application. Cancellation stops generation at a token
boundary. A cancelled partial reply remains in the in-memory transcript. A
worker failure does not save the transcript.

After a worker failure, use `/retry` to reload it and retry the pending
generation. The command also replaces the latest cancelled reply. Each retry
uses the next deterministic attempt seed. The command is unavailable after a
completed reply. Return to the registry and start a new session if recovery
fails.

## Saved snapshots

An explicit save creates this private artifact:

```text
var/chat/CHAT_ID/
├── manifest.json
└── turns.jsonl
```

The snapshot identity includes the assistant inputs, chat policy, generation
policy, title, parent snapshot, turns, attempts, finish reasons, seeds,
telemetry, and runtime versions. The creation time is not part of the identity.

A resumed snapshot creates a child when the transcript changes. Older snapshots
remain on disk. The registry shows verified lineage leaves. The loader rejects
an invalid schema, file digest, policy value, or lineage reference.

Snapshot directories use mode `0700`. Snapshot files use mode `0600`. A
snapshot can contain every user message and generated reply in the transcript.
Do not publish it.

## Operational limits

- Chat needs Apple silicon and the optional model packages.
- Chat can download a configured hub model during model resolution.
- Chat does not keep a durable transcript unless the user saves it.
- Chat does not read evaluation artifacts or claim that an adapter passed an
  evaluation.
- Chat does not contact Signal, DuckDB, or a remote inference service.

For model selection evidence, use the [evaluation procedure](eval.md). For
artifact handling rules, use the [security procedure](security.md).
