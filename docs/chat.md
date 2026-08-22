# Local Chat

Idiolect chat is a private text-only terminal interface for one user and one
verified adapter. It does not read Signal, query DuckDB, use tools, retrieve
documents, or send messages. The model worker is the only process that imports
MLX and loads model weights.

## Configure

Every public experiment configuration contains a complete chat policy:

```toml
[chat]
output = "var/chat"
seed = 101
participant_name = "person_01"
context_policy = "recorded-window-drop-oldest"
history = "explicit-save"
```

Chat validates this section only when a chat command runs. It also requires the
generation fields in `[infer]`. `participant_name` is the synthetic name used
inside model prompts. The interface always shows the local user as `You`.

Install and launch:

```console
just setup-chat
just chat
```

The landing screen searches verified run/dataset pairs and saved lineage leaves.
Discovery reads local manifests and never downloads a model. Each assistant has
the identity `IDIOLECT // NAME@run [MODEL]`. `run` is the first eight characters
of the full run ID. Both rows are unavailable if that prefix collides.

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
remains available during generation, but a second message is not queued.
After a generation failure, the pending user message must be retried before a
new user message can be submitted.

Commands are:

- `/assistant`: return to the assistant chooser;
- `/new`: start a new transcript with the loaded assistant;
- `/save [title]`: write one immutable snapshot;
- `/resume`: return to saved-chat selection;
- `/retry`: replace the latest assistant attempt;
- `/stats`: show recorded identity and measured runtime values;
- `/help`: show the command list;
- `/quit`: confirm and exit.

The TUI treats all transcript content as literal plain text. It does not enable
markup. A stopped partial reply remains an assistant turn with finish reason
`cancelled`. Retry removes that reply, increments its attempt, and derives a new
31-bit RNG seed from the configured chat seed, prompt digest, and attempt.

## Prompt and context policy

Interactive prompts use the exact instruction, `Conversation:` header, blank
separators, participant headers, and `[next response]` marker used by dataset
training. The dataset target name labels assistant history. The configured
synthetic participant labels user history.

The runtime first applies the dataset's recorded context-message count. It then
counts the complete tokenizer template and removes oldest whole messages until
the prompt fits `infer.max_prompt_tokens`. It never splits a message or removes
the newest user message. Input that cannot fit by itself is rejected.

## Worker and telemetry

One spawned worker owns the selected model session. It resolves and verifies the
recorded base model, loads the adapter, streams token text, and captures backend
stdout and stderr. Switching assistants shuts down that worker before starting
another. Unexpected exit or model failure keeps the memory-only transcript; use
`/retry` to reload and try the input again.

Model resolution, verification, and loading run outside the Textual event loop.
The interface stays responsive and reports the current loading state.

The footer keeps the last measured context tokens, context pressure, generated
tokens, generation throughput, and peak memory. Narrow terminals remove the
secondary throughput and memory fields. `/stats` includes full artifact IDs and
digests, recorded seed and revision, MLX device data, load duration, per-turn
token and timing measurements, aggregate token counts, dirty state, and saved
chat ID. It does not report estimated performance.

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
