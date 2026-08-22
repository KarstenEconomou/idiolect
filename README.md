# Idiolect

```text
     ╭─╮
  ╭──╯ ╰──╮
  │  · ·  │    IDIOLECT
  ╰──╮ ╭──╯    someone, reconstructed.
     ╰─╯
```

Idiolect is a local-first ML pipeline for fine-tuning models to reproduce
individual writing styles, linguistic patterns, and conversational behavior.
It collects whitelisted Signal group messages, preserves native mention, reply,
attachment, and reaction context, builds causally filtered and source-audited
immutable target-specific datasets, uses MLX-LM QLoRA to train adapters, and
generates reproducible local predictions.

It also provides a private multi-turn terminal chat for verified adapters. The
chat uses the same conversation grammar as training, streams local replies, and
saves only explicit immutable snapshots.

```text
                                                    Signal
                                                      │
                                                      ▼
                                            signal-cli collector
                                                      │
                                                      ▼
                                                normalization
                                                      │
                                                      ▼
                                                  DuckDB
                                                      │
                                                      ▼
                                          target-relative context
                                                      │
                                                      ▼
                                          immutable JSONL dataset
                                                      │
              ╭───────────────────────────────────────┴───────────────────╮
              ▼                                                           ▼
  recorded base inference                                      MLX-LM QLoRA training
              │                                                           │
              │                                       ╭───────────────────┴───────────────────╮
              │                                       ▼                                       ▼
              │                               adapter inference                     supervised MLX worker
              │                                       │                                       │
              ╰───────────────────────────────────────┤                                       │
                                                      ▼                                       ▼
                                        content-addressed predictions               private terminal chat
                                                      │                                       │
                                                      ▼                                       ▼
                                                    evals                       explicit immutable snapshots
```

The canonical configuration keeps Signal data, datasets, model files, adapters,
predictions, chat snapshots, evaluations, judgments, and panel reports under
the ignored `var/` directory. The repository tracks public settings in
`conf/idiolect.toml` and complete experiment settings in `conf/exp/`.
The evaluation runner compares a complete adapter policy with its exact recorded
base. It reports token-weighted corpus perplexity, paired example-level
likelihood, verified training-text matches, and private blind familiar-panel
judgments with example-and-rater uncertainty.

## Requirements

- Python 3.14
- [`uv`](https://docs.astral.sh/uv/)
- [`just`](https://just.systems/) 1.46.0 or later
- A current `signal-cli` release and a local QR code tool for collection
- An Apple silicon Mac for MLX-LM training, inference, chat, and automatic
  evaluation

Run commands from the repository root. Use only messages from people who consent
to the collection and model experiment.

Use the root `justfile` as the project command interface. Its recipes manage the
Python environment with `uv`. Use `uv` directly only when you change project
dependencies.

## Set up

Install the core environment:

```console
just setup
```

Link `signal-cli` as a secondary Signal device. See
[docs/signal.md](docs/signal.md). Then create the private environment file:

```console
touch .env
chmod 600 .env
```

Add the account first:

```sh
IDIOLECT_SIGNAL_ACCOUNT="+14165550123"
```

The values above are placeholders. Do not commit a real account, group ID,
message, token, or model artifact. Just recipes that launch Idiolect pass `.env`
to `uv`, so you do not need to load it in the current shell.

List groups, then add the selected IDs to `.env`:

```console
just idiolect signal groups
```

```sh
IDIOLECT_SIGNAL_CHATS='["GROUP_ID_ONE=", "GROUP_ID_TWO="]'
```

## Run the pipeline

Collect queued messages once, or use the documented macOS LaunchAgent for
continuous collection:

```console
just idiolect signal collect
just collect status
```

Build an immutable dataset for the linked Signal user:

```console
just idiolect data people
just data build TARGET_NAME
```

Install MLX-LM and run a short tracked experiment:

```console
just setup-train
just config train qwen3-8b-smoke var/data/DATASET_ID
```

Generate paired predictions from the exact base model and adapter recorded by
the run:

```console
just inference base-of var/runs/RUN_ID var/data/DATASET_ID test qwen3-8b-smoke
just inference run var/runs/RUN_ID var/data/DATASET_ID test qwen3-8b-smoke
```

Evaluate every configured training seed together on the fixed validation split:

```console
just eval policy var/data/DATASET_ID var/runs/RUN_ID_ONE var/runs/RUN_ID_TWO
```

Install the chat environment and open the searchable assistant chooser:

```console
just setup-chat
just chat
```

The chooser shows identities such as
`IDIOLECT // Karsten@7f3a91c2 [Qwen3-14B-4bit]`. Direct launch and resume are
also available:

```console
just chat run RUN_ID DATASET_ID
just chat resume CHAT_ID
```

Then collect private familiar-rater judgments and build a panel report. See
[docs/eval.md](docs/eval.md) for the consent, interpretation, and command rules.

Use `just config new NAME` to copy the complete canonical configuration before
you define another experiment. Do not change a configuration after you use it
for a recorded run.

## Documentation

See [docs/index.md](docs/index.md) for the replication entry point. It links the
procedures for Signal setup, security, collection, `launchd`, conversation
context, dataset construction, training, inference, chat, evaluation, and
development.

Important constraints:

- The collector receives new queued events. It does not import existing phone
  history.
- Stop the continuous collector during `reindex` and dataset construction.
- Collection can continue during training, inference, and evaluation because those operations
  use immutable files.
- Keep the Mac on, awake, and logged in for the LaunchAgent. Training, inference,
  and automatic evaluation recipes use `caffeinate`.
- Chat never reads Signal or DuckDB, and it never autosaves. Only `/save` or the
  Save choice in an unsaved-change prompt writes a private snapshot.
- Treat raw events, hashed records, chat transcripts, and saved snapshots as
  private data.

## Develop

Source code uses the `src` layout. Tests use synthetic fixtures, temporary
storage, and fake external boundaries.

```console
just check
just build
```

See [docs/development.md](docs/development.md) and
[AGENTS.md](AGENTS.md) for repository rules.
