# Development and Verification

## Repository boundaries

The package uses stage boundaries:

| Path | Responsibility |
|---|---|
| `ingest/` | Read source events and normalize Signal input. |
| `store/` | Define persistence ports and store local records. |
| `data/` | Select causal context and build immutable datasets. |
| `train/` | Define training contracts and run MLX-LM training. |
| `inference/` | Define generation contracts and create predictions. |
| `chat/` | Own assistant discovery, sessions, workers, and snapshots. |
| `tui/` | Present terminal state and collect user input. |
| `eval/` | Score policies and build judgment and panel artifacts. |

Shared contracts are in `artifact.py`, `config.py`, `model.py`, `prompt.py`,
`render.py`, and `types.py`. Keep optional MLX, Textual, DuckDB, and Signal
dependencies out of unrelated contract modules.

The root `AGENTS.md` applies to all changes. Each applicable nested `AGENTS.md`
adds rules for changes to stage code or tests.

## Environment

Install `just` 1.46.0 or later. On macOS, install the command-line tool:

```console
brew install just
just --version
```

Install the core Python environment:

```console
just setup
```

Use `uv add` or `uv add --dev` for dependency changes. Do not edit `uv.lock` by
hand.

## Required checks

Run focused tests during development. Run the complete checks before handoff:

```console
just check
just build
```

`just check` verifies Just files, runs Ruff, runs `ty`, and runs pytest. The
`just build` command creates the source distribution and wheel.

Setup, lint, type-check, test, and build commands do not load `.env`. Commands
that launch Idiolect use `uv run --env-file .env` through a Just recipe.

## Test isolation

Tests use synthetic messages, temporary files, and fake external boundaries.
They must not read these resources:

- `.env` or live configuration
- `var/` or user files
- the installed LaunchAgent
- live Signal state
- model hubs, model weights, or adapters
- private chats, datasets, predictions, or evaluations

Do not run collection, model downloads, training, inference, chat models, or GPU
work in pytest. Use fake model resolvers, tokenizers, sessions, command runners,
and clocks.

## Operational checks

Operational checks use private state and are outside standard repository
verification:

```console
just idiolect signal groups
just idiolect signal stats
just collect status
```

Use `just setup-chat` before a manual chat check. On Apple silicon, check one
model load, one streamed reply, cancellation, context limits, an explicit save,
and a clean exit. Do not make manual model work part of automated verification.
