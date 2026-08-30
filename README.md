# Idiolect

```text
     ╭─╮
  ╭──╯ ╰──╮
  │  · ·  │    IDIOLECT
  ╰──╮ ╭──╯    Someone, reconstructed.
     ╰─╯
```

Idiolect is a local-first system that adapts a language model to one person's
conversation style. It converts approved Signal group messages into causal
training examples. Each prompt contains conversation context that existed
before the target response. Each completion contains the full target response
episode; an episode can contain multiple Signal bubbles.

Idiolect uses MLX-LM to train a low-rank adapter on a fixed, quantized base
model. The base weights stay frozen. The experiment policy selects the model,
adapter targets, optimization settings, and sequence limits. Training minimizes
completion negative log-likelihood. Prompt and assistant-prefill tokens do not
contribute to the loss. A shared renderer applies the model chat template and
verifies the exact prompt-to-completion token boundary before training,
inference, or evaluation starts.

The result is a small adapter, not a new full model. Idiolect loads this adapter
with its recorded base model for local generation and private terminal chat. It
renders a generated response episode as separate messages at standalone
`[new message]` boundaries and gives each rendered message its own reply
reference. It compares each adapter policy with that same base model on held-out
responses:
the fidelity eval suite pillars are Likelihood, Voice, Validity, Memorization,
and Recognition. Source data, datasets, model files, adapters, predictions, chats,
and evaluation results stay on the local computer unless an operator enables
an external reporting service.

## Repository structure

```text
.
├── src/idiolect/            Python package and command-line application
│   ├── ingest/              Signal input and collection
│   ├── store/               Persistence contracts and DuckDB storage
│   ├── data/                Dataset selection and text construction
│   ├── train/               Training contracts and MLX-LM training
│   ├── inference/           Generation contracts and local inference
│   ├── chat/                Sessions, model workers, and snapshots
│   ├── tui/                 Textual terminal interface
│   ├── eval/                Policy evaluation and familiar-rater panels
│   ├── artifact.py          Artifact identity and file rules
│   ├── config.py            Strict TOML and environment configuration
│   ├── model.py             Model identity and verification
│   ├── prompt.py            Stage-neutral conversation meaning
│   ├── render.py            Model-token rendering contract
│   └── types.py             Shared immutable records
├── conf/                    Public and reproducible TOML policies
│   └── exp/                 Complete experiment policies
├── docs/                    Procedures and data-contract explanations
├── tests/                   Synthetic tests that mirror the package
├── justfile                 Setup, build, and verification recipes
└── var/                     Ignored private state and artifacts
```

Shared contracts are in `artifact.py`, `config.py`, `model.py`, `prompt.py`,
`render.py`, and `types.py`. Stage-specific behavior stays in the stage
directories.

## Requirements

- Python 3.14
- [`uv`](https://docs.astral.sh/uv/)
- [`just`](https://just.systems/) 1.46.0 or later
- A current `signal-cli` release for collection
- Apple silicon for local MLX-LM operations

Run commands from the repository root. Use `idiolect` for product operations.
Use the root `justfile` only for setup, build, and verification. Use `uv`
directly only for dependency maintenance.

## Set up

Install the core environment:

```console
just setup
```

Activate `.venv` before the `idiolect` examples below, and export the required
private environment values. To invoke the same CLI without activation, prefix a
command with `uv run --env-file .env`, for example
`uv run --env-file .env idiolect signal stats`.

See [docs/signal.md](docs/signal.md) before collection.
That procedure creates the private Signal state and `.env` file.

Install the optional local-model packages for training, inference, or
evaluation:

```console
just setup-train
```

## Workflow

**Collect.** Collect Signal events:

```console
idiolect signal collect
```

**Data.** List stored people and build a dataset for the linked Signal account:

```console
idiolect data people
idiolect data build TARGET_NAME
```

**Train.** Train the canonical policy or a named experiment policy:

```console
idiolect train var/data/DATASET_ID
idiolect -c qwen3-8b-smoke train var/data/DATASET_ID
```

**Inference.** Generate predictions from the base model recorded by a run and
from its adapter:

```console
idiolect -c qwen3-8b-smoke infer var/runs/RUN_ID --base --data var/data/DATASET_ID --split test
idiolect -c qwen3-8b-smoke infer var/runs/RUN_ID --data var/data/DATASET_ID --split test
```

**Evaluation.** Evaluate all runs from one policy:

```console
idiolect eval var/data/DATASET_ID \
  var/runs/RUN_ID_ONE \
  var/runs/RUN_ID_TWO \
  var/runs/RUN_ID_THREE
```

**Chat.** Open the private terminal chat:

```console
idiolect chat
```

See [docs/index.md](docs/index.md) for the complete procedures, required stop
conditions, artifact descriptions, and interpretation rules.

## Configuration and private state

`conf/idiolect.toml` is the canonical policy. `conf/exp/` contains complete
experiment policies. The configuration system does not merge files. Create a
new policy before changing an experiment:

```console
idiolect config list
idiolect config new EXPERIMENT_NAME
```

The default names `default` and `idiolect` select `conf/idiolect.toml`. Other
bare names select `conf/exp/NAME.toml`. `-c`, `--config`, and
`IDIOLECT_CONFIG` use these rules. A path selects that TOML file directly.

Commit a policy before its first run. Do not change a policy after a recorded
run uses it.

Keep credentials and Signal identifiers in `.env` or a system secret store.
Keep generated data under `var/`. Git ignores both locations, but ignore rules
do not remove data from Git history.

## Develop

[AGENTS.md](AGENTS.md) and the applicable nested `AGENTS.md` files define the
rules for code changes. Run the required checks before handoff:

```console
just check
just build
```

Tests use synthetic data and fake external boundaries. They must not read
`.env`, `var/`, live Signal state, model weights, or private artifacts. See
[docs/development.md](docs/development.md) for the package boundaries and
verification rules.
