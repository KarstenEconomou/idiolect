# idiolect

Idiolect collects Signal group messages for a local ML pipeline. It stores allowed raw events and identity-linked records in DuckDB. Native mentions and reply snapshots remain available for target-relative training context.

```text
signal-cli → group allowlist → normalized records → DuckDB
                                                   │
                                                   v
                   MLX-LM JSONL → LoRA → inference → evaluation
```

The package implements Signal collection, DuckDB storage, target-relative rendering, immutable dataset export, local MLX-LM adapter training, and reproducible local inference. Evaluation contains a typed contract.

## Start

Read the [operations index](docs/index.md). It contains the complete replication procedure for:

- Signal device setup and group selection
- Private configuration and credentials
- Collection and DuckDB behavior
- macOS `launchd` operation
- Development and verification

The minimum interactive commands are:

```console
uv sync
set -a
source .env
set +a
uv run idiolect signal groups
uv run idiolect signal collect --follow
```

Install and run local training separately:

```console
just sync-train
just train var/data/DATASET_ID
```

Create and run a named experiment configuration:

```console
just config new qwen3-14b-r16
just config train qwen3-14b-r16 var/data/DATASET_ID
```

Training reads all model, formatting, optimizer, seed, and path choices from one complete tracked TOML configuration. See [adapter training](docs/train.md).

Generate paired predictions from one fixed dataset split:

```console
just infer base var/data/DATASET_ID test
just infer run var/runs/RUN_ID var/data/DATASET_ID test
```

Inference reads its complete policy from TOML and writes private content-addressed artifacts under `var/infer/`. See [local inference](docs/infer.md).

Use native command groups for routine operations:

```console
just collect status
just data people
just data build Karsten
```

## Develop

The project requires Python 3.14 and uses `uv`.

```console
just check
uv build
```

Tests use synthetic fixtures, temporary databases, and fake Signal process boundaries. Do not use live data or model calls in tests.
