# idiolect

Idiolect collects Signal group messages for a local ML pipeline. It stores allowed raw events and identity-linked records in DuckDB. Native mentions and reply snapshots remain available for target-relative training context.

```text
signal-cli → group allowlist → normalized records → DuckDB
                                                   │
                                                   v
                         MLX-LM JSONL → LoRA → evaluation
```

Signal collection, DuckDB storage, target-relative rendering, immutable dataset export, and local MLX-LM adapter training operate now. Evaluation and inference contain typed contracts only.

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
cp conf/local.toml.example conf/local.toml
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

The training command reads all model, formatting, optimizer, seed, and path choices from private TOML configuration. See [adapter training](docs/train.md).

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
