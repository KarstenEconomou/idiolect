# idiolect

Idiolect collects Signal group messages for a local ML pipeline. It stores allowed raw events and normalized records in DuckDB.

```text
signal-cli → group allowlist → normalized records → DuckDB
                                                   │
                                                   v
                              Parquet → PEFT → evaluation
```

Signal collection and DuckDB storage operate now. Dataset creation, training, evaluation, and inference contain typed contracts only.

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

## Develop

The project requires Python 3.14 and uses `uv`.

```console
just check
uv build
```

Tests use synthetic fixtures, temporary databases, and fake Signal process boundaries. Do not use live data or model calls in tests.
