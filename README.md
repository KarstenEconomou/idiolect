# idiolect

Idiolect will build language model adapters that reproduce the writing style and conversation behavior of one person. The project is a typed scaffold. It does not yet collect messages, store data, train models, or generate replies.

## Data flow

```text
Signal live input ─┐
                   ├─> source events ─> normalized messages ─> DuckDB
Signal file input ─┘                                      │
                                                          v
                                    fixed Parquet dataset by person
                                                          │
                                                          v
                                    Hugging Face base model + PEFT
                                                          │
                                                          v
                                      test results + local adapter
```

DuckDB is the local source of truth. A training run reads only a fixed Parquet dataset. This rule makes each run repeatable and prevents live message changes from changing an active run.

## Packages

- `ingest` defines source and parser ports. The Signal module reserves adapters for live data and file data.
- `store` defines ports for DuckDB, Parquet datasets, and run files.
- `data` defines the build step for contextual reply datasets.
- `train` defines the model training port and reserves a Hugging Face PEFT adapter.
- `eval` defines the model test port.
- `infer` defines the reply generation port and reserves a Hugging Face adapter.
- `types.py` contains records that the pipeline shares.
- `config.py` contains fixed settings for each pipeline stage.

The adapter modules have no implementation. External runtime packages will be added when an adapter gets an implementation. The planned storage packages are DuckDB and PyArrow. The planned model packages are Transformers, Datasets, Torch, and PEFT.

## Local files

Idiolect will use the ignored `var/` directory by default:

```text
var/
├── idiolect.duckdb
├── data/<dataset-id>/
│   ├── train.parquet
│   ├── valid.parquet
│   ├── test.parquet
│   └── manifest.json
├── run/<run-id>/
│   ├── config.toml
│   ├── metrics.json
│   └── adapter/
└── log/
```

Do not commit messages, Signal identifiers, datasets, model files, credentials, or local settings. Use operating-system file permissions and disk encryption to protect local data. Idiolect does not enforce consent rules.

## Configuration

Copy `conf/local.toml.example` to `conf/local.toml` when configuration loading is implemented. Commit only safe example files. Put credentials and other secret values in environment variables.

## Development

The project requires Python 3.14 and uses `uv` for all Python commands.

```console
uv sync
uv run idiolect
just check
uv build
```

`just check` runs Ruff, ty, and pytest. The command-line entry point currently has no operation and writes no output.
