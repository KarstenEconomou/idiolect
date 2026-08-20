# Development and Verification

## Package boundaries

- `ingest` reads source events, parses Signal JSON, and runs the harvest operation.
- `store` defines storage ports and implements DuckDB storage.
- `types.py` defines immutable shared records.
- `config.py` loads strict TOML settings and environment overrides.
- `data` builds fixed training examples.
- `train` defines the training port and implements the MLX-LM backend.
- `eval` and `infer` contain future ML contracts.

Keep external behavior behind typed ports. Keep adapter code out of contract modules.

## Required checks

Install `just` 1.31.0 or later. On macOS, use the Homebrew package. Do not install the unrelated Python package with the same name.

```console
brew install just
just --version
```

Run:

```console
just sync
just check
uv build
```

The checks run Ruff, ty, and pytest. The build creates a source archive and a wheel.

## Test rules

Use only synthetic Signal JSON fixtures. Use a fake command runner for `signal-cli`. Use `tmp_path` for DuckDB and configuration files.

Do not read `.env`, `conf/idiolect.toml`, `conf/local.toml`, `var/`, or the installed launch agent in a test. Use fixture configuration files. Do not call Signal, a model hub, or another network service. Do not run a real model, model download, training operation, or GPU operation. Use a fake model resolver and a fake training command.

Each test must detect a possible implementation defect. Test the group allowlist, command safety options, message normalization, edit order, delete tombstones, reaction links, transaction behavior, duplicate events, strict configuration, and CLI results.

## Operational checks

Run these checks manually. Do not put them in pytest.

```console
uv run idiolect signal groups
uv run idiolect signal stats
just collect status
tail -f var/log/collect.err.log
```

These commands use private local state. An agent must not run them unless the user asks for a live operational check.
