# Development and Verification

## Package boundaries

- `ingest` reads source events, parses Signal JSON, and runs the harvest operation.
- `store` defines storage ports and implements DuckDB storage.
- `types.py` defines immutable shared records.
- `config.py` loads strict TOML settings and environment overrides.
- `data`, `train`, `eval`, and `infer` contain future ML contracts.

Keep external behavior behind typed ports. Keep adapter code out of contract modules.

## Required checks

Run:

```console
uv sync --frozen
just check
uv build
```

The checks run Ruff, ty, and pytest. The build creates a source archive and a wheel.

## Test rules

Use only synthetic Signal JSON fixtures. Use a fake command runner for `signal-cli`. Use `tmp_path` for DuckDB and configuration files.

Do not read `.env`, `conf/local.toml`, `var/`, or the installed launch agent in a test. Do not call Signal or another network service. Do not run a real model, model download, training operation, or GPU operation.

Each test must detect a possible implementation defect. Test the group allowlist, command safety options, message normalization, edit order, delete tombstones, reaction links, transaction behavior, duplicate events, strict configuration, and CLI results.

## Operational checks

Run these checks manually. Do not put them in pytest.

```console
uv run idiolect signal groups
uv run idiolect signal stats
launchctl print gui/$(id -u)/com.idiolect.collect
tail -f var/log/collect.err.log
```

These commands use private local state. An agent must not run them unless the user asks for a live operational check.
