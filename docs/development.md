# Development and Verification

## Package boundaries

- `ingest` reads source events, parses Signal JSON, and runs the harvest operation.
- `store` defines storage ports and implements DuckDB storage.
- `types.py` defines immutable shared records.
- `config.py` loads strict TOML settings and environment overrides.
- `data` builds fixed training examples.
- `train` defines the training port and implements the MLX-LM backend.
- `inference` defines generation ports and implements verified local MLX-LM inference.
- `eval` defines scoring ports and implements immutable policy and panel evaluation.

Keep external behavior behind typed ports. Keep adapter code out of contract modules.

## Required checks

Install `just` 1.46.0 or later. On macOS, use the Homebrew package. Do not install the unrelated Python package with the same name.

```console
brew install just
just --version
```

Use the root `justfile` as the project command interface. Its recipes use `uv`
for Python environment and package operations. Use `uv` directly only for
dependency maintenance.

Run:

```console
just setup
just check
just build
```

The checks verify Just formatting and run Ruff, ty, and pytest. The build creates a source archive and a wheel.

## Test rules

Use only synthetic Signal JSON fixtures. Use a fake command runner for `signal-cli`. Use `tmp_path` for DuckDB and configuration files.

Do not read `.env`, `conf/idiolect.toml`, `conf/local.toml`, `var/`, or the
installed launch agent in a test. Use fixture configuration files. Do not call
Signal, a model hub, or another network service. Do not run a real model, model
download, training operation, inference operation, evaluation operation, or GPU
operation. Use fake model resolvers, training commands, inference sessions, and
scoring sessions.

Each test must detect a possible implementation defect. Test the group whitelist, command safety options, message normalization, edit order, delete tombstones, reaction links, transaction behavior, duplicate events, strict configuration, and CLI results.

## Operational checks

Run these checks manually. Do not put them in pytest.

```console
just idiolect signal groups
just idiolect signal stats
just collect status
tail -f var/log/collect.err.log
```

These commands use private local state. An agent must not run them unless the user asks for a live operational check.

Install Textual and the local MLX packages before manual chat checks:

```console
just setup-chat
```

Automated chat tests use synthetic assistants, fake token counters, fake worker
events, and Textual pilot sessions. They do not load a model or use Metal. On
Apple silicon, manually check the landing probe, one assistant load, streaming,
prefill progress, stop and retry, the narrow footer, `/stats`, explicit save and
resume, worker reload, and clean exit.
