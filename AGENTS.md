# Idiolect Agent Guide

## Instruction scope

Read this file before you change the repository. Then read every `AGENTS.md`
that applies to the files you will touch. A file inherits instructions from the
repository root through its nearest parent directory. The nearest guide adds to
or narrows the broader guide.

- Read `src/idiolect/AGENTS.md` for Python package work.
- Read the guide inside a stage directory before you change that stage.
- Read `tests/AGENTS.md` for test work. Also read the matching source-stage guide
  when a test mirrors code under `src/idiolect/`.
- For a cross-stage change, read all affected guides and preserve every stage
  boundary. Do not copy scoped rules back into this root file.

## Project

Idiolect is a Python 3.14 package for a local-first ML pipeline. It collects
whitelisted Signal group messages, stores source and normalized data in DuckDB,
builds immutable target-specific JSONL datasets, trains content-addressed MLX-LM
LoRA adapters, generates verified local predictions, provides private terminal
chat, and evaluates adapter policies against their recorded base models.

Code lives in `src/idiolect/`. Tests mirror it in `tests/`. Public configuration
lives in `conf/`. Procedures live in `docs/`. Just modules live in `just/`.

Keep dependencies and ownership explicit. Put external systems behind typed
ports, keep protocol modules free of backend behavior, keep the CLI thin, and
put application behavior in the stage that owns it.

## Workflow

Use the root `justfile` as the project command interface. Its recipes use `uv`.

- `just setup`, `just setup-train`, and `just setup-chat` manage environments.
- `just idiolect` runs the CLI; `just chat` opens the local registry.
- `just config list`, `just config new`, and `just config train` manage complete
  experiment policies.
- `just data build`, `just train`, `just inference`, and `just eval policy` run
  the main pipeline stages. Read the applicable procedure before operational
  work.
- `just build` builds distributions.
- `just test`, `just lint`, `just typecheck`, and `just check` verify changes.

Use `uv` directly only for dependency maintenance. Add packages with `uv add` or
`uv add --dev`. Never edit `uv.lock` by hand. Recipes that launch Idiolect must
use `uv run --env-file .env`. Setup, build, lint, type-check, and test recipes
must not load `.env`.

Run focused checks while working. Run `just check` before handoff.

## Code and configuration

- Use focused modules, explicit imports, four-space indentation, and type
  annotations for public functions. Use `snake_case` for modules and functions,
  `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Write module and public API docstrings in ASD-STE100 Simplified Technical
  English. Use short sentences and one term for one meaning.
- Treat TOML as the complete experiment policy. Do not hide model, formatting,
  optimizer, sampling, seed, path, or reporting choices in code. Reject missing,
  unknown, or incompatible settings at the boundary that uses them.
- Keep dataset, run, inference, chat snapshot, evaluation, judgment, and panel
  artifacts immutable and content-addressed.
- Update `README.md` and the applicable procedure when behavior, setup,
  configuration, operation, or data flow changes. Document only the current
  implementation. Git history is the change record; do not write migration or
  historical narratives into current documentation.

## Data and security

Never commit credentials, local configuration, private messages, chat
transcripts, snapshots, Signal identifiers, databases, datasets, model weights,
adapters, checkpoints, logs, or generated artifacts. Use synthetic data in
examples and tests, and process data only from consenting participants.

Keep public, reproducible settings in `conf/idiolect.toml` and `conf/exp/`. Keep
secrets and Signal identifiers in environment variables or a system secret
store. Treat `.env`, optional `conf/local.toml`, `var/`, user files, and the
installed LaunchAgent as live private state. Do not read or change them unless
the user requests an operational task. Do not operate the collector as part of
code verification. Use `docs/` as the source for operational procedures.

## Changes and handoff

Preserve unrelated user changes in a dirty worktree. Use Conventional Commits in
the form `type(scope): description`; keep the description lowercase, concise,
and imperative. Pull requests must state the reason, summarize the approach,
identify dependency or configuration changes, and confirm that `just check`
passes.
