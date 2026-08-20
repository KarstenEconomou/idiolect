# Idiolect Agent Guide

## Project

Idiolect is a Python 3.14 package for a local-first ML pipeline. It collects Signal group messages, stores source and normalized data in DuckDB, builds immutable JSONL datasets, and trains content-addressed MLX-LM adapters for contextual reply generation.

Code lives in `src/idiolect/`. Tests mirror it in `tests/`. Keep Signal, storage, dataset, training, evaluation, and inference code behind the existing typed ports. Do not add working backend code to contract modules.

## Commands

Use `uv` for all Python work and the root `justfile` for checks.

- `uv sync`: update the local environment.
- `uv sync --extra train`: install optional local MLX-LM training packages.
- `uv run idiolect`: run the CLI.
- `uv build`: build distributions.
- `just collect status|start|stop`: operate the installed LaunchAgent.
- `just data people|build`: inspect authors or build a self dataset.
- `just train <dataset>`: train configured seeds while the Mac stays awake.
- `just test`: run pytest.
- `just lint`: run Ruff.
- `just typecheck`: run ty.
- `just check`: run all required checks.

Add packages with `uv add` or `uv add --dev`. Never edit `uv.lock` by hand. Run `just check` before handoff.

## Code

Use focused modules, explicit imports, four-space indentation, and type annotations for public functions. Use `snake_case` for modules and functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.

Write every Python docstring in ASD-STE100 Simplified Technical English. Add docstrings to modules and public classes, methods, and functions. Use short sentences and one term for one meaning.

Update `README.md` and the applicable file in `docs/` when behavior, setup, configuration, operation, or data flow changes.

## Tests

Every test must protect useful behavior or a stable contract. A test must fail for a plausible implementation defect. Do not add tests only to increase coverage, execute lines, confirm that imports succeed, or restate implementation details.

- Test public behavior, data contracts, boundary conditions, error handling, and regression cases. Do not bind tests to private helpers or incidental call order.
- Never read live configuration, credentials, environment-specific paths, `var/`, user files, or real Signal data. Use pytest fixtures, `tmp_path`, synthetic messages, and safe test settings.
- Never contact Signal, cloud services, model hubs, tracking services, or other networks. Replace external boundaries with fakes or mocks.
- Never download a model or run real training, fine-tuning, inference, GPU work, or another expensive routine in tests. Mock the expensive boundary and assert the inputs, outputs, state changes, and failures that the application owns.
- Prefer small in-memory fakes when behavior across a boundary matters. Use mocks only at defined ports. Do not mock the unit under test.
- Make tests deterministic. Fix clocks, random seeds, identifiers, and ordering when they affect results. Tests must not depend on execution order or machine state.
- Use temporary DuckDB databases, Parquet files, and artifact directories for storage integration tests. Fixtures must create and clean up all test state.
- Test dataset splitting for time-order preservation and leakage prevention. Test ingestion and storage for duplicate-event handling when those features exist.
- Keep tests fast enough for `just test` on a development machine. If a defect cannot be tested without live data or expensive work, test the local decision logic and validate the external integration outside pytest.

Name files `test_*.py` and tests `test_<behavior>`. Use a regression test for each bug fix.

## Data and Security

Never commit credentials, local configuration, private messages, Signal identifiers, DuckDB files, Parquet datasets, model weights, adapters, checkpoints, logs, or generated artifacts. Use synthetic data in examples and tests. Put secrets in environment variables and document them with safe placeholders.

Treat `conf/local.toml`, `.env`, `var/`, and the installed `launchd` agent as live private state. Do not read or change this state unless the user requests an operational task. Do not start, stop, install, or remove the collector as part of a code test. Use `docs/` as the source for operating procedures.

## Changes

Use Conventional Commits in the form `type(scope): description`. Keep the description lowercase, concise, and imperative. Pull requests must state the reason, summarize the approach, identify dependency or configuration changes, and confirm that `just check` passes.
