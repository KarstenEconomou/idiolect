# Idiolect Agent Guide

## Project

Idiolect is a Python 3.14 package for a local-first ML pipeline. It collects whitelisted Signal group messages, stores source and normalized data in DuckDB, builds immutable target-specific JSONL datasets, trains content-addressed MLX-LM LoRA adapters, generates verified local predictions, provides private multi-turn terminal chat for verified adapters, and evaluates adapter policies against their recorded base models with automatic metrics and private familiar-panel judgments.

Code lives in `src/idiolect/`. Tests mirror it in `tests/`. Public configuration lives in `conf/`. Operational and replication procedures live in `docs/`. Just modules live in `just/`.

Keep stage boundaries explicit:

- `ingest` reads and normalizes Signal events.
- `store` defines storage ports and implements DuckDB storage.
- `data` renders target-relative context and builds datasets.
- `train` defines training contracts and implements MLX-LM training.
- `infer` defines generation contracts and implements MLX-LM inference.
- `chat` discovers verified assistants, owns transcript and context policy, supervises the MLX worker, and stores explicit immutable snapshots.
- `tui` contains only Textual presentation and input handling. Do not put model loading, prompt policy, artifact verification, or snapshot identity logic in this package.
- `eval` defines scoring contracts, runs immutable local policy evaluations, and collects and summarizes private familiar-panel judgments.
- `config.py`, `model.py`, `prompt.py`, and `types.py` contain shared policy and data contracts.

Keep external systems behind the existing typed ports. Keep protocol modules free of backend behavior. Keep the CLI thin and put application behavior in its stage module.

Keep chat prompts byte-compatible with the shared training conversation grammar. Show the user as `You`, but serialize the configured participant name in prompts. Format every assistant identity as `IDIOLECT // NAME@run [MODEL]`, where `run` is a unique eight-character run prefix and `MODEL` is the final model repository or path component. Keep the TUI sparse and use its near-black, amber, cyan, warm off-white, muted gray, and restrained red tokens consistently. Render transcript content as literal plain text.

## Commands

Use the root `justfile` as the project command interface. Its recipes use `uv`
for Python environment and package operations.

- `just setup`: update the local environment.
- `just setup-train`: install optional local MLX-LM training packages.
- `just setup-chat`: install optional MLX-LM and Textual packages.
- `just idiolect`: run the CLI.
- `just chat`: open the searchable local chat chooser while the Mac stays awake.
- `just chat run <run> <dataset>`: open one verified assistant directly.
- `just chat resume <chat>`: resume one verified saved snapshot.
- `just build`: build distributions.
- `just collect status`, `start`, or `stop`: operate the installed LaunchAgent.
- `just idiolect data people`: list normalized authors.
- `just data build <name>`: build a dataset for the linked Signal user.
- `just config list`, `new`, or `train`: manage complete experiment configurations.
- `just train <dataset>`: train the canonical configuration while the Mac stays awake.
- `just infer base`, `base-of`, or `run`: generate one dataset split while the Mac stays awake.
- `just eval policy <dataset> <runs...>`: compare one complete training policy with its recorded base.
- `just idiolect eval rate`: complete one private blind familiar-rater session.
- `just idiolect eval panel`: summarize familiar-rater judgments.
- `just test`: run pytest.
- `just lint`: run Ruff.
- `just typecheck`: run ty.
- `just check`: run all required checks.

Use `uv` directly only for dependency maintenance. Add packages with `uv add`
or `uv add --dev`. Never edit `uv.lock` by hand. Run `just check` before
handoff.

Recipes that launch Idiolect must use `uv run --env-file .env`. Do not load
`.env` for setup, build, lint, type-check, or test recipes. Tests must stay
independent of private environment values.

## Code

Use focused modules, explicit imports, four-space indentation, and type annotations for public functions. Use `snake_case` for modules and functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.

Write every Python docstring in ASD-STE100 Simplified Technical English. Add docstrings to modules and public classes, methods, and functions. Use short sentences and one term for one meaning.

Treat TOML as the complete experiment policy. Do not add an implicit model, formatting, optimizer, sampling, seed, path, or reporting choice in code. Validate chat policy only at the chat boundary so unrelated stages can use configurations without it. Reject missing, unknown, or incompatible settings at the applicable boundary. Keep dataset, run, inference, chat snapshot, evaluation, judgment, and panel artifacts immutable and content-addressed.

Keep MLX ownership inside the supervised chat worker. Model verification and loading must not block the Textual event loop. Preserve the in-memory transcript when loading, generation, cancellation, saving, or worker recovery fails. An unsaved chat must not create a temporary transcript file.

Update `README.md` and the applicable file in `docs/` when behavior, setup, configuration, operation, or data flow changes.

Document only the present implementation. Never describe prior implementations, removed paths, replaced commands, migration narratives, or historical design choices. Git history is the change record.

## Tests

Every test must protect useful behavior or a stable contract. A test must fail for a plausible implementation defect. Do not add tests only to increase coverage, execute lines, confirm that imports succeed, or restate implementation details.

- Test public behavior, data contracts, boundary conditions, error handling, and regression cases. Do not bind tests to private helpers or incidental call order.
- Never read live configuration, credentials, environment-specific paths, `var/`, user files, or real Signal data. Use pytest fixtures, `tmp_path`, synthetic messages, and safe test settings.
- Never contact Signal, cloud services, model hubs, tracking services, or other networks. Replace external boundaries with fakes or mocks.
- Never download a model or run real training, fine-tuning, inference, GPU work, or another expensive routine in tests. Mock the expensive boundary and assert the inputs, outputs, state changes, and failures that the application owns.
- Test chat worker behavior with fake backends and TUI behavior with a fake runtime. Do not import MLX, load weights, contact a model hub, or require a GPU in chat tests.
- Prefer small in-memory fakes when behavior across a boundary matters. Use mocks only at defined ports. Do not mock the unit under test.
- Make tests deterministic. Fix clocks, random seeds, identifiers, and ordering when they affect results. Tests must not depend on execution order or machine state.
- Use temporary DuckDB databases, Parquet files, and artifact directories for storage integration tests. Fixtures must create and clean up all test state.
- Test dataset splitting for time-order preservation and leakage prevention. Test ingestion and storage for duplicate-event handling when those features exist.
- Keep tests fast enough for `just test` on a development machine. If a defect cannot be tested without live data or expensive work, test the local decision logic and validate the external integration outside pytest.
- Do not weaken an assertion, replace a meaningful test with an import check, or change production behavior only to make a test pass.

Name files `test_*.py` and tests `test_<behavior>`. Use a regression test for each bug fix.

## Data and Security

Never commit credentials, local configuration, private messages, chat transcripts, saved chat snapshots, Signal identifiers, DuckDB files, Parquet datasets, model weights, adapters, checkpoints, logs, or generated artifacts. Use only data from consenting participants. Use synthetic data in examples and tests. Put secrets in environment variables and document them with safe placeholders.

`conf/idiolect.toml` is the public canonical configuration. Files in `conf/exp/` are complete public experiment policies. Keep private values out of all tracked configurations. Store Signal account and chat identifiers only in `.env` or a system secret store. Treat optional `conf/local.toml`, `.env`, `var/`, and the installed `launchd` agent as live private state. Do not read or change this state unless the user requests an operational task. Do not start, stop, install, or remove the collector as part of a code test. Use `docs/` as the source for operating procedures.

## Changes

Use Conventional Commits in the form `type(scope): description`. Keep the description lowercase, concise, and imperative. Pull requests must state the reason, summarize the approach, identify dependency or configuration changes, and confirm that `just check` passes.
