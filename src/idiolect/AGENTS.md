# Idiolect Package Guide

These instructions apply to `src/idiolect/`. Read the guide in the nearest
stage directory before changing stage code.

## Package boundaries

- `ingest` reads and normalizes source events.
- `store` defines persistence ports and implements local storage.
- `data` renders target-relative context and builds fixed datasets.
- `train` defines training contracts and implements MLX-LM training.
- `inference` defines generation contracts and implements verified inference.
- `chat` owns assistant discovery, transcript policy, model-worker supervision,
  and immutable snapshots.
- `tui` owns Textual presentation and input handling only.
- `eval` defines scoring contracts and builds immutable policy and panel results.
- `config.py`, `model.py`, `prompt.py`, and `types.py` are shared contracts. Keep
  them stage-neutral and change them only when multiple consumers need the same
  rule.

Do not bypass a stage contract to reuse backend internals. Put protocols and
portable value objects in `base.py` or a shared contract module; put concrete
behavior in an explicit backend or application module. Avoid import cycles and
do not make an optional MLX, Textual, DuckDB, or Signal dependency mandatory for
unrelated stages.

## Shared behavior

- Keep prompt rendering centralized in `prompt.py`. Training, inference,
  evaluation, and chat must use the same conversation grammar where their
  contracts overlap.
- Keep model identity, resolution, digest, and verification rules centralized in
  `model.py`. Verify recorded inputs before using an artifact.
- Keep shared records immutable where practical. Preserve stable serialization,
  ordering, IDs, and hashes when they contribute to artifact identity.
- Keep the CLI as argument parsing and dependency wiring. Call stage functions
  for behavior and return stage errors with actionable messages.
- Validate configuration at the stage boundary. A stage must not require
  unrelated optional settings.
