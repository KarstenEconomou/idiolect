# Inference Guide

These instructions apply to `src/idiolect/inference/`.

- Keep generation ports, target values, streaming events, and cancellation
  contracts in `base.py`. Keep local orchestration and immutable prediction
  artifacts in `local.py`; keep MLX-LM session behavior in `mlx.py`.
- Distinguish the configured base, a run's recorded base, and a run adapter.
  Resolve the requested target explicitly and verify its recorded model and
  adapter digests before generation.
- Use the shared prompt grammar and the generation policy from TOML. Keep seed,
  token limits, sampling, formatting, and backend selection explicit and
  reproducible.
- Preserve input ordering and provenance. Make prediction artifacts immutable
  and content-addressed, and reject partial, incompatible, or tampered inputs.
- Keep cancellation and streaming semantics at the typed backend boundary. Do
  not leak MLX objects into local orchestration or other stages.
- Tests must use fake backends. Do not load weights, import MLX, contact model
  hubs, or run GPU generation.
- Use `idiolect infer --base`, `idiolect infer <run> --base`, or
  `idiolect infer <run>` for configured generation. Batch inference keeps the
  Mac awake and is operational, not a verification step.
- Review `docs/inference.md` when operator-visible target selection, commands,
  generation policy, output, or prediction artifacts change. Do not copy shared
  prompt or security contracts into this procedure.
