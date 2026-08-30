# Train Guide

These instructions apply to `src/idiolect/train/`.

- Keep training contracts and verified loaded-run values in `base.py`. Keep
  MLX-LM command construction, execution, and artifact handling in `mlx.py`.
- Treat the selected dataset, base model, revision, data format, LoRA policy,
  optimizer policy, run limit, reporting policy, and every seed as explicit TOML
  inputs. Do not supply hidden training defaults in Python.
- Resolve and verify the model and dataset before training. Record the complete
  effective policy, model digest, dataset identity, seed, backend information,
  adapter digest, and required output metadata.
- Make each run immutable and content-addressed. Use atomic temporary output and
  refuse incompatible or incomplete existing artifacts.
- Keep external reporting disabled unless configuration enables it. Never expose
  private training text, model credentials, or local paths in public examples.
- Tests must mock subprocess, MLX-LM, model resolution, and expensive work. Test
  the command, policy, verification, identity, and failure behavior owned here.
- Use `just setup-train` for optional packages and `idiolect train <dataset>` for a
  configured run. Training commands keep the Mac awake and are operational, not
  verification steps.
- Review `docs/train.md` when operator-visible setup, policy decisions,
  preflight behavior, commands, run operation, or artifacts change. Do not add
  backend command construction or other implementation detail.
