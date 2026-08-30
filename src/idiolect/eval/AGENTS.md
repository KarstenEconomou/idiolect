# Evaluation Guide

These instructions apply to `src/idiolect/eval/`.

- Keep scoring ports and portable score values in `base.py`. Keep MLX-LM scoring
  sessions in `mlx.py`, policy evaluation orchestration in `local.py`, familiar
  judgment and panel behavior in `panel.py`, and bounded text-match logic in
  `text.py`.
- Compare every adapter policy with the base model recorded by that run. Do not
  substitute the current configured base or combine incompatible policies,
  datasets, splits, prompts, or model revisions.
- Use fixed held-out examples and recorded prompt policy. Keep metric selection,
  sampling limits, bootstrap policy, confidence policy, ballot policy, controls,
  thresholds, and seeds explicit in TOML.
- Keep evaluation, judgment, and panel artifacts private, immutable,
  content-addressed, and reproducible. Record enough provenance to verify every
  input and reject incomplete, ineligible, or tampered artifacts.
- Keep familiar-panel ballots blind and deterministic. Do not reveal model
  identity before a judgment, reuse a ballot outside its recorded panel, or
  weaken minimum-rater and control requirements in code.
- Treat training-text overlap as a verification rule, not a heuristic shortcut.
  Keep normalization explicit and bound index memory growth.
- Tests must fake score and generation backends. Do not load a model, use a GPU,
  contact a model hub, or rely on private judgments.
- Use `idiolect eval <dataset> <runs...>` for a complete local policy
  comparison, `idiolect eval rate` for one private blind rating session,
  and `idiolect eval panel` to summarize judgments. These are operational,
  not verification commands.
- Review `docs/eval.md` when operator-visible metrics, eligibility, commands,
  ballots, panels, interpretation, or artifact formats change. Do not document
  internal scoring decomposition or every validation branch.
