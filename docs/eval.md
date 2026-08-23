# Model Evaluation

## Purpose

Evaluation tests whether one complete training policy improves the probability
and observable behavior of replies that the target person could plausibly send.
It compares every configured training seed with the exact base model recorded by
the runs.

The `fidelity` suite uses the validation split only. It does not read Signal or
DuckDB. It reads a verified dataset, verified training runs, and immutable
inference artifacts.

Evaluation does not produce one fidelity score. Open conversation has many valid
replies, and one recorded reply is not a complete reference distribution. Read
the five pillars together:

1. Likelihood: completion probability against the held-out human reply.
2. Voice: observable style and surface-distribution fidelity.
3. Validity: malformed, degenerate, or unstable generation diagnostics.
4. Memorization: training-text reproduction and leakage regression.
5. Recognition: familiar-rater preference for target likeness.

Likelihood, voice, validity, and memorization come from one automatic report.
Recognition comes from the separate familiar panel. No value is combined into
one score.

## Configuration

Each complete TOML file contains a required `[eval]` table. It selects:

- the private output directory, MLX-LM backend, and metric suite;
- `split = "valid"` and the stable example limit;
- the bootstrap seed, replication count, and confidence level;
- the minimum long training-text match;
- eligibility limits for empty output, format violations, truncation, and new
  training-text matches;
- the blind-ballot seed, ballot count, control fraction, and panel minimums.

`max_examples = 0` selects all validation examples. A positive value selects
examples by stable hash. The existing `[inference]` table supplies generation seeds,
token limits, and sampling behavior. Evaluation records the effective values in
its recipe.

Do not change a used experiment configuration. Create a complete new
configuration when the policy changes.

## Automatic policy evaluation

Install the optional local model packages:

```console
just setup-train
```

Supply every run produced by the configuration. For the canonical three-seed
policy:

```console
just eval policy var/data/DATASET_ID \
  var/runs/RUN_ID_ONE \
  var/runs/RUN_ID_TWO \
  var/runs/RUN_ID_THREE
```

Select the complete experiment configuration that created the runs:

```console
IDIOLECT_CONFIG=conf/exp/qwen3-8b-smoke.toml \
  just eval policy var/data/DATASET_ID var/runs/RUN_ID_ONE
```

The command rejects a selected configuration whose training policy does not
match the recorded runs. Use the same configuration for rating and panel
commands because it also contains the fixed ballot policy.

The runner rejects a partial seed set. It also rejects runs with different
datasets, models, text formats, training policies, or sequence limits.

The report contains four pillar sections. Each section compares the base model,
each adapter run, and the pooled policy with the held-out evidence.

### Likelihood

1. Score the real held-out reply with the base and every adapter. Only reply and
   reply-termination tokens contribute to negative log-likelihood. Static model
   prefill text does not contribute.
2. Report macro mean NLL across examples and token-weighted corpus perplexity
   as separate values.
3. Report paired macro-NLL policy deltas with bootstrap confidence intervals
   and the rate of examples that improve.

### Voice

1. Generate the same examples with identical derived random streams.
2. Compare message length, line structure, punctuation, capitalization, emoji,
   mentions, URLs, repeated characters, and character three-grams with the
   held-out human replies.
3. Report absolute feature differences and character three-gram JS divergence.

Character n-grams can measure topic as well as style. Read voice results with
the other pillars.

### Validity

1. Detect empty text, unknown mentions, template leakage, multi-role output,
   truncation, cross-prompt duplicates, and within-prompt duplicates.
2. Apply the configured empty-output, format-violation, and truncation gates.

Duplicates measure diversity. A policy that repeats one reply is not valid open
conversation, even when each copy is well formed.

### Memorization

1. Compare normalized generated text with training completions. Report exact
   duplicates and long contiguous matches. A sparse rolling-hash index finds
   candidates, and exact string matching verifies each reported match.
2. Measure the incremental memorization rate above the larger of the base rate
   and the held-out reference rate. Apply the configured delta gate.

## Eligibility

A policy is `eligible` only when all automatic gates pass. The bootstrap
resamples conversation examples. Tokens, generated samples, and training seeds
are not independent observations. Reports show each run, an equally weighted
policy estimate, paired macro-NLL confidence intervals, token-weighted corpus
perplexity, and run spread.

Each validity and memorization gate applies to every training run on its own,
not only to the pooled policy, so one degenerate seed cannot hide behind the
other runs. Eligibility is not a claim that the model has the target's voice
and is not a complete privacy audit. Use the recognition result as separate
evidence.

## Output

Automatic evaluation writes:

```text
var/eval/<evaluation-id>/
├── examples.jsonl
├── manifest.json
├── metrics.json
└── report.md
```

The evaluation ID includes the dataset and run digests, selected examples,
inference policy, evaluation policy, metric suite, and backend versions. An equal
request returns the existing verified directory.

`report.md` has one section per automatic pillar. `metrics.json` keeps the full
pillar values for the base, each run, and the pooled policy. `examples.jsonl`
contains identifiers, scores, numeric diagnostics, and failure flags. It does
not copy prompts, human replies, or generated replies. The manifest points to
the private source artifacts used by the blind workflow.

## Recognition: familiar-panel evaluation

Use raters who know the target's writing. Every rater must consent and must
already have permission to view the sampled conversation contexts. Do not show a
group conversation to a rater merely because that person knows the target.

Run one terminal session on the data owner's Mac:

```console
just idiolect eval rate var/eval/EVALUATION_ID --rater rater-01
```

The terminal shows randomized A/B replies and asks which reply:

- the target would be more likely to send in this context;
- sounds more like the target;
- fits the conversation better.

Most ballots compare the policy with the base. The configured control fraction
compares the real human reply with either the policy or base. Human controls
calibrate interpretation. They are not attention tests, because more than one
reply can be valid.

The command writes a pseudonymous immutable artifact under
`var/eval/judgments/`. It does not store prompts or reply text. Use a stable
pseudonym that does not contain a name, phone number, or Signal identifier.
Panel construction reconstructs the fixed ballot schedule from the verified
evaluation sources. It rejects missing, duplicate, changed, or unknown ballot
answers even when an artifact is internally content-addressed.

After the configured panel completes its sessions, create a report:

```console
just idiolect eval panel var/eval/EVALUATION_ID \
  var/eval/judgments/JUDGMENT_ID_ONE \
  var/eval/judgments/JUDGMENT_ID_TWO \
  var/eval/judgments/JUDGMENT_ID_THREE
```

The panel report contains policy/base wins, ties, neither choices, two-way
example-and-rater bootstrap confidence intervals, human-control recovery, A/B
position preference, and finite-sample nominal Krippendorff agreement for each
rating dimension. It remains `incomplete` until both configured panel minimums
are met.

Ballot construction draws the compared run from a rater-specific random stream,
so run identity is not paired with one generation seed across primary
comparisons. Judgment and panel artifacts record the ballot scheme version. The
current version is 2; older judgment artifacts do not match the current schedule
and must be rated again.

## Interpretation

Prefer evidence that converges across the five pillars:

- Likelihood: lower adapter completion NLL than the base across training seeds.
- Recognition: panel preference for the policy on target likelihood and voice
  without a contextual-fit regression.
- Voice: features closer to held-out human messages.
- Validity and memorization: no format, truncation, repetition, or memorization
  gate failure.
- Limited run-to-run variation.

Do not select a model from one metric. Character n-grams can measure topic as
well as style. Low likelihood can reward the one observed reply even when other
replies are valid. Familiar raters can prefer polished text that does not match
the target. The separate pillars expose these failure modes.

All evaluation, judgment, and panel files are private. Do not publish a report,
manifest, prompt, completion, prediction, or rater artifact.
