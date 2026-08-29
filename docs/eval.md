# Model Evaluation

## Purpose

Evaluation compares one complete adapter policy with the exact base model
recorded by its runs. It uses the validation split. It does not read Signal or
DuckDB.

Open conversation has many valid replies. One recorded reply is not a complete
reference distribution. Do not reduce the report to one score.

## Evidence pillars

Use five separate evidence pillars:

1. **Likelihood** measures the probability of the held-out human response.
2. **Voice** compares observable text and response-fragmentation features.
3. **Validity** detects empty, malformed, repeated, or truncated output.
4. **Memorization** detects training-text reproduction above the baseline.
5. **Recognition** measures blind preference from familiar raters.

The automatic report contains the first four pillars. The familiar-rater panel
provides recognition evidence.

## Policy requirements

The `[eval]` table fixes the validation split, example selection, bootstrap,
confidence level, overlap threshold, automatic gates, ballot schedule, and
panel minimums. `[inference]` supplies generation seeds and sampling behavior.

`max_examples = 0` selects all validation examples. A positive value selects
examples by stable hash.

Supply every seed from the training policy. Evaluation rejects a partial seed
set or a set with different datasets, models, formats, policies, or sequence
limits.

Do not change a used policy. Create a complete new TOML file when an evaluation
or inference choice changes.

## Run the automatic evaluation

Install the local model packages:

```console
just setup-train
```

Run the complete canonical seed set:

```console
idiolect eval var/data/DATASET_ID \
  var/runs/RUN_ID_ONE \
  var/runs/RUN_ID_TWO \
  var/runs/RUN_ID_THREE
```

Select a named experiment policy with `-c`:

```console
idiolect -c qwen3-8b-smoke eval var/data/DATASET_ID var/runs/RUN_ID_ONE
```

Use the same policy for rating and panel commands. It contains the fixed ballot
rules.

On macOS, automatic evaluation holds an idle-sleep assertion for its duration.
Rating and panel creation do not. Collection can run during evaluation.

## Read the automatic report

### Likelihood

Idiolect scores the held-out completion with the base and every adapter. Prompt
and static assistant-prefill tokens do not contribute to loss.

Read macro mean negative log-likelihood and token-weighted corpus perplexity as
different summaries. Use paired confidence intervals and the example
improvement rate. Inspect run spread; one good seed is not a stable policy.

### Voice

Idiolect generates the same examples with equal derived random streams. It
compares length, lines, punctuation, capitalization, emoji, mentions, URLs,
repeated characters, character three-grams, and message-bubble counts.

These features measure surface similarity. Character n-grams can also measure
topic. Use them with likelihood, validity, and recognition.

### Validity

Validity checks empty output, unknown mentions, template leakage, multi-role
text, truncation, and duplicate generations. Each configured gate uses the
worst run. The report also shows pooled-policy diagnostics.

### Memorization

Memorization compares normalized output with training completions. It reports
exact duplicates and verified long contiguous matches. The gate measures the
increase above the larger of the base rate and held-out reference rate.

## Eligibility

A policy is `eligible` only when every automatic gate passes. Eligibility is
not proof of target voice. It is not a complete privacy audit. Recognition is
separate evidence.

The bootstrap resamples conversation examples. Tokens, generations, and seeds
are not independent observations.

## Automatic artifact

Automatic evaluation creates:

```text
var/eval/<evaluation-id>/
├── examples.jsonl
├── manifest.json
├── metrics.json
└── report.md
```

`report.md` is the concise human report. `metrics.json` contains complete pillar
values. `examples.jsonl` contains identifiers, scores, numeric diagnostics, and
failure flags. It does not copy prompts, human replies, or generated replies.

The evaluation ID commits to all source digests, selected examples, policies,
metrics, and backend versions. An equal request returns the existing verified
artifact.

## Familiar-rater sessions

Use raters who know the target's writing. Each rater must consent and must have
permission to view every sampled conversation.

Run one private session on the data owner's computer:

```console
idiolect eval rate var/eval/EVALUATION_ID --rater rater-01
```

The session presents blind A/B comparisons. Most ballots compare policy output
with base output. Control ballots compare the real human response with a model
response. Controls calibrate interpretation; they are not attention tests.

The command writes an immutable artifact under `var/eval/judgments/`. Use a
stable pseudonym that contains no name, phone number, or Signal identifier. The
artifact stores choices and provenance, not prompt or reply text.

## Panel report

After the configured sessions finish, create the panel:

```console
idiolect eval panel var/eval/EVALUATION_ID \
  var/eval/judgments/JUDGMENT_ID_ONE \
  var/eval/judgments/JUDGMENT_ID_TWO \
  var/eval/judgments/JUDGMENT_ID_THREE
```

The panel artifact is under `var/eval/panels/`. It reports policy and base wins,
ties, neither choices, human-control recovery, position preference, agreement,
and example-and-rater confidence intervals. It remains incomplete until both
configured panel minimums are met.

## Model selection

Prefer evidence that agrees across pillars:

- lower held-out completion loss than the base across seeds;
- familiar-rater preference for target likeness without a context-fit loss;
- voice features closer to held-out human responses;
- no validity or memorization gate failure;
- limited variation across training seeds.

Do not select a model from one metric. Do not publish an evaluation, judgment,
panel, manifest, prompt, completion, or prediction.
