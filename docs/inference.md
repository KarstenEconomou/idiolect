# Local Inference

## Purpose

Inference generates local text from one verified model target. It accepts one
private prompt or one split from a verified dataset. It does not read Signal or
DuckDB.

Install the optional model packages:

```console
just setup-train
```

## Policy

The `[inference]` table defines output, seeds, example selection, prompt and
generation limits, sampling, and repetition behavior.

`max_examples = 0` selects all examples. A positive value selects examples by
stable hash. It does not select the first rows.

`max_prompt_tokens` applies after the model chat template renders the prompt.
Inference rejects an overlong prompt. It does not remove dataset context or
reduce `max_tokens`.

The text format comes from `[train.data]` for a configured base. A run target
uses the format recorded by that run. This rule keeps training, inference,
evaluation, and adapter chat consistent.

## Targets

Select exactly one target:

| Option | Target |
|---|---|
| `--base` | Base model and text policy from the selected TOML file. |
| `--base-of RUN` | Exact base model recorded by a verified run. |
| `--run RUN` | Base model and adapter recorded by a verified run. |

For run targets, the selected TOML file supplies only the inference policy. The
run supplies the model revision, model digest, text format, and adapter digest.

## One private prompt

Put prompt text in an ignored file:

```console
mkdir -p var/prompts
$EDITOR var/prompts/check.txt
just idiolect inference text --base var/prompts/check.txt
```

Use standard input when prompt text must not enter a process argument:

```console
just idiolect inference text --base
```

Use a run target:

```console
just idiolect inference text --base-of var/runs/RUN_ID var/prompts/check.txt
just idiolect inference text --run var/runs/RUN_ID var/prompts/check.txt
```

The command writes one JSON Lines result for each configured seed to standard
output. It does not create an inference artifact for a single prompt.

## Dataset inference

Generate a split with the configured base:

```console
just inference base var/data/DATASET_ID test
```

Generate the same split with a run's recorded base and adapter:

```console
just inference base-of var/runs/RUN_ID var/data/DATASET_ID test
just inference run var/runs/RUN_ID var/data/DATASET_ID test
```

Add an experiment-policy name as the final argument when its inference policy
must apply:

```console
just inference run var/runs/RUN_ID var/data/DATASET_ID test qwen3-8b-smoke
```

The recipes use `caffeinate -i`. Collection can run during inference.

## Prediction artifact

Dataset inference creates:

```text
var/inference/<inference-id>/
├── manifest.json
└── pred.jsonl
```

The recipe records the dataset digest, split, selected examples, target digests,
text format, inference policy, and MLX text-runtime fingerprint. Equal prompts
use equal derived random seeds across base and adapter targets.

Each prediction records the source index and digest, configured seed, derived
seed, generated text, finish reason, and token counts. It does not copy the
source prompt or expected completion.

Before reuse, the loader verifies the artifact identity, file digest, row order,
seeds, finish reasons, token limits, and recipe alignment. More than one valid
artifact for one recipe is an error because it indicates nondeterministic output
or an incomplete runtime fingerprint.

Do not publish prompts, predictions, manifests, models, or adapters.
