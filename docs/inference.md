# Local Inference

## Purpose

Inference generates text from one fixed model target. It reads explicit prompt text or one verified dataset split. It does not read Signal or DuckDB.

The implementation uses MLX-LM. Install the local model packages:

```console
just setup-train
```

## Configuration

Each complete TOML configuration contains one `[inference]` table. The table selects the output path, backend, seeds, example limit, token limits, and sampling values.

The canonical Qwen policy uses non-thinking generation. The training format adds `/no_think` to the prompt and uses an empty thinking block as the assistant prefill. Inference uses the same format.

`max_examples = 0` selects all examples. A positive value selects examples by their stable hash. It does not select the first messages in the split.

`max_prompt_tokens` is a strict input limit after the tokenizer applies the chat template. The operation stops if an input exceeds the limit. It does not remove context or reduce the output limit.

The generation-only part of `[inference]` is also the chat generation policy. It
contains the backend, prompt and output limits, temperature, probability
filters, and repetition settings. Batch-only output, seed lists, and example
selection stay specific to inference artifacts. Interactive chat applies its
recorded message window and then removes oldest whole messages until the
templated prompt fits `max_prompt_tokens`.

## Targets

Use one of these targets:

- `--base`: Use the model and text format in the selected TOML file.
- `--base-of RUN`: Use the exact base model and text format in a verified training run.
- `--run RUN`: Use the same frozen base with the verified run adapter.

The run reader verifies the run manifest, every run file, the adapter, and the resolved model digest. The selected TOML file supplies only the inference policy for run targets.

## One Prompt

Put private prompt text in an ignored file:

```console
mkdir -p var/prompts
$EDITOR var/prompts/check.txt
just idiolect inference text --base var/prompts/check.txt
```

Use standard input for text that must not enter a process argument:

```console
just idiolect inference text --base
```

Enter the prompt and send end-of-file. The command writes one JSON Lines record for each configured seed. It does not store the prompt or result.

Use a run target:

```console
just idiolect inference text --base-of var/runs/RUN_ID var/prompts/check.txt
just idiolect inference text --run var/runs/RUN_ID var/prompts/check.txt
```

## Dataset Batch

Generate the same verified split with the canonical base and one adapter:

```console
just inference base var/data/DATASET_ID test
just inference run var/runs/RUN_ID var/data/DATASET_ID test
```

Generate the exact base recorded by the run:

```console
just inference base-of var/runs/RUN_ID var/data/DATASET_ID test
```

Add an experiment configuration name as the final argument:

```console
just inference run var/runs/RUN_ID var/data/DATASET_ID test qwen3-8b-smoke
```

The recipes use `caffeinate -i`. Connect the Mac to power. Inference does not use `launchd`.

## Output

Each batch has this private structure:

```text
var/inference/<inference-id>/
├── manifest.json
└── pred.jsonl
```

The inference ID commits to the prediction-file digest and its counts. The recorded recipe includes the dataset digest, split, selected example IDs and source indexes, target digests, frozen text format, inference policy, and complete MLX text runtime fingerprint. The fingerprint covers MLX-LM, MLX, Transformers, tokenizers, Jinja, Python, and the operating-system platform.

Before generation, inference searches for a verified artifact with an equal recipe. It returns that artifact when exactly one match exists. Concurrent equal operations that produce equal content return the same artifact. More than one verified result for one recipe is an error because it indicates nondeterministic generation or an incomplete runtime fingerprint.

Each prediction contains the source index and digest, configured seed, derived MLX seed, generated text, finish reason, and token counts. The artifact reader verifies the complete prediction schema, order, seeds, finish reasons, token limits, counts, and recipe alignment. It does not contain the source prompt or expected completion. The manifest refers to the immutable dataset for these values.

The MLX boundary uses the same stream for batch and interactive generation.
Batch inference collects the text deltas and keeps its existing prediction
schema. The stream additionally exposes measured prompt throughput, generation
throughput, and peak memory for chat telemetry.

The derived seed is a hash of the configured seed and example ID. Equal examples use equal random streams across base and adapter targets. Input order does not change the seed.

The directory mode is `0700`. The file mode is `0600`. Do not publish a prompt, prediction, manifest, model, or adapter.
