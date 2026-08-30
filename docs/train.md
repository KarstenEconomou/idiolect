# Adapter Training

## Purpose

Training reads one verified dataset and creates one adapter for each configured
seed. It does not read Signal or DuckDB.

The objective is completion negative log-likelihood. Prompt masking limits loss
to the target response. One completion is one response episode, including its
internal `[new message]` boundaries.

Use a complete TOML policy for every experiment. Python code does not supply a
missing model, seed, optimizer, format, path, or reporting choice.

## Install

Install the optional MLX-LM packages:

```console
just setup-train
```

A hub model can download during the first run. Keep a required `HF_TOKEN` in
`.env`. Do not put it in TOML.

## Create an experiment policy

List available policies:

```console
idiolect config list
```

Copy the canonical policy:

```console
idiolect config new EXPERIMENT_NAME
```

Copy another experiment policy:

```console
idiolect config new NEW_NAME --from SOURCE_NAME
```

The command creates `conf/exp/NEW_NAME.toml`. Each file is complete. The
configuration system does not merge files.

The default names `default` and `idiolect` select `conf/idiolect.toml`. Other
bare names select `conf/exp/NAME.toml`. `-c`, `--config`, and
`IDIOLECT_CONFIG` use these rules. An explicit path remains unchanged.

Use a name that identifies the model and main policy change. Commit the policy
before its first run. Do not change it after a run uses it. Create a new policy
for the next change.

Review these policy groups:

| Group | Decision |
|---|---|
| Model | Name, source, revision, cache, and remote-code policy. |
| Run | Seeds and either epochs or iterations. |
| Optimization | Optimizer, learning rate, batch behavior, and schedule. |
| Sequence | Maximum length and prompt masking. |
| Format | Roles, system text, prefixes, suffixes, and assistant prefill. |
| LoRA | Target keys, rank, scale, and dropout. |
| Reporting | Local project name and approved external service. |

Set exactly one of `epochs` or `iterations`. Keep `mask_prompt = true`. Keep
`trust_remote_code = false` unless the model code has been audited. An empty
`report_to` value disables external reporting.

## Preflight validation

Before training, Idiolect performs these checks:

1. Verify the complete training policy.
2. Resolve the fixed model revision and verify its digest.
3. Load only the tokenizer.
4. Verify the dataset identity and files.
5. Apply the model format to every available split.
6. Verify a stable prompt-to-completion token boundary.
7. Reject an example that has no supervised token or exceeds
   `max_seq_length`.
8. Verify the required split and batch counts.

Idiolect validates the expected token stream. MLX-LM remains the authority for
training tokenization and prompt masking inside the training command.

The model-specific prefixes and suffixes change only the private run copy. They
do not change the canonical dataset.

## Run

Build a verified dataset first. Stop collection only during the dataset build.

Run the canonical policy:

```console
idiolect train var/data/DATASET_ID
```

Run a named experiment policy:

```console
idiolect -c qwen3-8b-smoke train var/data/DATASET_ID
```

On macOS, training holds an idle-sleep assertion for its duration. Connect the
Mac to power. Collection can run during training because training reads
immutable files.

## Artifact

Each seed creates one content-addressed run:

```text
var/runs/<run-id>/
├── adapter/
│   ├── adapter_config.json
│   └── adapters.safetensors
├── data/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
├── manifest.json
├── request.json
└── train.log
```

The run ID commits to the dataset, model digest, one seed, and complete policy.
An equal request returns the existing verified run. The loader verifies every
recorded file before inference, chat, or evaluation uses the run.

Do not edit a run. Do not publish the adapter, model-specific data, request,
manifest, or log.

## Experiment sequence

1. Use a small smoke policy to verify the operation.
2. Keep the validation set and inference policy fixed.
3. Train all configured seeds for a candidate policy.
4. Compare the complete seed set with its recorded base.
5. Use all evaluation pillars before model selection.

A small successful run verifies the system. It does not establish model
quality. Use [docs/eval.md](eval.md) for selection.
