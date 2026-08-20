# Adapter Training

## Purpose

Training is a local batch operation. It reads one immutable dataset and writes one adapter for each configured seed. It does not read Signal or DuckDB.

Each tracked TOML configuration contains one complete experiment policy. Python code does not select a model, seed, data format, optimizer value, adapter target, path, or external report service.

## Install

Install the optional local training packages:

```console
just sync-train
```

The canonical configuration uses `mlx-community/Qwen3-14B-4bit` at a fixed repository revision. The smoke configuration uses `mlx-community/Qwen3-8B-4bit`. The first run for each model downloads it to `train.model_cache`.

Do not put a model hub token in TOML. Put `HF_TOKEN` in `.env` only if the model requires authentication.

## Configure

`conf/idiolect.toml` is the canonical configuration. Complete experiment configurations are in `conf/exp/`. Git tracks both locations.

List the available configurations:

```console
just config list
```

Create a configuration from the canonical file:

```console
just config new qwen3-14b-r16
```

Create a configuration from another experiment:

```console
just config new qwen3-8b-r16 qwen3-8b-smoke
```

The command creates `conf/exp/<name>.toml` and stops if the target exists. Names use lowercase letters, numbers, and hyphens. Edit the new file directly. Each file is complete. The configuration system does not merge files or apply experiment overrides.

Use a name that identifies the model and the main changed dimensions. Do not use a sequence number as the only name. Commit a configuration before its training run. Keep a used configuration unchanged. Create another file for another policy.

The file name is a human label. The content-addressed run ID is the technical identity. It includes the complete resolved training policy, dataset digest, model digest, and seed.

The principal values are:

- `base_model`, `model_source`, and `model_revision`: Select one fixed model snapshot.
- `model_cache` and `output`: Select private model and run paths.
- `command`: Select the installed MLX-LM command.
- `seeds`: Select each independent run. The canonical configuration runs three adapters in sequence.
- `epochs` or `iterations`: Select one run limit. Do not set both values.
- `batch_size` and `grad_accumulation_steps`: Set the physical and effective batch operation.
- `train.data`: Set the private model-specific copy format.
- `train.lora`: Set the adapter layer keys, rank, scale, and dropout.
- `report_to`: Select an external report service. An empty value disables all report services.

The canonical configuration uses Qwen non-thinking text. It adds `/no_think` to each prompt and an empty thinking block before each completion. This change occurs only in the private run copy. The immutable canonical dataset does not change.

`trust_remote_code = false` rejects a model configuration that declares an `auto_map` code loader. Keep this value false unless you audit and accept the model code.

## Run

Build and note one dataset path:

```console
just data build Karsten
```

Stop collection only during this dataset build. Start collection after the dataset is complete. Training reads only fixed files, so collection can run during training.

Run the canonical configuration:

```console
just train var/data/DATASET_ID
```

Run a named experiment configuration:

```console
just config train qwen3-8b-smoke var/data/DATASET_ID
```

Connect the Mac to power. The recipe uses `caffeinate -i` to prevent idle sleep for the full operation. Training does not use `launchd`.

## Output

Each seed has one content-addressed path:

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

The run ID includes the dataset ID, dataset digest, model digest, model revision, seed, and complete training policy. A repeated equal run returns the existing adapter. The loader checks every run file against the manifest and rejects a changed file.

All output is private. Git ignores `var/`, model files, adapters, checkpoints, and report directories.

## Experiment sequence

Use a small dataset only to verify the operation. Do not interpret style quality from a few messages.

1. Record output from the unmodified model on fixed test prompts.
2. Run a short 8B smoke experiment with a separate TOML policy.
3. Run the configured 14B experiment when the corpus is sufficient.
4. Compare test loss, blind human preference, style statistics, mention behavior, reply behavior, and training-text overlap.
5. Do not change the test set or use it to select a checkpoint.

Do not publish a dataset, adapter, generated message, run log, or manifest. An adapter can retain private training information.

Use the [local inference procedure](infer.md) to generate paired base and adapter predictions. Use `--base-of` when the baseline must use the model snapshot and text format recorded by one run.
