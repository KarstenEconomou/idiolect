set shell := ["sh", "-eu", "-c"]

# Operate the macOS collector.
mod collect 'just/collect.just'

# Inspect data and build datasets.
mod data 'just/data.just'

# Manage committed experiment configurations.
mod config 'just/config.just'

# Generate text with local models and adapters.
mod infer 'just/infer.just'

# Evaluate trained model policies.
mod eval 'just/eval.just'

# List available commands.
default:
    @just --list

# Install the core project environment.
sync:
    uv sync

# Install the local training environment.
sync-train:
    uv sync --extra train

# Run the complete test suite.
test:
    uv run pytest

# Check Python style and common errors.
lint:
    @for file in justfile just/*.just; do just --justfile "$file" --fmt --check; done
    uv run ruff check .

# Type-check the source and tests.
typecheck:
    uv run ty check

# Run every required quality check.
check: lint typecheck test

# Train the canonical configuration and keep the Mac awake.
train dataset:
    caffeinate -i uv run idiolect --config conf/idiolect.toml train "{{ dataset }}"
