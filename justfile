set shell := ["sh", "-eu", "-c"]
set positional-arguments

# List available commands.
default:
    @just --list

# Install the core project environment.
setup:
    uv sync

# Install the local training environment.
setup-train:
    uv sync --extra train

# Install local model and terminal chat packages.
setup-chat:
    uv sync --extra train --extra chat

# Build the source archive and wheel.
build:
    uv build

# Run the complete test suite.
test:
    uv run pytest

# Check Python style and common errors.
lint:
    just --fmt --check
    uv run ruff check .

# Type-check the source and tests.
typecheck:
    uv run ty check

# Run every required quality check.
check: lint typecheck test
