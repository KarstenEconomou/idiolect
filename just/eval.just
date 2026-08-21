set shell := ["sh", "-eu", "-c"]
set working-directory := ".."
set positional-arguments

# List evaluation commands.
default:
    @just --justfile just/eval.just --list

# Compare one complete training policy with its recorded base.
policy dataset *runs:
    caffeinate -i uv run idiolect eval policy "$@"
