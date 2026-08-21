set shell := ["sh", "-eu", "-c"]
set working-directory := ".."
set positional-arguments

# List evaluation commands.
default:
    @just --justfile just/eval.just --list

# Compare one complete training policy with its recorded base.
policy dataset *runs:
    caffeinate -i uv run idiolect eval policy "$@"

# Complete one private familiar-rater session.
rate evaluation rater:
    uv run idiolect eval rate "$1" --rater "$2"

# Summarize one or more familiar-rater judgment artifacts.
panel evaluation *judgments:
    uv run idiolect eval panel "$@"
