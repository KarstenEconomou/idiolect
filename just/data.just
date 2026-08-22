set shell := ["sh", "-eu", "-c"]
set working-directory := ".."

# List data commands.
default:
    @just --justfile just/data.just --list

# Build an immutable dataset for the local Signal account.
build name:
    uv run --env-file .env idiolect data build --self --name "{{ name }}"
