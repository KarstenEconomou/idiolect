set shell := ["sh", "-eu", "-c"]
set working-directory := ".."

# List inference commands.
default:
    @just --justfile just/inference.just --list

# Generate one dataset split with a configured base model.
[arg('config', pattern='[a-z0-9][a-z0-9-]*')]
[arg('split', pattern='train|valid|test')]
base dataset split config="idiolect":
    #!/bin/sh
    set -eu
    config_path="{{ if config == "idiolect" { "conf/idiolect.toml" } else { "conf/exp/" + config + ".toml" } }}"
    if [ ! -f "$config_path" ]; then
        printf 'error: configuration does not exist: %s\n' "$config_path" >&2
        exit 2
    fi
    exec caffeinate -i uv run --env-file .env idiolect --config "$config_path" inference data --base "{{ dataset }}" --split "{{ split }}"

# Generate one dataset split with the base model from a run.
[arg('config', pattern='[a-z0-9][a-z0-9-]*')]
[arg('split', pattern='train|valid|test')]
base-of run dataset split config="idiolect":
    #!/bin/sh
    set -eu
    config_path="{{ if config == "idiolect" { "conf/idiolect.toml" } else { "conf/exp/" + config + ".toml" } }}"
    if [ ! -f "$config_path" ]; then
        printf 'error: configuration does not exist: %s\n' "$config_path" >&2
        exit 2
    fi
    exec caffeinate -i uv run --env-file .env idiolect --config "$config_path" inference data --base-of "{{ run }}" "{{ dataset }}" --split "{{ split }}"

# Generate one dataset split with a run adapter.
[arg('config', pattern='[a-z0-9][a-z0-9-]*')]
[arg('split', pattern='train|valid|test')]
run run dataset split config="idiolect":
    #!/bin/sh
    set -eu
    config_path="{{ if config == "idiolect" { "conf/idiolect.toml" } else { "conf/exp/" + config + ".toml" } }}"
    if [ ! -f "$config_path" ]; then
        printf 'error: configuration does not exist: %s\n' "$config_path" >&2
        exit 2
    fi
    exec caffeinate -i uv run --env-file .env idiolect --config "$config_path" inference data --run "{{ run }}" "{{ dataset }}" --split "{{ split }}"
