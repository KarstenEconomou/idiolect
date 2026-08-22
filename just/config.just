set shell := ["sh", "-eu", "-c"]
set working-directory := ".."

# List the canonical and experiment configurations.
list:
    #!/bin/sh
    set -eu
    printf '%s\n' 'idiolect (default)'
    for path in conf/exp/*.toml; do
        [ -e "$path" ] || continue
        name=${path##*/}
        printf '%s\n' "${name%.toml}"
    done

# Create a complete experiment configuration.
[arg('name', pattern='[a-z0-9][a-z0-9-]*')]
[arg('source', pattern='[a-z0-9][a-z0-9-]*')]
new name source="idiolect":
    #!/bin/sh
    set -eu
    if [ "{{ name }}" = idiolect ]; then
        printf '%s\n' 'error: idiolect is the canonical configuration' >&2
        exit 2
    fi
    if [ "{{ source }}" = idiolect ]; then
        source=conf/idiolect.toml
    else
        source=conf/exp/{{ source }}.toml
    fi
    target=conf/exp/{{ name }}.toml
    if [ ! -f "$source" ]; then
        printf 'error: configuration does not exist: %s\n' "$source" >&2
        exit 2
    fi
    if [ -e "$target" ]; then
        printf 'error: configuration already exists: %s\n' "$target" >&2
        exit 2
    fi
    install -m 644 "$source" "$target"
    printf 'created=%s\n' "$target"

# Train one experiment configuration and keep the Mac awake.
[arg('name', pattern='[a-z0-9][a-z0-9-]*')]
train name dataset:
    #!/bin/sh
    set -eu
    if [ "{{ name }}" = idiolect ]; then
        config=conf/idiolect.toml
    else
        config=conf/exp/{{ name }}.toml
    fi
    if [ ! -f "$config" ]; then
        printf 'error: configuration does not exist: %s\n' "$config" >&2
        exit 2
    fi
    exec caffeinate -i uv run --env-file .env idiolect --config "$config" train "{{ dataset }}"
