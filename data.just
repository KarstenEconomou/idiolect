set shell := ["sh", "-eu", "-c"]

# List normalized message authors.
people:
    uv run idiolect data people

# Build an immutable dataset for the local Signal account.
build name:
    uv run idiolect data build --self --name "{{name}}"
