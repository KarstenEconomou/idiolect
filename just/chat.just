set shell := ["sh", "-eu", "-c"]
set working-directory := ".."

# Open the searchable local chat chooser.
default:
    exec uv run --env-file .env --extra train --extra chat idiolect --config conf/idiolect.toml chat

# Open one verified run and dataset pair.
run run dataset:
    exec uv run --env-file .env --extra train --extra chat idiolect --config conf/idiolect.toml chat run "{{ run }}" "{{ dataset }}"

# Resume one immutable saved chat snapshot.
resume chat:
    exec uv run --env-file .env --extra train --extra chat idiolect --config conf/idiolect.toml chat resume "{{ chat }}"
