# macOS Launch Agent

## Purpose

`launchd` is the macOS service manager. A user LaunchAgent starts after user login. It can restart a process after a failure.

The Idiolect agent has the label `com.idiolect.collect`. It runs this operation:

```console
uv run idiolect signal collect --follow
```

The agent sources `.env`, sets the repository as its working directory, and uses explicit command paths. It writes logs to `var/log/`.

## Agent fields

| Field | Function |
|---|---|
| `ProgramArguments` | Source `.env` and replace the shell with Idiolect. |
| `WorkingDirectory` | Make relative configuration and data paths stable. |
| `EnvironmentVariables` | Add the package manager path to `PATH`. |
| `RunAtLoad` | Start the collector after the agent loads. |
| `KeepAlive.SuccessfulExit=false` | Restart the collector after a nonzero exit. |
| `ThrottleInterval=30` | Wait before a restart loop. |
| `Umask=63` | Create private files with mode `0600` or `0700`. |
| `StandardOutPath` | Store normal output in `var/log/collect.out.log`. |
| `StandardErrorPath` | Store errors in `var/log/collect.err.log`. |

## Create the agent

Find the command paths:

```console
command -v uv
command -v signal-cli
pwd
```

Copy [com.idiolect.collect.plist.example](com.idiolect.collect.plist.example) to the ignored runtime directory:

```console
mkdir -p var/launchd var/log
cp docs/com.idiolect.collect.plist.example var/launchd/com.idiolect.collect.plist
```

Replace each `REPO_PATH` and `UV_PATH` value with an absolute path. Add the directory that contains `signal-cli` to `PATH`. Do not add a credential to the plist.

Validate the file:

```console
plutil -lint var/launchd/com.idiolect.collect.plist
```

Install and load it:

```console
mkdir -p ~/Library/LaunchAgents
install -m 600 var/launchd/com.idiolect.collect.plist ~/Library/LaunchAgents/com.idiolect.collect.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.idiolect.collect.plist
```

Do not run a manual `collect --follow` process at the same time.

## Operate the agent

Show the state:

```console
launchctl print gui/$(id -u)/com.idiolect.collect
```

The expected state is `running`. `runs=1` and `last exit code = (never exited)` show that the first process still runs.

Restart the agent:

```console
launchctl kickstart -k gui/$(id -u)/com.idiolect.collect
```

Stop and remove the agent from the current login session:

```console
launchctl bootout gui/$(id -u)/com.idiolect.collect
```

Load it again with the `bootstrap` command. Remove the installed plist only after the `bootout` command.

Watch the logs:

```console
tail -f var/log/collect.err.log
tail -f var/log/collect.out.log
```

The normal output log can remain empty while `--follow` runs. The command writes its count summary when it exits.

## Power state

The agent runs only while the Mac is on, awake, and logged in. Use this command in a separate terminal to prevent idle system sleep:

```console
caffeinate -i
```

The `caffeinate` command must continue to run. A closed MacBook lid can still cause sleep unless macOS supports the current closed-display setup.

## Path changes

The plist contains absolute paths. If the repository, `uv`, or `signal-cli` moves, update the private plist. Run `bootout`, install the new file, and run `bootstrap` again.

The tracked example is not the installed agent. The installed file is `~/Library/LaunchAgents/com.idiolect.collect.plist`.
