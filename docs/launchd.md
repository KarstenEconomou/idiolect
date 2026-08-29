# macOS LaunchAgent

## Purpose

The optional user LaunchAgent runs continuous collection after login. It uses
the label `com.idiolect.collect`. It runs only while the Mac is on, awake, and
logged in.

The tracked plist is a template. The installed plist is private because it
contains absolute local paths.

## Prepare the plist

Find the required paths:

```console
command -v uv
command -v signal-cli
pwd
```

Copy the template to the ignored runtime directory:

```console
mkdir -p var/launchd var/log
cp docs/com.idiolect.collect.plist.example var/launchd/com.idiolect.collect.plist
```

Edit the copy. Replace each `REPO_PATH` and `UV_PATH`. Add the directory that
contains `signal-cli` to `PATH`. Do not put a credential in the plist.

The agent uses the repository as its working directory. It loads `.env`, uses
`conf/idiolect.toml`, and writes output under `var/log/`.

Validate the file:

```console
plutil -lint var/launchd/com.idiolect.collect.plist
```

## Install and start

Install the private file and load the agent:

```console
mkdir -p ~/Library/LaunchAgents
install -m 600 var/launchd/com.idiolect.collect.plist ~/Library/LaunchAgents/com.idiolect.collect.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.idiolect.collect.plist
```

Do not run a manual `collect --follow` process at the same time.

## Operate

Show the current state:

```console
just collect status
```

Restart the process without unloading the agent:

```console
launchctl kickstart -k gui/$(id -u)/com.idiolect.collect
```

Stop and unload the agent:

```console
just collect stop
```

Load the installed agent again:

```console
just collect start
```

Watch the private logs:

```console
tail -f var/log/collect.err.log
tail -f var/log/collect.out.log
```

Normal output can stay empty while continuous collection runs. The collector
writes its summary when it exits.

## Agent policy

The template applies these controls:

| Control | Effect |
|---|---|
| `RunAtLoad` | Start after the user agent loads. |
| `KeepAlive.SuccessfulExit=false` | Restart after a failure. |
| `ThrottleInterval=30` | Limit restart loops. |
| `Umask=63` | Create private files and directories. |
| `StandardOutPath` | Write normal output to `var/log/collect.out.log`. |
| `StandardErrorPath` | Write errors to `var/log/collect.err.log`. |

Use `caffeinate -i` in another terminal when the Mac must not enter idle sleep.
A closed laptop lid can still cause sleep.

## Path changes

The installed plist contains absolute paths. Update it when the repository,
`uv`, or `signal-cli` moves. Stop the agent. Install the updated file. Then
start the agent again.
