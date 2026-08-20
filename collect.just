set shell := ["sh", "-eu", "-c"]

# Show the collector service state.
status:
    launchctl print gui/$(id -u)/com.idiolect.collect

# Load the installed collector service.
start:
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.idiolect.collect.plist

# Stop the collector service.
stop:
    launchctl bootout gui/$(id -u)/com.idiolect.collect
