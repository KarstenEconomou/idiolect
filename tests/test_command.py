"""Test shared command-line process and path rules."""

from pathlib import Path

import idiolect.command
from idiolect.command import artifact_path, keep_awake


def test_artifact_path_resolves_only_content_ids() -> None:
    """Check that explicit paths cannot be mistaken for root-relative IDs."""
    digest = "a" * 64
    root = Path("var/eval")

    assert artifact_path(digest, root) == root / digest
    assert artifact_path(digest, root, "judgments") == root / "judgments" / digest
    assert artifact_path("short-name", root) == Path("short-name")
    assert artifact_path(Path("elsewhere") / digest, root) == Path("elsewhere") / digest


def test_keep_awake_holds_one_macos_assertion(monkeypatch) -> None:
    """Check the sleep assertion lifetime without running a system process."""
    events: list[str] = []
    started: list[tuple[object, object]] = []

    class FakeProcess:
        """Record process cleanup."""

        def poll(self):
            """Report one running process."""
            return

        def terminate(self) -> None:
            """Record termination."""
            events.append("terminate")

        def wait(self) -> None:
            """Record process completion."""
            events.append("wait")

    def start(command, **options):
        started.append((command, options))
        return FakeProcess()

    monkeypatch.setattr(idiolect.command.sys, "platform", "darwin")
    monkeypatch.setattr(idiolect.command.os, "getpid", lambda: 321)
    monkeypatch.setattr(idiolect.command.subprocess, "Popen", start)

    with keep_awake():
        events.append("operation")

    assert started[0][0] == ("caffeinate", "-i", "-w", "321")
    assert events == ["operation", "terminate", "wait"]
