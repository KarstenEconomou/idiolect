"""Test the message input contracts."""

from typing import is_protocol

from idiolect.ingest import Parser, Source


def test_input_ports_are_protocols() -> None:
    """Check that input ports are protocols."""
    assert is_protocol(Source)
    assert is_protocol(Parser)
