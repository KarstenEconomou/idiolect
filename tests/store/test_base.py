"""Test the data store contracts."""

from typing import is_protocol

from idiolect.store import DatasetStore, Repository, RunStore


def test_store_ports_are_protocols() -> None:
    """Check that store ports are protocols."""
    assert is_protocol(Repository)
    assert is_protocol(DatasetStore)
    assert is_protocol(RunStore)
