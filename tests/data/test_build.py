"""Test the dataset build contract."""

from typing import is_protocol

from idiolect.data import Builder


def test_builder_is_protocol() -> None:
    """Check that the builder is a protocol."""
    assert is_protocol(Builder)
