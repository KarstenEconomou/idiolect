"""Test the text generation contract."""

from typing import is_protocol

from idiolect.infer import Generator


def test_generator_is_protocol() -> None:
    """Check that the generator is a protocol."""
    assert is_protocol(Generator)
