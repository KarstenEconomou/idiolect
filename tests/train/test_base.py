"""Test the model training contract."""

from typing import is_protocol

from idiolect.train import Trainer


def test_trainer_is_protocol() -> None:
    """Check that the trainer is a protocol."""
    assert is_protocol(Trainer)
