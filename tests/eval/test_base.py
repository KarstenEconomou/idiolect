"""Test the model test contract."""

from typing import is_protocol

from idiolect.eval import Evaluator


def test_evaluator_is_protocol() -> None:
    """Check that the evaluator is a protocol."""
    assert is_protocol(Evaluator)
