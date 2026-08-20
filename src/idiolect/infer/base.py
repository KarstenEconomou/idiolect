"""Define the text generation port."""

from collections.abc import Sequence
from typing import Protocol

from idiolect.config import InferConfig
from idiolect.types import Message, RunRef


class Generator(Protocol):
    """Generate text with one model adapter."""

    def generate(
        self,
        run: RunRef,
        context: Sequence[Message],
        config: InferConfig,
    ) -> str:
        """Return one reply for the message context."""
        ...
