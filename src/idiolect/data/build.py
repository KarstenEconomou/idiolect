"""Define the contextual reply dataset port."""

from typing import Protocol

from idiolect.config import DataConfig
from idiolect.types import DatasetRef, PersonId


class Builder(Protocol):
    """Build one fixed dataset for one person."""

    def build(self, person_id: PersonId, name: str, config: DataConfig) -> DatasetRef:
        """Build and return one dataset."""
        ...
