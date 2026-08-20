"""Define data store contracts."""

from idiolect.store.base import DatasetStore, Repository, RunStore
from idiolect.store.duck import DuckRepository

__all__ = ["DatasetStore", "DuckRepository", "Repository", "RunStore"]
