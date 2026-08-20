"""Define message input contracts."""

from idiolect.ingest.base import Parser, Source
from idiolect.ingest.harvest import HarvestResult, harvest

__all__ = ["HarvestResult", "Parser", "Source", "harvest"]
