"""Define model training contracts."""

from idiolect.train.base import Trainer
from idiolect.train.mlx import LoadedRun, load_run

__all__ = ["LoadedRun", "Trainer", "load_run"]
