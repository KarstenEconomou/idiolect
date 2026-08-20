"""Define local inference contracts."""

from idiolect.infer.base import Backend, ModelTarget, Prediction, TargetMode
from idiolect.infer.local import (
    InferenceError,
    LocalInferencer,
    configured_target,
    recorded_target,
)

__all__ = [
    "Backend",
    "InferenceError",
    "LocalInferencer",
    "ModelTarget",
    "Prediction",
    "TargetMode",
    "configured_target",
    "recorded_target",
]
