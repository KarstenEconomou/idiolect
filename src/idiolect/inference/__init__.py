"""Define local inference contracts."""

from idiolect.inference.base import Backend, ModelTarget, Prediction, TargetMode
from idiolect.inference.local import (
    InferenceError,
    LocalInferencer,
    RecordedTargetResolver,
    configured_target,
    load_inference,
    recorded_target,
)

__all__ = [
    "Backend",
    "InferenceError",
    "LocalInferencer",
    "ModelTarget",
    "Prediction",
    "RecordedTargetResolver",
    "TargetMode",
    "configured_target",
    "load_inference",
    "recorded_target",
]
