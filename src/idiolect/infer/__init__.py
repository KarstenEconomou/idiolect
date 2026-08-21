"""Define local inference contracts."""

from idiolect.infer.base import Backend, ModelTarget, Prediction, TargetMode
from idiolect.infer.local import (
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
