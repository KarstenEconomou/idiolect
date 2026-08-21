"""Define model test contracts."""

from idiolect.eval.base import CompletionScore, Evaluator, ScoreBackend, ScoreSession
from idiolect.eval.local import EvaluationError, LocalEvaluator, load_evaluation
from idiolect.eval.mlx import EvalBackendError, MlxScoreBackend
from idiolect.eval.panel import collect_judgments, create_panel

__all__ = [
    "CompletionScore",
    "EvalBackendError",
    "EvaluationError",
    "Evaluator",
    "LocalEvaluator",
    "MlxScoreBackend",
    "ScoreBackend",
    "ScoreSession",
    "collect_judgments",
    "create_panel",
    "load_evaluation",
]
