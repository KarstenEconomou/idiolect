"""Resolve and verify local model snapshots."""

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path


class ModelError(RuntimeError):
    """Report an invalid or unavailable model."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Identify one fixed model snapshot."""

    name: str
    source: str
    revision: str
    cache: Path | None
    trust_remote_code: bool


def mlx_runtime_fingerprint() -> str:
    """Return the complete MLX text runtime fingerprint."""
    packages = ("mlx-lm", "mlx", "transformers", "tokenizers", "jinja2")
    values = [f"{name}={version(name)}" for name in packages]
    values.extend(
        (
            f"python={platform.python_version()}",
            f"implementation={sys.implementation.name}",
            f"system={platform.system()}",
            f"release={platform.release()}",
            f"machine={platform.machine()}",
        )
    )
    return ";".join(values)


def resolve_model(spec: ModelSpec) -> Path:
    """Resolve one model snapshot to a local directory."""
    if spec.source == "path":
        return Path(spec.name).expanduser().resolve()
    if spec.source != "hub":
        raise ModelError("Model source must be hub or path")
    if not spec.revision:
        raise ModelError("Hub models require a fixed revision")
    if spec.cache is None:
        raise ModelError("Hub models require a model cache")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ModelError(
            "Model packages are not installed. Run: uv sync --extra train"
        ) from error
    try:
        path = snapshot_download(
            repo_id=spec.name,
            revision=spec.revision,
            cache_dir=spec.cache.expanduser().resolve(),
        )
    except Exception as error:
        detail = str(error).strip() or type(error).__name__
        raise ModelError(f"Cannot resolve model {spec.name}: {detail}") from error
    return Path(path)


def verify_model(path: Path, spec: ModelSpec, expected_digest: str | None = None) -> str:
    """Verify one local model and return its directory digest."""
    if not path.is_dir():
        raise ModelError(f"Resolved model path does not exist: {path}")
    if not spec.trust_remote_code:
        _reject_remote_code(path)
    digest = directory_digest(path)
    if expected_digest is not None and digest != expected_digest:
        raise ModelError(f"Resolved model does not match its recorded digest: {path}")
    return digest


def directory_digest(root: Path) -> str:
    """Return one digest for all files in a directory."""
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _reject_remote_code(model_path: Path) -> None:
    for name in ("config.json", "tokenizer_config.json", "processor_config.json"):
        path = model_path / name
        if not path.exists():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModelError(f"Cannot inspect model configuration: {path}") from error
        if isinstance(value, dict) and value.get("auto_map"):
            raise ModelError(
                f"Model requires remote code but trust_remote_code is false: {path}"
            )
