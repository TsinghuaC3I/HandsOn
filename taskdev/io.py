"""YAML and filesystem helpers with conservative error handling."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any


class TaskDevError(Exception):
    """Expected task-development error suitable for a CLI response."""


def _yaml_module():
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise TaskDevError(
            "PyYAML is required; install it in the active CPU environment "
            "(for example: python -m pip install PyYAML)"
        ) from exc
    return yaml


def load_yaml(path: Path) -> Any:
    if not path.is_file():
        raise TaskDevError(f"YAML file does not exist: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TaskDevError(f"cannot read YAML file {path}: {exc}") from exc
    try:
        return _yaml_module().safe_load(text)
    except Exception as exc:
        raise TaskDevError(f"invalid YAML in {path}: {exc}") from exc


def dump_yaml(data: Any) -> str:
    return _yaml_module().safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TaskDevError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def resolve_inside(root: Path, candidate: str, *, label: str) -> Path:
    if not isinstance(candidate, str) or not candidate.strip():
        raise TaskDevError(f"{label} must be a non-empty path")
    raw = Path(candidate)
    if raw.is_absolute():
        raise TaskDevError(f"{label} must be repository-relative, got: {candidate}")
    root = root.resolve()
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise TaskDevError(f"{label} escapes repository root: {candidate}") from exc
    return resolved
