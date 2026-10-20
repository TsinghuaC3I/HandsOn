"""Portable, explicit paths and strict prediction validation."""
import hashlib
import json
from pathlib import Path
import sys

from ..interfaces import VIEWS, validate_observation

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def asset(model, name):
    return ROOT / "assets" / model / name


def vendor(model):
    """Upstream module names are isolated by the one-model-per-process server."""
    paths = [ROOT / "vendor" / model, ROOT / "vendor" / "openpi_client"]
    if model == "motus":
        paths.insert(1, ROOT / "vendor" / model / "bak")
    if model == "lingbot":
        paths.insert(1, ROOT / "vendor" / "lingbot_lerobot")
    sys.path[:0] = [str(p) for p in paths]


def path_option(options, key, default=None):
    value = getattr(options, key, None) or default
    require(value is not None, f"--{key.replace('_', '-')} is required for {options.model}")
    path = Path(value).expanduser().resolve()
    require(path.exists(), f"Missing {key}: {path}")
    return path


def load_policy(options):
    from .profiles import PROFILES
    return PROFILES[options.model].load(options)
