from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def asset_root() -> Path:
    checkout_assets = repo_root() / "assets"
    if checkout_assets.exists():
        return checkout_assets
    return Path(__file__).resolve().parent / "assets"


def default_artifact_root() -> Path:
    if Path("/work").exists():
        return Path("/work/rokae_wuji_ik_replay/artifacts")
    return repo_root() / "artifacts"


def default_wuji_description_root() -> Path:
    return default_artifact_root() / "sources" / "wuji-description"
