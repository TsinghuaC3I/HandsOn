"""Shared task-development contract constants."""

from __future__ import annotations

REQUIRED_CHECKS = (
    "schema",
    "assets",
    "reachability",
    "contact",
    "reset",
    "success",
    "camera",
    "stability",
    "collection",
)

TASK_ID_PATTERN = r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"

TAXONOMY_FAMILIES = (
    "pick_place",
    "assembly",
    "tool_use",
    "articulation",
    "deformable",
    "long_horizon",
    "interactive_control",
)

HORIZONS = ("short", "medium", "long")
HANDEDNESS = ("unimanual", "bimanual", "either")
SOURCE_ORIGINS = ("original", "adapted", "handson")
ASSET_KINDS = ("mjcf", "urdf", "usd", "mesh", "texture", "config", "other")
