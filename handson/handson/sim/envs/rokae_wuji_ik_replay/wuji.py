from __future__ import annotations

from pathlib import Path

import numpy as np

WUJI_DOF = 20
LIMITS: tuple[tuple[float, float], ...] = (
    (0.0475, 1.6033), (-0.1387, 0.9324), (-0.4642, 1.5623), (-0.4699, 1.5568),
    (-0.1585, 1.5604), (-0.3700, 0.3700), (-0.4777, 1.5485), (-0.4683, 1.5753),
    (-0.1644, 1.5516), (-0.3700, 0.3700), (-0.4739, 1.5512), (-0.4684, 1.5745),
    (-0.1554, 1.5585), (-0.3700, 0.3700), (-0.4765, 1.5487), (-0.4777, 1.5634),
    (-0.1626, 1.5585), (-0.3700, 0.3700), (-0.4768, 1.5490), (-0.4683, 1.5735),
)


def joint_names(side: str) -> list[str]:
    if side not in ("left", "right"):
        raise ValueError(f"side must be left/right, got {side!r}")
    return [f"{side}_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)]


def joint_limits() -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([lo for lo, _ in LIMITS], dtype=float)
    upper = np.array([hi for _, hi in LIMITS], dtype=float)
    return lower, upper


def neutral_q() -> np.ndarray:
    lower, _ = joint_limits()
    return lower


def urdf_path(root: str | Path, side: str) -> Path:
    return Path(root) / "hand" / "body" / "urdf" / f"{side}.urdf"
