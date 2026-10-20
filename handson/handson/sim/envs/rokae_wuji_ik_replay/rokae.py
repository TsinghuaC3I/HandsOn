from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .ik import DlsIkSolver
from .paths import asset_root
from .urdf import load_serial_chain_model

AR5_5_08 = "ar5_5_08"
LEFT_PREFIX = "AR5-5_08L-W4C6C11"
RIGHT_PREFIX = "AR5-5_08R-W4C1C9-ZY2"
DOF = 7
H = np.pi / 2.0
BASE_Y_SEPARATION = 0.130
BASE_Y_HALF_SEPARATION = BASE_Y_SEPARATION / 2.0
INITIAL_Q = {
    "left": np.array([H, H, -H, H, H, 0.0, 0.0]),
    "right": np.array([-H, H, H, H, -H, 0.0, 0.0]),
}
BASE_IN_WORLD = {
    "left": {"xyz": (0.0, BASE_Y_HALF_SEPARATION, 0.63), "rpy": (-H, 0.0, 0.0)},
    "right": {"xyz": (0.0, -BASE_Y_HALF_SEPARATION, 0.63), "rpy": (H, 0.0, 0.0)},
}
OPERATIONAL_LIMITS_DEG = np.array(
    [
        [-175.0, 175.0],
        [-117.0, 117.0],
        [-175.0, 175.0],
        [0.0, 143.0],
        [-175.0, 175.0],
        [-53.0, 53.0],
        [-53.0, 53.0],
    ]
)
J4_INDEX = 3
WRIST_INDICES = np.array([5, 6], dtype=np.int64)


def prefix(side: str) -> str:
    _validate_side(side)
    return LEFT_PREFIX if side == "left" else RIGHT_PREFIX


def end_link(side: str) -> str:
    return f"{prefix(side)}_flan_link"


def urdf_path(side: str) -> Path:
    path = asset_root() / "rokae" / "models" / f"{prefix(side)}.urdf"
    if not path.is_file():
        raise FileNotFoundError(f"Bundled ROKAE description is missing: {path}. Reinstall the complete HandsOn package.")
    return path


@cache
def load_model(side: str):
    return load_serial_chain_model(
        urdf_path(side),
        joint_pattern=rf"{prefix(side)}_joint_\d+",
        joint_count=DOF,
        end_link=end_link(side),
    )


def initial_q(side: str) -> np.ndarray:
    _validate_side(side)
    return INITIAL_Q[side].copy()


def base_in_world_matrix(side: str, *, y_separation: float | None = None) -> np.ndarray:
    _validate_side(side)
    spec = BASE_IN_WORLD[side]
    xyz = np.asarray(spec["xyz"], dtype=float).copy()
    if y_separation is not None:
        if y_separation <= 0:
            raise ValueError("y_separation must be > 0")
        sign = 1.0 if side == "left" else -1.0
        xyz[1] = sign * float(y_separation) * 0.5
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", spec["rpy"]).as_matrix()
    transform[:3, 3] = xyz
    return transform


def make_solver(side: str, *, position_only: bool = False) -> DlsIkSolver:
    model = load_model(side)
    lower = np.deg2rad(OPERATIONAL_LIMITS_DEG[:, 0])
    upper = np.deg2rad(OPERATIONAL_LIMITS_DEG[:, 1])
    posture_target = np.zeros(DOF)
    posture_weight = np.zeros(DOF)
    posture_range = np.maximum(upper - lower, 1e-9)
    posture_target[J4_INDEX] = max(0.0, float(model.mid[J4_INDEX]))
    posture_weight[J4_INDEX] = 0.015
    posture_weight[WRIST_INDICES] = 0.08
    return DlsIkSolver(
        model,
        lower=lower,
        upper=upper,
        orientation_weight=0.0 if position_only else 0.7,
        posture_target=posture_target,
        posture_weight=posture_weight,
        posture_range=posture_range,
    )


def make_seeds(side: str, current_q: np.ndarray) -> list[np.ndarray]:
    model = load_model(side)
    current = model.clip(current_q)
    elbow = current.copy()
    elbow[J4_INDEX] = max(elbow[J4_INDEX], model.mid[J4_INDEX], 0.0)
    return [elbow, model.mid.copy(), model.clip(np.zeros(DOF))]


def wrist_limit_ratio(q: np.ndarray) -> float:
    lower = np.deg2rad(OPERATIONAL_LIMITS_DEG[:, 0])
    upper = np.deg2rad(OPERATIONAL_LIMITS_DEG[:, 1])
    half = (upper[WRIST_INDICES] - lower[WRIST_INDICES]) * 0.5
    return float(np.max(np.abs(np.asarray(q)[WRIST_INDICES]) / half))


def arm_joint_names(side: str) -> list[str]:
    _validate_side(side)
    return [f"{side}_joint{i}" for i in range(1, 8)]


def _validate_side(side: str) -> None:
    if side not in ("left", "right"):
        raise ValueError(f"side must be left/right, got {side!r}")
