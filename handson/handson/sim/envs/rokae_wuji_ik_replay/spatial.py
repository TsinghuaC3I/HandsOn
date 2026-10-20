from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def pose_from_xyz_rpy(xyz, rpy) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", np.asarray(rpy, dtype=float)).as_matrix()
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def transform_from_axis_angle(axis, angle: float) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(np.asarray(axis, dtype=float) * float(angle)).as_matrix()
    return transform


def quat_wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(np.asarray(matrix, dtype=float)).as_quat()
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


def pose_error(target: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    pos = float(np.linalg.norm(target[:3, 3] - actual[:3, 3]))
    rel = target[:3, :3] @ actual[:3, :3].T
    rot = float(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec()))
    return pos, rot
