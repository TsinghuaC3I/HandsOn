from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spatial import pose_from_xyz_rpy, transform_from_axis_angle


@dataclass(frozen=True)
class JointSpec:
    name: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


class SerialChainModel:
    def __init__(self, joints: list[JointSpec], *, end_link: str, end_transform: np.ndarray) -> None:
        if not joints:
            raise ValueError("SerialChainModel requires at least one joint")
        self.joints = joints
        self.joint_names = [j.name for j in joints]
        self.dof = len(joints)
        self.lower = np.array([j.lower for j in joints], dtype=float)
        self.upper = np.array([j.upper for j in joints], dtype=float)
        finite = np.isfinite(self.lower) & np.isfinite(self.upper)
        self.mid = (np.where(finite, self.lower, 0.0) + np.where(finite, self.upper, 0.0)) * 0.5
        self.end_link = end_link
        self.arm_end_link = joints[-1].child
        self.end_transform = np.asarray(end_transform, dtype=float)
        self.origin_transforms = np.array([pose_from_xyz_rpy(j.xyz, j.rpy) for j in joints])
        self.joint_axes = np.array([j.axis for j in joints], dtype=float)

    def clip(self, q) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=float), self.lower, self.upper)

    def fk(self, q) -> np.ndarray:
        q_arr = np.asarray(q, dtype=float)
        if q_arr.shape != (self.dof,):
            raise ValueError(f"expected q shape ({self.dof},), got {q_arr.shape}")
        transform = np.eye(4)
        for origin_t, joint, angle in zip(self.origin_transforms, self.joints, q_arr, strict=True):
            transform = transform @ origin_t @ transform_from_axis_angle(joint.axis, float(angle))
        return transform @ self.end_transform
