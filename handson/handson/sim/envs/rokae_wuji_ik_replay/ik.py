from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrapper(func):
            return func

        return wrapper


@dataclass(frozen=True)
class IKSolution:
    q: np.ndarray
    success: bool
    iterations: int
    position_error: float
    orientation_error: float


class DlsIkSolver:
    def __init__(
        self,
        model,
        *,
        lower: np.ndarray,
        upper: np.ndarray,
        damping: float = 1e-3,
        step_limit: float = 0.2,
        orientation_weight: float = 0.7,
        posture_target: np.ndarray | None = None,
        posture_weight: np.ndarray | None = None,
        posture_range: np.ndarray | None = None,
    ) -> None:
        if damping <= 0:
            raise ValueError("damping must be > 0")
        self.model = model
        self.origin_transforms = np.ascontiguousarray(model.origin_transforms, dtype=np.float64)
        self.joint_axes = np.ascontiguousarray(model.joint_axes, dtype=np.float64)
        self.end_transform = np.ascontiguousarray(model.end_transform, dtype=np.float64)
        self.lower = np.ascontiguousarray(lower, dtype=np.float64)
        self.upper = np.ascontiguousarray(upper, dtype=np.float64)
        self.damping = float(damping)
        self.step_limit = float(step_limit)
        self.orientation_weight = float(orientation_weight)
        self.posture_target = (
            np.zeros(model.dof, dtype=np.float64)
            if posture_target is None
            else np.ascontiguousarray(posture_target, dtype=np.float64)
        )
        self.posture_weight = (
            np.zeros(model.dof, dtype=np.float64)
            if posture_weight is None
            else np.ascontiguousarray(posture_weight, dtype=np.float64)
        )
        self.posture_range = (
            np.ones(model.dof, dtype=np.float64)
            if posture_range is None
            else np.ascontiguousarray(posture_range, dtype=np.float64)
        )

    def warmup(self) -> float:
        t0 = time.perf_counter()
        self.solve(self.model.fk(self.model.mid), self.model.mid, max_iters=3)
        return time.perf_counter() - t0

    def solve(
        self,
        target: np.ndarray,
        seed: np.ndarray,
        *,
        max_iters: int = 120,
        pos_tol: float = 2e-4,
        rot_tol: float = 2e-3,
        stall_patience: int = 6,
    ) -> IKSolution:
        q, success, iters, pos_err, rot_err = _dls_solve(
            np.ascontiguousarray(target, dtype=np.float64),
            np.ascontiguousarray(seed, dtype=np.float64),
            self.origin_transforms,
            self.joint_axes,
            self.end_transform,
            self.lower,
            self.upper,
            self.damping,
            self.step_limit,
            self.orientation_weight,
            self.posture_target,
            self.posture_weight,
            self.posture_range,
            int(max_iters),
            float(pos_tol),
            float(rot_tol),
            int(stall_patience),
        )
        return IKSolution(np.asarray(q), bool(success), int(iters), float(pos_err), float(rot_err))

    def solve_multi_start(
        self,
        target: np.ndarray,
        seeds: Sequence[np.ndarray],
        *,
        reference_q: np.ndarray | None = None,
        first_success: bool = True,
        max_iters: int = 120,
        pos_tol: float = 2e-4,
        rot_tol: float = 2e-3,
    ) -> IKSolution:
        if not seeds:
            raise ValueError("at least one seed is required")
        ref = None if reference_q is None else np.asarray(reference_q, dtype=float)
        best: IKSolution | None = None
        best_score = float("inf")
        for idx, seed in enumerate(seeds):
            sol = self.solve(target, seed, max_iters=max_iters, pos_tol=pos_tol, rot_tol=rot_tol)
            score = sol.position_error + self.orientation_weight * sol.orientation_error
            if sol.success and ref is not None:
                score += 0.5 * float(np.linalg.norm(sol.q - ref))
            if best is None or (sol.success and not best.success) or (sol.success == best.success and score < best_score):
                best = sol
                best_score = score
            if first_success and idx == 0 and sol.success:
                return sol
        assert best is not None
        return best


@njit(cache=True)
def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    x, y, z = axis[0], axis[1], axis[2]
    r = np.empty((3, 3))
    r[0, 0] = t * x * x + c
    r[0, 1] = t * x * y - s * z
    r[0, 2] = t * x * z + s * y
    r[1, 0] = t * x * y + s * z
    r[1, 1] = t * y * y + c
    r[1, 2] = t * y * z - s * x
    r[2, 0] = t * x * z - s * y
    r[2, 1] = t * y * z + s * x
    r[2, 2] = t * z * z + c
    return r


@njit(cache=True)
def _axis_transform(axis: np.ndarray, angle: float) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = _rodrigues(axis, angle)
    return out


@njit(cache=True)
def _mat4_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty((4, 4))
    for row in range(4):
        for col in range(4):
            total = 0.0
            for idx in range(4):
                total += a[row, idx] * b[idx, col]
            out[row, col] = total
    return out


@njit(cache=True)
def _mat4_rot_vec_mul(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    out = np.empty(3)
    for row in range(3):
        total = 0.0
        for idx in range(3):
            total += mat[row, idx] * vec[idx]
        out[row] = total
    return out


@njit(cache=True)
def _rot_error_matrix(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    out = np.empty((3, 3))
    for row in range(3):
        for col in range(3):
            total = 0.0
            for idx in range(3):
                total += target[row, idx] * current[col, idx]
            out[row, col] = total
    return out


@njit(cache=True)
def _rotvec_from_matrix(rmat: np.ndarray) -> np.ndarray:
    trace = rmat[0, 0] + rmat[1, 1] + rmat[2, 2]
    cos_angle = (trace - 1.0) * 0.5
    if cos_angle > 1.0:
        cos_angle = 1.0
    elif cos_angle < -1.0:
        cos_angle = -1.0
    angle = np.arccos(cos_angle)
    rv = np.zeros(3)
    if angle < 1e-10:
        return rv
    if angle < np.pi - 1e-4:
        k = angle / (2.0 * np.sin(angle))
        rv[0] = (rmat[2, 1] - rmat[1, 2]) * k
        rv[1] = (rmat[0, 2] - rmat[2, 0]) * k
        rv[2] = (rmat[1, 0] - rmat[0, 1]) * k
        return rv
    m = rmat + np.eye(3)
    best = 0
    best_norm = 0.0
    for col in range(3):
        n = m[0, col] ** 2 + m[1, col] ** 2 + m[2, col] ** 2
        if n > best_norm:
            best_norm = n
            best = col
    axis = m[:, best]
    norm = np.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
    if norm > 1e-12:
        rv = axis / norm * angle
    return rv


@njit(cache=True)
def _fk_and_jac(q, origin_transforms, joint_axes, end_transform):
    dof = q.shape[0]
    transform = np.eye(4)
    origins = np.empty((dof, 3))
    axes = np.empty((dof, 3))
    for i in range(dof):
        transform = _mat4_mul(transform, origin_transforms[i])
        origins[i, :] = transform[:3, 3]
        axes[i, :] = _mat4_rot_vec_mul(transform, joint_axes[i])
        transform = _mat4_mul(transform, _axis_transform(joint_axes[i], q[i]))
    end_pose = _mat4_mul(transform, end_transform)
    end_pos = end_pose[:3, 3]
    jac = np.zeros((6, dof))
    for i in range(dof):
        rel = end_pos - origins[i]
        jac[0, i] = axes[i, 1] * rel[2] - axes[i, 2] * rel[1]
        jac[1, i] = axes[i, 2] * rel[0] - axes[i, 0] * rel[2]
        jac[2, i] = axes[i, 0] * rel[1] - axes[i, 1] * rel[0]
        jac[3:, i] = axes[i]
    return end_pose, jac


@njit(cache=True)
def _dls_solve(
    target,
    seed,
    origin_transforms,
    joint_axes,
    end_transform,
    lower,
    upper,
    damping,
    step_limit,
    orientation_weight,
    posture_target,
    posture_weight,
    posture_range,
    max_iters,
    pos_tol,
    rot_tol,
    stall_patience,
):
    dof = seed.shape[0]
    q = np.minimum(np.maximum(seed.copy(), lower), upper)
    best_err = 1e30
    stalled = 0
    pos_err = 1e30
    rot_err = 1e30
    success = False
    iters = 0
    for it in range(max_iters):
        iters = it + 1
        current, jac = _fk_and_jac(q, origin_transforms, joint_axes, end_transform)
        pos_delta = target[:3, 3] - current[:3, 3]
        rot_delta = _rotvec_from_matrix(_rot_error_matrix(target, current))
        pos_err = np.sqrt(np.dot(pos_delta, pos_delta))
        rot_err = np.sqrt(np.dot(rot_delta, rot_delta))
        orientation_active = orientation_weight > 0.0
        if pos_err <= pos_tol and (not orientation_active or rot_err <= rot_tol):
            success = True
            break

        err = np.empty(6)
        err[:3] = pos_delta
        err[3:] = rot_delta * orientation_weight
        j_weighted = jac.copy()
        j_weighted[3:, :] *= orientation_weight
        jj = j_weighted @ j_weighted.T
        for i in range(6):
            jj[i, i] += damping * damping
        y = np.linalg.solve(jj, err)
        j_pinv = j_weighted.T @ np.linalg.inv(jj)
        dq = j_weighted.T @ y

        posture = posture_weight * (posture_target - q) / posture_range
        dq += (np.eye(dof) - j_pinv @ j_weighted) @ posture

        norm = np.sqrt(np.dot(dq, dq))
        if norm > step_limit:
            dq *= step_limit / norm
        q = np.minimum(np.maximum(q + dq, lower), upper)

        total_err = pos_err + orientation_weight * rot_err
        if total_err < best_err - 1e-9:
            best_err = total_err
            stalled = 0
        else:
            stalled += 1
            if stalled >= stall_patience:
                break
    return q, success, iters, pos_err, rot_err
