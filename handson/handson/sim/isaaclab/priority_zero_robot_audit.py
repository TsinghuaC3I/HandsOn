"""Strict actual-state robot checks for live priority-zero acceptance.

The helpers in this module intentionally read the loaded IsaacLab
articulations.  Authored USD values, IK feasibility, or command targets are
not substitutes for the two bases, fourteen arm joints, and two palm links
that PhysX actually simulated.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import math
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_BASE_POSES_WXYZ = {
    "right": np.asarray(
        [-0.75, -0.065, 1.34, 0.7071067812, 0.7071067812, 0.0, 0.0],
        dtype=np.float64,
    ),
    "left": np.asarray(
        [-0.75, 0.065, 1.34, 0.7071067812, -0.7071067812, 0.0, 0.0],
        dtype=np.float64,
    ),
}
EXPECTED_ARM_HOME_RAD = {
    "right": np.asarray(
        [-1.5707963, 1.5707963, 1.5707963, 1.5707963, -1.5707963, 0.0, 0.0],
        dtype=np.float64,
    ),
    "left": np.asarray(
        [1.5707963, 1.5707963, -1.5707963, 1.5707963, 1.5707963, 0.0, 0.0],
        dtype=np.float64,
    ),
}
EXPECTED_PALM_POSES_WXYZ = {
    # Physical palm-link world poses for the standard ROKAE bases and mirrored
    # seven-joint home.  These are deliberately not the controller task-frame
    # TCP targets (whose x coordinate is -0.10 m rather than -0.3715 m).
    "right": np.asarray(
        [-0.3715, -0.2395, 0.9975, 0.7071067811865476, 0.0, 0.7071067811865476, 0.0],
        dtype=np.float64,
    ),
    "left": np.asarray(
        [-0.3715, 0.2395, 0.9975, 0.7071067811865476, 0.0, 0.7071067811865476, 0.0],
        dtype=np.float64,
    ),
}

PRIORITY_ZERO_LINEAR_SPEED_LIMIT_M_S = 0.001
PRIORITY_ZERO_ANGULAR_SPEED_LIMIT_RAD_S = 0.002
PRIORITY_ZERO_JOINT_SPEED_LIMIT_RAD_S = 0.001


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _rotation_error(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left[:3, :3]).T @ np.asarray(right[:3, :3])
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


def _pose_matrix_wxyz(pose: np.ndarray) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64).reshape(7)
    quaternion = values[3:7]
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return np.full((4, 4), np.nan, dtype=np.float64)
    w, x, y, z = quaternion / norm
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    result[:3, 3] = values[:3]
    return result


def pose_error_wxyz(
    actual: Any,
    target: Any,
) -> tuple[bool, float, float]:
    """Return finite, translation, and sign-invariant rotation pose error."""

    try:
        actual_pose = _array(actual).reshape(7)
        target_pose = _array(target).reshape(7)
    except (TypeError, ValueError):
        return False, math.inf, math.inf
    if not np.all(np.isfinite([*actual_pose, *target_pose])):
        return False, math.inf, math.inf
    actual_norm = float(np.linalg.norm(actual_pose[3:7]))
    target_norm = float(np.linalg.norm(target_pose[3:7]))
    if actual_norm <= 1.0e-12 or target_norm <= 1.0e-12:
        return False, math.inf, math.inf
    alignment = float(
        np.clip(
            abs(
                np.dot(
                    actual_pose[3:7] / actual_norm,
                    target_pose[3:7] / target_norm,
                )
            ),
            0.0,
            1.0,
        )
    )
    return (
        True,
        float(np.linalg.norm(actual_pose[:3] - target_pose[:3])),
        float(2.0 * math.acos(alignment)),
    )


def priority_zero_runtime_contract(
    environment: Any,
    *,
    requested_device: str,
    physics_hz: int = 120,
    control_hz: int = 30,
) -> dict[str, Any]:
    """Fail closed unless the live child is the production Isaac/PhysX runtime.

    The standalone acceptance programs are evidence generators, so labels or
    command-line defaults are not sufficient.  This checks the installed
    runtime versions, the actual environment backend/timesteps, and the device
    of the loaded articulation tensors.
    """

    requested = str(requested_device).strip().lower()
    actual_device = str(getattr(environment, "_isaac_device", "")).strip().lower()
    backend = str(getattr(environment, "_physics_backend", "")).strip().lower()
    physics_dt = float(getattr(environment, "_physics_dt", math.nan))
    sim_dt = float(getattr(environment, "_sim_dt", math.nan))
    control_dt = float(getattr(environment, "_control_dt", math.nan))
    simulation = getattr(environment, "_sim", None)
    if simulation is None:
        raise RuntimeError("priority-zero acceptance requires a live simulation")
    live_sim_dt = float(simulation.get_physics_dt())

    try:
        isaac_sim_version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Isaac Sim distribution metadata is unavailable") from exc
    if isaac_sim_version != "5.1.0.0":
        raise RuntimeError(
            "priority-zero acceptance requires Isaac Sim 5.1.0.0, got "
            f"{isaac_sim_version!r}"
        )

    import isaaclab
    import torch

    isaaclab_source = Path(isaaclab.__file__).resolve()
    version_path = next(
        (
            parent / "VERSION"
            for parent in (isaaclab_source.parent, *isaaclab_source.parents)
            if (parent / "VERSION").is_file()
        ),
        None,
    )
    if version_path is None:
        raise RuntimeError(
            f"could not locate IsaacLab VERSION near {isaaclab_source}"
        )
    isaaclab_version = version_path.read_text(encoding="utf-8").strip()
    if isaaclab_version != "2.3.2":
        raise RuntimeError(
            "priority-zero acceptance requires IsaacLab 2.3.2, got "
            f"{isaaclab_version!r}"
        )

    tensor_devices = {
        side: str(environment._robots[side].data.joint_pos.device).lower()
        for side in ("right", "left")
    }
    expected_physics_dt = 1.0 / float(physics_hz)
    expected_control_dt = 1.0 / float(control_hz)
    checks = {
        "requested_cuda": requested.startswith("cuda:"),
        "actual_cuda": actual_device.startswith("cuda:"),
        "requested_matches_actual": requested == actual_device,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "articulation_readback_on_cuda": all(
            device.startswith("cuda:") for device in tensor_devices.values()
        ),
        "physx_backend": backend == "physx",
        "physics_dt_120hz": bool(
            np.isclose(physics_dt, expected_physics_dt, rtol=0.0, atol=1.0e-9)
            and np.isclose(sim_dt, expected_physics_dt, rtol=0.0, atol=1.0e-9)
            and np.isclose(live_sim_dt, expected_physics_dt, rtol=0.0, atol=1.0e-9)
        ),
        "control_dt_30hz": bool(
            np.isclose(control_dt, expected_control_dt, rtol=0.0, atol=1.0e-9)
        ),
        "four_physics_substeps_per_control": int(
            getattr(environment, "_n_substeps", -1)
        )
        == int(physics_hz // control_hz),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(
            "priority-zero production runtime contract failed: " + ", ".join(failed)
        )
    return {
        "passed": True,
        "checks": checks,
        "isaac_sim_version": isaac_sim_version,
        "isaaclab_version": isaaclab_version,
        "isaaclab_source": str(isaaclab_source),
        "isaaclab_version_path": str(version_path),
        "requested_device": requested,
        "actual_device": actual_device,
        "articulation_tensor_devices": tensor_devices,
        "physics_backend": backend,
        "physics_dt_s": physics_dt,
        "live_sim_physics_dt_s": live_sim_dt,
        "control_dt_s": control_dt,
        "physics_hz": int(physics_hz),
        "control_hz": int(control_hz),
    }
def _body_velocity(robot: Any, body_index: int) -> np.ndarray:
    for attribute in ("body_link_vel_w", "body_vel_w", "body_state_w"):
        value = getattr(robot.data, attribute, None)
        if value is None:
            continue
        array = _array(value)
        try:
            if attribute == "body_state_w":
                return array[0, int(body_index), 7:13].copy()
            return array[0, int(body_index), :6].copy()
        except (IndexError, TypeError):
            continue
    return np.full(6, np.nan, dtype=np.float64)


def _base_body_index(side: str, robot: Any) -> int | None:
    for index, name in enumerate(robot.body_names):
        if name == f"base_{side}" or name.endswith(f"/base_{side}"):
            return int(index)
    return None


def robot_snapshot(environment: Any) -> dict[str, dict[str, np.ndarray]]:
    """Read both bases, all robot joints, and both physical palm links."""

    snapshot: dict[str, dict[str, np.ndarray]] = {}
    for side in ("right", "left"):
        robot = environment._robots[side]
        arm_ids = list(environment._arm_joint_ids[side])
        hand_ids = list(environment._hand_joint_ids[side])
        palm_index = int(environment._ee_body_indices[side])
        base_index = _base_body_index(side, robot)
        root_velocity = _array(robot.data.root_vel_w)[0, :6].copy()
        all_joint_position = _array(robot.data.joint_pos)[0].copy()
        all_joint_velocity = _array(robot.data.joint_vel)[0].copy()
        default_all_joint_position = _array(robot.data.default_joint_pos)[0].copy()
        expected_hand_joint_position = np.zeros(len(hand_ids), dtype=np.float64)
        for hand_index, robot_joint_index in enumerate(hand_ids):
            if robot.joint_names[robot_joint_index] == f"{side}_finger1_joint1":
                expected_hand_joint_position[hand_index] = 0.10
        snapshot[side] = {
            "base": environment._robot_base_pose_matrix(side, robot),
            "base_velocity": (
                root_velocity
                if base_index is None
                else _body_velocity(robot, base_index)
            ),
            "palm": environment._robot_body_pose_matrix(robot, palm_index),
            "palm_velocity": _body_velocity(robot, palm_index),
            "joint_position": all_joint_position[arm_ids].copy(),
            "joint_velocity": all_joint_velocity[arm_ids].copy(),
            "default_joint_position": default_all_joint_position[arm_ids].copy(),
            "joint_limits": _array(robot.data.soft_joint_pos_limits)[
                0, arm_ids, :
            ].copy(),
            "hand_joint_position": all_joint_position[hand_ids].copy(),
            "hand_joint_velocity": all_joint_velocity[hand_ids].copy(),
            "default_hand_joint_position": default_all_joint_position[
                hand_ids
            ].copy(),
            "expected_hand_joint_position": expected_hand_joint_position,
            "all_joint_position": all_joint_position,
            "all_joint_velocity": all_joint_velocity,
        }
    return snapshot


def initial_robot_state_report(environment: Any) -> tuple[bool, dict[str, Any]]:
    """Validate the first settled PhysX readback against the robot contract."""

    snapshot = robot_snapshot(environment)
    table_top = float(environment._current_table_top_z())
    rows: dict[str, Any] = {}
    for side in ("right", "left"):
        sample = snapshot[side]
        expected_base = _pose_matrix_wxyz(EXPECTED_BASE_POSES_WXYZ[side])
        expected_palm_pose = EXPECTED_PALM_POSES_WXYZ[side]
        expected_palm = _pose_matrix_wxyz(expected_palm_pose)
        expected_joint_position = EXPECTED_ARM_HOME_RAD[side]
        joint_position = sample["joint_position"]
        joint_velocity = sample["joint_velocity"]
        hand_joint_position = sample["hand_joint_position"]
        hand_joint_velocity = sample["hand_joint_velocity"]
        expected_hand_joint_position = sample["expected_hand_joint_position"]
        limits = sample["joint_limits"]
        base_linear_speed = float(np.linalg.norm(sample["base_velocity"][:3]))
        base_angular_speed = float(np.linalg.norm(sample["base_velocity"][3:]))
        palm_linear_speed = float(np.linalg.norm(sample["palm_velocity"][:3]))
        palm_angular_speed = float(np.linalg.norm(sample["palm_velocity"][3:]))
        margin = np.minimum(
            joint_position - limits[:, 0], limits[:, 1] - joint_position
        )
        finite = bool(
            all(np.all(np.isfinite(value)) for value in sample.values())
            and np.all(np.isfinite(expected_palm_pose))
        )
        checks = {
            "finite_actual_state": finite,
            "seven_arm_joints": joint_position.shape == (7,),
            "twenty_hand_joints": hand_joint_position.shape == (20,),
            "base_position": float(
                np.linalg.norm(sample["base"][:3, 3] - expected_base[:3, 3])
            )
            <= 0.002,
            "base_orientation": _rotation_error(sample["base"], expected_base)
            <= math.radians(0.5),
            "arm_home": float(
                np.max(np.abs(joint_position - expected_joint_position))
            )
            <= 0.01,
            "authored_default_is_canonical_home": float(
                np.max(
                    np.abs(
                        sample["default_joint_position"]
                        - expected_joint_position
                    )
                )
            )
            <= 1.0e-6,
            "arm_stationary": float(np.max(np.abs(joint_velocity))) <= 0.05,
            "hand_home": float(
                np.max(
                    np.abs(hand_joint_position - expected_hand_joint_position)
                )
            )
            <= 0.01,
            "authored_default_hand_is_canonical_home": float(
                np.max(
                    np.abs(
                        sample["default_hand_joint_position"]
                        - expected_hand_joint_position
                    )
                )
            )
            <= 1.0e-6,
            "hand_stationary": float(
                np.max(np.abs(hand_joint_velocity))
            )
            <= 0.05,
            "base_stationary": bool(
                base_linear_speed <= 0.01 and base_angular_speed <= 0.02
            ),
            "palm_stationary": bool(
                palm_linear_speed <= 0.10 and palm_angular_speed <= 0.50
            ),
            "joint_limit_margin": float(np.min(margin)) >= 0.05,
            "palm_position": float(
                np.linalg.norm(sample["palm"][:3, 3] - expected_palm[:3, 3])
            )
            <= 0.03,
            "palm_orientation": _rotation_error(sample["palm"], expected_palm)
            <= math.radians(5.0),
            "palm_above_table": float(sample["palm"][2, 3] - table_top) >= 0.08,
            "correct_side": bool(
                sample["palm"][1, 3] < 0.0
                if side == "right"
                else sample["palm"][1, 3] > 0.0
            ),
        }
        rows[side] = {
            "passed": all(checks.values()),
            "checks": checks,
            "expected_base_pose_wxyz": EXPECTED_BASE_POSES_WXYZ[side].tolist(),
            "actual_base_position_m": sample["base"][:3, 3].tolist(),
            "base_position_error_m": float(
                np.linalg.norm(sample["base"][:3, 3] - expected_base[:3, 3])
            ),
            "base_orientation_error_deg": math.degrees(
                _rotation_error(sample["base"], expected_base)
            ),
            "actual_arm_joint_position_rad": joint_position.tolist(),
            "expected_arm_joint_position_rad": expected_joint_position.tolist(),
            "default_arm_joint_position_rad": sample[
                "default_joint_position"
            ].tolist(),
            "actual_arm_joint_velocity_rad_s": joint_velocity.tolist(),
            "arm_home_max_error_rad": float(
                np.max(
                    np.abs(joint_position - expected_joint_position)
                )
            ),
            "arm_max_speed_rad_s": float(np.max(np.abs(joint_velocity))),
            "actual_hand_joint_position_rad": hand_joint_position.tolist(),
            "expected_hand_joint_position_rad": (
                expected_hand_joint_position.tolist()
            ),
            "default_hand_joint_position_rad": sample[
                "default_hand_joint_position"
            ].tolist(),
            "actual_hand_joint_velocity_rad_s": hand_joint_velocity.tolist(),
            "hand_home_max_error_rad": float(
                np.max(
                    np.abs(hand_joint_position - expected_hand_joint_position)
                )
            ),
            "hand_max_speed_rad_s": float(
                np.max(np.abs(hand_joint_velocity))
            ),
            "all_joint_count": int(sample["all_joint_position"].size),
            "all_joint_max_speed_rad_s": float(
                np.max(np.abs(sample["all_joint_velocity"]))
            ),
            "actual_base_velocity_linear_angular": sample[
                "base_velocity"
            ].tolist(),
            "base_linear_speed_m_s": base_linear_speed,
            "base_angular_speed_rad_s": base_angular_speed,
            "actual_palm_velocity_linear_angular": sample[
                "palm_velocity"
            ].tolist(),
            "palm_linear_speed_m_s": palm_linear_speed,
            "palm_angular_speed_rad_s": palm_angular_speed,
            "joint_limit_min_margin_rad": float(np.min(margin)),
            "expected_palm_pose_wxyz": expected_palm_pose.tolist(),
            "actual_palm_position_m": sample["palm"][:3, 3].tolist(),
            "palm_position_error_m": float(
                np.linalg.norm(sample["palm"][:3, 3] - expected_palm[:3, 3])
            ),
            "palm_orientation_error_deg": math.degrees(
                _rotation_error(sample["palm"], expected_palm)
            ),
            "palm_clearance_above_table_m": float(
                sample["palm"][2, 3] - table_top
            ),
        }
    return all(row["passed"] for row in rows.values()), rows


@dataclass
class RobotStabilityAudit:
    """Accumulate strict drift and velocity bounds over a zero-action run."""

    reference: dict[str, dict[str, np.ndarray]]
    samples: int = 0
    nonfinite_samples: int = 0
    base_max_drift_m: float = 0.0
    base_max_rotation_drift_rad: float = 0.0
    palm_max_drift_m: float = 0.0
    palm_max_rotation_drift_rad: float = 0.0
    arm_max_drift_rad: float = 0.0
    hand_max_drift_rad: float = 0.0
    arm_max_speed_rad_s: float = 0.0
    all_joint_max_speed_rad_s: float = 0.0
    final_arm_max_speed_rad_s: float = 0.0
    final_all_joint_max_speed_rad_s: float = 0.0
    base_max_linear_speed_m_s: float = 0.0
    base_max_angular_speed_rad_s: float = 0.0
    palm_max_linear_speed_m_s: float = 0.0
    palm_max_angular_speed_rad_s: float = 0.0
    final_base_max_linear_speed_m_s: float = 0.0
    final_base_max_angular_speed_rad_s: float = 0.0
    final_palm_max_linear_speed_m_s: float = 0.0
    final_palm_max_angular_speed_rad_s: float = 0.0

    @classmethod
    def start(cls, environment: Any) -> "RobotStabilityAudit":
        return cls(reference=robot_snapshot(environment))

    def sample(self, environment: Any) -> None:
        current = robot_snapshot(environment)
        self.samples += 1
        final_speed = 0.0
        final_all_joint_speed = 0.0
        final_base_linear_speed = 0.0
        final_base_angular_speed = 0.0
        final_palm_linear_speed = 0.0
        final_palm_angular_speed = 0.0
        for side in ("right", "left"):
            reference = self.reference[side]
            sample = current[side]
            if not all(np.all(np.isfinite(value)) for value in sample.values()):
                self.nonfinite_samples += 1
                continue
            self.base_max_drift_m = max(
                self.base_max_drift_m,
                float(
                    np.linalg.norm(
                        sample["base"][:3, 3] - reference["base"][:3, 3]
                    )
                ),
            )
            self.base_max_rotation_drift_rad = max(
                self.base_max_rotation_drift_rad,
                _rotation_error(sample["base"], reference["base"]),
            )
            self.palm_max_drift_m = max(
                self.palm_max_drift_m,
                float(
                    np.linalg.norm(
                        sample["palm"][:3, 3] - reference["palm"][:3, 3]
                    )
                ),
            )
            self.palm_max_rotation_drift_rad = max(
                self.palm_max_rotation_drift_rad,
                _rotation_error(sample["palm"], reference["palm"]),
            )
            self.arm_max_drift_rad = max(
                self.arm_max_drift_rad,
                float(
                    np.max(
                        np.abs(
                            sample["joint_position"]
                            - reference["joint_position"]
                        )
                    )
                ),
            )
            self.hand_max_drift_rad = max(
                self.hand_max_drift_rad,
                float(
                    np.max(
                        np.abs(
                            sample["hand_joint_position"]
                            - reference["hand_joint_position"]
                        )
                    )
                ),
            )
            speed = float(np.max(np.abs(sample["joint_velocity"])))
            all_joint_speed = float(
                np.max(np.abs(sample["all_joint_velocity"]))
            )
            self.arm_max_speed_rad_s = max(self.arm_max_speed_rad_s, speed)
            self.all_joint_max_speed_rad_s = max(
                self.all_joint_max_speed_rad_s,
                all_joint_speed,
            )
            final_speed = max(final_speed, speed)
            final_all_joint_speed = max(
                final_all_joint_speed,
                all_joint_speed,
            )
            base_linear_speed = float(
                np.linalg.norm(sample["base_velocity"][:3])
            )
            base_angular_speed = float(
                np.linalg.norm(sample["base_velocity"][3:])
            )
            palm_linear_speed = float(
                np.linalg.norm(sample["palm_velocity"][:3])
            )
            palm_angular_speed = float(
                np.linalg.norm(sample["palm_velocity"][3:])
            )
            self.base_max_linear_speed_m_s = max(
                self.base_max_linear_speed_m_s, base_linear_speed
            )
            self.base_max_angular_speed_rad_s = max(
                self.base_max_angular_speed_rad_s, base_angular_speed
            )
            self.palm_max_linear_speed_m_s = max(
                self.palm_max_linear_speed_m_s, palm_linear_speed
            )
            self.palm_max_angular_speed_rad_s = max(
                self.palm_max_angular_speed_rad_s, palm_angular_speed
            )
            final_base_linear_speed = max(
                final_base_linear_speed, base_linear_speed
            )
            final_base_angular_speed = max(
                final_base_angular_speed, base_angular_speed
            )
            final_palm_linear_speed = max(
                final_palm_linear_speed, palm_linear_speed
            )
            final_palm_angular_speed = max(
                final_palm_angular_speed, palm_angular_speed
            )
        self.final_arm_max_speed_rad_s = final_speed
        self.final_all_joint_max_speed_rad_s = final_all_joint_speed
        self.final_base_max_linear_speed_m_s = final_base_linear_speed
        self.final_base_max_angular_speed_rad_s = final_base_angular_speed
        self.final_palm_max_linear_speed_m_s = final_palm_linear_speed
        self.final_palm_max_angular_speed_rad_s = final_palm_angular_speed

    def passed(self, *, physics_steps: int, physics_hz: int = 120) -> bool:
        return bool(
            physics_steps >= 10 * physics_hz
            and self.samples >= physics_steps
            and self.nonfinite_samples == 0
            and self.base_max_drift_m <= 0.002
            and self.base_max_rotation_drift_rad <= math.radians(0.5)
            and self.palm_max_drift_m <= 0.02
            and self.palm_max_rotation_drift_rad <= math.radians(5.0)
            and self.arm_max_drift_rad <= 0.02
            and self.hand_max_drift_rad <= 0.02
            and self.arm_max_speed_rad_s
            <= PRIORITY_ZERO_JOINT_SPEED_LIMIT_RAD_S
            and self.all_joint_max_speed_rad_s
            <= PRIORITY_ZERO_JOINT_SPEED_LIMIT_RAD_S
            and self.base_max_linear_speed_m_s
            <= PRIORITY_ZERO_LINEAR_SPEED_LIMIT_M_S
            and self.base_max_angular_speed_rad_s
            <= PRIORITY_ZERO_ANGULAR_SPEED_LIMIT_RAD_S
            and self.palm_max_linear_speed_m_s
            <= PRIORITY_ZERO_LINEAR_SPEED_LIMIT_M_S
            and self.palm_max_angular_speed_rad_s
            <= PRIORITY_ZERO_ANGULAR_SPEED_LIMIT_RAD_S
            and self.final_arm_max_speed_rad_s
            <= PRIORITY_ZERO_JOINT_SPEED_LIMIT_RAD_S
            and self.final_all_joint_max_speed_rad_s
            <= PRIORITY_ZERO_JOINT_SPEED_LIMIT_RAD_S
            and self.final_base_max_linear_speed_m_s
            <= PRIORITY_ZERO_LINEAR_SPEED_LIMIT_M_S
            and self.final_base_max_angular_speed_rad_s
            <= PRIORITY_ZERO_ANGULAR_SPEED_LIMIT_RAD_S
            and self.final_palm_max_linear_speed_m_s
            <= PRIORITY_ZERO_LINEAR_SPEED_LIMIT_M_S
            and self.final_palm_max_angular_speed_rad_s
            <= PRIORITY_ZERO_ANGULAR_SPEED_LIMIT_RAD_S
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": int(self.samples),
            "nonfinite_sample_count": int(self.nonfinite_samples),
            "base_max_drift_m": float(self.base_max_drift_m),
            "base_max_rotation_drift_deg": math.degrees(
                self.base_max_rotation_drift_rad
            ),
            "palm_max_drift_m": float(self.palm_max_drift_m),
            "palm_max_rotation_drift_deg": math.degrees(
                self.palm_max_rotation_drift_rad
            ),
            "arm_max_drift_rad": float(self.arm_max_drift_rad),
            "hand_max_drift_rad": float(self.hand_max_drift_rad),
            "arm_max_speed_rad_s": float(self.arm_max_speed_rad_s),
            "all_joint_max_speed_rad_s": float(
                self.all_joint_max_speed_rad_s
            ),
            "final_arm_max_speed_rad_s": float(
                self.final_arm_max_speed_rad_s
            ),
            "final_all_joint_max_speed_rad_s": float(
                self.final_all_joint_max_speed_rad_s
            ),
            "base_max_linear_speed_m_s": float(self.base_max_linear_speed_m_s),
            "base_max_angular_speed_rad_s": float(
                self.base_max_angular_speed_rad_s
            ),
            "palm_max_linear_speed_m_s": float(self.palm_max_linear_speed_m_s),
            "palm_max_angular_speed_rad_s": float(
                self.palm_max_angular_speed_rad_s
            ),
            "final_base_max_linear_speed_m_s": float(
                self.final_base_max_linear_speed_m_s
            ),
            "final_base_max_angular_speed_rad_s": float(
                self.final_base_max_angular_speed_rad_s
            ),
            "final_palm_max_linear_speed_m_s": float(
                self.final_palm_max_linear_speed_m_s
            ),
            "final_palm_max_angular_speed_rad_s": float(
                self.final_palm_max_angular_speed_rad_s
            ),
            "thresholds": {
                "base_drift_m": 0.002,
                "base_rotation_drift_deg": 0.5,
                "palm_drift_m": 0.02,
                "palm_rotation_drift_deg": 5.0,
                "arm_drift_rad": 0.02,
                "hand_drift_rad": 0.02,
                "arm_speed_rad_s": PRIORITY_ZERO_JOINT_SPEED_LIMIT_RAD_S,
                "all_joint_speed_rad_s": (
                    PRIORITY_ZERO_JOINT_SPEED_LIMIT_RAD_S
                ),
                "base_linear_speed_m_s": PRIORITY_ZERO_LINEAR_SPEED_LIMIT_M_S,
                "base_angular_speed_rad_s": (
                    PRIORITY_ZERO_ANGULAR_SPEED_LIMIT_RAD_S
                ),
                "palm_linear_speed_m_s": PRIORITY_ZERO_LINEAR_SPEED_LIMIT_M_S,
                "palm_angular_speed_rad_s": (
                    PRIORITY_ZERO_ANGULAR_SPEED_LIMIT_RAD_S
                ),
                "final_arm_speed_rad_s": (
                    PRIORITY_ZERO_JOINT_SPEED_LIMIT_RAD_S
                ),
                "final_all_joint_speed_rad_s": (
                    PRIORITY_ZERO_JOINT_SPEED_LIMIT_RAD_S
                ),
                "final_base_linear_speed_m_s": (
                    PRIORITY_ZERO_LINEAR_SPEED_LIMIT_M_S
                ),
                "final_base_angular_speed_rad_s": (
                    PRIORITY_ZERO_ANGULAR_SPEED_LIMIT_RAD_S
                ),
                "final_palm_linear_speed_m_s": (
                    PRIORITY_ZERO_LINEAR_SPEED_LIMIT_M_S
                ),
                "final_palm_angular_speed_rad_s": (
                    PRIORITY_ZERO_ANGULAR_SPEED_LIMIT_RAD_S
                ),
            },
        }


__all__ = [
    "EXPECTED_BASE_POSES_WXYZ",
    "EXPECTED_ARM_HOME_RAD",
    "EXPECTED_PALM_POSES_WXYZ",
    "PRIORITY_ZERO_LINEAR_SPEED_LIMIT_M_S",
    "PRIORITY_ZERO_ANGULAR_SPEED_LIMIT_RAD_S",
    "PRIORITY_ZERO_JOINT_SPEED_LIMIT_RAD_S",
    "RobotStabilityAudit",
    "initial_robot_state_report",
    "pose_error_wxyz",
    "priority_zero_runtime_contract",
    "robot_snapshot",
]
