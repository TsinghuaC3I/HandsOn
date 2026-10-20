#!/usr/bin/env python3
"""Automatic physical collection and fresh replay for B320-HS014.

The controller emits only the native 54-D dual-ROKAE/Wuji action. It never
writes robot or task-object pose/velocity state after the reset boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

# Direct execution makes ``scripts/`` sys.path[0], where the legacy
# ``scripts/taskdev.py`` module would shadow the repository's ``taskdev``
# package.  Production collection needs the package for the hash-bound HS014
# asset composer, so make the repository root the unambiguous first entry.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "handson"))

import numpy as np
import zarr
from numcodecs import Blosc


TASK_NAME = "bench320_hs014_bottle_stabilized_cap_opening"
STABLE_RECORDING_ORIGIN_ENV = "HANDSON_STABLE_RECORDING_ORIGIN"
PHYSICS_HZ = 120
SCHEMA_VERSION = "handson.hs014.physx_collection_v1"
PRELIFT_CLEARANCE_M = 0.0
TWIST_PHASE_FRAMES = 100
DEFAULT_MAX_STEPS = 1650
TWIST_DIRECTION = 1.0
OBJECT_TRANSLATION_FEEDBACK_MAX_M = 0.020
LEFT_FIXTURE_APPROACH_OFFSET_M = np.asarray(
    [0.0, 0.075, -0.100], dtype=np.float64
)
LEFT_FIXTURE_PALM_OFFSET_M = np.asarray(
    [0.0, 0.050, -0.100], dtype=np.float64
)
RIGHT_TOP_DOWN_PALM_HEIGHT_M = 0.105
LEFT_GRASP_MOTION_FRAMES = 60
LEFT_GRASP_MAX_FRAMES = 75
LEFT_FIXTURE_PREGRIP = 0.40
LEFT_FIXTURE_LATCH_START_ALPHA = 0.40
LEFT_FIXTURE_GRIP = 0.75
# This bottle-relative path is the geometry used by the three randomized
# physical successes (seeds 7, 19 and 212) before the fixture hold switches
# from DLS to the post-contact IK controller.
LEFT_FIXTURE_GRASP_WORLD_QUAT_WXYZ = np.asarray(
    [0.0, 1.0, 0.0, 0.0], dtype=np.float64
)
LEFT_FIXTURE_GRASP_WORLD_QUAT_WXYZ /= np.linalg.norm(
    LEFT_FIXTURE_GRASP_WORLD_QUAT_WXYZ
)
# The cap is rotationally symmetric. Keep the physically validated top-down
# hand azimuth fixed in the world frame so reset-time cap yaw does not select a
# different arm IK/grasp branch.
RIGHT_TOP_DOWN_WORLD_QUAT_WXYZ = np.asarray(
    [
        -0.023910984644807905,
        -0.20907293270019867,
        0.9773478863414138,
        -0.022536252781348545,
    ],
    dtype=np.float64,
)
# Natural Wuji power-grip proportions already used by the benchmark's tool
# tasks. A uniform scalar command also drives the lateral spread joints and
# makes the fingers sweep the bottle during a top-down cap approach.
RIGHT_CAP_POWER_GRIP = np.asarray(
    [
        0.60, 0.10, 0.84, 0.70,
        0.68, 0.00, 0.90, 0.78,
        0.70, 0.00, 0.92, 0.80,
        0.68, 0.00, 0.90, 0.78,
        0.64, 0.00, 0.86, 0.74,
    ],
    dtype=np.float64,
)
REQUIRED_ZARR_FIELDS = (
    "action",
    "state",
    "next_state",
    "reward",
    "terminal",
    "succeed",
    "hs014_hidden_state",
    "next_hs014_hidden_state",
    "hs014_success_truth",
    "next_hs014_success_truth",
    "rgb_front",
    "rgb_wrist_left",
    "rgb_wrist_right",
    "physx_robot_joint_position",
    "next_physx_robot_joint_position",
    "physx_robot_joint_target",
    "next_physx_robot_joint_target",
    "physx_task_rigid_body_state",
    "next_physx_task_rigid_body_state",
    "physx_contact_signature_pair_metrics",
    "next_physx_contact_signature_pair_metrics",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        payload = file_path.read_bytes()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        count += 1
        size += len(payload)
    return {"sha256": digest.hexdigest(), "file_count": count, "bytes": size}


def _git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def _matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return quat / max(float(np.linalg.norm(quat)), 1.0e-12)


def _quat_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = max(math.sqrt(w * w + x * x + y * y + z * z), 1.0e-12)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _right_cap_grip(max_joint: float) -> np.ndarray:
    """Scale the natural cap grip without changing its joint proportions."""

    return RIGHT_CAP_POWER_GRIP * (
        float(max_joint) / float(np.max(RIGHT_CAP_POWER_GRIP))
    )


def _pose_to_matrix_wxyz(pose: np.ndarray) -> np.ndarray:
    vector = np.asarray(pose, dtype=np.float64).reshape(7)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_to_matrix_wxyz(vector[3:7])
    matrix[:3, 3] = vector[:3]
    return matrix


def _interpolate_pose(start: np.ndarray, goal: np.ndarray, alpha: float) -> np.ndarray:
    start = np.asarray(start, dtype=np.float64).reshape(7)
    goal = np.asarray(goal, dtype=np.float64).reshape(7)
    fraction = float(np.clip(alpha, 0.0, 1.0))
    quaternion = (1.0 - fraction) * start[3:7] + fraction * goal[3:7]
    quaternion /= max(float(np.linalg.norm(quaternion)), 1.0e-12)
    if float(np.dot(quaternion, goal[3:7])) < 0.0:
        quaternion *= -1.0
    return np.concatenate(
        [(1.0 - fraction) * start[:3] + fraction * goal[:3], quaternion]
    )


class HS014AutoController:
    """Event-gated action-only controller for the physical vertical slice."""

    PHASES = (
        "settle",
        "left_preorient",
        "left_approach",
        "left_grasp",
        "right_raise",
        "right_approach",
        "right_descend",
        "right_close_in",
        "right_grasp",
        "prelift",
        "twist_1",
        "twist_2",
        "twist_3",
        "twist_4",
        "twist_5",
        "twist_6",
        "twist_7",
        "lift",
        "hold",
    )
    DURATIONS = {
        "settle": 10,
        "left_preorient": 90,
        "left_approach": 45,
        "left_grasp": LEFT_GRASP_MAX_FRAMES,
        "right_raise": 75,
        "right_approach": 75,
        "right_descend": 90,
        "right_close_in": 120,
        "right_grasp": 90,
        "prelift": 45,
        "twist_1": TWIST_PHASE_FRAMES,
        "twist_2": TWIST_PHASE_FRAMES,
        "twist_3": TWIST_PHASE_FRAMES,
        "twist_4": TWIST_PHASE_FRAMES,
        "twist_5": TWIST_PHASE_FRAMES,
        "twist_6": TWIST_PHASE_FRAMES,
        "twist_7": TWIST_PHASE_FRAMES,
        "lift": 45,
        "hold": 60,
    }

    def __init__(self, raw_env: Any) -> None:
        self.raw = raw_env
        self.phase_index = 0
        self.phase_frame = 0
        self.phase_start_right = self._current_physical_tcp_pose("right")
        self.phase_start_left = self._current_physical_tcp_pose("left")
        self.right_goal = self.phase_start_right.copy()
        self.left_goal = self.phase_start_left.copy()
        self.left_hold = self.phase_start_left.copy()
        baseline = getattr(self.raw, "_hs014_bottle_baseline", None)
        if baseline is None or not np.all(np.isfinite(np.asarray(baseline))):
            baseline = self._bottle_pose()
        self.fixture_reference_pose = np.asarray(
            baseline, dtype=np.float64
        ).reshape(7).copy()
        self._twist_anchor_object_in_bottle: np.ndarray | None = None
        self.failed_reason = ""
        self.phase_history: list[dict[str, Any]] = []
        self._configure_goals()

    def _current_physical_tcp_pose(self, side: str) -> np.ndarray:
        fallback = np.asarray(
            self.raw._left_tcp_pose if side == "left" else self.raw._right_tcp_pose,
            dtype=np.float64,
        ).reshape(7)
        resolver = getattr(self.raw, "_native_end_effector_pose_matrix", None)
        if resolver is None:
            return fallback.copy()
        try:
            matrix = resolver(side, fallback)
        except Exception:
            matrix = None
        if matrix is None:
            return fallback.copy()
        matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        pose = np.concatenate([matrix[:3, 3], _matrix_to_quat_wxyz(matrix[:3, :3])])
        if not np.all(np.isfinite(pose)):
            return fallback.copy()
        return pose

    @property
    def phase(self) -> str:
        return self.PHASES[min(self.phase_index, len(self.PHASES) - 1)]

    @property
    def done(self) -> bool:
        return bool(self.failed_reason or self.phase == "hold" and self.phase_frame >= 60)

    def _bottle_pose(self) -> np.ndarray:
        return np.asarray(
            self.raw._physics_body_pose("hs014_water_bottle")
            if self.raw._physics_body_pose("hs014_water_bottle") is not None
            else self.raw._poses["hs014_water_bottle"],
            dtype=np.float64,
        )

    def _cap_pose(self) -> np.ndarray:
        return np.asarray(
            self.raw._physics_body_pose("hs014_bottle_cap")
            if self.raw._physics_body_pose("hs014_bottle_cap") is not None
            else self.raw._poses["hs014_bottle_cap"],
            dtype=np.float64,
        )

    def _twisted_cap_pose(self, degrees: float) -> np.ndarray:
        if self._twist_anchor_object_in_bottle is None:
            self._capture_twist_anchor()
        assert self._twist_anchor_object_in_bottle is not None
        bottle = _pose_to_matrix_wxyz(self._bottle_pose())
        relative = self._twist_anchor_object_in_bottle.copy()
        angle = TWIST_DIRECTION * math.radians(float(degrees))
        rotation_delta = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        # Keep the cap's contacted center fixed relative to the held bottle.
        # Advancing from the live cap pose made small IK tracking errors become
        # the next phase's reference and ratcheted the free cap off the mouth.
        relative[:3, :3] = rotation_delta @ relative[:3, :3]
        goal = bottle @ relative
        return np.concatenate(
            [goal[:3, 3], _matrix_to_quat_wxyz(goal[:3, :3])]
        )

    def _capture_twist_anchor(self) -> None:
        bottle = _pose_to_matrix_wxyz(self._bottle_pose())
        cap = _pose_to_matrix_wxyz(self._cap_pose())
        self._twist_anchor_object_in_bottle = np.linalg.inv(bottle) @ cap

    @staticmethod
    def _bounded_translation_feedback(
        desired_position: np.ndarray, actual_position: np.ndarray
    ) -> np.ndarray:
        correction = np.asarray(desired_position, dtype=np.float64).reshape(3) - np.asarray(
            actual_position, dtype=np.float64
        ).reshape(3)
        norm = float(np.linalg.norm(correction))
        if norm > OBJECT_TRANSLATION_FEEDBACK_MAX_M:
            correction *= OBJECT_TRANSLATION_FEEDBACK_MAX_M / norm
        return correction

    @staticmethod
    def _planned_top_down_palm_pose(cap_pose: np.ndarray) -> np.ndarray:
        cap = np.asarray(cap_pose, dtype=np.float64).reshape(7)
        position = cap[:3].copy()
        position[2] += RIGHT_TOP_DOWN_PALM_HEIGHT_M
        return np.concatenate([position, RIGHT_TOP_DOWN_WORLD_QUAT_WXYZ])

    def _configure_goals(self) -> None:
        phase = self.phase
        bottle = self._bottle_pose()
        cap = self._cap_pose()
        downward = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        self.right_goal = self.phase_start_right.copy()
        self.left_goal = self.phase_start_left.copy()
        if phase == "settle":
            return
        if phase == "left_preorient":
            # Rotate the open hand at its remote home position. Doing this
            # during the translational approach sweeps the fingers across the
            # bottle mouth and can strike the free cap before role contact.
            self.left_goal = np.concatenate([self.phase_start_left[:3], downward])
        elif phase == "left_approach":
            self.left_goal = np.concatenate(
                [bottle[:3] + LEFT_FIXTURE_APPROACH_OFFSET_M, downward]
            )
        elif phase == "left_grasp":
            self.left_goal = np.concatenate(
                [
                    bottle[:3] + LEFT_FIXTURE_PALM_OFFSET_M,
                    LEFT_FIXTURE_GRASP_WORLD_QUAT_WXYZ,
                ]
            )
        elif phase == "right_raise":
            self.left_goal = self.left_hold.copy()
            top_down = self._planned_top_down_palm_pose(cap)
            self.right_goal = np.concatenate(
                [
                    [
                        self.phase_start_right[0],
                        self.phase_start_right[1],
                        cap[2] + 0.150,
                    ],
                    top_down[3:7],
                ]
            )
        elif phase == "right_approach":
            self.left_goal = self.left_hold.copy()
            self.right_goal = self._planned_top_down_palm_pose(cap)
            self.right_goal[2] += 0.105
        elif phase == "right_grasp":
            self.left_goal = self.left_hold.copy()
            self.right_goal = self.phase_start_right.copy()
        elif phase == "right_close_in":
            self.left_goal = self.left_hold.copy()
            self.right_goal = self._planned_top_down_palm_pose(cap)
        elif phase == "right_descend":
            self.left_goal = self.left_hold.copy()
            self.right_goal = self._planned_top_down_palm_pose(cap)
            self.right_goal[2] += 0.035
        elif phase == "prelift":
            self.left_goal = self.left_hold.copy()
            bottle_axis = _quat_to_matrix_wxyz(bottle[3:7])[:, 2]
            cap_goal = cap.copy()
            # The free cap is already contact-free at its positive support gap.
            # Keep it seated for twisting; axial separation happens only after
            # the unchanged >180-degree rotation gate has been crossed.
            cap_goal[:3] += PRELIFT_CLEARANCE_M * bottle_axis
            palm_goal = self.raw.hs014_palm_pose_for_object_pose("right_cap", cap_goal)
            if palm_goal is None:
                self.failed_reason = "prelift_without_physical_right_cap_attachment"
            else:
                self.right_goal = palm_goal
        elif phase.startswith("twist_"):
            self.left_goal = self.left_hold.copy()
            turn_index = int(phase.rsplit("_", 1)[1])
            cap_goal = self._twisted_cap_pose(30.0 * turn_index)
            palm_goal = self.raw.hs014_palm_pose_for_object_pose("right_cap", cap_goal)
            if palm_goal is None:
                self.failed_reason = f"{phase}_without_physical_right_cap_attachment"
            else:
                if float(np.dot(self.phase_start_right[3:7], palm_goal[3:7])) < 0.0:
                    palm_goal[3:7] *= -1.0
                self.right_goal = palm_goal
        elif phase == "lift":
            self.left_goal = self.left_hold.copy()
            bottle_axis = _quat_to_matrix_wxyz(bottle[3:7])[:, 2]
            cap_goal = cap.copy()
            cap_goal[:3] += 0.055 * bottle_axis
            palm_goal = self.raw.hs014_palm_pose_for_object_pose("right_cap", cap_goal)
            if palm_goal is None:
                self.failed_reason = "lift_without_physical_right_cap_attachment"
            else:
                self.right_goal = palm_goal
        elif phase == "hold":
            self.left_goal = self.left_hold.copy()

    def _advance(self) -> None:
        self.phase_history.append(
            {"phase": self.phase, "control_frames": int(self.phase_frame)}
        )
        if self.phase == "left_grasp":
            self.left_hold = self._current_physical_tcp_pose("left")
        previous_right_goal = self.right_goal.copy()
        self.phase_index = min(self.phase_index + 1, len(self.PHASES) - 1)
        self.phase_frame = 0
        if self.phase.startswith("twist_"):
            # Preserve command-space continuity across twist segments.  Using
            # the lagging physical TCP here makes every segment inherit the
            # previous tracking error and ratchets the attached cap sideways.
            self.phase_start_right = previous_right_goal
        else:
            self.phase_start_right = self._current_physical_tcp_pose("right")
        self.phase_start_left = self._current_physical_tcp_pose("left")
        if self.phase == "twist_1":
            self._capture_twist_anchor()
        self._configure_goals()

    def _event_gates(self) -> None:
        events = self.raw._hs014_events
        phase = self.phase
        duration = self.DURATIONS[phase]
        if phase == "left_grasp" and events.get("left_fixture_contact", False):
            self._advance()
        elif phase in {"right_descend", "right_close_in"} and events.get(
            "right_cap_contact", False
        ):
            self._advance()
        elif phase == "right_grasp" and events.get("right_cap_contact", False):
            self._advance()
        elif phase == "twist_7" and self.phase_frame >= duration:
            if events.get("continuous_twist_seen", False):
                self._advance()
            else:
                self.failed_reason = "continuous_twist_event_timeout"
        elif phase == "lift" and events.get("cap_axially_separated", False):
            self._advance()
        elif (
            phase
            in {
                "settle",
                "left_preorient",
                "left_approach",
                "right_raise",
                "right_approach",
                "right_descend",
                "right_close_in",
                "prelift",
            }
            or phase.startswith("twist_")
        ) and self.phase_frame >= duration:
            self._advance()
        elif phase in {"left_grasp", "right_grasp", "lift"} and self.phase_frame >= duration:
            self.failed_reason = f"{phase}_event_timeout"

    def action(self) -> np.ndarray:
        self._event_gates()
        duration = max(1, self.DURATIONS[self.phase])
        motion_duration = (
            LEFT_GRASP_MOTION_FRAMES if self.phase == "left_grasp" else duration
        )
        alpha = min(1.0, float(self.phase_frame + 1) / float(motion_duration))
        if self.phase.startswith("twist_"):
            turn_index = int(self.phase.rsplit("_", 1)[1])
            degrees = 30.0 * (float(turn_index - 1) + alpha)
            cap_goal = self._twisted_cap_pose(degrees)
            palm_goal = self.raw.hs014_palm_pose_for_object_pose(
                "right_cap", cap_goal
            )
            if palm_goal is None:
                self.failed_reason = (
                    f"{self.phase}_without_physical_right_cap_attachment"
                )
                right = self._current_physical_tcp_pose("right")
            else:
                right = np.asarray(palm_goal, dtype=np.float64).reshape(7)
                if float(np.dot(self.right_goal[3:7], right[3:7])) < 0.0:
                    right[3:7] *= -1.0
        elif self.phase in {"right_approach", "right_descend", "right_close_in"}:
            live_goal = self._planned_top_down_palm_pose(self._cap_pose())
            if self.phase == "right_approach":
                live_goal[2] += 0.105
            elif self.phase == "right_descend":
                live_goal[2] += 0.035
            right = _interpolate_pose(self.phase_start_right, live_goal, alpha)
        else:
            right = _interpolate_pose(self.phase_start_right, self.right_goal, alpha)
        if (
            self.raw._hs014_events.get("left_fixture_contact", False)
            and self.raw._hs014_events.get("right_cap_contact", False)
        ):
            left_goal = self.raw.hs014_palm_pose_for_object_pose(
                "left_fixture", self.fixture_reference_pose
            )
            if left_goal is None:
                self.failed_reason = "bimanual_hold_without_left_fixture_attachment"
                left = self._current_physical_tcp_pose("left")
            else:
                left = np.asarray(left_goal, dtype=np.float64).reshape(7)
                left[:3] += self._bounded_translation_feedback(
                    self.fixture_reference_pose[:3], self._bottle_pose()[:3]
                )
        else:
            left = _interpolate_pose(self.phase_start_left, self.left_goal, alpha)
        right_hand = np.zeros(20, dtype=np.float64)
        left_hand = np.zeros(20, dtype=np.float64)
        if self.phase == "left_grasp" and alpha <= LEFT_FIXTURE_LATCH_START_ALPHA:
            # Stay below the environment's 0.45 role-ownership gate until the
            # arm reaches the stable fixture-contact portion of the path.
            left_hand.fill(LEFT_FIXTURE_PREGRIP)
        elif self.phase not in {"settle", "left_preorient", "left_approach"}:
            left_hand.fill(LEFT_FIXTURE_GRIP)
        if self.phase == "right_raise":
            right_hand[:] = _right_cap_grip(0.40)
        elif self.phase == "right_approach":
            # Finish closing at the observed safe height before the descent;
            # an open Wuji hand otherwise sweeps the free cap with its long
            # fingers on the first downward step. Role ownership still needs
            # four consecutive physical-contact substeps.
            right_hand[:] = _right_cap_grip(0.40 if alpha < 0.85 else 0.85)
        elif self.phase == "right_descend":
            right_hand[:] = _right_cap_grip(0.85)
        elif self.phase == "right_close_in":
            right_hand[:] = _right_cap_grip(0.85)
        elif self.phase == "right_grasp" or self.phase.startswith("twist_") or self.phase in {"prelift", "lift", "hold"}:
            right_hand[:] = _right_cap_grip(0.85)
        self.phase_frame += 1
        return np.concatenate([right, left, right_hand, left_hand]).astype(np.float64)


def _make_environment(seed: int, device: str, *, randomize: bool) -> Any:
    os.environ["HANDSON_ENABLE_ISAACSIM_NEWTON"] = "0"
    os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
    os.environ["HANDSON_SIM_BACKEND"] = "isaacsim"
    # The shared gate deliberately no-ops when this opt-in is absent.  HS014
    # collection and replay require the gate, so enable it before constructing
    # the environment as well as immediately before the episode reset.
    os.environ[STABLE_RECORDING_ORIGIN_ENV] = "1"
    from handson.tasks.mappings import CONFIG_MAPPING

    config = CONFIG_MAPPING[TASK_NAME]()
    return config.get_native_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=bool(randomize),
        randomize_dynamics=False,
        image_obs=True,
        launch_isaac=True,
        headless=True,
        render_resolution=(480, 640),
        isaac_device=device,
        physics_backend="physx",
        seed=int(seed),
    )


def _close_environment_fast(env: Any) -> None:
    """Release an RTX collection worker without blocking in SimulationContext.stop."""

    raw = env.unwrapped
    simulation_app = getattr(raw, "_simulation_app", None)
    raw._fast_shutdown_on_close = True
    env.close()
    if simulation_app is not None:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)


def _camera_arrays(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    front_key = "ego" if "ego" in observation else "random_camera"
    arrays = {
        "rgb_front": np.asarray(observation[front_key]),
        "rgb_wrist_left": np.asarray(observation["wrist_left"]),
        "rgb_wrist_right": np.asarray(observation["wrist_right"]),
    }
    for name, array in arrays.items():
        if array.ndim != 3 or array.shape[-1] < 3:
            raise RuntimeError(f"{name} is not an HWC RGB frame: {array.shape}")
        arrays[name] = array[..., :3].astype(np.uint8, copy=True)
    return arrays


def _frame_metrics(frame: np.ndarray) -> dict[str, Any]:
    rgb = np.asarray(frame, dtype=np.float64)[..., :3]
    return {
        "shape": list(rgb.shape),
        "mean": float(rgb.mean()),
        "variance": float(rgb.var()),
        "black_fraction": float(np.mean(rgb.mean(axis=-1) < 5.0)),
    }


def _reset_and_stabilize_physical_origin(
    env: Any,
    *,
    seed: int,
) -> tuple[Any, Mapping[str, Any]]:
    """Run the same strict reset-origin gate used by the production recorder."""

    from handson.sim.isaaclab.recording_origin import (
        stabilize_recording_origin,
        stable_recording_origin_kwargs_for_task,
    )

    os.environ[STABLE_RECORDING_ORIGIN_ENV] = "1"
    observation, reset_info = env.reset(seed=int(seed))
    profile = stable_recording_origin_kwargs_for_task(TASK_NAME)
    observation, reset_info = stabilize_recording_origin(
        env,
        observation,
        reset_info,
        **profile,
    )
    _require_hs014_physical_recording_origin(reset_info, profile=profile)
    return observation, reset_info


def _require_hs014_physical_recording_origin(
    info: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject a disabled, skipped, truncated, or unaccepted shared gate."""

    if not isinstance(info, Mapping):
        raise RuntimeError("HS014 stable-origin gate returned non-mapping info")
    raw_origin = info.get("physical_recording_origin")
    if not isinstance(raw_origin, Mapping):
        raise RuntimeError(
            "HS014 production reset is missing physical_recording_origin; "
            "the shared stable-origin gate was disabled or skipped"
        )
    origin = dict(raw_origin)
    if str(origin.get("task_name", "")).strip().lower() != TASK_NAME:
        raise RuntimeError("HS014 physical_recording_origin has the wrong task_name")
    stable_seconds = float(profile.get("stable_seconds", 0.0))
    if not math.isclose(stable_seconds, 10.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(
            "HS014 production stable-origin profile must require exactly 10 seconds"
        )
    expected_steps = int(round(stable_seconds * PHYSICS_HZ))
    required_steps = origin.get("stable_steps_required")
    observed_steps = origin.get("stable_steps_observed")
    if (
        type(required_steps) is not int
        or required_steps != expected_steps
        or type(observed_steps) is not int
        or observed_steps < required_steps
    ):
        raise RuntimeError(
            "HS014 physical_recording_origin does not prove the complete "
            f"{expected_steps}-frame production gate"
        )
    acceptance = origin.get("task_recording_origin_acceptance")
    if not isinstance(acceptance, Mapping) or acceptance.get("accepted") is not True:
        raise RuntimeError(
            "HS014 physical_recording_origin lacks exact task acceptance"
        )
    return origin


def _camera_visibility_errors(task_info: Mapping[str, Any]) -> list[str]:
    visibility = task_info.get("hs014_camera_visibility")
    if not isinstance(visibility, Mapping):
        return ["HS014 camera target-visibility truth is missing"]
    errors: list[str] = []
    for name in ("front", "wrist_left", "wrist_right"):
        record = visibility.get(name)
        if not isinstance(record, Mapping):
            errors.append(f"{name} target-visibility truth is missing")
            continue
        if int(record.get("required_target_count", 0)) <= 0:
            errors.append(f"{name} has no required task target")
        if not bool(record.get("targets_in_frame", False)):
            errors.append(f"{name} required task target is outside the frame")
    return errors


def _numeric_passthrough(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, value in observation.items():
        if not str(key).startswith(("physx_", "hs014_")):
            continue
        array = np.asarray(value)
        if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
            continue
        result[str(key)] = array.copy()
    return result


def _write_zarr(
    replay_path: Path,
    transitions: list[dict[str, Any]],
    *,
    seed: int,
    source_commit: str,
    randomize: bool,
) -> dict[str, Any]:
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(replay_path), mode="w")
    root.attrs.update(
        {
            "schema_version": SCHEMA_VERSION,
            "task_name": TASK_NAME,
            "simulator": "Isaac Sim 5.1.0.0",
            "isaaclab_checkout": "IsaacLab-v2.3.2",
            "physics_backend": "physx",
            "episode_reset_seed": int(seed),
            "source_commit": str(source_commit),
            "randomize": bool(randomize),
            "fresh_replay_required": True,
            "object_state_writes_after_action0_allowed": False,
        }
    )
    data = root.create_group("data")
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)

    fields: dict[str, list[np.ndarray]] = {
        "action": [],
        "state": [],
        "next_state": [],
        "reward": [],
        "terminal": [],
        "succeed": [],
        "rgb_front": [],
        "rgb_wrist_left": [],
        "rgb_wrist_right": [],
    }
    passthrough_names = sorted(
        set.intersection(
            *[
                set(item["passthrough"]) & set(item["next_passthrough"])
                for item in transitions
            ]
        )
    )
    for name in passthrough_names:
        fields[name] = []
        fields[f"next_{name}"] = []
    for item in transitions:
        fields["action"].append(np.asarray(item["action"], dtype=np.float64))
        fields["state"].append(np.asarray(item["observation"]["state"], dtype=np.float64))
        fields["next_state"].append(np.asarray(item["next_observation"]["state"], dtype=np.float64))
        fields["reward"].append(np.asarray(item["reward"], dtype=np.float64))
        fields["terminal"].append(np.asarray(item["terminal"], dtype=np.uint8))
        fields["succeed"].append(np.asarray(item["succeed"], dtype=np.uint8))
        for name, frame in item["cameras"].items():
            fields[name].append(frame)
        for name in passthrough_names:
            fields[name].append(item["passthrough"][name])
            fields[f"next_{name}"].append(item["next_passthrough"][name])

    shapes: dict[str, list[int]] = {}
    for name, values in fields.items():
        try:
            array = np.stack(values, axis=0)
        except ValueError as exc:
            raise RuntimeError(f"Zarr field {name} changed shape") from exc
        chunks = (1, *array.shape[1:]) if array.ndim >= 3 else None
        data.create_dataset(
            name,
            data=array,
            chunks=chunks,
            compressor=compressor,
            overwrite=True,
        )
        shapes[name] = list(array.shape)
    return {"fields": shapes, "steps": len(transitions)}


def episode_contract_errors(episode_dir: Path) -> list[str]:
    episode_dir = Path(episode_dir)
    replay_path = episode_dir / "replay.zarr"
    errors: list[str] = []
    if not replay_path.is_dir():
        return ["replay.zarr is missing"]
    try:
        root = zarr.open_group(str(replay_path), mode="r")
        data = root["data"]
    except Exception as exc:
        return [f"replay.zarr is unreadable: {exc}"]
    for field in REQUIRED_ZARR_FIELDS:
        if field not in data:
            errors.append(f"missing field: {field}")
    if errors:
        return errors
    steps = int(data["action"].shape[0])
    if steps <= 0:
        errors.append("action trajectory is empty")
    for field in REQUIRED_ZARR_FIELDS:
        if int(data[field].shape[0]) != steps:
            errors.append(f"step count mismatch: {field}")
    if tuple(data["action"].shape[1:]) != (54,):
        errors.append(f"action shape is not native Wuji 54: {data['action'].shape}")
    if not bool(np.asarray(data["succeed"][-1])):
        errors.append("recorded trajectory does not end in success")
    for camera in ("rgb_front", "rgb_wrist_left", "rgb_wrist_right"):
        metrics = _frame_metrics(np.asarray(data[camera][-1]))
        if metrics["mean"] <= 5.0 or metrics["variance"] <= 15.0:
            errors.append(f"{camera} final frame is black/blank")
    if bool(root.attrs.get("object_state_writes_after_action0_allowed", True)):
        errors.append("episode permits object-state writes after action0")
    return errors


def collect_episode(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    episode_dir = Path(args.output).expanduser().resolve()
    episode_dir.mkdir(parents=True, exist_ok=True)
    env = _make_environment(args.seed, args.device, randomize=args.randomize)
    raw = env.unwrapped
    transitions: list[dict[str, Any]] = []
    try:
        observation, reset_info = _reset_and_stabilize_physical_origin(
            env,
            seed=args.seed,
        )
        controller = HS014AutoController(raw)
        for _ in range(int(args.max_steps)):
            action = controller.action()
            cameras = _camera_arrays(observation)
            passthrough = _numeric_passthrough(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            transitions.append(
                {
                    "observation": observation,
                    "next_observation": next_observation,
                    "action": action,
                    "reward": reward,
                    "terminal": bool(terminated or truncated),
                    "succeed": bool(info.get("succeed", False)),
                    "cameras": cameras,
                    "passthrough": passthrough,
                    "next_passthrough": _numeric_passthrough(next_observation),
                }
            )
            observation = next_observation
            if bool(info.get("succeed", False)):
                break
            if bool(info.get("failed", False)) or controller.failed_reason:
                break
        source_commit = _git_commit(repo_root)
        zarr_summary = _write_zarr(
            episode_dir / "replay.zarr",
            transitions,
            seed=args.seed,
            source_commit=source_commit,
            randomize=bool(args.randomize),
        )
        errors = episode_contract_errors(episode_dir)
        final_info = raw._task_info()
        errors.extend(_camera_visibility_errors(final_info))
        state_write_count = int(
            final_info.get("hs014_trajectory_state_write_count", -1)
        )
        if state_write_count != 0:
            errors.append(
                f"trajectory contains {state_write_count} task/robot state writes"
            )
        camera_metrics = {
            name: _frame_metrics(frame)
            for name, frame in _camera_arrays(observation).items()
        }
        camera_visibility = final_info.get("hs014_camera_visibility", {})
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "task_name": TASK_NAME,
            "task_complete": False,
            "mode": "automatic_action_collection",
            "source_commit": source_commit,
            "seed": int(args.seed),
            "randomize": bool(args.randomize),
            "device": str(args.device),
            "physics_backend": "physx",
            "isaac_sim_version": "5.1.0.0",
            "isaaclab_checkout": "IsaacLab-v2.3.2",
            "action_semantics": "native_wuji_54_exact_environment_actions",
            "settle_action_semantics": "hold_reset_physical_tcp_with_open_native_wuji_hands",
            "full_trajectory_fresh_replay_required": True,
            "controller_phase_history": controller.phase_history,
            "controller_failure_reason": controller.failed_reason,
            "reset_info": reset_info,
            "final_task_info": final_info,
            "camera_metrics": camera_metrics,
            "camera_visibility": camera_visibility,
            "zarr": zarr_summary,
            "zarr_identity": _tree_identity(episode_dir / "replay.zarr"),
            "trajectory_state_write_count": state_write_count,
            "collection_pass": not errors,
            "errors": errors,
        }
        (episode_dir / "collection_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
        return 0 if not errors else 2
    finally:
        _close_environment_fast(env)


def replay_episode(args: argparse.Namespace) -> int:
    episode_dir = Path(args.episode).expanduser().resolve()
    errors = episode_contract_errors(episode_dir)
    if errors:
        raise RuntimeError("Cannot replay invalid episode: " + "; ".join(errors))
    root = zarr.open_group(str(episode_dir / "replay.zarr"), mode="r")
    data = root["data"]
    actions = np.asarray(data["action"], dtype=np.float64)
    seed = int(root.attrs["episode_reset_seed"])
    recorded_randomize = bool(root.attrs.get("randomize", False))
    if bool(args.randomize) != recorded_randomize:
        raise RuntimeError("Replay --randomize must match the recorded episode")
    env = _make_environment(seed, args.device, randomize=recorded_randomize)
    raw = env.unwrapped
    replay_errors: list[str] = []
    try:
        observation, reset_info = _reset_and_stabilize_physical_origin(
            env,
            seed=seed,
        )
        final_info: dict[str, Any] = {}
        full_steps = 0
        for index, action in enumerate(actions):
            observation, _reward, terminated, truncated, final_info = env.step(action)
            full_steps = index + 1
            if (terminated or truncated) and index + 1 < len(actions):
                replay_errors.append(
                    f"replay terminated at prefix {index + 1}/{len(actions)}"
                )
                break
        if full_steps != len(actions):
            replay_errors.append("fresh replay did not consume the complete action array")
        if not bool(final_info.get("succeed", False)):
            replay_errors.append("fresh full-action replay did not reach success")
        task_info = raw._task_info()
        replay_errors.extend(_camera_visibility_errors(task_info))
        state_write_count = int(
            task_info.get("hs014_trajectory_state_write_count", -1)
        )
        if state_write_count != 0:
            replay_errors.append(
                f"fresh replay contains {state_write_count} post-reset state writes"
            )
        camera_metrics = {
            name: _frame_metrics(frame)
            for name, frame in _camera_arrays(observation).items()
        }
        for name, metrics in camera_metrics.items():
            if metrics["mean"] <= 5.0 or metrics["variance"] <= 15.0:
                replay_errors.append(f"fresh replay {name} is black/blank")
        report = {
            "schema_version": SCHEMA_VERSION,
            "task_name": TASK_NAME,
            "task_complete": not replay_errors,
            "mode": "fresh_process_full_action_replay",
            "source_episode": str(episode_dir),
            "source_zarr_identity": _tree_identity(episode_dir / "replay.zarr"),
            "seed": seed,
            "full_action_count": int(len(actions)),
            "executed_action_count": int(full_steps),
            "reset_info": reset_info,
            "final_task_info": task_info,
            "camera_visibility": task_info.get("hs014_camera_visibility", {}),
            "trajectory_state_write_count": state_write_count,
            "camera_metrics": camera_metrics,
            "replay_pass": not replay_errors,
            "errors": replay_errors,
            "timestamp_unix": time.time(),
        }
        report_path = episode_dir / "fresh_replay_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0 if not replay_errors else 2
    finally:
        _close_environment_fast(env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--output", required=True)
    collect.add_argument("--seed", type=int, default=19)
    collect.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    collect.add_argument("--device", default="cuda:0")
    collect.add_argument("--randomize", action="store_true")
    replay = subparsers.add_parser("replay")
    replay.add_argument("--episode", required=True)
    replay.add_argument("--device", default="cuda:0")
    replay.add_argument("--randomize", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "collect":
        return collect_episode(args)
    return replay_episode(args)


if __name__ == "__main__":
    raise SystemExit(main())
