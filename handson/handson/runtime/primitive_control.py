"""AI- and keyboard-facing Cartesian hand primitives for HandsOn.

The controller in this module deliberately has no Isaac Sim dependency.  It
owns palm-root and native hand-joint targets.  In the hands-only IsaacLab
runtime a finite-wrench PhysX servo follows those palm targets and finite-effort
joint drives follow the finger targets.  There is no arm or online IK, and
contacts can stop or displace the actual hand instead of being overwritten.
Network handlers only enqueue commands; callers must apply them from the
simulation thread.
"""

from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from queue import Empty, Queue
import threading
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qsl, urlsplit

import numpy as np

from ..sim.envs.hand_specs import (
    hand_action_dim,
    policy_side_to_native_hand_values,
)


_SIDES = ("right", "left")
_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_FINGER_ALIASES = {
    "1": "thumb",
    "finger1": "thumb",
    "thumb": "thumb",
    "2": "index",
    "finger2": "index",
    "index": "index",
    "index_finger": "index",
    "3": "middle",
    "finger3": "middle",
    "middle": "middle",
    "middle_finger": "middle",
    "4": "ring",
    "finger4": "ring",
    "ring": "ring",
    "ring_finger": "ring",
    "5": "pinky",
    "finger5": "pinky",
    "little": "pinky",
    "little_finger": "pinky",
    "pinky": "pinky",
}
_GRASP_ALIASES = {
    "full": "power",
    "fist": "power",
    "close": "power",
    "closed": "power",
    "power_grasp": "power",
    "ball": "spherical",
    "sphere": "spherical",
    "round": "spherical",
    "cup": "cylindrical",
    "cup_body": "cylindrical",
    "cylinder": "cylindrical",
    "bottle": "cylindrical",
    "handle": "hook",
    "handle_grasp": "hook",
    "three_finger": "tripod",
    "three_finger_pinch": "tripod",
    "precision": "pinch",
    "pinch_grasp": "pinch",
    "side_pinch": "lateral",
    "key_grasp": "lateral",
    "release": "open",
}
_GRASP_POLICIES = {
    # Values use the repository's 16-value Allegro policy order and are then
    # retargeted through hand_specs.py.  Keeping all poses in this one table
    # makes the API deterministic across right/left Wuji hands.
    "relax": (
        0.00, 0.28, 0.32, 0.24,
        0.00, 0.30, 0.34, 0.26,
        0.00, 0.32, 0.36, 0.28,
        0.42, 0.14, 0.26, 0.20,
    ),
    "power": (
        0.00, 1.20, 1.30, 1.15,
        0.00, 1.20, 1.30, 1.15,
        0.00, 1.10, 1.25, 1.10,
        0.80, 0.65, 0.95, 0.85,
    ),
    "cylindrical": (
        -0.03, 1.02, 1.34, 1.18,
        0.00, 1.08, 1.38, 1.22,
        0.04, 1.02, 1.32, 1.18,
        0.84, 0.54, 1.08, 0.96,
    ),
    "spherical": (
        -0.18, 0.86, 1.02, 0.88,
        0.00, 0.90, 1.04, 0.90,
        0.18, 0.84, 0.98, 0.84,
        0.92, 0.58, 0.88, 0.78,
    ),
    "pinch": (
        0.00, 0.95, 1.05, 0.85,
        0.00, 0.10, 0.10, 0.10,
        0.00, 0.05, 0.05, 0.05,
        0.95, 0.55, 0.95, 0.85,
    ),
    "tripod": (
        -0.05, 0.96, 1.06, 0.86,
        0.08, 0.82, 0.94, 0.76,
        0.00, 0.12, 0.12, 0.10,
        0.98, 0.56, 0.98, 0.86,
    ),
    "lateral": (
        0.00, 0.72, 0.82, 0.64,
        0.00, 0.18, 0.18, 0.14,
        0.00, 0.14, 0.14, 0.12,
        1.08, 0.18, 0.72, 0.56,
    ),
    "hook": (
        0.00, 0.62, 1.52, 1.42,
        0.00, 0.66, 1.56, 1.45,
        0.00, 0.64, 1.54, 1.44,
        0.36, 0.10, 0.20, 0.16,
    ),
    "point": (
        0.00, 0.06, 0.06, 0.05,
        0.00, 1.18, 1.34, 1.18,
        0.00, 1.16, 1.32, 1.16,
        0.72, 0.42, 0.70, 0.60,
    ),
}
_GRASP_PRESETS = ("open", *_GRASP_POLICIES.keys())
_COMMANDS = frozenset(
    {
        "move",
        "rotate",
        "set_pose",
        "grasp",
        "fingers",
        "grasp_adaptive",
        "lift_test",
        "cancel",
        "select_hand",
        "toggle_hand",
        "reset",
        "quit",
    }
)


class PrimitiveCommandError(ValueError):
    """Raised when a primitive command is malformed or unsupported."""


def _finite_vector(value: Any, size: int, *, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(size)
    except (TypeError, ValueError) as exc:
        raise PrimitiveCommandError(f"{name} must contain {size} numbers") from exc
    if not np.all(np.isfinite(vector)):
        raise PrimitiveCommandError(f"{name} must contain only finite numbers")
    return vector


def _normalize_quaternion_wxyz(value: Any) -> np.ndarray:
    quaternion = _finite_vector(value, 4, name="quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise PrimitiveCommandError("quaternion norm must be positive")
    return quaternion / norm


def _quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _axis_angle_quaternion_wxyz(axis: Any, angle_radians: float) -> np.ndarray:
    axis_vector = _finite_vector(axis, 3, name="axis")
    norm = float(np.linalg.norm(axis_vector))
    if norm <= 1e-12:
        raise PrimitiveCommandError("rotation axis norm must be positive")
    axis_vector /= norm
    half_angle = 0.5 * float(angle_radians)
    return np.concatenate(
        ([math.cos(half_angle)], axis_vector * math.sin(half_angle))
    )


def _rotate_vector_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    unit = _normalize_quaternion_wxyz(quaternion)
    xyz = unit[1:]
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + unit[0] * twice_cross + np.cross(xyz, twice_cross)


def _policy_grasp(preset: str) -> np.ndarray:
    """Return one 16-value Allegro-order grasp used for native retargeting."""

    values = _GRASP_POLICIES.get(preset)
    if values is None:
        supported = ", ".join(_GRASP_PRESETS)
        raise PrimitiveCommandError(
            f"unsupported grasp preset: {preset!r}; expected one of: {supported}"
        )
    return np.asarray(values, dtype=np.float64)


def available_grasp_presets() -> tuple[str, ...]:
    """Return stable public names accepted by the grasp primitive."""

    return tuple(_GRASP_PRESETS)


def native_hand_preset(
    hand_type: str,
    preset: str,
    *,
    side: str,
    bimanual: bool,
) -> np.ndarray:
    """Build one native hand target from a named grasp preset."""

    if side not in _SIDES:
        raise PrimitiveCommandError(f"unsupported hand side: {side!r}")
    normalized = _GRASP_ALIASES.get(str(preset).strip().lower(), str(preset).strip().lower())
    if normalized == "open":
        return np.zeros(hand_action_dim(hand_type), dtype=np.float64)
    return policy_side_to_native_hand_values(
        hand_type,
        _policy_grasp(normalized),
        side=side,
        dual_arm=bimanual,
    )


class HandPrimitiveController:
    """Maintain Cartesian palm targets and safely slewed hand presets."""

    def __init__(
        self,
        *,
        hand_type: str = "wuji1",
        bimanual: bool = False,
        workspace_lower: Sequence[float] = (-0.70, -0.80, 0.70),
        workspace_upper: Sequence[float] = (0.70, 0.80, 1.85),
        hand_slew_rate_rad_s: float = 1.5,
    ) -> None:
        self.hand_type = str(hand_type)
        self.bimanual = bool(bimanual)
        self.sides = _SIDES if self.bimanual else ("right",)
        self.selected_hand = "right"
        self.workspace_lower = _finite_vector(
            workspace_lower, 3, name="workspace_lower"
        )
        self.workspace_upper = _finite_vector(
            workspace_upper, 3, name="workspace_upper"
        )
        if np.any(self.workspace_lower >= self.workspace_upper):
            raise PrimitiveCommandError(
                "workspace_lower must be smaller than workspace_upper"
            )
        self.hand_slew_rate_rad_s = max(0.0, float(hand_slew_rate_rad_s))
        dimension = hand_action_dim(self.hand_type)
        self.poses = {
            side: np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
            for side in self.sides
        }
        self.hands = {
            side: np.zeros(dimension, dtype=np.float64) for side in self.sides
        }
        self.hand_goals = {
            side: hand.copy() for side, hand in self.hands.items()
        }
        self.grasp_presets = {side: "open" for side in self.sides}

    def _side(self, side: str | None) -> str:
        resolved = self.selected_hand if side is None else str(side).strip().lower()
        if resolved not in self.sides:
            supported = ", ".join(self.sides)
            raise PrimitiveCommandError(
                f"hand {resolved!r} is unavailable; expected one of: {supported}"
            )
        return resolved

    def sync_side(self, side: str, pose: Any, hand: Any) -> None:
        resolved = self._side(side)
        pose_vector = _finite_vector(pose, 7, name=f"{resolved}_pose")
        pose_vector[3:7] = _normalize_quaternion_wxyz(pose_vector[3:7])
        pose_vector[:3] = np.clip(
            pose_vector[:3], self.workspace_lower, self.workspace_upper
        )
        hand_vector = _finite_vector(
            hand,
            hand_action_dim(self.hand_type),
            name=f"{resolved}_hand",
        )
        self.poses[resolved] = pose_vector
        self.hands[resolved] = hand_vector
        self.hand_goals[resolved] = hand_vector.copy()
        self.grasp_presets[resolved] = "custom"

    def sync_from_env(self, env: Any) -> None:
        for side in self.sides:
            self.sync_side(
                side,
                getattr(env, f"_{side}_tcp_pose"),
                getattr(env, f"_{side}_hand"),
            )

    def select_hand(self, side: str) -> str:
        self.selected_hand = self._side(side)
        return self.selected_hand

    def toggle_hand(self) -> str:
        if self.bimanual:
            self.selected_hand = (
                "left" if self.selected_hand == "right" else "right"
            )
        return self.selected_hand

    def move(
        self,
        delta_position_m: Any,
        *,
        side: str | None = None,
        frame: str = "world",
    ) -> np.ndarray:
        resolved = self._side(side)
        delta = _finite_vector(delta_position_m, 3, name="delta_position_m")
        normalized_frame = str(frame).strip().lower()
        if normalized_frame in {"hand", "local", "tcp"}:
            delta = _rotate_vector_wxyz(self.poses[resolved][3:7], delta)
        elif normalized_frame != "world":
            raise PrimitiveCommandError(
                "move frame must be 'world' or 'hand'"
            )
        self.poses[resolved][:3] = np.clip(
            self.poses[resolved][:3] + delta,
            self.workspace_lower,
            self.workspace_upper,
        )
        return self.poses[resolved].copy()

    def rotate_axis(
        self,
        axis: Any,
        degrees: float,
        *,
        side: str | None = None,
        frame: str = "local",
    ) -> np.ndarray:
        resolved = self._side(side)
        if not math.isfinite(float(degrees)):
            raise PrimitiveCommandError("rotation degrees must be finite")
        delta = _axis_angle_quaternion_wxyz(axis, math.radians(float(degrees)))
        current = self.poses[resolved][3:7]
        normalized_frame = str(frame).strip().lower()
        if normalized_frame in {"hand", "local", "tcp"}:
            result = _quaternion_multiply_wxyz(current, delta)
        elif normalized_frame == "world":
            result = _quaternion_multiply_wxyz(delta, current)
        else:
            raise PrimitiveCommandError(
                "rotation frame must be 'world' or 'local'"
            )
        self.poses[resolved][3:7] = _normalize_quaternion_wxyz(result)
        return self.poses[resolved].copy()

    def rotate_rpy(
        self,
        rpy_degrees: Any,
        *,
        side: str | None = None,
        frame: str = "local",
    ) -> np.ndarray:
        resolved = self._side(side)
        rpy = _finite_vector(rpy_degrees, 3, name="rpy_degrees")
        for axis, degrees in zip(np.eye(3), rpy):
            if abs(float(degrees)) > 1e-12:
                self.rotate_axis(
                    axis,
                    float(degrees),
                    side=resolved,
                    frame=frame,
                )
        return self.poses[resolved].copy()

    def set_pose(self, pose: Any, *, side: str | None = None) -> np.ndarray:
        """Set an absolute bounded palm target without changing finger state."""

        resolved = self._side(side)
        value = _finite_vector(pose, 7, name=f"{resolved}_pose")
        value[:3] = np.clip(
            value[:3], self.workspace_lower, self.workspace_upper
        )
        value[3:7] = _normalize_quaternion_wxyz(value[3:7])
        self.poses[resolved] = value
        return value.copy()

    def set_grasp(self, preset: str, *, side: str | None = None) -> np.ndarray:
        resolved = self._side(side)
        normalized = _GRASP_ALIASES.get(
            str(preset).strip().lower(), str(preset).strip().lower()
        )
        target = native_hand_preset(
            self.hand_type,
            normalized,
            side=resolved,
            bimanual=self.bimanual,
        )
        self.hand_goals[resolved] = target
        self.grasp_presets[resolved] = normalized
        return target.copy()

    def set_fingers(
        self,
        curls: Mapping[str, Any] | Sequence[Any],
        *,
        side: str | None = None,
        base_preset: str = "power",
    ) -> np.ndarray:
        """Set five Wuji fingers independently using curls or four-joint values.

        Scalar values are normalized curls in ``[0, 1]`` and interpolate from
        the open hand to the requested base pose.  A four-value sequence sets
        that finger's native joints directly; the environment's physical
        joint limits and finite-effort drives remain authoritative.
        """

        resolved = self._side(side)
        dimension = hand_action_dim(self.hand_type)
        if dimension != 20:
            raise PrimitiveCommandError(
                "independent finger control currently requires a 20-joint Wuji hand"
            )
        if isinstance(curls, Mapping):
            entries = dict(curls)
        else:
            values = list(curls)
            if len(values) != len(_FINGER_NAMES):
                raise PrimitiveCommandError(
                    "finger curls must contain five values in thumb/index/middle/ring/pinky order"
                )
            entries = dict(zip(_FINGER_NAMES, values))

        normalized_base = _GRASP_ALIASES.get(
            str(base_preset).strip().lower(), str(base_preset).strip().lower()
        )
        if normalized_base == "open":
            normalized_base = "power"
        base = native_hand_preset(
            self.hand_type,
            normalized_base,
            side=resolved,
            bimanual=self.bimanual,
        )
        target = self.hand_goals[resolved].copy()
        for raw_name, raw_value in entries.items():
            name = _FINGER_ALIASES.get(str(raw_name).strip().lower())
            if name is None:
                supported = ", ".join(_FINGER_NAMES)
                raise PrimitiveCommandError(
                    f"unsupported finger {raw_name!r}; expected one of: {supported}"
                )
            index = _FINGER_NAMES.index(name)
            finger_slice = slice(index * 4, (index + 1) * 4)
            value_array = np.asarray(raw_value, dtype=np.float64)
            if value_array.ndim == 0:
                curl = float(value_array)
                if not math.isfinite(curl) or not 0.0 <= curl <= 1.0:
                    raise PrimitiveCommandError(
                        f"{name} curl must be a finite value in [0, 1]"
                    )
                target[finger_slice] = curl * base[finger_slice]
            else:
                joints = _finite_vector(
                    value_array, 4, name=f"{name}_joint_values"
                )
                # This is an input sanity envelope, not a substitute for the
                # articulation's authored joint limits.
                target[finger_slice] = np.clip(joints, -1.60, 2.10)
        self.hand_goals[resolved] = target
        self.grasp_presets[resolved] = "custom_fingers"
        return target.copy()

    def stop(self, *, side: str | None = None) -> None:
        """Freeze queued palm/finger targets at their current commanded values."""

        resolved = self._side(side)
        self.hand_goals[resolved] = self.hands[resolved].copy()

    def advance(self, dt: float) -> None:
        seconds = max(0.0, float(dt))
        max_step = self.hand_slew_rate_rad_s * seconds
        for side in self.sides:
            delta = self.hand_goals[side] - self.hands[side]
            if max_step > 0.0:
                delta = np.clip(delta, -max_step, max_step)
            self.hands[side] += delta

    def action(self) -> np.ndarray:
        if not self.bimanual:
            return np.concatenate(
                (self.poses["right"], self.hands["right"])
            ).astype(np.float32)
        return np.concatenate(
            (
                self.poses["right"],
                self.poses["left"],
                self.hands["right"],
                self.hands["left"],
            )
        ).astype(np.float32)

    def state(self) -> dict[str, Any]:
        return {
            "selected_hand": self.selected_hand,
            "bimanual": self.bimanual,
            "hands": {
                side: {
                    "tcp_pose_wxyz": self.poses[side].tolist(),
                    "grasp": self.grasp_presets[side],
                    "hand_target": self.hand_goals[side].tolist(),
                }
                for side in self.sides
            },
        }


def adaptive_grasp_strategies(object_name: str | None) -> tuple[str, ...]:
    """Return deterministic fallback grasps for a task-object name."""

    name = str(object_name or "").strip().lower()
    if any(token in name for token in ("cup", "mug", "glass", "pitcher")):
        return ("hook", "cylindrical", "lateral", "power")
    if any(token in name for token in ("peg", "pin", "key", "nail")):
        return ("tripod", "pinch", "lateral")
    if any(token in name for token in ("bread", "toast", "card", "lid")):
        return ("lateral", "tripod", "power")
    if any(token in name for token in ("banana", "handle", "tool")):
        return ("cylindrical", "hook", "power")
    if any(token in name for token in ("apple", "orange", "pear", "ball")):
        return ("spherical", "power", "tripod")
    return ("power", "cylindrical", "spherical", "tripod", "pinch")


class AdaptiveGraspExecutor:
    """Run contact-aware close, bounded lift-test, and grasp retries.

    The executor consumes only physical readback dictionaries.  It never
    writes an object pose or bypasses PhysX: the palm still moves through the
    finite-wrench servo and the fingers through their finite-effort drives.
    """

    def __init__(self) -> None:
        self._sequence = 0
        self._job: dict[str, Any] | None = None
        self._last_state: dict[str, Any] = {
            "active": False,
            "phase": "idle",
        }

    @staticmethod
    def _bounded_float(
        value: Any,
        *,
        name: str,
        lower: float,
        upper: float,
    ) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise PrimitiveCommandError(f"{name} must be numeric") from exc
        if not math.isfinite(result):
            raise PrimitiveCommandError(f"{name} must be finite")
        return float(np.clip(result, lower, upper))

    @staticmethod
    def _object_pose(
        objects: Mapping[str, Any],
        name: str,
    ) -> np.ndarray | None:
        entry = objects.get(name)
        if isinstance(entry, Mapping):
            entry = entry.get("pose_wxyz", entry.get("pose"))
        if entry is None:
            return None
        try:
            pose = np.asarray(entry, dtype=np.float64).reshape(7)
        except (TypeError, ValueError):
            return None
        return pose if np.all(np.isfinite(pose)) else None

    @classmethod
    def _nearest_object(
        cls,
        objects: Mapping[str, Any],
        palm_position: np.ndarray,
        requested: str | None,
    ) -> tuple[str | None, np.ndarray | None]:
        if requested:
            requested_pose = cls._object_pose(objects, requested)
            if requested_pose is not None:
                return requested, requested_pose
        candidates: list[tuple[float, str, np.ndarray]] = []
        for name in objects:
            pose = cls._object_pose(objects, name)
            if pose is None:
                continue
            candidates.append(
                (float(np.linalg.norm(pose[:3] - palm_position)), str(name), pose)
            )
        if not candidates:
            return None, None
        _distance, name, pose = min(candidates, key=lambda item: item[0])
        return name, pose

    @staticmethod
    def _required_contacts(strategy: str, contacts: set[str]) -> bool:
        if strategy in {"pinch", "lateral"}:
            return {"thumb", "index"}.issubset(contacts)
        if strategy == "tripod":
            return (
                {"thumb", "index"}.issubset(contacts)
                and len(contacts & {"index", "middle"}) >= 1
            )
        if strategy == "hook":
            return len(contacts & {"index", "middle", "ring", "pinky"}) >= 2
        return "thumb" in contacts and len(
            contacts & {"index", "middle", "ring", "pinky"}
        ) >= 2

    def start(
        self,
        controller: HandPrimitiveController,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        side = controller._side(payload.get("hand"))
        object_name_value = payload.get("object", payload.get("object_name"))
        object_name = (
            str(object_name_value).strip() if object_name_value else None
        )
        requested_strategy = str(
            payload.get("strategy", payload.get("type", "auto"))
        ).strip().lower()
        requested_strategy = _GRASP_ALIASES.get(
            requested_strategy, requested_strategy
        )
        if requested_strategy == "auto":
            strategies = adaptive_grasp_strategies(object_name)
        else:
            # Validate before a job becomes active.
            native_hand_preset(
                controller.hand_type,
                requested_strategy,
                side=side,
                bimanual=controller.bimanual,
            )
            strategies = (requested_strategy,)
        max_attempts = int(
            np.clip(int(payload.get("max_attempts", 3)), 1, 5)
        )
        lift_distance = self._bounded_float(
            payload.get("lift_distance_m", payload.get("z", 0.03)),
            name="lift_distance_m",
            lower=0.01,
            upper=0.08,
        )
        self._sequence += 1
        self._job = {
            "id": self._sequence,
            "mode": "adaptive_grasp",
            "side": side,
            "object": object_name,
            "requested_strategy": requested_strategy,
            "strategies": strategies,
            "strategy": strategies[0],
            "attempt": 1,
            "max_attempts": min(max_attempts, len(strategies)),
            "lift_distance_m": lift_distance,
            "phase": "closing",
            "phase_elapsed_s": 0.0,
            "contact_streaks": {name: 0 for name in _FINGER_NAMES},
            "contacts": set(),
            "baseline_object_pose": None,
            "baseline_palm_pose": None,
            "prelift_target_pose": None,
            "reason": None,
            "result": None,
        }
        controller.set_grasp(strategies[0], side=side)
        return self.state()

    def start_lift_test(
        self,
        controller: HandPrimitiveController,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        side = controller._side(payload.get("hand"))
        object_name_value = payload.get("object", payload.get("object_name"))
        object_name = (
            str(object_name_value).strip() if object_name_value else None
        )
        lift_distance = self._bounded_float(
            payload.get("lift_distance_m", payload.get("z", 0.03)),
            name="lift_distance_m",
            lower=0.01,
            upper=0.08,
        )
        self._sequence += 1
        self._job = {
            "id": self._sequence,
            "mode": "lift_test",
            "side": side,
            "object": object_name,
            "requested_strategy": controller.grasp_presets[side],
            "strategies": (controller.grasp_presets[side],),
            "strategy": controller.grasp_presets[side],
            "attempt": 1,
            "max_attempts": 1,
            "lift_distance_m": lift_distance,
            "phase": "prepare_lift",
            "phase_elapsed_s": 0.0,
            "contact_streaks": {name: 0 for name in _FINGER_NAMES},
            "contacts": set(),
            "baseline_object_pose": None,
            "baseline_palm_pose": None,
            "prelift_target_pose": None,
            "reason": None,
            "result": None,
        }
        return self.state()

    def cancel(self) -> dict[str, Any]:
        if self._job is not None:
            self._job["phase"] = "cancelled"
            self._job["result"] = "cancelled"
            self._job["reason"] = "cancel_requested"
            self._last_state = self._public_state(self._job, active=False)
            self._job = None
        return self.state()

    @staticmethod
    def _public_state(job: Mapping[str, Any], *, active: bool) -> dict[str, Any]:
        contacts = sorted(str(value) for value in job.get("contacts", set()))
        return {
            "active": bool(active),
            "id": int(job["id"]),
            "mode": str(job["mode"]),
            "side": str(job["side"]),
            "object": job.get("object"),
            "requested_strategy": str(job["requested_strategy"]),
            "strategy": str(job["strategy"]),
            "attempt": int(job["attempt"]),
            "max_attempts": int(job["max_attempts"]),
            "lift_distance_m": float(job["lift_distance_m"]),
            "phase": str(job["phase"]),
            "contacts_inferred": contacts,
            "result": job.get("result"),
            "reason": job.get("reason"),
        }

    def state(self) -> dict[str, Any]:
        if self._job is None:
            return dict(self._last_state)
        return self._public_state(self._job, active=True)

    def _finish(self, result: str, reason: str) -> None:
        assert self._job is not None
        self._job["phase"] = result
        self._job["result"] = result
        self._job["reason"] = reason
        self._last_state = self._public_state(self._job, active=False)
        self._job = None

    def _begin_lift(
        self,
        controller: HandPrimitiveController,
        physical_side: Mapping[str, Any],
        objects: Mapping[str, Any],
    ) -> None:
        assert self._job is not None
        job = self._job
        actual_pose_value = physical_side.get("actual_pose_wxyz")
        if actual_pose_value is None:
            job["reason"] = "physical_palm_feedback_unavailable"
            job["phase"] = "failed"
            return
        actual_pose = np.asarray(actual_pose_value, dtype=np.float64).reshape(7)
        object_name, object_pose = self._nearest_object(
            objects,
            actual_pose[:3],
            job.get("object"),
        )
        if object_name is None or object_pose is None:
            job["reason"] = "physical_object_feedback_unavailable"
            job["phase"] = "failed"
            return
        job["object"] = object_name
        job["baseline_object_pose"] = object_pose.copy()
        job["baseline_palm_pose"] = actual_pose.copy()
        job["prelift_target_pose"] = controller.poses[job["side"]].copy()
        controller.move(
            [0.0, 0.0, float(job["lift_distance_m"])],
            side=job["side"],
            frame="world",
        )
        job["phase"] = "lifting"
        job["phase_elapsed_s"] = 0.0

    def _retry_or_fail(
        self,
        controller: HandPrimitiveController,
        reason: str,
    ) -> None:
        assert self._job is not None
        job = self._job
        if job["mode"] != "adaptive_grasp" or job["attempt"] >= job["max_attempts"]:
            prelift = job.get("prelift_target_pose")
            if prelift is not None:
                controller.set_pose(prelift, side=job["side"])
            self._finish("failed", reason)
            return
        prelift = job.get("prelift_target_pose")
        if prelift is not None:
            controller.set_pose(prelift, side=job["side"])
        controller.set_grasp("open", side=job["side"])
        job["phase"] = "retry_open"
        job["phase_elapsed_s"] = 0.0
        job["reason"] = reason
        job["contacts"] = set()
        job["contact_streaks"] = {name: 0 for name in _FINGER_NAMES}

    def update(
        self,
        controller: HandPrimitiveController,
        *,
        dt: float,
        physical_hands: Mapping[str, Any],
        objects: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Advance the active job once using the previous PhysX frame."""

        if self._job is None:
            return self.state()
        job = self._job
        seconds = max(0.0, float(dt))
        job["phase_elapsed_s"] += seconds
        side = str(job["side"])
        physical_side = physical_hands.get(side, {})
        if not isinstance(physical_side, Mapping):
            physical_side = {}
        phase = str(job["phase"])

        if phase == "closing":
            tracking = physical_side.get("finger_tracking", {})
            if isinstance(tracking, Mapping):
                for finger_name in _FINGER_NAMES:
                    finger = tracking.get(finger_name, {})
                    likely = bool(
                        isinstance(finger, Mapping)
                        and finger.get("contact_likely", False)
                    )
                    streak = int(job["contact_streaks"][finger_name])
                    streak = streak + 1 if likely else max(0, streak - 1)
                    job["contact_streaks"][finger_name] = streak
                    if streak >= 3:
                        job["contacts"].add(finger_name)
            ready = self._required_contacts(
                str(job["strategy"]), set(job["contacts"])
            )
            if ready or job["phase_elapsed_s"] >= 1.65:
                job["phase"] = "settling"
                job["phase_elapsed_s"] = 0.0

        elif phase == "settling":
            if job["phase_elapsed_s"] >= 0.25:
                self._begin_lift(controller, physical_side, objects)

        elif phase == "prepare_lift":
            self._begin_lift(controller, physical_side, objects)

        elif phase == "lifting":
            baseline_palm = job.get("baseline_palm_pose")
            actual_pose_value = physical_side.get("actual_pose_wxyz")
            palm_rise = 0.0
            if baseline_palm is not None and actual_pose_value is not None:
                actual_pose = np.asarray(
                    actual_pose_value, dtype=np.float64
                ).reshape(7)
                palm_rise = float(actual_pose[2] - baseline_palm[2])
            if (
                palm_rise >= 0.72 * float(job["lift_distance_m"])
                or job["phase_elapsed_s"] >= 1.15
            ):
                object_pose = self._object_pose(objects, str(job["object"]))
                baseline_object = job.get("baseline_object_pose")
                if object_pose is None or baseline_object is None:
                    self._retry_or_fail(
                        controller, "physical_object_feedback_lost"
                    )
                else:
                    object_rise = float(object_pose[2] - baseline_object[2])
                    required_rise = max(
                        0.008, 0.35 * float(job["lift_distance_m"])
                    )
                    if object_rise >= required_rise:
                        self._finish("succeeded", "object_followed_physical_lift")
                    else:
                        reason = (
                            "palm_force_saturated"
                            if bool(physical_side.get("force_saturated", False))
                            and palm_rise < required_rise
                            else "object_did_not_follow_lift"
                        )
                        self._retry_or_fail(controller, reason)

        elif phase == "retry_open":
            if job["phase_elapsed_s"] >= 0.42:
                job["attempt"] += 1
                strategy_index = min(
                    int(job["attempt"]) - 1,
                    len(job["strategies"]) - 1,
                )
                job["strategy"] = job["strategies"][strategy_index]
                job["phase"] = "closing"
                job["phase_elapsed_s"] = 0.0
                job["reason"] = None
                controller.set_grasp(str(job["strategy"]), side=side)

        elif phase == "failed":
            self._finish("failed", str(job.get("reason") or "adaptive_failed"))

        return self.state()


@dataclass(frozen=True)
class PrimitiveCommand:
    sequence: int
    operation: str
    payload: dict[str, Any]


class PrimitiveCommandBroker:
    """Thread-safe bridge between HTTP callbacks and the simulation thread."""

    def __init__(self) -> None:
        self._queue: Queue[PrimitiveCommand] = Queue()
        self._lock = threading.Lock()
        self._sequence = 0
        self._state: dict[str, Any] = {"ready": False}

    def submit(
        self,
        operation: str,
        payload: Mapping[str, Any] | None = None,
    ) -> PrimitiveCommand:
        normalized = str(operation).strip().lower()
        if normalized not in _COMMANDS:
            raise PrimitiveCommandError(
                f"unsupported primitive operation: {operation!r}"
            )
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        command = PrimitiveCommand(sequence, normalized, dict(payload or {}))
        self._queue.put(command)
        return command

    def drain(self, limit: int = 128) -> list[PrimitiveCommand]:
        commands: list[PrimitiveCommand] = []
        for _ in range(max(0, int(limit))):
            try:
                commands.append(self._queue.get_nowait())
            except Empty:
                break
        return commands

    def update_state(self, state: Mapping[str, Any]) -> None:
        with self._lock:
            self._state = dict(state)

    def state(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))


def apply_primitive_command(
    controller: HandPrimitiveController,
    command: PrimitiveCommand,
) -> dict[str, Any]:
    """Apply one queued command and return control-plane effects."""

    payload = command.payload
    side = payload.get("hand")
    operation = command.operation
    effects: dict[str, Any] = {
        "reset": False,
        "quit": False,
        "adaptive_grasp": None,
        "lift_test": None,
        "cancel": False,
    }
    if operation == "move":
        if "delta_position_m" in payload:
            delta = payload["delta_position_m"]
        else:
            delta = [
                payload.get("x", 0.0),
                payload.get("y", 0.0),
                payload.get("z", 0.0),
            ]
        controller.move(
            delta,
            side=side,
            frame=str(payload.get("frame", "world")),
        )
    elif operation == "rotate":
        frame = str(payload.get("frame", "local"))
        if "rpy_degrees" in payload:
            controller.rotate_rpy(
                payload["rpy_degrees"], side=side, frame=frame
            )
        elif "clockwise_degrees" in payload:
            controller.rotate_axis(
                [0.0, 0.0, 1.0],
                -float(payload["clockwise_degrees"]),
                side=side,
                frame="local",
            )
        elif "clockwise" in payload:
            controller.rotate_axis(
                [0.0, 0.0, 1.0],
                -float(payload["clockwise"]),
                side=side,
                frame="local",
            )
        else:
            controller.rotate_axis(
                payload.get("axis", [0.0, 0.0, 1.0]),
                float(payload.get("degrees", 0.0)),
                side=side,
                frame=frame,
            )
    elif operation == "set_pose":
        pose = payload.get("pose_wxyz", payload.get("pose"))
        if pose is None:
            pose = [
                payload.get("x", 0.0),
                payload.get("y", 0.0),
                payload.get("z", 1.0),
                payload.get("qw", 1.0),
                payload.get("qx", 0.0),
                payload.get("qy", 0.0),
                payload.get("qz", 0.0),
            ]
        controller.set_pose(pose, side=side)
    elif operation == "grasp":
        controller.set_grasp(
            str(payload.get("type", payload.get("preset", "power"))),
            side=side,
        )
    elif operation == "fingers":
        curls = payload.get("curls")
        if curls is None:
            if "finger" in payload and "curl" in payload:
                curls = {str(payload["finger"]): payload["curl"]}
            else:
                curls = {
                    key: value
                    for key, value in payload.items()
                    if str(key).strip().lower() in _FINGER_ALIASES
                }
        if not curls:
            raise PrimitiveCommandError(
                "fingers requires curls or finger/curl values"
            )
        controller.set_fingers(
            curls,
            side=side,
            base_preset=str(payload.get("base_preset", "power")),
        )
    elif operation == "grasp_adaptive":
        effects["adaptive_grasp"] = dict(payload)
    elif operation == "lift_test":
        effects["lift_test"] = dict(payload)
    elif operation == "cancel":
        controller.stop(side=side)
        effects["cancel"] = True
    elif operation == "select_hand":
        controller.select_hand(str(payload.get("hand", "right")))
    elif operation == "toggle_hand":
        controller.toggle_hand()
    elif operation == "reset":
        effects["reset"] = True
    elif operation == "quit":
        effects["quit"] = True
    return effects


class _PrimitiveHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class PrimitiveHTTPService:
    """Expose the command broker as a loopback JSON HTTP service."""

    def __init__(
        self,
        broker: PrimitiveCommandBroker,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.broker = broker
        handler = self._handler_type(broker)
        self._server = _PrimitiveHTTPServer((host, int(port)), handler)
        self._thread: threading.Thread | None = None

    @staticmethod
    def _handler_type(broker: PrimitiveCommandBroker):
        class Handler(BaseHTTPRequestHandler):
            server_version = "HandsOnPrimitiveControl/2.0"

            def _reply(self, status: int, payload: Mapping[str, Any]) -> None:
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            @staticmethod
            def _query_payload(query: str) -> dict[str, Any]:
                payload: dict[str, Any] = {}
                for key, raw_value in parse_qsl(query, keep_blank_values=True):
                    value: Any = raw_value
                    normalized = raw_value.strip().lower()
                    if normalized in {"true", "false"}:
                        value = normalized == "true"
                    elif raw_value:
                        try:
                            value = float(raw_value)
                        except ValueError:
                            if raw_value[:1] in {"[", "{"}:
                                try:
                                    value = json.loads(raw_value)
                                except json.JSONDecodeError:
                                    value = raw_value
                    payload[key] = value
                return payload

            @staticmethod
            def _route(path: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
                normalized = path.rstrip("/")
                if normalized.startswith("/v1/"):
                    normalized = normalized[4:]
                else:
                    normalized = normalized.lstrip("/")
                normalized = normalized.replace("/", "_").lower()
                route_aliases = {
                    "finger": "fingers",
                    "pose": "set_pose",
                    "grasp_adaptive": "grasp_adaptive",
                }
                operation = route_aliases.get(normalized, normalized)
                distance = payload.get(
                    "distance_m",
                    payload.get("distance", payload.get("meters", 0.01)),
                )
                if operation in {"lift", "lower", "approach", "retreat"}:
                    signed_distance = abs(float(distance))
                    if operation == "lift":
                        payload["delta_position_m"] = [0.0, 0.0, signed_distance]
                        payload["frame"] = "world"
                    elif operation == "lower":
                        payload["delta_position_m"] = [0.0, 0.0, -signed_distance]
                        payload["frame"] = "world"
                    elif operation == "approach":
                        payload["delta_position_m"] = [signed_distance, 0.0, 0.0]
                        payload["frame"] = "hand"
                    else:
                        payload["delta_position_m"] = [-signed_distance, 0.0, 0.0]
                        payload["frame"] = "hand"
                    operation = "move"
                if operation in {"roll", "pitch", "yaw"}:
                    degrees = float(payload.get("degrees", payload.get("angle", 5.0)))
                    axes = {
                        "roll": [degrees, 0.0, 0.0],
                        "pitch": [0.0, degrees, 0.0],
                        "yaw": [0.0, 0.0, degrees],
                    }
                    payload["rpy_degrees"] = axes[operation]
                    payload.setdefault("frame", "local")
                    operation = "rotate"
                if operation in _GRASP_PRESETS or operation in _GRASP_ALIASES:
                    payload.setdefault("type", operation)
                    operation = "grasp"
                if operation == "pinch":
                    payload.setdefault("type", "pinch")
                    operation = "grasp"
                return operation, payload

            def _queue(self, operation: str, payload: dict[str, Any]) -> None:
                try:
                    command = broker.submit(operation, payload)
                except PrimitiveCommandError as exc:
                    self._reply(400, {"ok": False, "error": str(exc)})
                    return
                self._reply(
                    202,
                    {
                        "ok": True,
                        "status": "queued",
                        "sequence": command.sequence,
                        "operation": command.operation,
                    },
                )

            def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
                parsed = urlsplit(self.path)
                path = parsed.path.rstrip("/")
                if path in {"", "/health", "/v1/health"}:
                    self._reply(200, {"ok": True, "service": "handson-primitives"})
                    return
                if path == "/v1/state":
                    self._reply(200, broker.state())
                    return
                if path in {"/primitives", "/v1/primitives"}:
                    self._reply(
                        200,
                        {
                            "ok": True,
                            "operations": sorted(_COMMANDS),
                            "motion_shortcuts": [
                                "lift",
                                "lower",
                                "approach",
                                "retreat",
                                "roll",
                                "pitch",
                                "yaw",
                            ],
                            "grasp_presets": list(available_grasp_presets()),
                            "finger_order": list(_FINGER_NAMES),
                        },
                    )
                    return
                try:
                    payload = self._query_payload(parsed.query)
                    operation, payload = self._route(path, payload)
                except (PrimitiveCommandError, TypeError, ValueError) as exc:
                    self._reply(400, {"ok": False, "error": str(exc)})
                    return
                # Accept the compact form /grasp?full and /grasp?pinch.
                if operation == "grasp" and "type" not in payload:
                    blank_keys = [
                        key for key, value in payload.items() if value == ""
                    ]
                    if blank_keys:
                        payload["type"] = blank_keys[0]
                if operation in _COMMANDS:
                    self._queue(operation, payload)
                    return
                self._reply(404, {"ok": False, "error": "not_found"})

            def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
                parsed = urlsplit(self.path)
                path = parsed.path.rstrip("/")
                if not path.startswith("/v1/"):
                    self._reply(404, {"ok": False, "error": "not_found"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length < 0 or length > 65536:
                        raise PrimitiveCommandError("request body is too large")
                    raw = self.rfile.read(length) if length else b"{}"
                    payload = json.loads(raw.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise PrimitiveCommandError("request body must be a JSON object")
                    query_payload = self._query_payload(parsed.query)
                    query_payload.update(payload)
                    operation, payload = self._route(path, query_payload)
                except (PrimitiveCommandError, ValueError, json.JSONDecodeError) as exc:
                    self._reply(400, {"ok": False, "error": str(exc)})
                    return
                self._queue(operation, payload)

            def log_message(self, _format: str, *_args: Any) -> None:
                return None

        return Handler

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> tuple[str, int]:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="handson-primitive-http",
                daemon=True,
            )
            self._thread.start()
        return self.address

    def close(self) -> None:
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=3.0)
            self._thread = None
        self._server.server_close()


__all__ = [
    "AdaptiveGraspExecutor",
    "HandPrimitiveController",
    "PrimitiveCommand",
    "PrimitiveCommandBroker",
    "PrimitiveCommandError",
    "PrimitiveHTTPService",
    "apply_primitive_command",
    "adaptive_grasp_strategies",
    "available_grasp_presets",
    "native_hand_preset",
]
