"""Pure NumPy Plan1 command boundary for arm-free E01 runtimes.

No MuJoCo, Genesis, Isaac, or Torch import is required.  The Wuji table guard
evaluates the authored primitive MJCF with a small hinge-tree FK routine so the
same constraints are available inside the isolated Isaac Python environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import numpy as np

from .e01_detached_hand_scene import (
    E01_DETACHED_HAND_HOME_POSES,
    E01_WUJI_HAND_DIM,
    E01_WUJI_HAND_HOME,
)


E01_TABLE_TOP_Z = 0.89
E01_HAND_TABLE_CLEARANCE = 0.001
E01_PLAN1_MAX_TCP_TRANSLATION_STEP = 0.060
E01_PLAN1_MAX_TCP_ROTATION_STEP = math.radians(10.0)
E01_PLAN1_MAX_HAND_JOINT_STEP = 0.12

_ROKAE_REACH_RADIUS = 0.88
_ROKAE_REACH_MAX_POINT_DIST = 0.08
_ROKAE_REACH_CACHE_PATH = (
    Path(__file__).resolve().parents[1]
    / "envs"
    / "c01_rokae_wuji_reachability.npz"
)
_WUJI_HAND_MJCF_DIR = (
    Path(__file__).resolve().parents[1]
    / "envs"
    / "xmls"
    / "wuji_hand1"
    / "mjcf"
)
_ROKAE_ARM_ROOT_POS = {
    "right": np.asarray([-0.75, -0.065, 1.34], dtype=np.float64),
    "left": np.asarray([-0.75, 0.065, 1.34], dtype=np.float64),
}
_ROKAE_HAND_WORKSPACE_BOUNDS = {
    "right": (
        np.asarray([-0.42, -0.72, E01_TABLE_TOP_Z + 0.001]),
        np.asarray([0.18, -0.02, 1.38]),
    ),
    "left": (
        np.asarray([-0.42, 0.02, E01_TABLE_TOP_Z + 0.001]),
        np.asarray([0.18, 0.72, 1.38]),
    ),
}
_DETACHED_HAND_WORKSPACE_BOUNDS = (
    np.asarray([-0.70, -0.80, E01_TABLE_TOP_Z + 0.001], dtype=np.float64),
    np.asarray([0.70, 0.80, 1.85], dtype=np.float64),
)


def _vector(value: str | None, length: int, default: tuple[float, ...]) -> np.ndarray:
    if value is None or not value.strip():
        return np.asarray(default, dtype=np.float64)
    result = np.fromstring(value, sep=" ", dtype=np.float64)
    if result.size != length:
        raise ValueError(f"Expected {length} MJCF values, got {value!r}")
    return result


def _quat_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    w, x, y, z = quat
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    one_minus = 1.0 - cosine
    return np.asarray(
        [
            [cosine + x * x * one_minus, x * y * one_minus - z * sine, x * z * one_minus + y * sine],
            [y * x * one_minus + z * sine, cosine + y * y * one_minus, y * z * one_minus - x * sine],
            [z * x * one_minus - y * sine, z * y * one_minus + x * sine, cosine + z * z * one_minus],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class _Body:
    parent: int
    position: np.ndarray
    rotation: np.ndarray
    joint_name: str | None
    joint_position: np.ndarray
    joint_axis: np.ndarray


@dataclass(frozen=True)
class _Geom:
    body: int
    kind: str
    position: np.ndarray
    rotation: np.ndarray
    size: np.ndarray


class NumPyWujiPrimitiveCollisionGuard:
    """Exact primitive lower bound from the standalone Wuji MJCF hinge tree."""

    def __init__(
        self,
        side: str,
        primitive_mjcf: Path | None = None,
        *,
        ensure_proximal_finger_collisions: bool = False,
    ) -> None:
        if side not in {"right", "left"}:
            raise ValueError(f"Unsupported Wuji side: {side!r}")
        self.side = side
        path = (
            primitive_mjcf
            or _WUJI_HAND_MJCF_DIR / f"{side}_primitive.xml"
        ).resolve()
        root = ET.parse(path).getroot()
        palm = next(
            (
                body
                for body in root.iter("body")
                if body.attrib.get("name") == f"{side}_palm_link"
            ),
            None,
        )
        if palm is None:
            raise ValueError(f"Missing {side} palm in {path}")

        actuator_ranges: dict[str, np.ndarray] = {}
        actuator_root = root.find("actuator")
        if actuator_root is not None:
            for actuator in actuator_root:
                joint_name = actuator.attrib.get("joint")
                if joint_name and "ctrlrange" in actuator.attrib:
                    actuator_ranges[joint_name] = _vector(
                        actuator.attrib["ctrlrange"],
                        2,
                        (0.0, 0.0),
                    )

        bodies: list[_Body] = []
        geoms: list[_Geom] = []
        authored_ranges: dict[str, np.ndarray] = {}

        def visit(element: ET.Element, parent: int) -> None:
            direct_joints = element.findall("joint")
            if len(direct_joints) > 1:
                raise ValueError(
                    f"NumPy Wuji FK expects at most one joint per body in {path}"
                )
            joint = direct_joints[0] if direct_joints else None
            joint_name = None if joint is None else joint.attrib.get("name")
            if joint is not None and joint_name:
                authored_ranges[joint_name] = _vector(
                    joint.attrib.get("range"),
                    2,
                    (-math.inf, math.inf),
                )
            body_index = len(bodies)
            bodies.append(
                _Body(
                    parent=parent,
                    position=_vector(element.attrib.get("pos"), 3, (0.0, 0.0, 0.0)),
                    rotation=_quat_matrix(
                        _vector(element.attrib.get("quat"), 4, (1.0, 0.0, 0.0, 0.0))
                    ),
                    joint_name=joint_name,
                    joint_position=_vector(
                        None if joint is None else joint.attrib.get("pos"),
                        3,
                        (0.0, 0.0, 0.0),
                    ),
                    joint_axis=_vector(
                        None if joint is None else joint.attrib.get("axis"),
                        3,
                        (0.0, 0.0, 1.0),
                    ),
                )
            )
            for geom_index, geom in enumerate(element.findall("geom")):
                contype = int(geom.attrib.get("contype", "1"))
                conaffinity = int(geom.attrib.get("conaffinity", "1"))
                body_name = element.attrib.get("name", "")
                proximal_collision = bool(
                    ensure_proximal_finger_collisions
                    and geom_index == 0
                    and body_name
                    in {
                        f"{side}_finger{finger}_link1"
                        for finger in range(1, 6)
                    }
                )
                if contype == 0 and conaffinity == 0 and not proximal_collision:
                    continue
                kind = geom.attrib.get("type", "sphere")
                size = np.fromstring(geom.attrib.get("size", ""), sep=" ")
                if kind not in {"sphere", "capsule", "box", "cylinder", "ellipsoid"}:
                    raise ValueError(
                        f"Unsupported Wuji primitive geom {kind!r} in {path}"
                    )
                geoms.append(
                    _Geom(
                        body=body_index,
                        kind=kind,
                        position=_vector(
                            geom.attrib.get("pos"),
                            3,
                            (0.0, 0.0, 0.0),
                        ),
                        rotation=_quat_matrix(
                            _vector(
                                geom.attrib.get("quat"),
                                4,
                                (1.0, 0.0, 0.0, 0.0),
                            )
                        ),
                        size=np.asarray(size, dtype=np.float64),
                    )
                )
            for child in element.findall("body"):
                visit(child, body_index)

        visit(palm, -1)
        self._bodies = tuple(bodies)
        self._geoms = tuple(geoms)
        self._joint_names = tuple(
            f"{side}_finger{finger}_joint{joint}"
            for finger in range(1, 6)
            for joint in range(1, 5)
        )
        self._joint_indices = {
            name: index for index, name in enumerate(self._joint_names)
        }
        ranges = []
        for name in self._joint_names:
            if name not in authored_ranges:
                raise ValueError(f"Missing Wuji joint range {name} in {path}")
            joint_range = authored_ranges[name].copy()
            control_range = actuator_ranges.get(name)
            if control_range is not None and control_range[0] < control_range[1]:
                joint_range[0] = max(joint_range[0], control_range[0])
                joint_range[1] = min(joint_range[1], control_range[1])
            ranges.append(joint_range)
        self.joint_ranges = np.asarray(ranges, dtype=np.float64)
        if not self._geoms:
            raise ValueError(f"No collision primitives found in {path}")

    @staticmethod
    def _vertical_extent(geom: _Geom, rotation: np.ndarray) -> float:
        size = geom.size
        if geom.kind == "sphere":
            return float(size[0])
        if geom.kind == "capsule":
            return float(size[0] + abs(rotation[2, 2]) * size[1])
        if geom.kind == "box":
            return float(np.sum(np.abs(rotation[2]) * size[:3]))
        if geom.kind == "cylinder":
            axial_z = abs(float(rotation[2, 2]))
            radial_z = math.sqrt(max(0.0, 1.0 - axial_z * axial_z))
            return float(axial_z * size[1] + radial_z * size[0])
        return float(np.linalg.norm(rotation[2] * size[:3]))

    def minimum_collision_z(self, pose: np.ndarray, hand: np.ndarray) -> float:
        pose = np.asarray(pose, dtype=np.float64).reshape(7)
        hand = np.asarray(hand, dtype=np.float64).reshape(E01_WUJI_HAND_DIM)
        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(hand)):
            raise ValueError("Wuji collision FK requires finite inputs")
        hand = np.clip(hand, self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        positions: list[np.ndarray] = []
        rotations: list[np.ndarray] = []
        for body_index, body in enumerate(self._bodies):
            if body.parent < 0:
                position = pose[:3].copy()
                rotation = _quat_matrix(pose[3:7])
            else:
                parent_position = positions[body.parent]
                parent_rotation = rotations[body.parent]
                position = parent_position + parent_rotation @ body.position
                reference_rotation = parent_rotation @ body.rotation
                if body.joint_name is None:
                    rotation = reference_rotation
                else:
                    angle = hand[self._joint_indices[body.joint_name]]
                    joint_rotation = _axis_angle_matrix(body.joint_axis, angle)
                    rotation = reference_rotation @ joint_rotation
                    position += (
                        reference_rotation @ body.joint_position
                        - rotation @ body.joint_position
                    )
            positions.append(position)
            rotations.append(rotation)

        minimum = math.inf
        for geom in self._geoms:
            body_position = positions[geom.body]
            body_rotation = rotations[geom.body]
            center = body_position + body_rotation @ geom.position
            rotation = body_rotation @ geom.rotation
            minimum = min(
                minimum,
                float(center[2]) - self._vertical_extent(geom, rotation),
            )
        return float(minimum)

    def project_above_table(
        self,
        pose: np.ndarray,
        hand: np.ndarray,
        *,
        table_top_z: float = E01_TABLE_TOP_Z,
        clearance: float = E01_HAND_TABLE_CLEARANCE,
    ) -> tuple[np.ndarray, bool, float, float]:
        projected = np.asarray(pose, dtype=np.float64).reshape(7).copy()
        floor_z = float(table_top_z) + max(0.0, float(clearance))
        minimum_z = self.minimum_collision_z(projected, hand)
        signed_correction = floor_z - minimum_z
        correction = max(0.0, signed_correction)
        required_palm_z = float(projected[2] + signed_correction)
        projected[2] += correction
        return projected, correction > 1e-9, correction, required_palm_z


@lru_cache(maxsize=4)
def _cached_reach_points(path_string: str) -> dict[str, np.ndarray]:
    path = Path(path_string)
    if not path.is_file():
        return {}
    try:
        with np.load(path, allow_pickle=False) as cache:
            result: dict[str, np.ndarray] = {}
            for side in ("right", "left"):
                key = f"{side}_points"
                if key not in cache.files:
                    continue
                points = np.asarray(cache[key], dtype=np.float64)
                if points.ndim != 2 or points.shape[1] != 3:
                    continue
                lower, upper = _ROKAE_HAND_WORKSPACE_BOUNDS[side]
                root = _ROKAE_ARM_ROOT_POS[side]
                valid = np.all(np.isfinite(points), axis=1)
                valid &= np.all(points >= lower, axis=1)
                valid &= np.all(points <= upper, axis=1)
                valid &= np.linalg.norm(points - root, axis=1) <= _ROKAE_REACH_RADIUS + 1e-9
                points = np.ascontiguousarray(points[valid])
                if points.size:
                    points.setflags(write=False)
                    result[side] = points
            return result
    except (OSError, ValueError, KeyError):
        return {}


def load_rokae_reach_points(path: Path | None = None) -> dict[str, np.ndarray]:
    resolved = (path or _ROKAE_REACH_CACHE_PATH).expanduser().resolve()
    return dict(_cached_reach_points(str(resolved)))


def project_rokae_hand_position(
    side: str,
    target: np.ndarray,
    fallback: np.ndarray,
    *,
    reach_points: np.ndarray | None = None,
    minimum_z: float | None = None,
    step_origin: np.ndarray | None = None,
    max_step: float | None = None,
    allow_continuous_projection_without_reach_sample: bool = False,
) -> tuple[np.ndarray, bool, float]:
    requested = np.asarray(target, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(requested)):
        requested = np.asarray(fallback, dtype=np.float64).reshape(3).copy()
    projected = requested.copy()
    base_lower, upper = _ROKAE_HAND_WORKSPACE_BOUNDS[side]
    lower = base_lower.copy()
    if minimum_z is not None and np.isfinite(minimum_z):
        lower[2] = min(upper[2], max(lower[2], float(minimum_z)))
    root = _ROKAE_ARM_ROOT_POS[side]
    for _ in range(2):
        projected = np.clip(projected, lower, upper)
        offset = projected - root
        distance = float(np.linalg.norm(offset))
        if distance > _ROKAE_REACH_RADIUS and distance > 1e-12:
            projected = root + offset * (_ROKAE_REACH_RADIUS / distance)
    projected = np.clip(projected, lower, upper)

    if reach_points is not None:
        points = np.asarray(reach_points, dtype=np.float64).reshape(-1, 3)
        valid = np.all(np.isfinite(points), axis=1)
        valid &= np.all(points >= lower, axis=1)
        valid &= np.all(points <= upper, axis=1)
        valid &= np.linalg.norm(points - root, axis=1) <= _ROKAE_REACH_RADIUS + 1e-9
        safe_points = points[valid]
        if safe_points.size:
            delta = safe_points - projected
            nearest_index = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
            nearest_distance = float(np.linalg.norm(delta[nearest_index]))
            projected_within_step = True
            origin = None
            if step_origin is not None and max_step is not None and max_step > 0.0:
                origin = np.asarray(step_origin, dtype=np.float64).reshape(3)
                projected_within_step = bool(
                    np.linalg.norm(projected - origin) <= float(max_step) + 1e-9
                )
                within_step = np.linalg.norm(safe_points - origin, axis=1) <= float(max_step) + 1e-9
            else:
                within_step = np.ones(safe_points.shape[0], dtype=bool)
            if nearest_distance <= _ROKAE_REACH_MAX_POINT_DIST and projected_within_step:
                pass
            elif np.any(within_step):
                candidates = safe_points[within_step]
                candidate_delta = candidates - projected
                index = int(np.argmin(np.einsum("ij,ij->i", candidate_delta, candidate_delta)))
                projected = candidates[index].copy()
            elif (
                origin is not None
                and not (
                    allow_continuous_projection_without_reach_sample
                    and projected_within_step
                )
            ):
                projected = origin.copy()
    correction = float(np.linalg.norm(projected - requested))
    return projected, correction > 1e-6, correction


def limit_pose_step(
    current: np.ndarray,
    target: np.ndarray,
    *,
    max_translation: float,
    max_rotation: float,
) -> np.ndarray:
    current = np.asarray(current, dtype=np.float64).reshape(7)
    target = np.asarray(target, dtype=np.float64).reshape(7).copy()
    delta = target[:3] - current[:3]
    distance = float(np.linalg.norm(delta))
    if max_translation > 0.0 and distance > max_translation:
        target[:3] = current[:3] + delta * (max_translation / distance)
    current_quat = current[3:7] / max(float(np.linalg.norm(current[3:7])), 1e-12)
    target_norm = float(np.linalg.norm(target[3:7]))
    target_quat = target[3:7] / target_norm if target_norm > 1e-12 else current_quat.copy()
    dot = float(np.clip(np.dot(current_quat, target_quat), -1.0, 1.0))
    if dot < 0.0:
        target_quat = -target_quat
        dot = -dot
    angle = 2.0 * math.acos(min(1.0, dot))
    if max_rotation > 0.0 and angle > max_rotation:
        fraction = max_rotation / angle
        half_angle = math.acos(min(1.0, dot))
        if half_angle < 1e-8:
            blended = (1.0 - fraction) * current_quat + fraction * target_quat
        else:
            denominator = math.sin(half_angle)
            blended = (
                math.sin((1.0 - fraction) * half_angle) / denominator * current_quat
                + math.sin(fraction * half_angle) / denominator * target_quat
            )
        target_quat = blended / max(float(np.linalg.norm(blended)), 1e-12)
    target[3:7] = target_quat
    return target


class E01Plan1CommandFilter:
    """Stateful 54D reach, rate, joint-range and full-hand table boundary."""

    def __init__(
        self,
        *,
        max_tcp_translation_step: float,
        max_tcp_rotation_step: float,
        max_hand_joint_step: float = E01_PLAN1_MAX_HAND_JOINT_STEP,
        reach_cache_path: Path | None = None,
        ensure_proximal_finger_collisions: bool = False,
        enforce_arm_reach: bool = True,
    ) -> None:
        self.max_tcp_translation_step = float(max_tcp_translation_step)
        self.max_tcp_rotation_step = float(max_tcp_rotation_step)
        self.max_hand_joint_step = float(max_hand_joint_step)
        self.enforce_arm_reach = bool(enforce_arm_reach)
        self.reach_points = load_rokae_reach_points(reach_cache_path)
        self.collision_guards = {
            side: NumPyWujiPrimitiveCollisionGuard(
                side,
                ensure_proximal_finger_collisions=(
                    ensure_proximal_finger_collisions
                ),
            )
            for side in ("right", "left")
        }
        self.joint_ranges = {
            side: guard.joint_ranges.copy()
            for side, guard in self.collision_guards.items()
        }
        self.reset()

    def reset(self) -> None:
        home_hand = np.asarray(E01_WUJI_HAND_HOME, dtype=np.float64)
        self.poses = {
            side: np.asarray(pose, dtype=np.float64)
            for side, pose in E01_DETACHED_HAND_HOME_POSES.items()
        }
        self.hands = {"right": home_hand.copy(), "left": home_hand.copy()}
        self.arm_reach_guard_active = {"right": False, "left": False}
        self.arm_reach_guard_distance = {"right": 0.0, "left": 0.0}
        self.hand_table_guard_active = {"right": False, "left": False}
        self.hand_table_guard_distance = {"right": 0.0, "left": 0.0}
        self.constraint_hold_active = {"right": False, "left": False}
        self.hand_minimum_collision_z = {
            side: self.collision_guards[side].minimum_collision_z(
                self.poses[side], self.hands[side]
            )
            for side in ("right", "left")
        }

    @staticmethod
    def split_action(action: Any) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(action, Mapping):
            right = np.asarray(action.get("right", np.zeros(27)), dtype=np.float64).ravel()
            left = np.asarray(action.get("left", np.zeros(27)), dtype=np.float64).ravel()
        else:
            flat = np.asarray(action, dtype=np.float64).ravel()
            if flat.size < 54:
                raise ValueError(f"Expected a 54-value E01 action, got {flat.size}")
            right = np.concatenate([flat[:7], flat[14:34]])
            left = np.concatenate([flat[7:14], flat[34:54]])
        if right.size < 27 or left.size < 27:
            raise ValueError(
                f"Expected right/left 27-value actions, got {right.size}/{left.size}"
            )
        return right[:27], left[:27]

    def _sanitize_hand(self, side: str, requested: np.ndarray) -> np.ndarray:
        ranges = self.joint_ranges[side]
        current = self.hands[side]
        target = np.asarray(requested, dtype=np.float64).reshape(E01_WUJI_HAND_DIM)
        target = np.where(np.isfinite(target), target, current)
        target = np.clip(target, ranges[:, 0], ranges[:, 1])
        if self.max_hand_joint_step > 0.0:
            target = np.clip(
                target,
                current - self.max_hand_joint_step,
                current + self.max_hand_joint_step,
            )
        return np.clip(target, ranges[:, 0], ranges[:, 1])

    def apply(self, action: Any, *, table_top_z: float = E01_TABLE_TOP_Z) -> None:
        right, left = self.split_action(action)
        floor_z = float(table_top_z) + E01_HAND_TABLE_CLEARANCE
        for side, side_action in (("right", right), ("left", left)):
            current_pose = self.poses[side]
            current_hand = self.hands[side]
            side_action = side_action.copy()
            # Teleoperation uses an all-zero 7D pose as an explicit "this side
            # has not engaged / hold" sentinel.  Preserve finger commands but
            # never interpret that sentinel as a request to move toward world
            # zero or the nearest reach-envelope boundary.
            if np.allclose(side_action[:7], 0.0, rtol=0.0, atol=1e-12):
                side_action[:7] = current_pose
            requested_pose = side_action[:7].astype(np.float64, copy=True)
            raw_requested_position = requested_pose[:3].copy()
            if self.enforce_arm_reach:
                requested_pose[:3], reach_active, reach_distance = (
                    project_rokae_hand_position(
                        side,
                        requested_pose[:3],
                        current_pose[:3],
                        reach_points=self.reach_points.get(
                            side, np.empty((0, 3))
                        ),
                    )
                )
            else:
                lower, upper = _DETACHED_HAND_WORKSPACE_BOUNDS
                requested_pose[:3] = np.clip(
                    requested_pose[:3], lower, upper
                )
                reach_distance = float(
                    np.linalg.norm(requested_pose[:3] - raw_requested_position)
                )
                reach_active = reach_distance > 1e-6
            pose = limit_pose_step(
                current_pose,
                requested_pose,
                max_translation=self.max_tcp_translation_step,
                max_rotation=self.max_tcp_rotation_step,
            )
            hand = self._sanitize_hand(side, side_action[7:27])
            guard = self.collision_guards[side]
            previous_minimum_z = float(
                self.hand_minimum_collision_z.get(
                    side,
                    guard.minimum_collision_z(current_pose, current_hand),
                )
            )
            same_hand_shape = bool(
                np.max(np.abs(hand - current_hand)) <= 1e-9
            )
            pose, table_active, table_correction, required_palm_z = guard.project_above_table(
                pose,
                hand,
                table_top_z=table_top_z,
                clearance=E01_HAND_TABLE_CLEARANCE,
            )
            post_reach_active = False
            post_reach_distance = 0.0
            final_minimum_z = guard.minimum_collision_z(pose, hand)
            table_resolving_upward = False
            if (
                table_active
                and self.max_tcp_translation_step > 0.0
                and requested_pose[2] > current_pose[2] + 1e-9
                and pose[2] > current_pose[2] + self.max_tcp_translation_step + 1e-9
            ):
                limited_pose = limit_pose_step(
                    current_pose,
                    pose,
                    max_translation=self.max_tcp_translation_step,
                    max_rotation=self.max_tcp_rotation_step,
                )
                limited_minimum_z = guard.minimum_collision_z(limited_pose, hand)
                if limited_pose[2] > current_pose[2] + 1e-9:
                    pose = limited_pose
                    final_minimum_z = limited_minimum_z
                    table_resolving_upward = final_minimum_z < floor_z - 1e-9
            if not table_resolving_upward and self.enforce_arm_reach:
                for _ in range(2):
                    pose[:3], corrected, correction = project_rokae_hand_position(
                        side,
                        pose[:3],
                        current_pose[:3],
                        reach_points=self.reach_points.get(side, np.empty((0, 3))),
                        minimum_z=required_palm_z,
                        step_origin=current_pose[:3],
                        max_step=self.max_tcp_translation_step,
                        allow_continuous_projection_without_reach_sample=True,
                    )
                    post_reach_active = post_reach_active or corrected
                    post_reach_distance = max(post_reach_distance, correction)
                    final_minimum_z = guard.minimum_collision_z(pose, hand)
                    deficit = floor_z - final_minimum_z
                    if deficit <= 1e-9:
                        break
                    required_palm_z = pose[2] + deficit
                    table_active = True
                    table_correction += deficit

            step_distance = float(np.linalg.norm(pose[:3] - current_pose[:3]))
            step_exceeded = bool(
                self.max_tcp_translation_step > 0.0
                and step_distance > self.max_tcp_translation_step + 1e-9
            )
            unsafe_table = bool(final_minimum_z < floor_z - 1e-9)
            if (
                not table_resolving_upward
                and step_exceeded
                and table_active
                and same_hand_shape
                and pose[2] > current_pose[2] + 1e-9
            ):
                limited_pose = limit_pose_step(
                    current_pose,
                    pose,
                    max_translation=self.max_tcp_translation_step,
                    max_rotation=self.max_tcp_rotation_step,
                )
                limited_minimum_z = guard.minimum_collision_z(
                    limited_pose,
                    hand,
                )
                if (
                    limited_pose[2] > current_pose[2] + 1e-9
                    and limited_minimum_z >= previous_minimum_z - 1e-9
                ):
                    pose = limited_pose
                    final_minimum_z = limited_minimum_z
                    step_distance = float(
                        np.linalg.norm(pose[:3] - current_pose[:3])
                    )
                    step_exceeded = bool(
                        self.max_tcp_translation_step > 0.0
                        and step_distance > self.max_tcp_translation_step + 1e-9
                    )
                    unsafe_table = bool(final_minimum_z < floor_z - 1e-9)
                    table_resolving_upward = unsafe_table
            constraint_hold = bool(
                step_exceeded or (unsafe_table and not table_resolving_upward)
            )
            if constraint_hold:
                pose = current_pose.copy()
                hand = current_hand.copy()
                final_minimum_z = guard.minimum_collision_z(pose, hand)
                if final_minimum_z < floor_z - 1e-9:
                    raise RuntimeError(
                        f"Previous {side} Wuji command is below the table safety floor"
                    )
                table_active = table_active or unsafe_table

            self.arm_reach_guard_active[side] = bool(
                reach_active or post_reach_active
            )
            self.arm_reach_guard_distance[side] = max(
                float(reach_distance),
                float(post_reach_distance),
                float(np.linalg.norm(pose[:3] - raw_requested_position))
                if reach_active
                else 0.0,
            )
            self.hand_table_guard_active[side] = bool(table_active)
            self.hand_table_guard_distance[side] = float(table_correction)
            self.constraint_hold_active[side] = constraint_hold
            self.hand_minimum_collision_z[side] = float(final_minimum_z)
            self.poses[side] = pose.copy()
            self.hands[side] = hand.copy()


__all__ = [
    "E01_HAND_TABLE_CLEARANCE",
    "E01_PLAN1_MAX_HAND_JOINT_STEP",
    "E01_PLAN1_MAX_TCP_ROTATION_STEP",
    "E01_PLAN1_MAX_TCP_TRANSLATION_STEP",
    "E01Plan1CommandFilter",
    "E01_TABLE_TOP_Z",
    "NumPyWujiPrimitiveCollisionGuard",
    "limit_pose_step",
    "load_rokae_reach_points",
    "project_rokae_hand_position",
]
