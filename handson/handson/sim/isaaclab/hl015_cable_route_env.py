"""Native Isaac Sim environment for B320-HL015.

HL015 turns the existing C10 connector pair into a longer, genuinely
bimanual episode.  The cable must pass two fixture gates before either
connector is seated; the operator then completes the ordered dual insertion,
places the cover, and clears both hands for a physical stability hold.
"""

from __future__ import annotations

import math
from typing import Any

from gymnasium import spaces
import numpy as np

from .abc_collection_env import (
    HandsOnABCIsaacLabEnv,
    _pose_vector_to_matrix,
)


HL015_TASK_NAME = "bench320_hl015_bimanual_cable_route_dual_plug"
HL015_STAGE_NAMES = (
    "route_gate_a",
    "route_gate_b",
    "plug_first",
    "cable_second",
    "cover_closed",
    "hands_clear",
    "complete",
)
HL015_ROUTE_LOCAL_POINTS = np.asarray(
    [[0.0, -0.112, 0.041], [0.0, 0.112, 0.041]],
    dtype=np.float64,
)
HL015_ROUTE_DISTANCE_M = 0.085
HL015_COVER_XY_TOLERANCE_M = 0.045
HL015_COVER_Z_TOLERANCE_M = 0.025
HL015_COVER_ALIGNMENT = 0.90
HL015_HAND_OBJECT_DISTANCE_M = 0.20
HL015_HAND_CLEAR_DISTANCE_M = 0.19
HL015_HAND_CLOSED_THRESHOLD = 0.05
HL015_HAND_OPEN_THRESHOLD = 0.08
HL015_STABLE_LINEAR_SPEED_M_S = 0.06
HL015_STABLE_ANGULAR_SPEED_RAD_S = 1.5
HL015_STABLE_SECONDS = 0.50

# Interface-only previews do not have USD rigid-body views to read back.  The
# ABC base class therefore falls back to a synthetic pose for each task body.
# Do not derive those poses from Python's salted ``hash()``: a new process can
# otherwise put the cable on a route gate and make the initial stage depend on
# process randomness.  Keep this table aligned with the authored HL015 scene
# so the first observation has the same explicit, stable layout every time.
_HL015_INTERFACE_RESET_POSES = {
    "c10_charger_plug": np.asarray(
        [-0.23, -0.17, 0.920, 0.7071068, 0.0, 0.0, 0.7071068],
        dtype=np.float64,
    ),
    "c10_charging_cable": np.asarray(
        [-0.23, 0.17, 0.912, 0.7071068, 0.0, 0.0, 0.7071068],
        dtype=np.float64,
    ),
    "hl015_cover": np.asarray(
        [-0.21, 0.42, 0.948, 0.7071068, 0.0, 0.0, 0.7071068],
        dtype=np.float64,
    ),
    "c10_power_strip": np.asarray(
        [-0.39, 0.0, 0.893, 0.7071068, 0.0, 0.0, 0.7071068],
        dtype=np.float64,
    ),
}


def hl015_route_progress(
    cable_position: np.ndarray,
    waypoint_positions: np.ndarray,
    seen: tuple[bool, bool] = (False, False),
    *,
    tolerance_m: float = HL015_ROUTE_DISTANCE_M,
) -> tuple[bool, bool]:
    """Latch ordered cable-gate passage from read-only pose geometry."""

    cable = np.asarray(cable_position, dtype=np.float64).reshape(3)
    waypoints = np.asarray(waypoint_positions, dtype=np.float64).reshape(2, 3)
    flags = [bool(seen[0]), bool(seen[1])]
    if not flags[0] and float(np.linalg.norm(cable - waypoints[0])) <= tolerance_m:
        flags[0] = True
    if flags[0] and not flags[1] and float(np.linalg.norm(cable - waypoints[1])) <= tolerance_m:
        flags[1] = True
    return bool(flags[0]), bool(flags[1])


def hl015_cover_is_closed(
    cover_pose: np.ndarray,
    strip_pose: np.ndarray,
    *,
    xy_tolerance_m: float = HL015_COVER_XY_TOLERANCE_M,
    z_tolerance_m: float = HL015_COVER_Z_TOLERANCE_M,
    alignment_threshold: float = HL015_COVER_ALIGNMENT,
) -> bool:
    """Return whether the cover is seated over the strip without writing state."""

    cover = np.asarray(cover_pose, dtype=np.float64).reshape(7)
    strip = np.asarray(strip_pose, dtype=np.float64).reshape(7)
    if not np.all(np.isfinite(cover)) or not np.all(np.isfinite(strip)):
        return False
    cover_matrix = _pose_vector_to_matrix(cover)
    strip_matrix = _pose_vector_to_matrix(strip)
    target = (strip_matrix @ np.asarray([0.0, 0.0, 0.048, 1.0]))[:3]
    xy_error = float(np.linalg.norm(cover[:2] - target[:2]))
    z_error = abs(float(cover[2] - target[2]))
    rotation_alignment = float(np.trace(cover_matrix[:3, :3].T @ strip_matrix[:3, :3]) - 1.0) / 2.0
    return bool(
        xy_error <= float(xy_tolerance_m)
        and z_error <= float(z_tolerance_m)
        and rotation_alignment >= float(alignment_threshold)
    )


def hl015_stage_index(
    route_a_seen: bool,
    route_b_seen: bool,
    first_connected: bool,
    second_connected: bool,
    cover_closed: bool,
    hands_clear: bool,
    complete: bool,
) -> int:
    """Map ordered latches to the seven public stage labels."""

    if complete:
        return 6
    if not route_a_seen:
        return 0
    if not route_b_seen:
        return 1
    if not first_connected:
        return 2
    if not second_connected:
        return 3
    if not cover_closed:
        return 4
    if not hands_clear:
        return 5
    return 5


class HL015CableRouteIsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Physics-only ordered cable routing and dual-plug task."""

    def __init__(self, *args: Any, **kwargs: Any):
        self._reset_hl015_state()
        super().__init__(*args, **kwargs)

    def _fallback_pose(self, name: str) -> np.ndarray:
        """Return deterministic authored poses when no Isaac stage exists."""

        if self.task_name == HL015_TASK_NAME:
            pose = _HL015_INTERFACE_RESET_POSES.get(name)
            if pose is not None:
                return pose.copy()
        return super()._fallback_pose(name)

    def _reset_hl015_state(self) -> None:
        self._hl015_route_seen = [False, False]
        self._hl015_first_connected_seen = False
        self._hl015_second_connected_seen = False
        self._hl015_cover_closed_seen = False
        self._hl015_bimanual_seen = False
        self._hl015_order_violation = False
        self._hl015_hands_clear_seen = False
        self._hl015_stability_steps = 0
        self._hl015_stage_index = 0
        self._hl015_last_metrics: dict[str, Any] = {}

    def reset(self, *args: Any, **kwargs: Any):
        self._reset_hl015_state()
        # The parent reset owns all PhysX tensors; this hook clears only task
        # latches and never writes a body pose, velocity, or kinematic flag.
        result = super().reset(*args, **kwargs)
        # Interface-only runs have no PhysX substep to invoke the normal
        # high-frequency hook.  Refresh the read-only geometry once so the
        # very first observation/info pair exposes explicit false conditions
        # instead of an empty metrics dictionary.  Live Isaac runs already
        # refresh during the reset substep; this call is idempotent there.
        if self.task_name == HL015_TASK_NAME:
            self._update_hl015_state()
            if isinstance(result, tuple) and len(result) == 2:
                observation, info = result
                if isinstance(observation, dict):
                    state = observation.get("state")
                    if isinstance(state, dict):
                        state.update(self._hl015_observation_state())
                if isinstance(info, dict):
                    info.update(self._task_info())
                result = (observation, info)
        return result

    def _waypoint_positions(self, strip_pose: np.ndarray) -> np.ndarray:
        strip_matrix = _pose_vector_to_matrix(strip_pose)
        points = np.concatenate(
            [HL015_ROUTE_LOCAL_POINTS, np.ones((2, 1), dtype=np.float64)],
            axis=1,
        )
        return (strip_matrix @ points.T).T[:, :3]

    def _hand_position(self, side: str) -> np.ndarray:
        pose = self._left_tcp_pose if side == "left" else self._right_tcp_pose
        return np.asarray(pose, dtype=np.float64).reshape(7)[:3]

    def _hand_is_closed(self, side: str) -> bool:
        try:
            return self._hand_command_max(side) >= HL015_HAND_CLOSED_THRESHOLD
        except Exception:
            return False

    def _hand_is_open(self, side: str) -> bool:
        try:
            return self._hand_command_max(side) <= HL015_HAND_OPEN_THRESHOLD
        except Exception:
            return True

    def _body_speed_is_stable(self, name: str) -> bool:
        velocity = self._physics_body_velocity(name)
        if velocity is None:
            return True
        values = np.asarray(velocity, dtype=np.float64).reshape(-1)
        if values.size < 6 or not np.all(np.isfinite(values)):
            return False
        return bool(
            float(np.linalg.norm(values[:3])) <= HL015_STABLE_LINEAR_SPEED_M_S
            and float(np.linalg.norm(values[3:6])) <= HL015_STABLE_ANGULAR_SPEED_RAD_S
        )

    def _update_hl015_state(self) -> None:
        plug = self._current_body_pose("c10_charger_plug")
        cable = self._current_body_pose("c10_charging_cable")
        strip = self._current_body_pose("c10_power_strip")
        cover = self._current_body_pose("hl015_cover")
        if plug is None or cable is None or strip is None or cover is None:
            self._hl015_last_metrics = {
                "hl015_runtime_valid": False,
                "hl015_success_now": False,
                "hl015_stage_index": int(self._hl015_stage_index),
            }
            return

        geometry = self._c10_connection_geometry(plug, cable, strip)
        waypoints = self._waypoint_positions(strip)
        self._hl015_route_seen[:] = hl015_route_progress(
            cable[:3], waypoints, tuple(self._hl015_route_seen)
        )
        route_complete = bool(all(self._hl015_route_seen))
        first_now = bool(geometry["first_connected_now"])
        second_now = bool(geometry["second_connected_now"])
        if first_now and not route_complete:
            self._hl015_order_violation = True
        if second_now and not self._hl015_first_connected_seen:
            self._hl015_order_violation = True
        if first_now and route_complete:
            self._hl015_first_connected_seen = True
        if second_now and self._hl015_first_connected_seen:
            self._hl015_second_connected_seen = True

        left_pos = self._hand_position("left")
        right_pos = self._hand_position("right")
        left_near_strip = float(np.linalg.norm(left_pos - strip[:3])) <= HL015_HAND_OBJECT_DISTANCE_M
        right_near_cable_or_plug = min(
            float(np.linalg.norm(right_pos - cable[:3])),
            float(np.linalg.norm(right_pos - plug[:3])),
        ) <= HL015_HAND_OBJECT_DISTANCE_M
        if (
            left_near_strip
            and right_near_cable_or_plug
            and self._hand_is_closed("left")
            and self._hand_is_closed("right")
        ):
            self._hl015_bimanual_seen = True

        cover_now = bool(
            self._hl015_second_connected_seen
            and first_now
            and second_now
            and hl015_cover_is_closed(cover, strip)
        )
        if cover_now:
            self._hl015_cover_closed_seen = True

        object_points = (strip[:3], plug[:3], cable[:3], cover[:3])
        hand_distances = {
            side: min(
                float(np.linalg.norm(self._hand_position(side) - point))
                for point in object_points
            )
            for side in ("left", "right")
        }
        hands_clear_now = bool(
            all(value >= HL015_HAND_CLEAR_DISTANCE_M for value in hand_distances.values())
            and self._hand_is_open("left")
            and self._hand_is_open("right")
        )
        if hands_clear_now and self._hl015_cover_closed_seen:
            self._hl015_hands_clear_seen = True

        stable_now = bool(
            self._hl015_hands_clear_seen
            and self._hl015_bimanual_seen
            and all(
                self._body_speed_is_stable(name)
                for name in (
                    "c10_power_strip",
                    "c10_charger_plug",
                    "c10_charging_cable",
                    "hl015_cover",
                )
            )
        )
        if stable_now:
            self._hl015_stability_steps += 1
        else:
            self._hl015_stability_steps = 0
        required_stability_steps = max(
            1,
            int(math.ceil(HL015_STABLE_SECONDS / max(float(self._physics_dt), 1e-9))),
        )
        complete = bool(
            self._hl015_second_connected_seen
            and first_now
            and second_now
            and self._hl015_cover_closed_seen
            and self._hl015_hands_clear_seen
            and self._hl015_bimanual_seen
            and not self._hl015_order_violation
            and self._hl015_stability_steps >= required_stability_steps
        )
        self._hl015_stage_index = hl015_stage_index(
            *self._hl015_route_seen,
            self._hl015_first_connected_seen,
            self._hl015_second_connected_seen,
            self._hl015_cover_closed_seen,
            self._hl015_hands_clear_seen,
            complete,
        )
        self._hl015_last_metrics = {
            "hl015_runtime_valid": True,
            "hl015_stage_index": int(self._hl015_stage_index),
            "hl015_route_gate_a": bool(self._hl015_route_seen[0]),
            "hl015_route_gate_b": bool(self._hl015_route_seen[1]),
            "hl015_route_complete": route_complete,
            "hl015_first_connected": bool(self._hl015_first_connected_seen and first_now),
            "hl015_second_connected": bool(self._hl015_second_connected_seen and second_now),
            "hl015_first_connected_now": first_now,
            "hl015_second_connected_now": second_now,
            "hl015_order_valid": not self._hl015_order_violation,
            "hl015_order_violation": bool(self._hl015_order_violation),
            "hl015_cover_closed": bool(self._hl015_cover_closed_seen),
            "hl015_bimanual_seen": bool(self._hl015_bimanual_seen),
            "hl015_hands_clear": bool(self._hl015_hands_clear_seen),
            "hl015_hand_clear_distance_left_m": float(hand_distances["left"]),
            "hl015_hand_clear_distance_right_m": float(hand_distances["right"]),
            "hl015_stability_steps": int(self._hl015_stability_steps),
            "hl015_stability_steps_required": int(required_stability_steps),
            "hl015_stability_seconds": float(self._hl015_stability_steps * self._physics_dt),
            "hl015_success_now": complete,
            "hl015_waypoint_positions": waypoints.tolist(),
            "hl015_cover_target_pose": np.concatenate(
                [(
                    _pose_vector_to_matrix(strip)
                    @ np.asarray([0.0, 0.0, 0.048, 1.0])
                )[:3],
                strip[3:7],
                ],
            ).tolist(),
        }

    def _after_task_physics_substep(self) -> None:
        super()._after_task_physics_substep()
        if self.task_name == HL015_TASK_NAME:
            self._update_hl015_state()

    def _compute_success_now(self) -> bool:
        if self.task_name != HL015_TASK_NAME:
            return super()._compute_success_now()
        self._update_hl015_state()
        return bool(self._hl015_last_metrics.get("hl015_success_now", False))

    def _build_observation_space(self) -> spaces.Dict:
        base = super()._build_observation_space()
        if self.task_name != HL015_TASK_NAME:
            return base
        state_spaces = dict(base["state"].spaces)
        state_spaces.update(
            {
                "hl015_task_stage": spaces.Box(0.0, 1.0, shape=(len(HL015_STAGE_NAMES),), dtype=np.float64),
                "hl015_task_stage_index": spaces.Box(0.0, 6.0, shape=(1,), dtype=np.float64),
                "hl015_route_progress": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float64),
                "hl015_connection_state": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float64),
                "hl015_role_state": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float64),
                "hl015_stability_state": spaces.Box(0.0, np.inf, shape=(4,), dtype=np.float64),
                "hl015_waypoint_poses": spaces.Box(-np.inf, np.inf, shape=(2, 7), dtype=np.float64),
            }
        )
        result: dict[str, Any] = {"state": spaces.Dict(state_spaces)}
        if "images" in base.spaces:
            result["images"] = base["images"]
        return spaces.Dict(result)

    def _hl015_observation_state(self) -> dict[str, np.ndarray]:
        metrics = self._hl015_last_metrics
        stage = np.zeros(len(HL015_STAGE_NAMES), dtype=np.float64)
        stage[int(np.clip(self._hl015_stage_index, 0, len(stage) - 1))] = 1.0
        waypoints = np.asarray(
            metrics.get("hl015_waypoint_positions", np.zeros((2, 3))),
            dtype=np.float64,
        ).reshape(2, 3)
        waypoint_poses = np.zeros((2, 7), dtype=np.float64)
        waypoint_poses[:, :3] = waypoints
        waypoint_poses[:, 3] = 1.0
        return {
            "hl015_task_stage": stage,
            "hl015_task_stage_index": np.asarray([float(self._hl015_stage_index)], dtype=np.float64),
            "hl015_route_progress": np.asarray(
                [float(self._hl015_route_seen[0]), float(self._hl015_route_seen[1])],
                dtype=np.float64,
            ),
            "hl015_connection_state": np.asarray(
                [
                    float(metrics.get("hl015_first_connected", False)),
                    float(metrics.get("hl015_second_connected", False)),
                    float(metrics.get("hl015_order_valid", False)),
                    float(metrics.get("hl015_cover_closed", False)),
                ],
                dtype=np.float64,
            ),
            "hl015_role_state": np.asarray(
                [
                    float(metrics.get("hl015_bimanual_seen", False)),
                    float(metrics.get("hl015_hands_clear", False)),
                    float(metrics.get("hl015_order_violation", False)),
                    float(metrics.get("hl015_success_now", False)),
                ],
                dtype=np.float64,
            ),
            "hl015_stability_state": np.asarray(
                [
                    float(metrics.get("hl015_stability_steps", 0)),
                    float(metrics.get("hl015_stability_steps_required", 0)),
                    float(metrics.get("hl015_stability_seconds", 0.0)),
                    float(metrics.get("hl015_success_now", False)),
                ],
                dtype=np.float64,
            ),
            "hl015_waypoint_poses": waypoint_poses,
        }

    def _compute_observation(self) -> dict[str, Any]:
        observation = super()._compute_observation()
        if self.task_name == HL015_TASK_NAME:
            observation["state"].update(self._hl015_observation_state())
            metrics = self._hl015_last_metrics
            observation["state"]["connection_state"] = np.asarray(
                [
                    float(metrics.get("hl015_first_connected", False)),
                    float(metrics.get("hl015_second_connected", False)),
                    float(metrics.get("hl015_order_valid", False)),
                ],
                dtype=np.float64,
            )
        return observation

    def _task_info(self) -> dict[str, Any]:
        if self.task_name != HL015_TASK_NAME:
            return super()._task_info()
        info = dict(self._hl015_last_metrics)
        info.update(
            {
                "task_stage": HL015_STAGE_NAMES[
                    int(np.clip(self._hl015_stage_index, 0, len(HL015_STAGE_NAMES) - 1))
                ],
                "task_stage_index": int(self._hl015_stage_index),
                "hl015_success_counter": int(self._success_counter),
                "hl015_success_steps_required": int(self._success_steps_required()),
                "hl015_benchmark_task_id": "B320-HL015",
                "hl015_physics_only": True,
                "hl015_scripted_body_writes_forbidden": True,
            }
        )
        return info

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        if self.task_name != HL015_TASK_NAME:
            return super()._success_condition_rows()
        metrics = self._hl015_last_metrics
        return [
            self._condition_row("hl015_route_gate_a", "电缆通过第一理线框", metrics.get("hl015_route_gate_a", False)),
            self._condition_row("hl015_route_gate_b", "电缆按顺序通过第二理线框", metrics.get("hl015_route_gate_b", False)),
            self._condition_row("hl015_first_connected", "充电头先插入排插", metrics.get("hl015_first_connected", False)),
            self._condition_row("hl015_second_connected", "电缆接头再插入充电头", metrics.get("hl015_second_connected", False)),
            self._condition_row("hl015_bimanual", "左右手均完成有效分工", metrics.get("hl015_bimanual_seen", False)),
            self._condition_row("hl015_cover_closed", "扣盖压合到绿色目标框", metrics.get("hl015_cover_closed", False)),
            self._condition_row("hl015_hands_clear", "双手打开并撤离工作区", metrics.get("hl015_hands_clear", False)),
            self._condition_row("hl015_stable", "组装体稳定保持 0.5 秒", metrics.get("hl015_success_now", False)),
        ]

    def _task_status_text(self) -> str:
        if self.task_name != HL015_TASK_NAME:
            return super()._task_status_text()
        metrics = self._hl015_last_metrics
        route_count = int(bool(metrics.get("hl015_route_gate_a", False))) + int(
            bool(metrics.get("hl015_route_gate_b", False))
        )
        return (
            f"阶段 {HL015_STAGE_NAMES[int(np.clip(self._hl015_stage_index, 0, 6))]}"
            f" | 理线 {route_count}/2"
            f" | 双手分工 {'✓' if metrics.get('hl015_bimanual_seen', False) else '—'}"
            f" | 稳定 {float(metrics.get('hl015_stability_seconds', 0.0)):.2f}/0.50 s"
        )
