"""Native IsaacLab environment for Synchronized Dual-Knob Setting."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .abc_collection_env import HandsOnABCIsaacLabEnv
from .dual_knob_panel import (
    DUAL_KNOB_LEFT_BODY_PATH,
    DUAL_KNOB_LEFT_JOINT_PATH,
    DUAL_KNOB_RIGHT_BODY_PATH,
    DUAL_KNOB_RIGHT_JOINT_PATH,
    DUAL_KNOB_ROOT_PATH,
    DUAL_KNOB_TASK_NAME,
    author_dual_knob_panel,
)
from .dual_knob_semantics import (
    CONTACT_THRESHOLD_N,
    DUAL_KNOB_TARGETS,
    HOLD_FRAMES,
    DualKnobEpisodeTracker,
)


_TASK_STATE_SHAPES = {
    "knob_joint_state": (4,),
    "relation_family": (2,),
    "relation_parameters": (3,),
    "relation_targets": (2,),
    "relation_residuals": (2,),
    "knob_contact_state": (4,),
    "synchrony_state": (4,),
    "dual_knob_phase": (1,),
    "dual_knob_hold": (2,),
}
_PHASE_INDEX = {
    "reset": 0.0,
    "engage": 1.0,
    "rotate": 2.0,
    "acquire": 3.0,
    "hold": 4.0,
    "success": 5.0,
    "failed": 6.0,
}


class SynchronizedDualKnobIsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Two physical knobs, assigned contacts, relation, synchrony, and hold."""

    def __init__(self, task_name: str = DUAL_KNOB_TASK_NAME, **kwargs: Any):
        if task_name != DUAL_KNOB_TASK_NAME:
            raise ValueError(
                f"Expected task_name={DUAL_KNOB_TASK_NAME!r}; got {task_name!r}"
            )
        control_dt = kwargs.get("control_dt")
        hz = int(kwargs.get("hz", 30))
        tracker_dt = float(control_dt) if control_dt is not None else 1.0 / max(1, hz)
        self._dual_knob_target_index = 0
        self._dual_knob_target = DUAL_KNOB_TARGETS[0]
        self._dual_knob_tracker = DualKnobEpisodeTracker(
            self._dual_knob_target,
            control_dt=tracker_dt,
        )
        self._dual_knob_contact_views: dict[tuple[str, str], Any] = {}
        self._dual_knob_contact_errors: set[tuple[str, str]] = set()
        self._dual_knob_articulation_view: Any | None = None
        self._dual_knob_dof_indices: dict[str, int] = {}
        self._dual_knob_initial_positions = np.zeros(2, dtype=np.float64)
        self._dual_knob_previous_positions: np.ndarray | None = None
        self._dual_knob_previous_frame: int | None = None
        self._dual_knob_finite_difference_velocities = np.zeros(
            2, dtype=np.float64
        )
        self._dual_knob_last_metrics: dict[str, Any] = {}
        self._dual_knob_panel_origin_at_start: np.ndarray | None = None
        self._dual_knob_panel_displaced = False
        self._dual_knob_panel_displacement_m = 0.0
        self._dual_knob_failure_reported = False
        super().__init__(task_name=task_name, **kwargs)

    def _build_observation_space(self):
        from gymnasium import spaces

        observation_space = super()._build_observation_space()
        state_spaces = dict(observation_space.spaces["state"].spaces)
        for key, shape in _TASK_STATE_SHAPES.items():
            state_spaces[key] = spaces.Box(-np.inf, np.inf, shape=shape)
        observation_spaces = dict(observation_space.spaces)
        observation_spaces["state"] = spaces.Dict(state_spaces)
        return spaces.Dict(observation_spaces)

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        del sim_utils
        if self._stage is None:
            raise RuntimeError("Dual-knob stage is unavailable before asset spawn")
        author_dual_knob_panel(self._stage)
        print(
            "[HandsOn/IsaacLab] Authored self-owned dual-knob panel: "
            "rigid_bodies=3 colliders=3 revolute_joints=2",
            flush=True,
        )

    def _capture_task_scene_state(self, task_mjcf) -> None:
        """Keep panel links under their reduced-coordinate articulation owner."""

        super()._capture_task_scene_state(task_mjcf)
        panel_link_prefix = f"{DUAL_KNOB_ROOT_PATH}/"
        self._task_reset_body_states = [
            state
            for state in self._task_reset_body_states
            if not str(state[0]).startswith(panel_link_prefix)
        ]

    def _apply_task_pose_randomization(self) -> None:
        """Keep the fixed panel pose canonical; relation targets still vary."""

        self.delta_h[:] = 0.0

    def _hand_body_paths(self, side: str) -> list[str]:
        if self._stage is None:
            return []
        from pxr import Usd, UsdPhysics

        root = self._stage.GetPrimAtPath(
            f"/World/HandsOnRobots/{side.capitalize()}Robot"
        )
        if not root.IsValid():
            return []
        prefix = f"{side}_finger"
        palm = f"{side}_palm_link"
        return [
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and (prim.GetName() == palm or prim.GetName().startswith(prefix))
        ]

    def _init_task_specific_runtime(self) -> None:
        self._dual_knob_contact_views = {}
        self._dual_knob_contact_errors = set()
        if not self._robots or self._stage is None:
            return
        physics_sim_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_sim_view is None:
            raise RuntimeError("Dual-knob PhysX simulation view is unavailable")

        articulation = physics_sim_view.create_articulation_view(DUAL_KNOB_ROOT_PATH)
        if articulation is None or int(articulation.count) != 1:
            count = 0 if articulation is None else int(articulation.count)
            raise RuntimeError(
                f"Dual-knob articulation view must contain one panel; got {count}"
            )
        metatype = articulation.get_metatype(0)
        dof_indices = {
            str(name): int(index) for name, index in metatype.dof_indices.items()
        }
        expected = {"left_knob_joint", "right_knob_joint"}
        if set(dof_indices) != expected:
            raise RuntimeError(
                "Dual-knob articulation DOFs changed: "
                f"actual={sorted(dof_indices)}, expected={sorted(expected)}"
            )
        positions = articulation.get_dof_positions()
        initial = positions[0]
        if hasattr(initial, "detach"):
            initial = initial.detach().cpu().numpy()
        self._dual_knob_articulation_view = articulation
        self._dual_knob_dof_indices = dof_indices
        self._dual_knob_initial_positions = np.asarray(
            [
                initial[dof_indices["left_knob_joint"]],
                initial[dof_indices["right_knob_joint"]],
            ],
            dtype=np.float64,
        )
        self._dual_knob_previous_positions = self._dual_knob_initial_positions.copy()
        self._dual_knob_previous_frame = int(self.env_step)
        self._dual_knob_finite_difference_velocities[:] = 0.0
        root_transform = articulation.get_root_transforms()[0]
        if hasattr(root_transform, "detach"):
            root_transform = root_transform.detach().cpu().numpy()
        self._dual_knob_panel_origin_at_start = np.asarray(
            root_transform[:3], dtype=np.float64
        ).copy()

        hand_paths = {
            side: self._hand_body_paths(side) for side in ("left", "right")
        }
        knob_paths = {
            "left": DUAL_KNOB_LEFT_BODY_PATH,
            "right": DUAL_KNOB_RIGHT_BODY_PATH,
        }
        max_contacts = max(
            64, max((len(paths) for paths in hand_paths.values()), default=0) * 8
        )
        for hand_side in ("left", "right"):
            for knob_side in ("left", "right"):
                key = (hand_side, knob_side)
                paths = hand_paths[hand_side]
                try:
                    view = physics_sim_view.create_rigid_contact_view(
                        knob_paths[knob_side],
                        filter_patterns=paths,
                        max_contact_data_count=max_contacts,
                    )
                    if (
                        view is not None
                        and int(view.sensor_count) > 0
                        and int(view.filter_count) > 0
                    ):
                        self._dual_knob_contact_views[key] = view
                    else:
                        self._dual_knob_contact_errors.add(key)
                except Exception as exc:
                    self._dual_knob_contact_errors.add(key)
                    self._add_task_state_warning(
                        f"Could not bind dual-knob contact {key}: {exc}"
                    )
        if self._dual_knob_contact_errors:
            missing = ",".join(
                f"{hand}->{knob}" for hand, knob in sorted(self._dual_knob_contact_errors)
            )
            raise RuntimeError(
                f"Dual-knob contact reporting must fail closed; missing={missing}"
            )
        print(
            "[HandsOn/IsaacLab] Dual-knob runtime ready: "
            f"dofs={sorted(dof_indices)} contact_views=4 threshold={CONTACT_THRESHOLD_N:.3f}N",
            flush=True,
        )

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self.delta_h[:] = 0.0
        if self.randomize:
            self._dual_knob_target_index = int(
                self._random.randint(0, len(DUAL_KNOB_TARGETS))
            )
        else:
            self._dual_knob_target_index = 0
        self._dual_knob_target = DUAL_KNOB_TARGETS[self._dual_knob_target_index]
        self._dual_knob_tracker.reset(self._dual_knob_target)
        self._dual_knob_last_metrics = dict(self._dual_knob_tracker.metrics)
        self._dual_knob_previous_positions = None
        self._dual_knob_previous_frame = None
        self._dual_knob_finite_difference_velocities[:] = 0.0
        self._dual_knob_panel_displaced = False
        self._dual_knob_panel_displacement_m = 0.0
        self._dual_knob_failure_reported = False
        self._update_success_condition_overlay()
        print(
            "[HandsOn/KA005/TARGET] " + self._dual_knob_operator_target_text(),
            flush=True,
        )

    def _dual_knob_operator_target_text(self) -> str:
        target = self._dual_knob_target
        relation = target.label.replace("deg", "°").replace(", ", "，")
        return (
            f"蓝色左旋钮逆时针 {target.x_target_deg:g}°；"
            f"红色右旋钮顺时针 {target.y_target_deg:g}°；"
            f"关系：{relation}"
        )

    def _task_status_text(self) -> str:
        target = self._dual_knob_target
        return (
            f"Targets: BLUE LEFT CCW {target.x_target_deg:g} deg | "
            f"RED RIGHT CW {target.y_target_deg:g} deg | {target.label}"
        )

    def _task_articulation_root_is_writable(self, path: str) -> bool:
        if path == DUAL_KNOB_ROOT_PATH:
            return False
        return super()._task_articulation_root_is_writable(path)

    def _task_articulation_root_velocity_is_writable(self, path: str) -> bool:
        if path == DUAL_KNOB_ROOT_PATH:
            return False
        return super()._task_articulation_root_velocity_is_writable(path)

    def _joint_reading(self) -> tuple[float, float, float, float]:
        view = self._dual_knob_articulation_view
        if view is None:
            return 0.0, 0.0, 0.0, 0.0
        positions = view.get_dof_positions()[0]
        if hasattr(positions, "detach"):
            positions = positions.detach().cpu().numpy()
        left_index = self._dual_knob_dof_indices["left_knob_joint"]
        right_index = self._dual_knob_dof_indices["right_knob_joint"]
        current = np.asarray(
            [positions[left_index], positions[right_index]], dtype=np.float64
        )
        frame = int(self.env_step)
        previous = self._dual_knob_previous_positions
        previous_frame = self._dual_knob_previous_frame
        if previous is None or previous_frame is None or frame < previous_frame:
            self._dual_knob_finite_difference_velocities[:] = 0.0
        elif frame > previous_frame:
            elapsed = (
                (frame - previous_frame) * self._dual_knob_tracker.control_dt
            )
            self._dual_knob_finite_difference_velocities[:] = (
                current - previous
            ) / elapsed
        self._dual_knob_previous_positions = current.copy()
        self._dual_knob_previous_frame = frame
        return (
            float(current[0] - self._dual_knob_initial_positions[0]),
            float(current[1] - self._dual_knob_initial_positions[1]),
            float(self._dual_knob_finite_difference_velocities[0]),
            float(self._dual_knob_finite_difference_velocities[1]),
        )

    def _contact_force(self, hand_side: str, knob_side: str) -> float:
        return float(
            self._c04_contact_force(
                self._dual_knob_contact_views.get((hand_side, knob_side))
            )
        )

    def _advance_dual_knob_state(self) -> dict[str, Any]:
        if (
            self._dual_knob_articulation_view is not None
            and self._dual_knob_panel_origin_at_start is not None
        ):
            root_transform = self._dual_knob_articulation_view.get_root_transforms()[0]
            if hasattr(root_transform, "detach"):
                root_transform = root_transform.detach().cpu().numpy()
            root_xyz = np.asarray(root_transform[:3], dtype=np.float64)
            self._dual_knob_panel_displacement_m = (
                float(
                    np.linalg.norm(
                        root_xyz - self._dual_knob_panel_origin_at_start
                    )
                )
                if np.all(np.isfinite(root_xyz))
                else math.inf
            )
            self._dual_knob_panel_displaced = bool(
                self._dual_knob_panel_displacement_m > 0.002
            )
        q_left, q_right, qd_left, qd_right = self._joint_reading()
        metrics = self._dual_knob_tracker.advance(
            self.env_step,
            q_left_rad=q_left,
            q_right_rad=q_right,
            qd_left_rad_s=qd_left,
            qd_right_rad_s=qd_right,
            left_assigned_force_n=self._contact_force("left", "left"),
            right_assigned_force_n=self._contact_force("right", "right"),
            left_cross_force_n=self._contact_force("left", "right"),
            right_cross_force_n=self._contact_force("right", "left"),
        )
        self._dual_knob_last_metrics = metrics
        return metrics

    def _compute_success_now(self) -> bool:
        metrics = self._advance_dual_knob_state()
        return bool(metrics["success"] and not self._dual_knob_panel_displaced)

    def _compute_failure(self) -> bool:
        base_failed = bool(super()._compute_failure())
        tracker_failed = bool(self._dual_knob_tracker.failed)
        failed = bool(
            base_failed or tracker_failed or self._dual_knob_panel_displaced
        )
        if failed and not self._dual_knob_failure_reported:
            metrics = self._dual_knob_last_metrics
            reason = str(self._dual_knob_tracker.failure_reason or "")
            if not reason and self._dual_knob_panel_displaced:
                reason = "panel_displaced"
            if not reason and base_failed:
                reason = "base_task_failure"
            print(
                "[HandsOn/KA005/TERMINAL] "
                f"reason={reason or 'unknown'} "
                f"base_failed={base_failed} tracker_failed={tracker_failed} "
                f"panel_displaced={self._dual_knob_panel_displaced} "
                f"panel_displacement_m={self._dual_knob_panel_displacement_m:.6f} "
                f"q_left_deg={float(metrics.get('q_left_deg', 0.0)):.3f} "
                f"q_right_deg={float(metrics.get('q_right_deg', 0.0)):.3f} "
                f"x_deg={float(metrics.get('x_deg', 0.0)):.3f} "
                f"y_deg={float(metrics.get('y_deg', 0.0)):.3f} "
                f"left_assigned_n={float(metrics.get('left_assigned_force_n', 0.0)):.3f} "
                f"right_assigned_n={float(metrics.get('right_assigned_force_n', 0.0)):.3f} "
                f"left_cross_n={float(metrics.get('left_cross_force_n', 0.0)):.3f} "
                f"right_cross_n={float(metrics.get('right_cross_force_n', 0.0)):.3f} "
                f"cross_frames={int(metrics.get('cross_contact_frames', 0))} "
                f"forbidden_frames={int(metrics.get('forbidden_direction_frames', 0))}",
                flush=True,
            )
            self._dual_knob_failure_reported = True
        return failed

    def _task_info(self) -> dict[str, Any]:
        metrics = dict(self._dual_knob_last_metrics)
        metrics.update(
            {
                "target_index": int(self._dual_knob_target_index),
                "target_family": self._dual_knob_target.family,
                "target_label": self._dual_knob_target.label,
                "x_target_deg": float(self._dual_knob_target.x_target_deg),
                "y_target_deg": float(self._dual_knob_target.y_target_deg),
                "relation_parameters": self._dual_knob_target.parameters,
                "panel_displaced": bool(self._dual_knob_panel_displaced),
                "panel_displacement_m": float(
                    self._dual_knob_panel_displacement_m
                ),
                "contact_reporting_complete": bool(
                    len(self._dual_knob_contact_views) == 4
                    and not self._dual_knob_contact_errors
                ),
            }
        )
        return {f"dual_knob_{key}": value for key, value in metrics.items()}

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._dual_knob_last_metrics
        target = self._dual_knob_target
        rows = [
            self._condition_row("left_target", f"Blue left knob CCW to {target.x_target_deg:g} deg", bool(metrics.get("left_target_ok"))),
            self._condition_row("right_target", f"Red right knob CW to {target.y_target_deg:g} deg", bool(metrics.get("right_target_ok"))),
            self._condition_row("relation", f"Relation satisfied: {target.label}", bool(metrics.get("relation_ok"))),
            self._condition_row("assigned_contacts", "Each hand stays on its assigned knob", bool(metrics.get("assigned_contacts_ok"))),
            self._condition_row("cross_contacts", "No cross-hand knob contact", bool(metrics.get("cross_contacts_clear"))),
            self._condition_row("start_sync", "Rotation starts within 0.8 s", bool(metrics.get("start_sync_ok"))),
            self._condition_row("entry_sync", "Targets reached within 1.0 s", bool(metrics.get("entry_sync_ok"))),
            self._condition_row("stable", "Both knob speeds below 5 deg/s", bool(metrics.get("low_velocity_ok"))),
        ]
        return self._append_success_hold_row(rows)

    @staticmethod
    def _success_condition_display_label(row: dict[str, Any]) -> str:
        return str(row.get("label") or row.get("key") or "Condition")

    def _compute_observation(self) -> dict[str, Any]:
        observation = super()._compute_observation()
        metrics = self._dual_knob_last_metrics
        target = self._dual_knob_target
        start_delta = float(metrics.get("start_delta_s", math.inf))
        entry_delta = float(metrics.get("entry_delta_s", math.inf))
        observation["state"].update(
            {
                "knob_joint_state": np.asarray(
                    [
                        metrics.get("q_left_deg", 0.0),
                        metrics.get("q_right_deg", 0.0),
                        metrics.get("qd_left_deg_s", 0.0),
                        metrics.get("qd_right_deg_s", 0.0),
                    ],
                    dtype=np.float64,
                ),
                "relation_family": np.asarray(target.family_one_hot, dtype=np.float64),
                "relation_parameters": np.asarray(target.parameters, dtype=np.float64),
                "relation_targets": np.asarray(
                    [target.x_target_deg, target.y_target_deg], dtype=np.float64
                ),
                "relation_residuals": np.asarray(
                    [
                        metrics.get("relation_primary_residual", 0.0),
                        metrics.get("relation_secondary_residual", 0.0),
                    ],
                    dtype=np.float64,
                ),
                "knob_contact_state": np.asarray(
                    [
                        metrics.get("left_assigned_force_n", 0.0),
                        metrics.get("right_assigned_force_n", 0.0),
                        metrics.get("left_cross_force_n", 0.0),
                        metrics.get("right_cross_force_n", 0.0),
                    ],
                    dtype=np.float64,
                ),
                "synchrony_state": np.asarray(
                    [
                        start_delta if math.isfinite(start_delta) else -1.0,
                        entry_delta if math.isfinite(entry_delta) else -1.0,
                        float(bool(metrics.get("start_sync_ok"))),
                        float(bool(metrics.get("entry_sync_ok"))),
                    ],
                    dtype=np.float64,
                ),
                "dual_knob_phase": np.asarray(
                    [_PHASE_INDEX.get(str(metrics.get("phase", "reset")), -1.0)],
                    dtype=np.float64,
                ),
                "dual_knob_hold": np.asarray(
                    [float(metrics.get("hold_frames", 0)), float(HOLD_FRAMES)],
                    dtype=np.float64,
                ),
            }
        )
        return observation


__all__ = ["SynchronizedDualKnobIsaacLabEnv"]
