"""Continuous B01 -> B05 composition using the two original physical judges."""
from __future__ import annotations

import math
from typing import Any, Mapping

from gymnasium import spaces
import numpy as np

from ...tasks.eval_comb01_cover_pitcher_and_box import layout
from ...tasks.eval_comb01_cover_pitcher_and_box.recording import hidden_field_schema, task_state_envelope
from .abc_collection_env import (
    HandsOnABCIsaacLabEnv,
    _B05_RELEASE_STABLE_SECONDS,
    _B05_HALF_HEIGHTS,
    _B05_OBJECT_HAND_CLEARANCE_MARGIN,
    _quat_wxyz_to_matrix,
    _TABLE_TOP_Z,
)


class XY5PitcherThenBoxLidEnv(HandsOnABCIsaacLabEnv):
    def __init__(self, **kwargs):
        self._xy5_layout: dict[str, np.ndarray] = {}
        self._reset_xy5_history()
        kwargs.setdefault("control_dt", 1.0 / 30.0)
        kwargs.setdefault("physics_dt", 1.0 / 120.0)
        kwargs.setdefault("hz", 30)
        kwargs.setdefault("time_limit", 240.0)
        # The ordinary recorder rebuilds this concrete class with task_name
        # already present in its saved constructor arguments.
        kwargs["task_name"] = layout.TASK_NAME
        super().__init__(**kwargs)

    def _reset_xy5_history(self) -> None:
        self._xy5_phase = 0
        self._xy5_b05_was_moved = False
        self._xy5_b05_start_pose = np.asarray([0., 0., 0., 1., 0., 0., 0.])
        self._xy5_b01_completed_step = -1
        self._xy5_b05_completed_step = -1

    def _resolve_target_object(self, target_object):
        # Roles remain fixed throughout the episode: no stage oracle is fed
        # through target_selector or target_object_pose to the policy.
        if target_object not in (None, "", "random", "pitcher_lid"):
            raise ValueError("XY5 has a fixed ordered pair, not a selectable target")
        return "pitcher_lid"

    def _build_observation_space(self):
        result = super()._build_observation_space()
        result["xy5_hidden_state"] = spaces.Box(-np.inf, np.inf, shape=(21,), dtype=np.float64)
        result["xy5_reset_poses"] = spaces.Box(-np.inf, np.inf, shape=(4, 7), dtype=np.float64)
        return result

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        self._reset_xy5_history()
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self._xy5_layout = layout.sample_layout(
            self._random, randomize=self.randomize,
            table_top_z=_TABLE_TOP_Z + float(self.delta_h[0]),
        )
        self._lid_start_pose = self._xy5_layout["pitcher_lid"].copy()
        self._xy5_b05_start_pose = self._xy5_layout["Box_lid"].copy()
        self._target_object_name = "pitcher_lid"
        self._refresh_object_poses()

    def _fallback_pose(self, name: str) -> np.ndarray:
        if name in self._xy5_layout:
            return self._xy5_layout[name].copy()
        return super()._fallback_pose(name)

    def _customize_task_reset_pose(self, name, randomized_pose, *, applied_height):
        if name not in self._xy5_layout:
            return super()._customize_task_reset_pose(name, randomized_pose, applied_height=applied_height)
        pose = self._xy5_layout[name].copy()
        if name == "Box_lid":
            pose = layout.propped_lid_pose(self._xy5_layout["Clear_Box"], _TABLE_TOP_Z + applied_height)
        else:
            pose[2] = _TABLE_TOP_Z + applied_height + layout.support_offset(name)
        self._xy5_layout[name] = pose.copy()
        # Shared PhysX reset tensors are xyzw, public poses are wxyz.
        result = np.asarray(randomized_pose).copy()
        result[:, :3] = pose[:3]
        result[:, 3:7] = pose[[4, 5, 6, 3]]
        return result

    def _reset_hand_layout_margin(self, right_points, left_points=()):
        # Reuse B01's right-arm clearance and B05's 3-D swept box clearance.
        # These are conservative initialization checks, not a replacement for
        # final-shape queries or a change to either success predicate.
        pitcher_xy = {name: self._xy5_layout[name][:2]
                      for name in ("pitcher_lid", "019_pitcher_base")}
        b01_margin = self._b01_reset_layout_margin(pitcher_xy, tuple(right_points))
        margins = []
        for point in (*right_points, *left_points):
            for name in ("Box_lid", "Clear_Box"):
                pose = self._xy5_layout[name]
                local = _quat_wxyz_to_matrix(pose[3:]).T @ (np.asarray(point) - pose[:3])
                extent = np.r_[layout.half_extents(name), _B05_HALF_HEIGHTS[name]]
                offset = np.abs(local) - extent
                distance = float(np.linalg.norm(np.maximum(offset, 0.))) if np.any(offset > 0) else float(offset.max())
                margins.append(distance - _B05_OBJECT_HAND_CLEARANCE_MARGIN)
        b05_margin = min(margins, default=math.inf)
        return min(b01_margin, b05_margin)

    def _apply_task_pose_randomization(self):
        if self._robots:
            margin = self._reset_hand_layout_margin(
                self._reset_robot_body_points("right"),
                self._reset_robot_body_points("left"),
            )
            if not np.isfinite(margin) or margin < 0.0:
                raise RuntimeError(f"XY5 reset conflicts with robot home: clearance={margin:.6f} m")
        super()._apply_task_pose_randomization()

    def _reset_scene_state(self):
        super()._reset_scene_state()
        if not self._robots or self._direct_hand_control:
            return
        # The source controller home differs from the native palm frame.
        # Start operator deltas at the actual reset pose without moving a
        # body, changing the source joint home, or touching episode physics.
        for side, robot in self._robots.items():
            robot.update(0.0)
            matrix = self._native_end_effector_pose_matrix(side, getattr(self, f"_{side}_tcp_pose"))
            pose = self._pose_matrix_to_vector(matrix)
            if not np.isfinite(pose).all():
                raise RuntimeError(f"XY5 reset returned a nonfinite {side} palm pose")
            setattr(self, f"_{side}_tcp_pose", pose.copy())

    def _configure_lid_grasp_physics(self, sim_utils):
        first = super()._configure_lid_grasp_physics(
            sim_utils, reference_task="t04_cover_pitcher", lid_name="pitcher_lid",
        )
        second = super()._configure_lid_grasp_physics(
            sim_utils, reference_task="t03_cover_box", lid_name="Box_lid",
        )
        return first[0] + second[0], first[1] and second[1]

    def _success_steps_required(self) -> int:
        return max(15, int(math.ceil(_B05_RELEASE_STABLE_SECONDS / self._control_dt)))

    def _compute_success_now(self) -> bool:
        pitcher_lid = self._poses["pitcher_lid"]
        pitcher = self._poses["019_pitcher_base"]
        box_lid = self._poses["Box_lid"]
        box = self._poses["Clear_Box"]
        b01 = self._b01_lid_insert_metrics(pitcher_lid, pitcher, update_hand_clear_hold=True)
        self._last_b01_lid_success_metrics = b01
        b05 = self._b05_lid_fit_metrics(
            box_lid, box, start_pose=self._xy5_b05_start_pose,
            was_moved=self._xy5_b05_was_moved,
        )
        self._last_b05_lid_success_metrics = b05
        if self._xy5_phase == 0:
            if b01["b01_success_now"]:
                self._xy5_phase = 1
                self._xy5_b01_completed_step = int(self.env_step)
                # A pre-covered box cannot become an automatic B completion.
                # B must be moved/lifted and seated after this boundary.
                self._xy5_b05_start_pose = box_lid.copy()
                self._xy5_b05_was_moved = False
            return False
        self._xy5_b05_was_moved = bool(b05["b05_moved"])
        return bool(b01["b01_success_now"] and b05["b05_success_now"])

    def _compute_success(self) -> bool:
        success = super()._compute_success()
        if success:
            self._xy5_phase = 2
            if self._xy5_b05_completed_step < 0:
                self._xy5_b05_completed_step = int(self.env_step)
        return success

    def _task_info(self) -> dict[str, Any]:
        info = super()._task_info()
        info.update({
            "xy5_phase": self._xy5_phase,
            "xy5_b01_completed_step": self._xy5_b01_completed_step,
            "xy5_b05_completed_step": self._xy5_b05_completed_step,
            "xy5_final_stable_steps": self._success_counter,
            "xy5_final_stable_steps_required": self._success_steps_required(),
            **self._last_b01_lid_success_metrics,
            **self._last_b05_lid_success_metrics,
        })
        return info

    def _success_condition_rows(self):
        b01 = self._last_b01_lid_success_metrics
        b05 = self._last_b05_lid_success_metrics
        return [
            self._condition_row("xy5_b01_first", "Pitcher lid completed first", self._xy5_phase >= 1),
            self._condition_row("xy5_b01_kept", "Pitcher lid kept seated/released", bool(b01.get("b01_success_now", False))),
            self._condition_row("xy5_b05_moved", "Box lid moved afterwards", self._xy5_b05_was_moved),
            self._condition_row("xy5_b05_covered", "Box lid seated; box stable", all(b05.get(k, False) for k in (
                "b05_full_coverage", "b05_height_close", "b05_top_aligned", "b05_base_stable", "b05_base_upright", "b05_base_on_table",
            ))),
            self._condition_row("xy5_b05_released", "Hand clear of box and lid", bool(b05.get("b05_hand_released", False))),
            self._condition_row("xy5_final_hold", "Both lids stable for 0.5 s", self._success_counter >= self._success_steps_required()),
        ]

    def _compute_observation(self):
        obs = super()._compute_observation()
        start_pose = self._lid_start_pose if self._lid_start_pose is not None else self._fallback_pose("pitcher_lid")
        obs["xy5_hidden_state"] = np.asarray([
            self._xy5_phase, self._lid_was_moved, self._b01_hand_lid_clear_steps,
            self._xy5_b05_was_moved, self._success_counter,
            self._xy5_b01_completed_step, self._xy5_b05_completed_step,
            *start_pose, *self._xy5_b05_start_pose,
        ], dtype=np.float64)
        obs["xy5_reset_poses"] = np.stack([self._fallback_pose(name) for name in layout.BODY_NAMES])
        return obs

    def get_recording_schema(self):
        schema = super().get_recording_schema()
        if schema is None:
            return None
        result = dict(schema)
        result["fields"] = {**schema["fields"], "xy5_hidden_state": hidden_field_schema()}
        result["task_state"] = task_state_envelope()
        return result

    def restore_recorded_initial_state(self, state_parts: Mapping[str, np.ndarray]):
        result = super().restore_recorded_initial_state(state_parts)
        # Shared restorers restore four physical bodies from fixed object_poses
        # and receptacle_pose. These fields restore only the judge's history.
        if "xy5_hidden_state" in state_parts:
            hidden = np.asarray(state_parts["xy5_hidden_state"], dtype=np.float64)
            if hidden.shape != (21,) or not np.all(np.isfinite(hidden)):
                raise ValueError("XY5 replay requires a finite 21-value hidden state")
            self._xy5_phase = int(hidden[0])
            self._lid_was_moved = bool(hidden[1])
            self._b01_hand_lid_clear_steps = int(hidden[2])
            self._xy5_b05_was_moved = bool(hidden[3])
            self._success_counter = int(hidden[4])
            self._xy5_b01_completed_step = int(hidden[5])
            self._xy5_b05_completed_step = int(hidden[6])
            self._lid_start_pose = hidden[7:14].copy()
            self._xy5_b05_start_pose = hidden[14:21].copy()
        if "xy5_reset_poses" in state_parts:
            poses = np.asarray(state_parts["xy5_reset_poses"], dtype=np.float64)
            if poses.shape != (4, 7) or not np.all(np.isfinite(poses)):
                raise ValueError("XY5 replay requires all four reset poses")
            self._xy5_layout = dict(zip(layout.BODY_NAMES, poses.copy()))
        self._target_object_name = "pitcher_lid"
        return result
