from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from handson.sim.isaaclab.recording_origin import (
    JOINT_VELOCITY_LIMIT_ENV,
    RecordingOriginStabilityError,
    STABLE_ORIGIN_ENV,
    STABLE_ORIGIN_RESET_ATTEMPTS_ENV,
    VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS,
    VELOCITY_CLAMP_SCOPE_GATE_ONLY,
    VELOCITY_CLAMP_SCOPE_NONE,
    _contact_penetration_summary,
    _task_pose_delta,
    is_abc_style_collection_task,
    stable_recording_origin_kwargs_for_task,
    stable_recording_origin_kwargs_from_report,
    stable_recording_origin_reset_attempts_for_task,
    stabilize_recording_origin,
)
from handson.sim.isaaclab.bimanual_assembly_env import (
    BimanualAssemblyIsaacLabEnv,
)
from handson.sim.isaaclab.e01_detached_hand_env import (
    HandsOnE01DetachedHandsIsaacLabEnv,
)


class _Tensor:
    def __init__(self, value) -> None:
        self._value = np.asarray(value, dtype=np.float64)

    @property
    def shape(self):
        return self._value.shape

    def clone(self):
        return _Tensor(self._value.copy())

    def zero_(self):
        self._value[...] = 0.0
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._value.copy()

    def __getitem__(self, item):
        return _Tensor(self._value[item])


class _Robot:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            joint_pos=_Tensor([[0.2, 0.4]]),
            joint_vel=_Tensor([[0.03, -0.04]]),
            default_joint_pos=_Tensor([[0.1, 0.25]]),
            # Deliberately nonzero: the reset contract must create exact zero
            # instead of trusting an asset's default velocity buffer.
            default_joint_vel=_Tensor([[5.0, -2.0]]),
        )
        self.targets: list[np.ndarray] = []
        self.velocity_targets: list[np.ndarray] = []
        self.written_joint_states: list[tuple[np.ndarray, np.ndarray]] = []
        self.write_count = 0
        self.update_count = 0

    def write_joint_state_to_sim(
        self,
        position: _Tensor,
        velocity: _Tensor,
    ) -> None:
        self.data.joint_pos = position.clone()
        self.data.joint_vel = velocity.clone()
        self.written_joint_states.append((position.numpy(), velocity.numpy()))

    def set_joint_position_target(self, target: _Tensor) -> None:
        self.targets.append(target.numpy())

    def set_joint_velocity_target(self, target: _Tensor) -> None:
        self.velocity_targets.append(target.numpy())

    def write_data_to_sim(self) -> None:
        self.write_count += 1

    def update(self, _dt: float) -> None:
        self.update_count += 1


class _Controller:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


def _snapshot(
    *,
    robot_joint_velocity: float,
    task_linear_velocity: float = 0.0,
    task_angular_velocity: float = 0.0,
    task_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    robot_joint_position: tuple[float, float] = (0.2, 0.4),
    robot_joint_target: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    if robot_joint_target is None:
        robot_joint_target = robot_joint_position
    robot_root = np.zeros((1, 13), dtype=np.float64)
    robot_root[:, 3] = 1.0
    task_body = np.zeros((1, 13), dtype=np.float64)
    task_body[:, :3] = task_position
    task_body[:, 3] = 1.0
    task_body[:, 7] = task_linear_velocity
    task_body[:, 10] = task_angular_velocity
    return {
        "physx_robot_root_state": robot_root,
        "physx_robot_root_valid": np.ones(1, dtype=np.uint8),
        "physx_tcp_state": robot_root.copy(),
        "physx_tcp_valid": np.ones(1, dtype=np.uint8),
        "physx_robot_joint_velocity": np.asarray(
            [[robot_joint_velocity, robot_joint_velocity * 0.5]],
            dtype=np.float64,
        ),
        "physx_robot_joint_position": np.asarray(
            [robot_joint_position],
            dtype=np.float64,
        ),
        "physx_robot_joint_target": np.asarray(
            [robot_joint_target],
            dtype=np.float64,
        ),
        "physx_robot_joint_valid": np.ones((1, 2), dtype=np.uint8),
        "physx_robot_joint_target_valid": np.ones(
            (1, 2),
            dtype=np.uint8,
        ),
        "physx_task_rigid_body_state": task_body,
        "physx_task_rigid_body_valid": np.ones(1, dtype=np.uint8),
        "physx_task_articulation_root_state": np.zeros(
            (0, 13),
            dtype=np.float64,
        ),
        "physx_task_articulation_root_valid": np.zeros(0, dtype=np.uint8),
        "physx_task_articulation_dof_velocity": np.zeros(
            (0, 0),
            dtype=np.float64,
        ),
        "physx_task_articulation_dof_position": np.zeros(
            (0, 0),
            dtype=np.float64,
        ),
        "physx_task_articulation_dof_valid": np.zeros(
            (0, 0),
            dtype=np.uint8,
        ),
        "physx_deformable_position": np.zeros((0, 3), dtype=np.float64),
        "physx_deformable_velocity": np.zeros((0, 3), dtype=np.float64),
        "physx_deformable_valid": np.zeros(0, dtype=np.uint8),
        "physx_command_hand": np.zeros((1, 2), dtype=np.float64),
        "physx_command_hand_valid": np.ones((1, 2), dtype=np.uint8),
        "physx_contact_report_valid": np.ones(1, dtype=np.uint8),
        "physx_contact_capture_source": np.asarray([2], dtype=np.uint8),
        "physx_contact_header_count": np.zeros(1, dtype=np.uint32),
        "physx_contact_header_ids": np.zeros((0, 5), dtype=np.uint64),
        "physx_contact_header_meta": np.zeros((0, 7), dtype=np.uint32),
        "physx_contact_point_count": np.zeros(1, dtype=np.uint32),
        "physx_contact_point_data": np.zeros((0, 10), dtype=np.float32),
        "physx_contact_point_face_indices": np.zeros(
            (0, 2), dtype=np.uint32
        ),
        "physx_contact_point_material_ids": np.zeros(
            (0, 2), dtype=np.uint64
        ),
        "physx_friction_anchor_count": np.zeros(1, dtype=np.uint32),
        "physx_friction_anchor_data": np.zeros(
            (0, 6), dtype=np.float32
        ),
        "physx_contact_signature_valid": np.ones(1, dtype=np.uint8),
        "physx_contact_signature_pair_count": np.zeros(
            1, dtype=np.uint32
        ),
        "physx_contact_signature_pair_path_indices": np.zeros(
            (0, 4), dtype=np.uint32
        ),
        "physx_contact_signature_pair_counts": np.zeros(
            (0, 2), dtype=np.uint32
        ),
        "physx_contact_signature_pair_metrics": np.zeros(
            (0, 3), dtype=np.float64
        ),
    }


def _set_single_contact(
    snapshot: dict[str, np.ndarray],
    *,
    separation: float,
    impulse: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    snapshot["physx_contact_header_count"] = np.ones(1, dtype=np.uint32)
    snapshot["physx_contact_header_ids"] = np.asarray(
        [[1, 2, 1, 2, 0]], dtype=np.uint64
    )
    snapshot["physx_contact_header_meta"] = np.asarray(
        [[1, 0, 1, 0, 0, 0, 0]], dtype=np.uint32
    )
    snapshot["physx_contact_point_count"] = np.ones(1, dtype=np.uint32)
    snapshot["physx_contact_point_data"] = np.zeros(
        (1, 10), dtype=np.float32
    )
    snapshot["physx_contact_point_data"][0, 5] = 1.0
    snapshot["physx_contact_point_data"][0, 6:9] = impulse
    snapshot["physx_contact_point_data"][0, 9] = separation
    snapshot["physx_contact_point_face_indices"] = np.zeros(
        (1, 2), dtype=np.uint32
    )
    snapshot["physx_contact_point_material_ids"] = np.zeros(
        (1, 2), dtype=np.uint64
    )
    snapshot["physx_contact_signature_pair_count"] = np.ones(
        1, dtype=np.uint32
    )
    snapshot["physx_contact_signature_pair_path_indices"] = np.asarray(
        [[1, 1, 2, 2]], dtype=np.uint32
    )
    snapshot["physx_contact_signature_pair_counts"] = np.asarray(
        [[1, 0]], dtype=np.uint32
    )
    stored_separation = float(snapshot["physx_contact_point_data"][0, 9])
    stored_impulse_z = float(snapshot["physx_contact_point_data"][0, 8])
    snapshot["physx_contact_signature_pair_metrics"] = np.asarray(
        [[max(0.0, -stored_separation), abs(stored_impulse_z), 0.0]],
        dtype=np.float64,
    )


def test_rigid_contact_zero_padding_matches_canonical_empty_signature() -> None:
    snapshot = _snapshot(robot_joint_velocity=0.0)
    _set_single_contact(snapshot, separation=0.0)
    snapshot["physx_contact_point_data"].fill(0.0)
    snapshot["physx_contact_signature_pair_count"].fill(0)
    snapshot["physx_contact_signature_pair_path_indices"] = np.zeros(
        (0, 4), dtype=np.uint32
    )
    snapshot["physx_contact_signature_pair_counts"] = np.zeros(
        (0, 2), dtype=np.uint32
    )
    snapshot["physx_contact_signature_pair_metrics"] = np.zeros(
        (0, 3), dtype=np.float64
    )

    summary = _contact_penetration_summary(
        snapshot,
        _Trace([snapshot]).schema(),
    )

    assert summary["valid"] is True
    assert summary["contact_point_count"] == 1
    assert summary["maximum_penetration_depth_m"] == 0.0
    assert summary["canonical_contact_signature"] == {
        "pair_count": 0,
        "pair_path_indices": [],
        "pair_counts": [],
        "pair_metrics": [],
    }


def test_rigid_contact_partial_zero_normal_row_still_fails_closed() -> None:
    snapshot = _snapshot(robot_joint_velocity=0.0)
    _set_single_contact(snapshot, separation=0.0)
    snapshot["physx_contact_point_data"][0, :].fill(0.0)
    snapshot["physx_contact_point_data"][0, 0] = 0.25

    summary = _contact_penetration_summary(
        snapshot,
        _Trace([snapshot]).schema(),
    )

    assert summary["valid"] is False
    assert summary["error"].startswith(
        "contact_normal_invalid:point=0,actors=(/World/A,/World/B),"
        "colliders=(/World/A,/World/B),impulse=[0.0, 0.0, 0.0],"
        "separation=0"
    )


def test_canonical_contact_metrics_accept_four_ulp_zero_boundary() -> None:
    snapshot = _snapshot(robot_joint_velocity=0.0)
    _set_single_contact(snapshot, separation=0.0, impulse=(0.0, 0.0, 1.0))
    snapshot["physx_contact_signature_pair_metrics"][0, 2] += (
        4.0 * np.spacing(0.0)
    )

    summary = _contact_penetration_summary(
        snapshot,
        _Trace([snapshot]).schema(),
    )

    assert summary["valid"] is True
    assert summary["error"] is None


def test_canonical_contact_metrics_reject_five_ulp_zero_delta() -> None:
    snapshot = _snapshot(robot_joint_velocity=0.0)
    _set_single_contact(snapshot, separation=0.0, impulse=(0.0, 0.0, 1.0))
    snapshot["physx_contact_signature_pair_metrics"][0, 2] += (
        5.0 * np.spacing(0.0)
    )

    summary = _contact_penetration_summary(
        snapshot,
        _Trace([snapshot]).schema(),
    )

    assert summary["valid"] is False
    assert summary["error"].startswith(
        "canonical_contact_metrics_mismatch:max_abs="
    )
    assert ",recomputed=" in summary["error"]
    assert ",absolute_difference=" in summary["error"]
    assert ",ulp_tolerance=" in summary["error"]


def test_canonical_contact_metrics_apply_four_ulp_tolerance() -> None:
    eps = np.finfo(np.float64).eps
    snapshot = _snapshot(robot_joint_velocity=0.0)
    _set_single_contact(snapshot, separation=0.0, impulse=(0.0, 0.0, 1.0))
    snapshot["physx_contact_signature_pair_metrics"][0, 1] += 4.0 * eps

    accepted = _contact_penetration_summary(
        snapshot,
        _Trace([snapshot]).schema(),
    )

    assert accepted["valid"] is True
    snapshot["physx_contact_signature_pair_metrics"][0, 1] += eps
    rejected = _contact_penetration_summary(
        snapshot,
        _Trace([snapshot]).schema(),
    )
    assert rejected["valid"] is False
    assert rejected["error"].startswith(
        "canonical_contact_metrics_mismatch:max_abs="
    )


@pytest.mark.parametrize(
    ("field", "column", "expected_error"),
    (
        (
            "physx_contact_signature_pair_path_indices",
            0,
            "canonical_contact_paths_mismatch",
        ),
        (
            "physx_contact_signature_pair_counts",
            0,
            "canonical_contact_counts_mismatch",
        ),
    ),
)
def test_canonical_contact_nonmetric_gates_remain_exact(
    field: str,
    column: int,
    expected_error: str,
) -> None:
    snapshot = _snapshot(robot_joint_velocity=0.0)
    _set_single_contact(snapshot, separation=0.0, impulse=(0.0, 0.0, 1.0))
    snapshot[field][0, column] += 1

    summary = _contact_penetration_summary(
        snapshot,
        _Trace([snapshot]).schema(),
    )

    assert summary["valid"] is False
    assert summary["error"] == expected_error


class _Trace:
    def __init__(self, snapshots: list[dict[str, np.ndarray]]) -> None:
        self._snapshots = snapshots
        self._snapshot_index = 0
        self.robots = [
            SimpleNamespace(
                side="right",
                joint_names=["right_j0", "right_j1"],
            )
        ]
        self.rigid_bodies = [
            SimpleNamespace(name="/World/Task/object/object")
        ]
        self.articulations = []
        self.restored_snapshot: dict[str, np.ndarray] | None = None
        self.restore_count = 0
        self.restore_sync_values: list[bool] = []
        self.boundary_write_count = 0
        self.boundary_sync_values: list[bool] = []
        self.velocity_clamp_count = 0
        self.sleep_count = 0
        self.substep_capture_started = False
        self._immediate_restore_readback: dict[str, np.ndarray] | None = None

    def snapshot(self) -> dict[str, np.ndarray]:
        if self._immediate_restore_readback is not None:
            snapshot = self._immediate_restore_readback
            self._immediate_restore_readback = None
            return {
                name: np.asarray(value).copy()
                for name, value in snapshot.items()
            }
        index = min(self._snapshot_index, len(self._snapshots) - 1)
        self._snapshot_index += 1
        return {
            name: np.asarray(value).copy()
            for name, value in self._snapshots[index].items()
        }

    def schema(self) -> dict:
        has_deformable = any(
            np.asarray(
                snapshot.get(
                    "physx_deformable_position",
                    np.zeros((0, 3)),
                )
            ).shape[0]
            for snapshot in self._snapshots
        )
        schema = {
            "fields": {},
            "physx": {
                "rigid_body_names": [
                    str(entry.name) for entry in self.rigid_bodies
                ],
                "articulation_names": [
                    str(entry.name) for entry in self.articulations
                ],
                "articulation_dof_names": {
                    str(entry.name): list(entry.dof_names)
                    for entry in self.articulations
                },
                "deformable_name": "surface" if has_deformable else None,
                "contact_report": {
                    "path_table": [
                        {"id": 1, "path": "/World/A"},
                        {"id": 2, "path": "/World/B"},
                    ]
                },
            }
        }
        if has_deformable:
            node_count = max(
                np.asarray(
                    snapshot["physx_deformable_position"]
                ).shape[0]
                for snapshot in self._snapshots
            )
            schema["fields"]["physx_deformable_position"] = {
                "shape": [node_count, 3]
            }
        return schema

    def restore(self, snapshot, _schema, *, sync: bool):
        self.restore_count += 1
        self.restore_sync_values.append(bool(sync))
        self.restored_snapshot = {
            name: np.asarray(value).copy()
            for name, value in snapshot.items()
        }
        for name in (
            "physx_robot_joint_velocity",
            "physx_task_articulation_dof_velocity",
            "physx_deformable_velocity",
        ):
            np.testing.assert_array_equal(self.restored_snapshot[name], 0.0)
        np.testing.assert_array_equal(
            self.restored_snapshot["physx_robot_root_state"][..., 7:13],
            0.0,
        )
        np.testing.assert_array_equal(
            self.restored_snapshot["physx_task_rigid_body_state"][..., 7:13],
            0.0,
        )
        self._immediate_restore_readback = {
            name: np.asarray(value).copy()
            for name, value in self.restored_snapshot.items()
        }
        return {"restored": 1}

    def write_zero_velocity_boundary(
        self,
        snapshot,
        _schema,
        *,
        sync: bool,
    ):
        self.boundary_write_count += 1
        self.boundary_sync_values.append(bool(sync))
        self.restored_snapshot = {
            name: np.asarray(value).copy()
            for name, value in snapshot.items()
        }
        self.restored_snapshot["physx_robot_root_state"][..., 7:13] = 0.0
        self.restored_snapshot["physx_robot_joint_velocity"][...] = 0.0
        self.restored_snapshot["physx_robot_joint_target"] = (
            self.restored_snapshot["physx_robot_joint_position"].copy()
        )
        self.restored_snapshot["physx_robot_joint_target_valid"] = (
            self.restored_snapshot["physx_robot_joint_valid"].copy()
        )
        self.restored_snapshot["physx_task_rigid_body_state"][..., 7:13] = 0.0
        self.restored_snapshot[
            "physx_task_articulation_root_state"
        ][..., 7:13] = 0.0
        self.restored_snapshot[
            "physx_task_articulation_dof_velocity"
        ][...] = 0.0
        self.restored_snapshot["physx_deformable_velocity"][...] = 0.0
        self._immediate_restore_readback = {
            name: np.asarray(value).copy()
            for name, value in self.restored_snapshot.items()
        }
        return {
            "robot_joints": 2,
            "rigid_bodies": 1,
            "articulation_roots": 0,
            "articulation_dofs": 0,
            "deformable_nodes": 0,
        }

    def physics_advanced(self) -> None:
        self._immediate_restore_readback = None

    def begin_substep_capture(self) -> None:
        self.substep_capture_started = True

    def zero_task_velocities(self):
        self.velocity_clamp_count += 1
        return {
            "rigid_bodies": 1,
            "articulation_roots": 0,
            "articulation_dofs": 0,
            "deformable_nodes": 0,
        }

    def put_task_bodies_to_sleep(self):
        self.sleep_count += 1
        return {"rigid_bodies": 1, "body_paths": ["/World/Task/object"]}


class _Environment:
    def __init__(
        self,
        snapshots: list[dict[str, np.ndarray]],
        *,
        task_name: str,
        joint_positions_after_step: list[np.ndarray] | None = None,
    ) -> None:
        self.task_name = task_name
        self._isaac_device = "cuda:0"
        self._physics_dt = 0.25
        self._robots = {"right": _Robot()}
        self._joint_targets: dict[str, _Tensor] = {}
        self._hand_joint_ids = {"right": [0, 1]}
        self._right_hand = np.zeros(2, dtype=np.float64)
        self._left_hand = np.zeros(0, dtype=np.float64)
        self._right_pose_commanded = True
        self._left_pose_commanded = True
        self._arm_pose_filters = {"right": object(), "left": object()}
        self.controller = _Controller()
        self._ik_controllers = {"right": self.controller}
        self._physx_ground_truth_trace = _Trace(snapshots)
        self._recording_origin_authored_reset_snapshot = {
            name: np.asarray(value).copy()
            for name, value in snapshots[0].items()
        }
        self._recording_origin_authored_reset_serial = 1
        self._physx_scene_generation = 0
        self.env_step = 17
        self._success_counter = 4
        self._failure_counter = 5
        self.apply_count = 0
        self.physics_step_count = 0
        self.refresh_count = 0
        self.joint_positions_after_step = list(joint_positions_after_step or [])

    @property
    def unwrapped(self):
        return self

    def _apply_joint_targets(self) -> None:
        self.apply_count += 1

    def _step_isaaclab(self) -> None:
        self._physx_ground_truth_trace.physics_advanced()
        self.physics_step_count += 1
        if self.joint_positions_after_step:
            index = min(
                self.physics_step_count - 1,
                len(self.joint_positions_after_step) - 1,
            )
            self._robots["right"].data.joint_pos = _Tensor(
                self.joint_positions_after_step[index]
            )

    def _refresh_object_poses(self) -> None:
        self.refresh_count += 1

    def _compute_observation(self):
        return {"rebuilt": np.asarray([self.physics_step_count])}

    def observation(self, observation):
        return {**observation, "wrapped": np.asarray([1])}


class _DefaultJointResetEnvironment(_Environment):
    """Isaac-like fixture exposing the atomic default-joint reset hook."""

    stable_recording_origin_default_robot_restore_enabled = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.default_joint_position = _Tensor([[0.1, 0.25]])
        self.default_restore_calls: list[tuple[bool, int]] = []

    def _restore_default_robot_joint_state(
        self,
        *,
        reset_controllers: bool,
        sync: bool,
    ) -> dict[str, _Tensor]:
        assert reset_controllers is True
        target = self.default_joint_position.clone()
        self.default_restore_calls.append((sync, self.physics_step_count))
        self._robots["right"].data.joint_pos = target.clone()
        self._robots["right"].set_joint_position_target(target)
        self._robots["right"].write_data_to_sim()
        self._joint_targets["right"] = target.clone()
        self._right_hand[:] = target.numpy().reshape(-1)
        self._right_pose_commanded = False
        self._arm_pose_filters = {"right": None, "left": None}
        self.controller.reset()
        return {"right": target}


def _enable_stable_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(STABLE_ORIGIN_ENV, "1")
    monkeypatch.delenv(JOINT_VELOCITY_LIMIT_ENV, raising=False)


def test_robot_default_restore_clears_state_and_stale_drive_targets() -> None:
    env = BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
    )
    robots = {"right": _Robot(), "left": _Robot()}
    env._scene = {
        "right_robot": robots["right"],
        "left_robot": robots["left"],
    }
    env._hand_joint_ids = {"right": [0, 1], "left": [0, 1]}
    env._joint_targets = {
        "right": _Tensor([[0.9, -0.8]]),
        "left": _Tensor([[-0.7, 0.6]]),
    }
    env._right_hand.fill(0.8)
    env._left_hand.fill(-0.7)
    env._right_pose_commanded = True
    env._left_pose_commanded = True
    env._arm_pose_filters = {"right": object(), "left": object()}
    controllers = {"right": _Controller(), "left": _Controller()}
    env._ik_controllers = controllers

    class _Simulation:
        def __init__(self) -> None:
            self.forward_count = 0

        def forward(self) -> None:
            self.forward_count += 1

    simulation = _Simulation()
    env._sim = simulation

    targets = env._restore_default_robot_joint_state(
        reset_controllers=True,
        sync=False,
    )

    assert set(targets) == {"right", "left"}
    assert simulation.forward_count == 0
    for side, robot in robots.items():
        assert len(robot.written_joint_states) == 1
        position, velocity = robot.written_joint_states[0]
        np.testing.assert_allclose(position, [[0.1, 0.25]])
        np.testing.assert_array_equal(velocity, [[0.0, 0.0]])
        np.testing.assert_allclose(robot.targets[-1], [[0.1, 0.25]])
        np.testing.assert_array_equal(
            robot.velocity_targets[-1],
            [[0.0, 0.0]],
        )
        np.testing.assert_allclose(
            env._joint_targets[side].numpy(),
            [[0.1, 0.25]],
        )
        assert robot.write_count == 1
        assert robot.update_count == 0
        assert controllers[side].reset_count == 1
    np.testing.assert_allclose(env._right_hand[:2], [0.1, 0.25])
    np.testing.assert_allclose(env._left_hand[:2], [0.1, 0.25])
    assert env._right_pose_commanded is False
    assert env._left_pose_commanded is False
    assert env._arm_pose_filters == {"right": None, "left": None}


def test_stable_origin_holds_current_targets_until_all_motion_stays_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [
            _snapshot(
                robot_joint_velocity=0.8,
                task_linear_velocity=0.2,
            ),
            _snapshot(robot_joint_velocity=0.02),
            _snapshot(robot_joint_velocity=0.0005),
            _snapshot(robot_joint_velocity=0.0004),
        ],
        task_name="t50_water_plant",
        joint_positions_after_step=[
            np.asarray([[0.3, 0.5]]),
            np.asarray([[0.35, 0.55]]),
            np.asarray([[0.4, 0.6]]),
        ],
    )

    observation, info = stabilize_recording_origin(
        env,
        {"old": np.asarray([1])},
        {"kept": True},
        stable_seconds=0.5,
        timeout_seconds=2.0,
    )

    report = info["physical_recording_origin"]
    assert report["strategy"] == "continuous_current_target_all_velocity_settle"
    assert report["physics_steps"] == 3
    assert report["stable_steps_required"] == 2
    assert report["stable_steps_observed"] == 2
    assert report["joint_velocity_limit"] == pytest.approx(0.001)
    assert report["joint_velocity_limit_source"] == "default"
    assert report["gate_max_robot_joint_velocity"] == pytest.approx(0.0004)
    assert report["final_max_robot_joint_velocity"] == 0.0
    assert report["final_max_task_linear_velocity"] == 0.0
    assert report["final_max_task_angular_velocity"] == 0.0
    assert report["hold_target_sync"] == {
        "robot_joints": 2,
        "hand_commands": 2,
        "passes": 4,
    }
    assert report["per_frame_state_writes"] is False
    assert info["kept"] is True
    assert observation["rebuilt"].tolist() == [3]
    assert observation["wrapped"].tolist() == [1]

    robot = env._robots["right"]
    assert robot.write_count == 5
    assert len(robot.targets) == 5
    for target in robot.targets[:-1]:
        np.testing.assert_allclose(target, [[0.2, 0.4]])
    np.testing.assert_allclose(robot.targets[-1], [[0.4, 0.6]])
    np.testing.assert_allclose(env._right_hand, [0.4, 0.6])
    assert env._right_pose_commanded is False
    # The fixture contains no left robot, so its unrelated flag is untouched.
    assert env._left_pose_commanded is True
    assert env._arm_pose_filters == {"right": None, "left": None}
    assert env.controller.reset_count == 2
    assert env.apply_count == 3
    assert env.physics_step_count == 3
    assert env.env_step == 0
    assert env._success_counter == 0
    assert env._failure_counter == 0
    assert env.refresh_count == 1
    assert env._physx_ground_truth_trace.substep_capture_started is True


def test_stable_origin_publishes_diagnostic_progress_without_extra_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [
            _snapshot(robot_joint_velocity=0.02),
            _snapshot(robot_joint_velocity=0.0005),
            _snapshot(robot_joint_velocity=0.0004),
        ],
        task_name="t25_place_cup_on_plate",
    )
    progress: list[dict[str, object]] = []

    stabilize_recording_origin(
        env,
        {},
        {},
        stable_seconds=0.5,
        timeout_seconds=1.0,
        progress_callback=lambda payload: progress.append(dict(payload)),
    )

    assert env.physics_step_count == 2
    assert progress
    assert progress[-1]["phase"] == "stability_gate"
    assert progress[-1]["physics_steps"] == 2
    assert progress[-1]["stable_steps_observed"] == 2
    assert progress[-1]["stable_steps_required"] == 2


@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        "cuda",
        "CUDA:0",
        " cuda:0",
        "cuda:0 ",
        "cuda:00",
        "cuda:01",
        "cuda:-1",
        "cuda:²",
    ),
)
def test_stable_origin_rejects_cpu_or_noncanonical_device(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [_snapshot(robot_joint_velocity=0.0)],
        task_name="t25_place_cup_on_plate",
    )
    env._isaac_device = device

    with pytest.raises(RuntimeError, match="canonical CUDA PhysX device"):
        stabilize_recording_origin(
            env,
            {},
            {},
            stable_seconds=0.5,
            timeout_seconds=0.5,
        )

    assert env.physics_step_count == 0


def test_native_origin_restores_default_then_holds_the_accepted_settled_q(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _DefaultJointResetEnvironment(
        [
            _snapshot(robot_joint_velocity=0.4),
            _snapshot(robot_joint_velocity=0.2),
            _snapshot(robot_joint_velocity=0.0005),
            _snapshot(robot_joint_velocity=0.0004),
        ],
        task_name="t09_insert_peg",
        joint_positions_after_step=[
            np.asarray([[0.152, 0.31]]),
            np.asarray([[0.148, 0.30]]),
            np.asarray([[0.145, 0.29]]),
            np.asarray([[0.14, 0.28]]),
        ],
    )
    # Model stale commands left by a previous teleoperation episode.
    env._joint_targets["right"] = _Tensor([[0.9, -0.8]])
    env._right_hand[:] = [0.8, 0.7]

    _observation, info = stabilize_recording_origin(
        env,
        {},
        {},
        initial_settle_seconds=0.5,
        post_settle_velocity_zeroing=True,
        stable_seconds=0.5,
        timeout_seconds=2.0,
    )

    report = info["physical_recording_origin"]
    assert report["robot_hold_target_source"] == "default_joint_state"
    assert report["post_settle_robot_hold_target_source"] == (
        "settled_after_default_joint_state"
    )
    assert report["default_joint_state_restores"] == 1
    assert report["final_default_joint_state_restore"] is False
    assert report["final_boundary_restore_sync"] is True
    assert report["final_robot_hold_target_source"] == (
        "accepted_settled_joint_state"
    )
    assert report["final_robot_position_target_error"] == 0.0
    assert env.default_restore_calls == [(True, 0)]
    # The final target is captured from the state that actually passed the
    # gate; no subsequent physics step can let a stale target chase it.
    assert env.physics_step_count == 4
    np.testing.assert_allclose(
        env._joint_targets["right"].numpy(),
        [[0.14, 0.28]],
    )
    np.testing.assert_allclose(env._right_hand, [0.14, 0.28])
    assert env._right_pose_commanded is False
    # Origin setup never rewrites an already-authored pose.  Its initial,
    # post-settle, and final transactions are synchronous velocity-only writes.
    assert env._physx_ground_truth_trace.restore_sync_values == []
    assert env._physx_ground_truth_trace.boundary_sync_values == [
        True,
        True,
        True,
    ]
    for target in env._robots["right"].targets[:4]:
        np.testing.assert_allclose(target, [[0.1, 0.25]])
    for target in env._robots["right"].targets[4:-1]:
        np.testing.assert_allclose(target, [[0.148, 0.30]])
    np.testing.assert_allclose(
        env._robots["right"].targets[-1],
        [[0.14, 0.28]],
    )


def test_initial_settle_precedes_the_strict_recording_origin_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [
            _snapshot(robot_joint_velocity=0.0005),
            _snapshot(robot_joint_velocity=0.0004),
        ],
        task_name="t26_place_fruit_in_bucket",
    )

    _observation, info = stabilize_recording_origin(
        env,
        {},
        {},
        initial_settle_seconds=0.5,
        stable_seconds=0.5,
        timeout_seconds=2.0,
    )

    report = info["physical_recording_origin"]
    assert report["initial_settle_seconds"] == pytest.approx(0.5)
    assert report["initial_settle_steps"] == 2
    assert report["physics_steps"] == 4
    assert env.physics_step_count == 4
    assert env.apply_count == 4
    assert report["hold_target_sync"]["passes"] == 5


def test_a_series_fast_settle_then_zeroes_once_before_strict_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [
            _snapshot(
                robot_joint_velocity=0.4,
                task_linear_velocity=0.2,
            ),
            _snapshot(
                robot_joint_velocity=0.2,
                task_angular_velocity=0.3,
            ),
            _snapshot(robot_joint_velocity=0.0005),
            _snapshot(robot_joint_velocity=0.0004),
        ],
        task_name="t26_place_fruit_in_bucket",
    )

    _observation, info = stabilize_recording_origin(
        env,
        {},
        {},
        initial_settle_seconds=0.5,
        post_settle_velocity_zeroing=True,
        stabilization_velocity_clamp=True,
        stabilization_velocity_clamp_scope=(
            VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS
        ),
        stable_seconds=0.5,
        timeout_seconds=2.0,
    )

    report = info["physical_recording_origin"]
    assert report["stability_metric"] == "velocity"
    assert report["post_settle_velocity_zeroing"] is True
    assert report["velocity_zeroing_passes"] == 3
    assert report["stabilization_velocity_clamp"] is True
    assert report["stabilization_velocity_clamp_steps"] == 4
    assert report["stabilization_velocity_clamp_write"] == {
        "rigid_bodies": 1,
        "articulation_roots": 0,
        "articulation_dofs": 0,
        "deformable_nodes": 0,
    }
    assert report["initialization_per_physics_step_velocity_writes"] is True
    assert report["post_settle_state_write"] == {
        "robot_joints": 2,
        "rigid_bodies": 1,
        "articulation_roots": 0,
        "articulation_dofs": 0,
        "deformable_nodes": 0,
    }
    assert report["physics_steps"] == 4
    assert report["final_boundary_task_and_joint_velocity_exact_zero"] is True
    assert report["final_boundary_physics_steps_after_write"] == 0
    assert env._physx_ground_truth_trace.restore_count == 0
    assert env._physx_ground_truth_trace.boundary_write_count == 3
    assert env._physx_ground_truth_trace.velocity_clamp_count == 4


def test_collection_tasks_naturally_settle_then_zero_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [
            _snapshot(robot_joint_velocity=0.0),
            _snapshot(
                robot_joint_velocity=0.02,
                task_linear_velocity=0.2,
            ),
            _snapshot(robot_joint_velocity=0.0005),
            _snapshot(robot_joint_velocity=0.0004),
        ],
        task_name="a02_fruit_on_plate",
    )

    _observation, info = stabilize_recording_origin(
        env,
        {},
        {},
        **stable_recording_origin_kwargs_for_task(env.task_name),
    )

    report = info["physical_recording_origin"]
    assert report["initial_settle_seconds"] == pytest.approx(1.0)
    assert report["timeout_seconds"] == pytest.approx(39.0)
    assert report["physics_steps"] == 6
    assert report["stabilization_velocity_clamp"] is False
    assert report["stabilization_velocity_clamp_scope"] == "none"
    assert report["post_settle_sleeping"] is False
    assert report["post_settle_sleep"] is None
    assert report["boundary_state_write_mode"] == "velocity_only"
    assert report["contact_warm_start_preserved"] is True
    assert report["initialization_per_physics_step_velocity_writes"] is False
    assert report["final_boundary_task_and_joint_velocity_exact_zero"] is True
    assert report["final_boundary_sleep"] is None
    assert env._physx_ground_truth_trace.restore_count == 0
    assert env._physx_ground_truth_trace.boundary_write_count == 3
    assert env._physx_ground_truth_trace.sleep_count == 0
    assert report["stabilization_velocity_clamp_steps"] == 0
    assert report["stabilization_velocity_clamp_write"] is None
    assert env._physx_ground_truth_trace.velocity_clamp_count == 0
    assert report["reset_pose_drift_checked"] is True
    assert report["reset_max_translation_drift_m"] == 0.0
    assert report["reset_max_rotation_drift_rad"] == 0.0
    assert report["reset_max_coordinate_drift"] == 0.0
    assert report["authored_reset_snapshot_source"] == (
        "environment_after_authored_writes_before_first_physics_step"
    )
    assert report["initialization_contact_summary"] == {
        "valid": True,
        "sample_count": 7,
        "invalid_sample_count": 0,
        "capture_sources": [2],
        "maximum_contact_point_count": 0,
        "maximum_penetration_depth_m": 0.0,
        "deepest_sample_index": None,
        "deepest_contact_point_index": None,
        "error": None,
        "expected_sample_count": 7,
        "all_physics_frames_sampled": True,
    }
    assert report["gate_contact_summary"] == {
        "valid": True,
        "capture_source": 2,
        "contact_header_count": 0,
        "contact_point_count": 0,
        "friction_anchor_count": 0,
        "maximum_penetration_depth_m": 0.0,
        "deepest_contact_point_index": None,
        "canonical_contact_signature": {
            "pair_count": 0,
            "pair_path_indices": [],
            "pair_counts": [],
            "pair_metrics": [],
        },
        "error": None,
    }
    assert report["initial_geometry_gate_passed"] is True
    assert report["scene_rebuilt_before_reset"] is False
    assert report["scene_generation"] == 0
    expected_active_layout = {
        "format": "handson.active_physics_layout",
        "version": 1,
        "rigid_body": {
            "names": ["/World/Task/object/object"],
            "valid": [1],
        },
        "articulation_root": {"names": [], "valid": []},
        "articulation_dof": {"names": [], "valid": []},
        "deformable_node": {"names": [], "valid": []},
    }
    assert report["active_physics_layout"] == expected_active_layout
    assert report["gate_active_physics_layout"] == expected_active_layout
    assert report["final_active_physics_layout"] == expected_active_layout
    assert report["active_physics_layout_gate_final_exact_match"] is True
    assert report["final_boundary_physics_gate_passed"] is True
    assert report["gate_final_max_task_translation_delta_m"] == 0.0
    assert report["gate_final_max_task_rotation_delta_rad"] == 0.0
    assert report["final_contact_summary"] == report["gate_contact_summary"]


def test_stable_origin_reports_the_fresh_physx_scene_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [_snapshot(robot_joint_velocity=0.0) for _ in range(3)],
        task_name="t09_insert_peg",
    )
    env._physx_scene_generation = 2

    _observation, info = stabilize_recording_origin(
        env,
        {},
        {},
        stable_seconds=0.5,
        timeout_seconds=0.5,
    )

    report = info["physical_recording_origin"]
    assert report["scene_rebuilt_before_reset"] is True
    assert report["scene_generation"] == 2


@pytest.mark.parametrize(
    "invalid_rigid_names",
    [None, [""], ["duplicate", "duplicate"]],
)
def test_geometry_gate_rejects_missing_or_invalid_active_layout_names(
    monkeypatch: pytest.MonkeyPatch,
    invalid_rigid_names,
) -> None:
    _enable_stable_origin(monkeypatch)
    snapshots = [
        _snapshot(robot_joint_velocity=0.0),
        _snapshot(robot_joint_velocity=0.0),
        _snapshot(robot_joint_velocity=0.0),
    ]
    env = _Environment(snapshots, task_name="t09_insert_peg")
    trace = env._physx_ground_truth_trace
    valid_schema = trace.schema()

    def invalid_schema():
        schema = {
            "physx": dict(valid_schema["physx"]),
        }
        if invalid_rigid_names is None:
            schema["physx"].pop("rigid_body_names")
        else:
            schema["physx"]["rigid_body_names"] = invalid_rigid_names
        return schema

    trace.schema = invalid_schema
    with pytest.raises(RecordingOriginStabilityError) as error:
        stabilize_recording_origin(
            env,
            {},
            {},
            stable_seconds=0.5,
            timeout_seconds=0.5,
        )

    assert error.value.diagnostics["failure_phase"] == (
        "expected_active_physics_layout"
    )
    assert "expected_active_physics_layout_invalid" in error.value.diagnostics[
        "geometry_failures"
    ]


def test_geometry_gate_rejects_non_binary_active_layout_bitmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    snapshots = [
        _snapshot(robot_joint_velocity=0.0) for _ in range(3)
    ]
    for snapshot in snapshots:
        snapshot["physx_task_rigid_body_valid"] = np.asarray(
            [2], dtype=np.uint8
        )
    env = _Environment(snapshots, task_name="t09_insert_peg")

    with pytest.raises(RecordingOriginStabilityError) as error:
        stabilize_recording_origin(
            env,
            {},
            {},
            stable_seconds=0.5,
            timeout_seconds=0.5,
        )

    assert "active_physics_layout_invalid" in error.value.diagnostics[
        "geometry_failures"
    ]


def test_initial_layout_rejects_a_schema_body_hidden_from_the_first_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    snapshots = [_snapshot(robot_joint_velocity=0.0) for _ in range(3)]
    for snapshot in snapshots:
        snapshot["physx_task_rigid_body_state"] = np.repeat(
            snapshot["physx_task_rigid_body_state"], 2, axis=0
        )
        snapshot["physx_task_rigid_body_valid"] = np.asarray(
            [1, 0], dtype=np.uint8
        )
    env = _Environment(snapshots, task_name="t09_insert_peg")
    env._physx_ground_truth_trace.rigid_bodies = [
        SimpleNamespace(name="required_a"),
        SimpleNamespace(name="required_b"),
    ]

    with pytest.raises(RecordingOriginStabilityError) as error:
        stabilize_recording_origin(
            env,
            {},
            {},
            stable_seconds=0.5,
            timeout_seconds=0.5,
        )

    assert error.value.diagnostics["geometry_failures"] == [
        "authored_active_physics_layout_expected_mismatch",
        "live_reset_active_physics_layout_expected_mismatch",
    ]


def test_motion_gate_rejects_non_finite_velocity_in_any_valid_task_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    snapshots = [_snapshot(robot_joint_velocity=0.0) for _ in range(3)]
    for snapshot in snapshots:
        snapshot["physx_task_rigid_body_state"] = np.repeat(
            snapshot["physx_task_rigid_body_state"], 2, axis=0
        )
        snapshot["physx_task_rigid_body_valid"] = np.ones(
            2, dtype=np.uint8
        )
    snapshots[1]["physx_task_rigid_body_state"][1, 7] = np.nan
    env = _Environment(snapshots, task_name="t09_insert_peg")
    env._physx_ground_truth_trace.rigid_bodies = [
        SimpleNamespace(name="required_a"),
        SimpleNamespace(name="required_b"),
    ]

    with pytest.raises(RuntimeError, match="no task-object state readback"):
        stabilize_recording_origin(
            env,
            {},
            {},
            stable_seconds=0.5,
            timeout_seconds=0.5,
        )


def test_geometry_gate_rejects_a_reset_that_falls_then_becomes_quiet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    settled = _snapshot(
        robot_joint_velocity=0.0,
        task_position=(0.006, 0.0, 0.0),
    )
    env = _Environment(
        [
            _snapshot(robot_joint_velocity=0.0),
            settled,
            settled,
        ],
        task_name="t09_insert_peg",
    )

    with pytest.raises(RecordingOriginStabilityError) as error:
        stabilize_recording_origin(
            env,
            {},
            {},
            reset_translation_drift_limit=0.005,
            reset_rotation_drift_limit=np.deg2rad(3.0),
            stable_seconds=0.5,
            timeout_seconds=0.5,
        )

    assert error.value.diagnostics["failure_phase"] == (
        "initial_geometry_gate"
    )
    assert error.value.diagnostics["geometry_failures"] == [
        "reset_translation_drift_exceeded"
    ]
    assert error.value.diagnostics["reset_pose_drift_peaks"][0][
        "name"
    ] == "/World/Task/object/object"


def test_geometry_gate_uses_the_pre_first_step_authored_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    authored = _snapshot(robot_joint_velocity=0.0)
    displaced_on_reset_return = _snapshot(
        robot_joint_velocity=0.0,
        task_position=(0.006, 0.0, 0.0),
    )
    env = _Environment(
        [displaced_on_reset_return, displaced_on_reset_return],
        task_name="t09_insert_peg",
    )
    env._recording_origin_authored_reset_snapshot = authored

    with pytest.raises(RecordingOriginStabilityError) as error:
        stabilize_recording_origin(
            env,
            displaced_on_reset_return,
            {},
            reset_translation_drift_limit=0.005,
            reset_rotation_drift_limit=np.deg2rad(3.0),
            stable_seconds=0.5,
            timeout_seconds=0.5,
        )

    assert error.value.diagnostics["geometry_failures"] == [
        "reset_translation_drift_exceeded"
    ]


def test_geometry_gate_keeps_intermediate_drift_after_return_to_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    authored = _snapshot(robot_joint_velocity=0.0)
    displaced = _snapshot(
        robot_joint_velocity=0.0,
        task_position=(0.006, 0.0, 0.0),
    )
    env = _Environment(
        [authored, displaced, authored, authored],
        task_name="t09_insert_peg",
    )

    with pytest.raises(RecordingOriginStabilityError) as error:
        stabilize_recording_origin(
            env,
            authored,
            {},
            reset_translation_drift_limit=0.005,
            reset_rotation_drift_limit=np.deg2rad(3.0),
            stable_seconds=0.5,
            timeout_seconds=0.75,
        )

    assert error.value.diagnostics["geometry_failures"] == [
        "reset_translation_drift_exceeded"
    ]
    assert error.value.diagnostics["reset_max_translation_drift_m"] == (
        pytest.approx(0.006)
    )
    assert error.value.diagnostics["reset_pose_drift_peaks"][0][
        "sample_index"
    ] == 1


def test_geometry_gate_rejects_excessive_solved_contact_penetration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    penetrating = _snapshot(robot_joint_velocity=0.0)
    _set_single_contact(penetrating, separation=-0.001)
    env = _Environment(
        [penetrating, penetrating, penetrating],
        task_name="t09_insert_peg",
    )

    with pytest.raises(RecordingOriginStabilityError) as error:
        stabilize_recording_origin(
            env,
            {},
            {},
            require_valid_contact_report=True,
            contact_penetration_limit=0.0005,
            stable_seconds=0.5,
            timeout_seconds=0.5,
        )

    assert error.value.diagnostics["geometry_failures"] == [
        "contact_penetration_exceeded"
    ]
    assert error.value.diagnostics["contact_summary"][
        "maximum_penetration_depth_m"
    ] == pytest.approx(0.001)


def test_geometry_gate_keeps_the_deepest_transient_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    authored = _snapshot(robot_joint_velocity=0.0)
    penetrating = _snapshot(robot_joint_velocity=0.0)
    _set_single_contact(penetrating, separation=-0.001)
    quiet = _snapshot(robot_joint_velocity=0.0)
    env = _Environment(
        [authored, penetrating, quiet, quiet],
        task_name="t09_insert_peg",
    )

    with pytest.raises(RecordingOriginStabilityError) as error:
        stabilize_recording_origin(
            env,
            authored,
            {},
            require_valid_contact_report=True,
            contact_penetration_limit=0.0005,
            stable_seconds=0.5,
            timeout_seconds=0.75,
        )

    assert error.value.diagnostics["geometry_failures"] == [
        "contact_penetration_exceeded"
    ]
    assert error.value.diagnostics["contact_summary"][
        "deepest_sample_index"
    ] == 1


def test_geometry_gate_uses_radian_limit_for_robot_joint_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    authored = _snapshot(
        robot_joint_velocity=0.0,
        robot_joint_position=(0.2, 0.4),
    )
    settled = _snapshot(
        robot_joint_velocity=0.0,
        robot_joint_position=(0.21, 0.4),
    )
    env = _Environment(
        [authored, settled, settled, settled],
        task_name="t09_insert_peg",
    )

    _observation, info = stabilize_recording_origin(
        env,
        authored,
        {},
        reset_translation_drift_limit=0.005,
        reset_rotation_drift_limit=np.deg2rad(3.0),
        stable_seconds=0.5,
        timeout_seconds=0.5,
    )

    report = info["physical_recording_origin"]
    assert report["reset_max_rotation_drift_rad"] == pytest.approx(0.01)
    assert report["reset_max_coordinate_drift"] == 0.0


def test_geometry_gate_rejects_robot_joint_angular_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    authored = _snapshot(
        robot_joint_velocity=0.0,
        robot_joint_position=(0.2, 0.4),
    )
    settled = _snapshot(
        robot_joint_velocity=0.0,
        robot_joint_position=(0.26, 0.4),
    )
    env = _Environment(
        [authored, settled, settled],
        task_name="t09_insert_peg",
    )

    with pytest.raises(RecordingOriginStabilityError) as error:
        stabilize_recording_origin(
            env,
            authored,
            {},
            reset_translation_drift_limit=0.005,
            reset_rotation_drift_limit=np.deg2rad(3.0),
            stable_seconds=0.5,
            timeout_seconds=0.5,
        )

    assert "reset_rotation_drift_exceeded" in error.value.diagnostics[
        "geometry_failures"
    ]
    assert error.value.diagnostics["reset_pose_drift_peaks"][0][
        "kind"
    ] == "robot_joint"


def test_geometry_gate_rejects_articulation_dof_and_deformable_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    authored = _snapshot(robot_joint_velocity=0.0)
    settled = _snapshot(robot_joint_velocity=0.0)
    for snapshot, dof, node_x in (
        (authored, 0.0, 0.0),
        (settled, 0.01, 0.006),
    ):
        snapshot["physx_task_articulation_root_state"] = np.zeros(
            (1, 13), dtype=np.float64
        )
        snapshot["physx_task_articulation_root_state"][:, 3] = 1.0
        snapshot["physx_task_articulation_root_valid"] = np.ones(
            1, dtype=np.uint8
        )
        snapshot["physx_task_articulation_dof_position"] = np.asarray(
            [[dof]], dtype=np.float64
        )
        snapshot["physx_task_articulation_dof_velocity"] = np.zeros(
            (1, 1), dtype=np.float64
        )
        snapshot["physx_task_articulation_dof_valid"] = np.ones(
            (1, 1), dtype=np.uint8
        )
        snapshot["physx_deformable_position"] = np.asarray(
            [[node_x, 0.0, 0.0]], dtype=np.float64
        )
        snapshot["physx_deformable_velocity"] = np.zeros(
            (1, 3), dtype=np.float64
        )
        snapshot["physx_deformable_valid"] = np.ones(1, dtype=np.uint8)
    env = _Environment(
        [authored, settled, settled],
        task_name="t09_insert_peg",
    )
    env._physx_ground_truth_trace.articulations = [
        SimpleNamespace(name="drawer", dof_names=("slide",))
    ]

    with pytest.raises(RecordingOriginStabilityError) as error:
        stabilize_recording_origin(
            env,
            authored,
            {},
            reset_translation_drift_limit=0.005,
            reset_rotation_drift_limit=np.deg2rad(3.0),
            stable_seconds=0.5,
            timeout_seconds=0.5,
        )

    failures = error.value.diagnostics["geometry_failures"]
    assert "reset_coordinate_drift_exceeded" in failures
    assert "reset_translation_drift_exceeded" in failures


def test_final_boundary_rejects_a_nonzero_immediate_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [
            _snapshot(robot_joint_velocity=0.0),
            _snapshot(robot_joint_velocity=0.0005),
            _snapshot(robot_joint_velocity=0.0004),
        ],
        task_name="t09_insert_peg",
    )
    trace = env._physx_ground_truth_trace
    boundary_write = trace.write_zero_velocity_boundary

    def boundary_with_residual(snapshot, schema, *, sync: bool):
        result = boundary_write(snapshot, schema, sync=sync)
        if trace.boundary_write_count == 2:
            trace._immediate_restore_readback = _snapshot(
                robot_joint_velocity=0.0002,
                task_linear_velocity=0.0001,
            )
        return result

    trace.write_zero_velocity_boundary = boundary_with_residual

    with pytest.raises(
        RecordingOriginStabilityError,
        match="did not read back exact zero",
    ):
        stabilize_recording_origin(
            env,
            {},
            {},
            stable_seconds=0.5,
            timeout_seconds=1.0,
        )

    assert env.physics_step_count == 2
    assert trace.restore_count == 0
    assert trace.boundary_write_count == 2


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    [
        ("hidden_body", "active_physics_layout_invalid"),
        (
            "robot_target_validity",
            "final_robot_target_validity_mismatch",
        ),
        (
            "robot_coordinate",
            "gate_final_coordinates_changed_without_physics_step",
        ),
        (
            "nan_valid_task_row",
            "final_valid_physics_rows_non_finite",
        ),
        ("pose", "gate_final_translation_delta_exceeded"),
        ("penetration", "final_contact_penetration_exceeded"),
        ("nan_impulse", "final_contact_report_invalid"),
        ("signature", "final_contact_report_invalid"),
        ("missing_contact", "final_contact_report_invalid"),
    ],
)
def test_final_boundary_revalidates_layout_pose_and_contact_after_restore(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_failure: str,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [
            _snapshot(robot_joint_velocity=0.0),
            _snapshot(robot_joint_velocity=0.0005),
            _snapshot(robot_joint_velocity=0.0004),
        ],
        task_name="t09_insert_peg",
    )
    trace = env._physx_ground_truth_trace
    boundary_write = trace.write_zero_velocity_boundary

    def boundary_with_bad_final_readback(snapshot, schema, *, sync: bool):
        result = boundary_write(snapshot, schema, sync=sync)
        if trace.boundary_write_count != 2:
            return result
        assert trace._immediate_restore_readback is not None
        readback = {
            name: np.asarray(value).copy()
            for name, value in trace._immediate_restore_readback.items()
        }
        if mutation == "hidden_body":
            readback["physx_task_rigid_body_valid"][0] = 0
        elif mutation == "robot_target_validity":
            readback["physx_robot_joint_target_valid"][0, 1] = 0
        elif mutation == "robot_coordinate":
            readback["physx_robot_joint_position"][0, 0] += 1.0e-6
            readback["physx_robot_joint_target"][0, 0] += 1.0e-6
        elif mutation == "nan_valid_task_row":
            readback["physx_task_rigid_body_state"][0, 7] = np.nan
        elif mutation == "pose":
            readback["physx_task_rigid_body_state"][0, 0] += 0.001
        elif mutation == "penetration":
            _set_single_contact(readback, separation=-0.001)
        elif mutation == "nan_impulse":
            _set_single_contact(
                readback,
                separation=0.0,
                impulse=(np.nan, 0.0, 0.0),
            )
        elif mutation == "signature":
            readback["physx_contact_signature_pair_count"] = np.ones(
                1, dtype=np.uint32
            )
            readback["physx_contact_signature_pair_path_indices"] = (
                np.asarray([[1, 1, 2, 2]], dtype=np.uint32)
            )
            readback["physx_contact_signature_pair_counts"] = np.zeros(
                (1, 2), dtype=np.uint32
            )
            readback["physx_contact_signature_pair_metrics"] = np.zeros(
                (1, 3), dtype=np.float64
            )
        else:
            readback.pop("physx_contact_report_valid")
        trace._immediate_restore_readback = readback
        return result

    trace.write_zero_velocity_boundary = boundary_with_bad_final_readback

    with pytest.raises(RecordingOriginStabilityError) as error:
        stabilize_recording_origin(
            env,
            {},
            {},
            stable_seconds=0.5,
            timeout_seconds=1.0,
        )

    assert error.value.diagnostics["failure_phase"] == (
        "final_boundary_physics_gate"
    )
    assert expected_failure in error.value.diagnostics["geometry_failures"]
    # The final verification is a read-only boundary check.
    assert env.physics_step_count == 2


def test_stability_readback_is_sampled_without_pausing_every_physics_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [
            _snapshot(robot_joint_velocity=0.0),
            _snapshot(robot_joint_velocity=0.0005),
            _snapshot(robot_joint_velocity=0.0004),
        ],
        task_name="t44_stack_blocks",
    )
    env._physics_dt = 0.01

    _observation, info = stabilize_recording_origin(
        env,
        {},
        {},
        stable_seconds=0.1,
        stability_sample_seconds=0.05,
        timeout_seconds=0.2,
    )

    report = info["physical_recording_origin"]
    assert report["physics_steps"] == 10
    assert report["stability_sample_steps"] == 5
    assert report["stability_samples"] == 2
    assert report["stable_steps_observed"] == 10
    # Every post-step contact buffer is consumed immediately, while velocity
    # stability is still evaluated only at the configured two readbacks.
    assert env._physx_ground_truth_trace._snapshot_index == 12


def test_abc_collection_settle_is_bounded_at_forty_simulated_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [
            _snapshot(robot_joint_velocity=0.1),
            _snapshot(robot_joint_velocity=0.1),
        ],
        task_name="a02_fruit_on_plate",
    )

    with pytest.raises(RecordingOriginStabilityError):
        stabilize_recording_origin(
            env,
            {},
            {},
            **stable_recording_origin_kwargs_for_task(env.task_name),
        )

    assert env.physics_step_count == 160


def test_shared_task_and_manifest_parameters_preserve_strict_contract() -> None:
    assert is_abc_style_collection_task("a02_fruit_on_plate")
    assert is_abc_style_collection_task("B03_PEG_IN_SOCKET")
    assert is_abc_style_collection_task("loaded_tray_through_narrow_gate")
    assert is_abc_style_collection_task(
        "t29_place_wine_glass"
    )
    assert not is_abc_style_collection_task("d01_bimanual_pick_laptop")

    task_kwargs = stable_recording_origin_kwargs_for_task(
        "a02_fruit_on_plate"
    )
    assert task_kwargs == {
        "stability_metric": "velocity",
        "initial_settle_seconds": 1.0,
        "post_settle_velocity_zeroing": True,
        "post_settle_sleeping": False,
        "stabilization_velocity_clamp": False,
        "stabilization_velocity_clamp_scope": VELOCITY_CLAMP_SCOPE_NONE,
        "reset_translation_drift_limit": 0.005,
        "reset_rotation_drift_limit": pytest.approx(np.deg2rad(3.0)),
        "require_valid_contact_report": True,
        "contact_penetration_limit": 0.0005,
        "timeout_seconds": 39.0,
        "stability_sample_seconds": 0.0,
    }
    assert stable_recording_origin_reset_attempts_for_task(
        "a02_fruit_on_plate"
    ) == 3
    assert stable_recording_origin_reset_attempts_for_task(
        "t26_place_fruit_in_bucket"
    ) == 12
    assert stable_recording_origin_kwargs_for_task(
        "t26_place_fruit_in_bucket"
    ) == {
        "stability_metric": "velocity_or_static_pose",
        "initial_settle_seconds": 1.0,
        "post_settle_velocity_zeroing": True,
        "post_settle_sleeping": False,
        "stabilization_velocity_clamp": False,
        "stabilization_velocity_clamp_scope": VELOCITY_CLAMP_SCOPE_NONE,
        "reset_translation_drift_limit": 0.005,
        "reset_rotation_drift_limit": pytest.approx(np.deg2rad(3.0)),
        "require_valid_contact_report": True,
        "contact_penetration_limit": 0.0005,
        "pose_translation_limit": 1.0e-6,
        "pose_rotation_limit": 1.0e-5,
        "timeout_seconds": 39.0,
        "stability_sample_seconds": 0.0,
    }
    assert stable_recording_origin_reset_attempts_for_task(
        "t09_insert_peg"
    ) == 3
    assert stable_recording_origin_kwargs_for_task(
        "t09_insert_peg"
    ) == {
        "stability_metric": "velocity_or_static_pose",
        "initial_settle_seconds": 1.0,
        "post_settle_velocity_zeroing": True,
        "post_settle_sleeping": False,
        "stabilization_velocity_clamp": False,
        "stabilization_velocity_clamp_scope": VELOCITY_CLAMP_SCOPE_NONE,
        "reset_translation_drift_limit": 0.005,
        "reset_rotation_drift_limit": pytest.approx(np.deg2rad(3.0)),
        "require_valid_contact_report": True,
        "contact_penetration_limit": 0.0005,
        "pose_translation_limit": 1.0e-6,
        "pose_rotation_limit": 1.0e-5,
        "timeout_seconds": 39.0,
        "stability_sample_seconds": 0.0,
    }
    for task_name in (
        "t25_place_cup_on_plate",
        "t24_place_ball_in_bowl",

        "t02_connect_charger_and_cable",
    ):
        candidate = stable_recording_origin_kwargs_for_task(task_name)
        assert candidate["stability_metric"] == "velocity_or_static_pose"
        assert candidate["pose_translation_limit"] == pytest.approx(1.0e-6)
        assert candidate["pose_rotation_limit"] == pytest.approx(1.0e-5)
    assert stable_recording_origin_kwargs_for_task(
        "t29_place_wine_glass"
    ) == {
        "stability_metric": "velocity_or_static_pose",
        "initial_settle_seconds": 1.0,
        "post_settle_velocity_zeroing": True,
        "post_settle_sleeping": False,
        "stabilization_velocity_clamp": False,
        "stabilization_velocity_clamp_scope": VELOCITY_CLAMP_SCOPE_NONE,
        "reset_translation_drift_limit": 0.005,
        "reset_rotation_drift_limit": pytest.approx(np.deg2rad(3.0)),
        "require_valid_contact_report": True,
        "contact_penetration_limit": 0.0005,
        "pose_translation_limit": 1.0e-6,
        "pose_rotation_limit": 1.0e-5,
        "timeout_seconds": 39.0,
        "stability_sample_seconds": 0.0,
    }
    assert stable_recording_origin_kwargs_for_task(
        "t14_move_and_click_mouse"
    ) == {
        "stability_metric": "velocity_or_static_pose",
        "initial_settle_seconds": 1.0,
        "post_settle_velocity_zeroing": True,
        "post_settle_sleeping": False,
        "stabilization_velocity_clamp": False,
        "stabilization_velocity_clamp_scope": VELOCITY_CLAMP_SCOPE_NONE,
        "reset_translation_drift_limit": 0.005,
        "reset_rotation_drift_limit": pytest.approx(np.deg2rad(3.0)),
        "require_valid_contact_report": True,
        "contact_penetration_limit": 0.0005,
        "pose_translation_limit": 1.0e-6,
        "pose_rotation_limit": 1.0e-5,
        "timeout_seconds": 39.0,
        "stability_sample_seconds": 0.0,
    }
    assert stable_recording_origin_kwargs_for_task(
        "t19_orient_cup_upright"
    ) == {
        "stability_metric": "velocity_or_static_pose",
        "initial_settle_seconds": 1.0,
        "post_settle_velocity_zeroing": True,
        "post_settle_sleeping": False,
        "stabilization_velocity_clamp": False,
        "stabilization_velocity_clamp_scope": VELOCITY_CLAMP_SCOPE_NONE,
        "reset_translation_drift_limit": 0.005,
        "reset_rotation_drift_limit": pytest.approx(np.deg2rad(3.0)),
        "require_valid_contact_report": True,
        "contact_penetration_limit": 0.0005,
        "pose_translation_limit": 1.0e-6,
        "pose_rotation_limit": 1.0e-5,
        "timeout_seconds": 39.0,
        "stability_sample_seconds": 0.0,
    }
    assert stable_recording_origin_kwargs_from_report(
        {
            "stability_metric": "velocity",
            "initial_settle_seconds": 5.0,
            "post_settle_velocity_zeroing": True,
            "post_settle_sleeping": True,
            "stabilization_velocity_clamp": True,
            "linear_velocity_limit": 0.001,
            "angular_velocity_limit": 0.002,
            "joint_velocity_limit": 0.001,
            "reset_translation_drift_limit": 0.005,
            "reset_rotation_drift_limit": 0.05235987755982989,
            "require_valid_contact_report": True,
            "contact_penetration_limit": 0.0005,
            "stable_seconds_required": 0.5,
            "stability_sample_seconds": 0.05,
            "timeout_seconds": 30.0,
        }
    ) == {
        "stability_metric": "velocity",
        "initial_settle_seconds": 5.0,
        "post_settle_velocity_zeroing": True,
        "post_settle_sleeping": True,
        "stabilization_velocity_clamp": True,
        "stabilization_velocity_clamp_scope": (
            VELOCITY_CLAMP_SCOPE_GATE_ONLY
        ),
        "linear_velocity_limit": 0.001,
        "angular_velocity_limit": 0.002,
        "joint_velocity_limit": 0.001,
        "reset_translation_drift_limit": 0.005,
        "reset_rotation_drift_limit": 0.05235987755982989,
        "require_valid_contact_report": True,
        "contact_penetration_limit": 0.0005,
        "stable_seconds": 0.5,
        "stability_sample_seconds": 0.05,
        "timeout_seconds": 30.0,
    }


@pytest.mark.parametrize(
    "report",
    [
        {"stabilization_velocity_clamp_scope": "none"},
        {
            "initialization_per_physics_step_velocity_writes": False,
            "stabilization_velocity_clamp_scope": "none",
        },
    ],
)
def test_manifest_parser_normalizes_historical_disabled_clamp_flags(
    report: dict[str, object],
) -> None:
    assert stable_recording_origin_kwargs_from_report(report) == {
        "stabilization_velocity_clamp": False,
        "stabilization_velocity_clamp_scope": "none",
    }


def test_reset_attempts_env_can_force_one_fresh_process_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(STABLE_ORIGIN_RESET_ATTEMPTS_ENV, raising=False)
    assert stable_recording_origin_reset_attempts_for_task(
        "t09_insert_peg"
    ) == 3

    monkeypatch.setenv(STABLE_ORIGIN_RESET_ATTEMPTS_ENV, "1")
    assert stable_recording_origin_reset_attempts_for_task(
        "t09_insert_peg"
    ) == 1

    for invalid in ("0", "-1", "not-an-integer"):
        monkeypatch.setenv(STABLE_ORIGIN_RESET_ATTEMPTS_ENV, invalid)
        with pytest.raises(ValueError, match="positive integer"):
            stable_recording_origin_reset_attempts_for_task(
                "t09_insert_peg"
            )


def test_pose_metric_accepts_settled_positions_despite_velocity_readback_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [
            _snapshot(robot_joint_velocity=0.8, task_linear_velocity=0.2),
            _snapshot(robot_joint_velocity=0.7, task_angular_velocity=0.3),
        ],
        task_name="t26_place_fruit_in_bucket",
    )

    _observation, info = stabilize_recording_origin(
        env,
        {},
        {},
        stability_metric="pose",
        stable_seconds=0.5,
        timeout_seconds=2.0,
    )

    report = info["physical_recording_origin"]
    assert report["stability_metric"] == "pose"
    assert report["physics_steps"] == 2
    assert report["final_max_task_translation_delta"] == pytest.approx(0.0)
    assert report["final_max_task_rotation_delta"] == pytest.approx(0.0)


def test_a01_static_pose_gate_accepts_only_after_sixty_unclamped_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [
            _snapshot(
                robot_joint_velocity=0.0004,
                task_linear_velocity=0.0002,
                task_angular_velocity=0.0100245,
            )
        ],
        task_name="t26_place_fruit_in_bucket",
    )
    env._physics_dt = 1.0 / 120.0

    _observation, info = stabilize_recording_origin(
        env,
        {},
        {},
        **stable_recording_origin_kwargs_for_task(env.task_name),
    )

    report = info["physical_recording_origin"]
    assert report["initial_settle_steps"] == 120
    assert report["physics_steps"] == 180
    assert report["stable_steps_required"] == 60
    assert report["stable_steps_observed"] == 60
    assert report["stability_sample_steps"] == 1
    assert report["stability_samples"] == 60
    assert report["full_frame_pose_gate"] is True
    assert report["gate_task_velocity_stable"] is False
    assert report["gate_task_static_pose_stable"] is True
    assert report["gate_robot_velocity_stable"] is True
    assert report["gate_static_pose_fallback_samples"] == 60
    assert report["gate_pose_samples"] == 60
    assert report["gate_max_task_translation_delta"] == 0.0
    assert report["gate_max_task_rotation_delta"] == 0.0
    assert report["stabilization_velocity_clamp"] is False
    assert report["stabilization_velocity_clamp_scope"] == "none"
    assert report["stabilization_velocity_clamp_steps"] == 0
    assert report["post_settle_sleeping"] is False
    assert report["boundary_state_write_mode"] == "velocity_only"
    assert report["final_boundary_physics_steps_after_write"] == 0
    assert report["final_boundary_task_and_joint_velocity_exact_zero"] is True
    assert env._physx_ground_truth_trace.velocity_clamp_count == 0
    assert env._physx_ground_truth_trace.sleep_count == 0
    assert env._physx_ground_truth_trace.boundary_write_count == 3


def test_a01_static_pose_gate_rejects_real_per_frame_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [
            _snapshot(
                robot_joint_velocity=0.0004,
                task_angular_velocity=0.0100245,
                task_position=(index * 2.0e-6, 0.0, 0.0),
            )
            for index in range(61)
        ],
        task_name="t26_place_fruit_in_bucket",
    )
    env._physics_dt = 1.0 / 120.0

    with pytest.raises(RecordingOriginStabilityError):
        stabilize_recording_origin(
            env,
            {},
            {},
            stability_metric="velocity_or_static_pose",
            pose_translation_limit=1.0e-6,
            pose_rotation_limit=1.0e-5,
            stable_seconds=0.5,
            # The approved hybrid metric must still inspect every physics frame.
            stability_sample_seconds=0.5,
            timeout_seconds=0.5,
        )

    assert env.physics_step_count == 60
    assert env._physx_ground_truth_trace.velocity_clamp_count == 0
    assert env._physx_ground_truth_trace.sleep_count == 0


def test_static_pose_gate_rejects_moving_articulation_dof_under_static_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    snapshots = []
    for index in range(62):
        snapshot = _snapshot(
            robot_joint_velocity=0.0,
            task_angular_velocity=0.01,
        )
        articulation_root = np.zeros((1, 13), dtype=np.float64)
        articulation_root[:, 3] = 1.0
        snapshot["physx_task_articulation_root_state"] = articulation_root
        snapshot["physx_task_articulation_root_valid"] = np.ones(
            1, dtype=np.uint8
        )
        snapshot["physx_task_articulation_dof_position"] = np.asarray(
            [[index * 2.0e-6]], dtype=np.float64
        )
        snapshot["physx_task_articulation_dof_velocity"] = np.asarray(
            [[0.01]], dtype=np.float64
        )
        snapshot["physx_task_articulation_dof_valid"] = np.ones(
            (1, 1), dtype=np.uint8
        )
        snapshots.append(snapshot)

    env = _Environment(snapshots, task_name="t50_water_plant")
    env._physics_dt = 1.0 / 120.0
    env._physx_ground_truth_trace.articulations = [
        SimpleNamespace(name="spray", dof_names=("trigger",))
    ]

    with pytest.raises(RecordingOriginStabilityError) as error:
        stabilize_recording_origin(
            env,
            {},
            {},
            stability_metric="velocity_or_static_pose",
            pose_translation_limit=1.0e-6,
            pose_rotation_limit=1.0e-5,
            stable_seconds=0.5,
            timeout_seconds=0.5,
        )

    assert error.value.diagnostics["failure_phase"] == "stability_timeout"
    assert env.physics_step_count == 60


@pytest.mark.parametrize(
    "relaxed_limit",
    [
        {"pose_translation_limit": 1.000001e-6},
        {"pose_rotation_limit": 1.000001e-5},
    ],
)
def test_a01_static_pose_gate_rejects_relaxed_pose_limits(
    monkeypatch: pytest.MonkeyPatch,
    relaxed_limit: dict[str, float],
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [_snapshot(robot_joint_velocity=0.0)],
        task_name="t26_place_fruit_in_bucket",
    )
    env._physics_dt = 1.0 / 120.0

    kwargs = {
        "stability_metric": "velocity_or_static_pose",
        "pose_translation_limit": 1.0e-6,
        "pose_rotation_limit": 1.0e-5,
        **relaxed_limit,
    }
    with pytest.raises(ValueError, match="approved 120 Hz stability contract"):
        stabilize_recording_origin(env, {}, {}, **kwargs)


@pytest.mark.parametrize(
    "forbidden_mode",
    [
        {"stabilization_velocity_clamp": True},
        {"post_settle_sleeping": True},
    ],
)
def test_a01_static_pose_gate_rejects_clamp_and_sleep(
    monkeypatch: pytest.MonkeyPatch,
    forbidden_mode: dict[str, bool],
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [_snapshot(robot_joint_velocity=0.0)],
        task_name="t26_place_fruit_in_bucket",
    )
    env._physics_dt = 1.0 / 120.0

    with pytest.raises(ValueError, match="awake, unclamped physics"):
        stabilize_recording_origin(
            env,
            {},
            {},
            stability_metric="velocity_or_static_pose",
            pose_translation_limit=1.0e-6,
            pose_rotation_limit=1.0e-5,
            **forbidden_mode,
        )


def test_a06_static_pose_fallback_checks_every_frame_and_keeps_robot_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [
            _snapshot(
                robot_joint_velocity=0.0004,
            ),
            _snapshot(
                robot_joint_velocity=0.0004,
            ),
        ],
        task_name="t19_orient_cup_upright",
    )
    env._physics_dt = 1.0 / 120.0

    _observation, info = stabilize_recording_origin(
        env,
        {},
        {},
        stability_metric="velocity_or_static_pose",
        stabilization_velocity_clamp=True,
        stabilization_velocity_clamp_scope=(
            VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS
        ),
        pose_translation_limit=1.0e-6,
        pose_rotation_limit=1.0e-5,
        stable_seconds=0.5,
        # The hybrid metric must override this coarse interval.
        stability_sample_seconds=0.5,
        timeout_seconds=1.0,
    )

    report = info["physical_recording_origin"]
    assert report["stability_metric"] == "velocity_or_static_pose"
    assert report["physics_steps"] == 60
    assert report["stability_sample_steps"] == 1
    assert report["stability_samples"] == 60
    assert report["full_frame_pose_gate"] is True
    assert report["stabilization_velocity_clamp"] is True
    assert report["stabilization_velocity_clamp_steps"] == 60
    assert report["gate_task_velocity_stable"] is False
    assert report["gate_task_static_pose_stable"] is True
    assert report["gate_robot_velocity_stable"] is True
    assert report["gate_static_pose_fallback_samples"] == 60
    assert report["gate_pose_samples"] == 60
    assert report["gate_max_task_translation_delta"] == 0.0
    assert report["gate_max_task_rotation_delta"] == 0.0
    assert report["gate_max_fallback_translation_delta"] == 0.0
    assert report["gate_max_fallback_rotation_delta"] == 0.0


def test_a06_all_step_clamp_cannot_hide_continuous_pose_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [
            _snapshot(
                robot_joint_velocity=0.0004,
                task_position=(index * 2.0e-6, 0.0, 0.0),
            )
            for index in range(62)
        ],
        task_name="t19_orient_cup_upright",
    )
    env._physics_dt = 1.0 / 120.0

    with pytest.raises(RecordingOriginStabilityError):
        stabilize_recording_origin(
            env,
            {},
            {},
            stability_metric="velocity_or_static_pose",
            stabilization_velocity_clamp=True,
            stabilization_velocity_clamp_scope=(
                VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS
            ),
            pose_translation_limit=1.0e-6,
            pose_rotation_limit=1.0e-5,
            stable_seconds=0.5,
            timeout_seconds=0.5,
        )


def test_a06_static_pose_fallback_rejects_robot_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [
            _snapshot(
                robot_joint_velocity=0.02,
                task_angular_velocity=0.02,
            ),
            _snapshot(
                robot_joint_velocity=0.02,
                task_angular_velocity=0.02,
            ),
        ],
        task_name="t19_orient_cup_upright",
    )
    env._physics_dt = 1.0 / 120.0

    with pytest.raises(RecordingOriginStabilityError):
        stabilize_recording_origin(
            env,
            {},
            {},
            stability_metric="velocity_or_static_pose",
            stabilization_velocity_clamp=True,
            stabilization_velocity_clamp_scope=(
                VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS
            ),
            pose_translation_limit=1.0e-6,
            pose_rotation_limit=1.0e-5,
            stable_seconds=0.5,
            timeout_seconds=0.5,
        )


def test_a06_hybrid_gate_does_not_mix_two_incomplete_stability_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [
            _snapshot(robot_joint_velocity=0.0004),
            # Velocity is stable, but the pose jumps beyond the fallback limit.
            _snapshot(
                robot_joint_velocity=0.0004,
                task_position=(0.001, 0.0, 0.0),
            ),
            # The pose is now static, but the velocity tensor is noisy.
            _snapshot(
                robot_joint_velocity=0.0004,
                task_angular_velocity=0.02,
                task_position=(0.001, 0.0, 0.0),
            ),
        ],
        task_name="t19_orient_cup_upright",
    )
    env._physics_dt = 1.0 / 120.0

    with pytest.raises(RecordingOriginStabilityError):
        stabilize_recording_origin(
            env,
            {},
            {},
            stability_metric="velocity_or_static_pose",
            stabilization_velocity_clamp=True,
            stabilization_velocity_clamp_scope=(
                VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS
            ),
            pose_translation_limit=1.0e-6,
            pose_rotation_limit=1.0e-5,
            stable_seconds=0.5,
            timeout_seconds=0.5,
        )


def test_a06_static_pose_fallback_rejects_relaxed_robot_joint_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [_snapshot(robot_joint_velocity=0.5, task_angular_velocity=0.02)],
        task_name="t19_orient_cup_upright",
    )
    env._physics_dt = 1.0 / 120.0

    with pytest.raises(ValueError, match="approved 120 Hz stability contract"):
        stabilize_recording_origin(
            env,
            {},
            {},
            stability_metric="velocity_or_static_pose",
            pose_translation_limit=1.0e-6,
            pose_rotation_limit=1.0e-5,
            joint_velocity_limit=1.0,
        )

    monkeypatch.setenv(JOINT_VELOCITY_LIMIT_ENV, "1.0")
    with pytest.raises(ValueError, match="approved 120 Hz stability contract"):
        stabilize_recording_origin(
            env,
            {},
            {},
            stability_metric="velocity_or_static_pose",
            pose_translation_limit=1.0e-6,
            pose_rotation_limit=1.0e-5,
        )


def test_static_pose_fallback_is_restricted_to_approved_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [_snapshot(robot_joint_velocity=0.0)],
        task_name="a02_fruit_on_plate",
    )

    with pytest.raises(ValueError, match="restricted to approved tasks"):
        stabilize_recording_origin(
            env,
            {},
            {},
            stability_metric="velocity_or_static_pose",
        )


def test_task_pose_delta_rejects_changed_body_validity_identity() -> None:
    previous = _snapshot(robot_joint_velocity=0.0)
    current = _snapshot(robot_joint_velocity=0.0)
    previous["physx_task_rigid_body_state"] = np.repeat(
        previous["physx_task_rigid_body_state"],
        2,
        axis=0,
    )
    current["physx_task_rigid_body_state"] = np.repeat(
        current["physx_task_rigid_body_state"],
        2,
        axis=0,
    )
    previous["physx_task_rigid_body_valid"] = np.asarray([1, 0], dtype=np.uint8)
    current["physx_task_rigid_body_valid"] = np.asarray([0, 1], dtype=np.uint8)

    assert _task_pose_delta(previous, current) is None


def test_c06_uses_its_validated_near_zero_joint_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [
            _snapshot(robot_joint_velocity=0.5),
            _snapshot(robot_joint_velocity=0.0025),
            _snapshot(robot_joint_velocity=0.0024),
        ],
        task_name="t23_photograph_logo",
    )

    _observation, info = stabilize_recording_origin(
        env,
        {},
        {},
        stable_seconds=0.5,
        timeout_seconds=0.5,
    )

    report = info["physical_recording_origin"]
    assert report["joint_velocity_limit"] == pytest.approx(0.003)
    assert report["joint_velocity_limit_source"] == "task_default"
    assert report["task_name"] == "t23_photograph_logo"
    assert report["physics_steps"] == 2


def test_stable_origin_times_out_when_a_joint_never_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_stable_origin(monkeypatch)
    env = _Environment(
        [
            _snapshot(robot_joint_velocity=0.5),
            _snapshot(robot_joint_velocity=0.0025),
            _snapshot(robot_joint_velocity=0.0024),
        ],
        task_name="t50_water_plant",
    )

    with pytest.raises(
        RuntimeError,
        match=r"did not remain physically stable.*right_j0",
    ):
        stabilize_recording_origin(
            env,
            {},
            {},
            stable_seconds=0.5,
            timeout_seconds=0.5,
        )
