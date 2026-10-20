from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from handson.data.recording_fields import (
    stack_transition_numeric_observation_fields,
)
from handson.sim.isaaclab.bimanual_assembly_env import (
    BimanualAssemblyIsaacLabEnv,
)
from handson.sim.isaaclab.e_collection_env import HandsOnEIsaacLabEnv
from handson.sim.isaaclab.physx_ground_truth import PhysxGroundTruthTrace


class _Robot:
    def __init__(self, side: str, joint_count: int) -> None:
        self.joint_names = [f"{side}_{index}" for index in range(joint_count)]
        body_state = np.zeros((1, 2, 13), dtype=np.float64)
        body_state[0, 1] = np.arange(13, dtype=np.float64) + 10.0
        self.data = SimpleNamespace(
            joint_pos=np.arange(joint_count, dtype=np.float64)[None, :] + 0.1,
            joint_vel=np.arange(joint_count, dtype=np.float64)[None, :] + 0.2,
            body_state_w=body_state,
        )
        self.position_target = None
        self.joint_velocity_writes: list[np.ndarray] = []

    def write_joint_state_to_sim(
        self,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
    ) -> None:
        self.data.joint_pos = np.asarray(joint_position).copy()
        self.data.joint_vel = np.asarray(joint_velocity).copy()

    def set_joint_position_target(self, target: np.ndarray) -> None:
        self.position_target = np.asarray(target).copy()

    def write_joint_velocity_to_sim(self, velocity: np.ndarray) -> None:
        self.data.joint_vel = np.asarray(velocity).copy()
        self.joint_velocity_writes.append(self.data.joint_vel.copy())

    def write_data_to_sim(self) -> None:
        pass

    def update(self, _dt: float) -> None:
        pass


class _RigidView:
    def __init__(self) -> None:
        self.transforms = np.asarray(
            [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9]]
        )
        self.velocities = np.asarray(
            [[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]
        )

    def get_transforms(self) -> np.ndarray:
        return self.transforms

    def get_velocities(self) -> np.ndarray:
        return self.velocities

    def set_transforms(self, values: np.ndarray) -> None:
        self.transforms = np.asarray(values).copy()

    def set_velocities(self, values: np.ndarray) -> None:
        self.velocities = np.asarray(values).copy()


class _ArticulationView:
    def __init__(self) -> None:
        self.root_transforms = np.asarray(
            [[3.0, 2.0, 1.0, 0.4, 0.5, 0.6, 0.7]]
        )
        self.root_velocities = np.asarray(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
        )
        self.dof_positions = np.asarray([[0.25, 0.5]])
        self.dof_velocities = np.asarray([[0.75, 1.0]])

    def get_root_transforms(self) -> np.ndarray:
        return self.root_transforms

    def get_root_velocities(self) -> np.ndarray:
        return self.root_velocities

    def get_dof_positions(self) -> np.ndarray:
        return self.dof_positions

    def get_dof_velocities(self) -> np.ndarray:
        return self.dof_velocities

    def set_root_transforms(self, values: np.ndarray) -> None:
        self.root_transforms = np.asarray(values).copy()

    def set_root_velocities(self, values: np.ndarray) -> None:
        self.root_velocities = np.asarray(values).copy()

    def set_dof_positions(self, values: np.ndarray) -> None:
        self.dof_positions = np.asarray(values).copy()

    def set_dof_velocities(self, values: np.ndarray) -> None:
        self.dof_velocities = np.asarray(values).copy()

    def get_metatype(self, _index: int) -> SimpleNamespace:
        return SimpleNamespace(dof_indices={"hinge": 0, "slider": 1})


class _FixedArticulationView(_ArticulationView):
    def get_dof_positions(self) -> np.ndarray:
        return np.zeros((1, 0), dtype=np.float64)

    def get_dof_velocities(self) -> np.ndarray:
        return np.zeros((1, 0), dtype=np.float64)

    def get_metatype(self, _index: int) -> SimpleNamespace:
        return SimpleNamespace(dof_indices={})


class _DeformableView:
    def __init__(self) -> None:
        self.positions = np.asarray(
            [[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]],
            dtype=np.float64,
        )
        self.velocities = np.asarray(
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
            dtype=np.float64,
        )

    def get_simulation_nodal_positions(self) -> np.ndarray:
        return self.positions

    def get_simulation_nodal_velocities(self) -> np.ndarray:
        return self.velocities

    def set_simulation_nodal_positions(self, values: np.ndarray) -> None:
        self.positions = np.asarray(values).copy()

    def set_simulation_nodal_velocities(self, values: np.ndarray) -> None:
        self.velocities = np.asarray(values).copy()


class _IndexedDeformableView:
    def __init__(self) -> None:
        self.positions = np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
            ],
            dtype=np.float64,
        )
        self.velocities = np.ones_like(self.positions)
        self.velocity_writes: list[tuple[np.ndarray, np.ndarray]] = []

    def get_simulation_nodal_positions(self) -> np.ndarray:
        return self.positions

    def get_simulation_nodal_velocities(self) -> np.ndarray:
        return self.velocities

    def set_simulation_nodal_positions(self, *_args) -> None:
        raise AssertionError("velocity clamp must not rewrite nodal positions")

    def set_simulation_nodal_velocities(
        self,
        values: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        value_array = np.asarray(values)
        index_array = np.asarray(indices, dtype=np.int64)
        self.velocity_writes.append(
            (value_array.copy(), index_array.copy())
        )
        self.velocities[index_array] = value_array[index_array]


class _SceneRigidObject:
    def __init__(self, *, kinematic: bool) -> None:
        self.cfg = SimpleNamespace(
            spawn=SimpleNamespace(
                rigid_props=SimpleNamespace(
                    kinematic_enabled=kinematic,
                )
            )
        )
        self.data = SimpleNamespace(
            root_state_w=np.arange(13, dtype=np.float64)[None, :]
        )
        self.velocity_writes: list[np.ndarray] = []

    def write_root_velocity_to_sim(self, values: np.ndarray) -> None:
        self.velocity_writes.append(np.asarray(values).copy())


class _TorchArrayTemplate:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values)
        self.dtype = object()
        self.device = object()

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class _ProxyArray:
    def __init__(self, values: np.ndarray) -> None:
        self.torch = _TorchArrayTemplate(values)
        self.dtype = object()
        self.device = object()


class _FakeTorchModule:
    def __init__(self) -> None:
        self.payload = object()
        self.calls: list[tuple[np.ndarray, object, object]] = []

    def as_tensor(self, values, *, dtype, device):
        self.calls.append((np.asarray(values).copy(), dtype, device))
        return self.payload


class _ProxySceneRigidObject(_SceneRigidObject):
    def __init__(self) -> None:
        super().__init__(kinematic=False)
        self.data.root_state_w = _ProxyArray(
            np.arange(13, dtype=np.float32)[None, :]
        )
        self.velocity_writes: list[object] = []

    def write_root_velocity_to_sim(self, values) -> None:
        self.velocity_writes.append(values)


class _ContactReportInterface:
    def __init__(self, report: tuple[list, list, list]) -> None:
        self.report = report
        self.callback = None

    def get_full_contact_report(self) -> tuple[list, list, list]:
        return self.report

    def subscribe_contact_report_events(self, callback):
        self.callback = callback
        return SimpleNamespace(active=True)

    def emit(self) -> None:
        self.callback(self.report[0], self.report[1])


def _contact_report(*, point_count: int = 1) -> tuple[list, list, list]:
    points = [
        SimpleNamespace(
            position=(1.0 + index, 2.0, 3.0),
            normal=(0.0, 0.0, 1.0),
            impulse=(0.1, 0.2, 0.3 + index),
            separation=-0.004,
            face_index0=-1,
            face_index1=4 + index,
            material0=101,
            material1=102,
        )
        for index in range(point_count)
    ]
    anchors = [
        SimpleNamespace(
            position=(4.0, 5.0, 6.0),
            impulse=(0.4, 0.5, 0.6),
        )
    ]
    headers = [
        SimpleNamespace(
            actor0=11,
            actor1=12,
            collider0=21,
            collider1=22,
            stage_id=31,
            contact_data_offset=0,
            num_contact_data=point_count,
            friction_anchors_offset=0,
            num_friction_anchors_data=1,
            proto_index0=-1,
            proto_index1=7,
            type=SimpleNamespace(value=2),
        )
    ]
    return headers, points, anchors


def _configure_contact_signature_paths(trace: PhysxGroundTruthTrace) -> None:
    paths = {
        11: "/World/A",
        12: "/World/B",
        13: "/World/C",
        14: "/World/D",
        21: "/World/A/Collider",
        22: "/World/B/Collider",
        23: "/World/C/Collider",
        24: "/World/D/Collider",
    }
    trace._contact_report_paths = ("/World/A", "/World/C")
    trace._contact_report_path_table = tuple(
        {"id": process_id, "path": path}
        for process_id, path in paths.items()
    )


def _two_pair_contact_report(
    *,
    reverse_headers: bool = False,
    reverse_points: bool = False,
) -> tuple[list, list, list]:
    pairs = [
        ((11, 12, 21, 22), -0.004, 0.3, 0.6),
        ((13, 14, 23, 24), -0.002, 0.7, 0.9),
    ]
    if reverse_headers:
        pairs.reverse()
    headers = []
    points = []
    anchors = []
    for pair_index, (ids, separation, normal_impulse, friction_impulse) in enumerate(
        pairs
    ):
        pair_points = [
            SimpleNamespace(
                position=(float(pair_index), float(point_index), 0.0),
                normal=(0.0, 0.0, 1.0),
                impulse=(0.0, 0.0, normal_impulse + point_index * 0.1),
                separation=separation - point_index * 0.001,
                face_index0=-1,
                face_index1=-1,
                material0=0,
                material1=0,
            )
            for point_index in range(2)
        ]
        pair_anchors = [
            SimpleNamespace(
                position=(float(pair_index), float(anchor_index), 0.0),
                impulse=(friction_impulse + anchor_index * 0.1, 0.0, 0.0),
            )
            for anchor_index in range(2)
        ]
        if reverse_points:
            pair_points.reverse()
            pair_anchors.reverse()
        point_offset = len(points)
        anchor_offset = len(anchors)
        points.extend(pair_points)
        anchors.extend(pair_anchors)
        actor0, actor1, collider0, collider1 = ids
        if reverse_headers:
            actor0, actor1 = actor1, actor0
            collider0, collider1 = collider1, collider0
        headers.append(
            SimpleNamespace(
                actor0=actor0,
                actor1=actor1,
                collider0=collider0,
                collider1=collider1,
                stage_id=999 if reverse_headers else 31,
                contact_data_offset=point_offset,
                num_contact_data=len(pair_points),
                friction_anchors_offset=anchor_offset,
                num_friction_anchors_data=len(pair_anchors),
                proto_index0=-1,
                proto_index1=-1,
                type=SimpleNamespace(value=2),
            )
        )
    return headers, points, anchors


def _fake_env() -> SimpleNamespace:
    right = _Robot("right", 3)
    left = _Robot("left", 2)
    rigid = _RigidView()
    articulation = _ArticulationView()
    deformable = _DeformableView()
    return SimpleNamespace(
        _robots={"right": right, "left": left},
        _ee_body_indices={"right": 1, "left": 1},
        _joint_targets={
            "right": np.asarray([[1.0, 2.0, 3.0]]),
            "left": np.asarray([[4.0, 5.0]]),
        },
        _task_reset_body_states=[
            ("/World/Task/cube", rigid, None, None, None)
        ],
        _task_reset_articulation_states=[
            (
                "/World/Task/drawer",
                articulation,
                None,
                None,
                None,
                None,
                None,
            )
        ],
        _e_nodes_np=np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
        _e_velocities_np=np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        _e_garment_view=deformable,
        _e_garment_indices=None,
        _sim=SimpleNamespace(current_time=2.5),
        _physics_dt=1.0 / 120.0,
        _control_dt=1.0 / 30.0,
        _n_substeps=4,
        _isaac_device="cuda:0",
        env_step=7,
        _hand_action_dim=20,
        _scene=None,
    )


def test_snapshot_records_actual_robot_and_task_readback() -> None:
    trace = PhysxGroundTruthTrace(_fake_env())
    snapshot = trace.snapshot()

    assert snapshot["physx_robot_joint_position"].shape == (2, 3)
    np.testing.assert_allclose(
        snapshot["physx_robot_joint_position"][0],
        [0.1, 1.1, 2.1],
    )
    np.testing.assert_allclose(
        snapshot["physx_robot_joint_position"][1, :2],
        [0.1, 1.1],
    )
    assert snapshot["physx_robot_joint_position"][1, 2] == 0.0
    np.testing.assert_array_equal(
        snapshot["physx_robot_joint_valid"],
        [[1, 1, 1], [1, 1, 0]],
    )
    np.testing.assert_allclose(
        snapshot["physx_tcp_state"][0],
        np.arange(13, dtype=np.float64) + 10.0,
    )
    np.testing.assert_allclose(
        snapshot["physx_task_rigid_body_state"][0],
        [1, 2, 3, 0.9, 0.1, 0.2, 0.3, 4, 5, 6, 7, 8, 9],
    )
    np.testing.assert_allclose(
        snapshot["physx_task_articulation_root_state"][0],
        [3, 2, 1, 0.7, 0.4, 0.5, 0.6, 1, 2, 3, 4, 5, 6],
    )
    np.testing.assert_allclose(
        snapshot["physx_task_articulation_dof_position"][0],
        [0.25, 0.5],
    )
    np.testing.assert_allclose(
        snapshot["physx_deformable_position"],
        [[0, 0, 0], [1, 2, 3]],
    )
    np.testing.assert_allclose(snapshot["physx_sim_time"], [2.5])
    np.testing.assert_allclose(snapshot["physx_control_step"], [7.0])


def test_zero_task_velocities_does_not_rewrite_task_poses() -> None:
    env = _fake_env()
    trace = PhysxGroundTruthTrace(env)
    rigid = env._task_reset_body_states[0][1]
    articulation = env._task_reset_articulation_states[0][1]
    rigid_transform = rigid.get_transforms().copy()
    articulation_transform = articulation.get_root_transforms().copy()
    articulation_dof_position = articulation.get_dof_positions().copy()

    zeroed = trace.zero_task_velocities()

    assert zeroed == {
        "rigid_bodies": 1,
        "articulation_roots": 1,
        "articulation_dofs": 2,
        "deformable_nodes": 2,
    }
    np.testing.assert_array_equal(rigid.get_velocities(), 0.0)
    np.testing.assert_array_equal(articulation.get_root_velocities(), 0.0)
    np.testing.assert_array_equal(articulation.get_dof_velocities(), 0.0)
    np.testing.assert_array_equal(env._e_garment_view.velocities, 0.0)
    np.testing.assert_array_equal(env._e_velocities_np, 0.0)
    np.testing.assert_array_equal(rigid.get_transforms(), rigid_transform)
    np.testing.assert_array_equal(
        articulation.get_root_transforms(),
        articulation_transform,
    )
    np.testing.assert_array_equal(
        articulation.get_dof_positions(),
        articulation_dof_position,
    )


def test_zero_velocity_boundary_covers_robot_and_task_without_pose_write() -> None:
    env = _fake_env()
    trace = PhysxGroundTruthTrace(env)
    snapshot = trace.snapshot()
    rigid = env._task_reset_body_states[0][1]
    articulation = env._task_reset_articulation_states[0][1]
    robot_positions = {
        side: robot.data.joint_pos.copy()
        for side, robot in env._robots.items()
    }
    rigid_transform = rigid.get_transforms().copy()
    articulation_transform = articulation.get_root_transforms().copy()
    articulation_dof_position = articulation.get_dof_positions().copy()

    counts = trace.write_zero_velocity_boundary(
        snapshot,
        trace.schema(),
        sync=False,
    )

    assert counts == {
        "robot_joints": 5,
        "rigid_bodies": 1,
        "articulation_roots": 1,
        "articulation_dofs": 2,
        "deformable_nodes": 2,
    }
    for side, robot in env._robots.items():
        np.testing.assert_array_equal(robot.data.joint_vel, 0.0)
        np.testing.assert_array_equal(robot.data.joint_pos, robot_positions[side])
        np.testing.assert_array_equal(
            robot.position_target,
            robot_positions[side],
        )
    np.testing.assert_array_equal(rigid.get_transforms(), rigid_transform)
    np.testing.assert_array_equal(
        articulation.get_root_transforms(),
        articulation_transform,
    )
    np.testing.assert_array_equal(
        articulation.get_dof_positions(),
        articulation_dof_position,
    )
    np.testing.assert_array_equal(rigid.get_velocities(), 0.0)
    np.testing.assert_array_equal(articulation.get_root_velocities(), 0.0)
    np.testing.assert_array_equal(articulation.get_dof_velocities(), 0.0)
    np.testing.assert_array_equal(env._e_garment_view.velocities, 0.0)


def test_zero_velocity_boundary_fails_before_writes_on_invalid_mask() -> None:
    env = _fake_env()
    trace = PhysxGroundTruthTrace(env)
    snapshot = trace.snapshot()
    snapshot["physx_task_articulation_root_valid"][0] = 0
    rigid_velocity = env._task_reset_body_states[0][1].get_velocities().copy()

    with pytest.raises(ValueError, match="articulation root"):
        trace.write_zero_velocity_boundary(
            snapshot,
            trace.schema(),
            sync=False,
        )

    for robot in env._robots.values():
        assert robot.joint_velocity_writes == []
    np.testing.assert_array_equal(
        env._task_reset_body_states[0][1].get_velocities(),
        rigid_velocity,
    )


def test_zero_velocity_boundary_accepts_snapshot_without_articulations() -> None:
    env = _fake_env()
    env._task_reset_articulation_states = []
    trace = PhysxGroundTruthTrace(env)
    snapshot = trace.snapshot()

    assert "physx_task_articulation_root_state" not in snapshot
    assert "physx_task_articulation_dof_position" not in snapshot

    counts = trace.write_zero_velocity_boundary(
        snapshot,
        trace.schema(),
        sync=False,
    )

    assert counts["articulation_roots"] == 0
    assert counts["articulation_dofs"] == 0


def test_zero_task_velocities_syncs_live_deformable_position_cache() -> None:
    env = _fake_env()
    live_positions = np.asarray(
        [[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]]],
        dtype=np.float64,
    )
    env._e_garment_view.positions = live_positions.copy()
    env._e_last_valid_nodes_np = np.full((2, 3), -1.0)
    env._e_last_valid_velocities_np = np.full((2, 3), -1.0)
    env._e_last_control_nodes_np = np.full((2, 3), -1.0)
    accepted_diagnostics: list[dict[str, float]] = []

    def accept_readback(nodes, velocities, diagnostics) -> None:
        env._e_nodes_np = np.asarray(nodes).copy()
        env._e_velocities_np = np.asarray(velocities).copy()
        env._e_last_valid_nodes_np = env._e_nodes_np.copy()
        env._e_last_valid_velocities_np = env._e_velocities_np.copy()
        env._e_last_control_nodes_np = env._e_nodes_np.copy()
        accepted_diagnostics.append(dict(diagnostics))

    env._accept_garment_readback = accept_readback
    trace = PhysxGroundTruthTrace(env)

    def forbid_position_write(*_args) -> None:
        raise AssertionError("velocity clamp must not rewrite nodal positions")

    env._e_garment_view.set_simulation_nodal_positions = (
        forbid_position_write
    )

    trace.zero_task_velocities()

    np.testing.assert_array_equal(env._e_nodes_np, live_positions[0])
    np.testing.assert_array_equal(
        env._e_garment_view.positions,
        live_positions,
    )
    np.testing.assert_array_equal(env._e_velocities_np, 0.0)
    np.testing.assert_array_equal(
        env._e_last_valid_nodes_np,
        live_positions[0],
    )
    np.testing.assert_array_equal(env._e_last_valid_velocities_np, 0.0)
    np.testing.assert_array_equal(
        env._e_last_control_nodes_np,
        live_positions[0],
    )
    assert accepted_diagnostics == [
        {
            "max_control_node_displacement": 0.0,
            "max_nodal_speed": 0.0,
        }
    ]


def test_zero_task_velocities_rejects_invalid_surface_without_poisoning_cache(
) -> None:
    env = _fake_env()
    last_valid_nodes = env._e_nodes_np.copy()
    last_valid_velocities = env._e_velocities_np.copy()
    env._e_last_valid_nodes_np = last_valid_nodes.copy()
    env._e_last_valid_velocities_np = last_valid_velocities.copy()
    rejected: list[str] = []

    def validate(nodes, _velocities, *, expected_node_count):
        assert expected_node_count == 2
        if not np.all(np.isfinite(nodes)):
            return "nonfinite_nodal_positions", {"readback_node_count": 2}
        return None, {}

    def reject(reason, _diagnostics) -> None:
        rejected.append(reason)
        env._e_nodes_np = env._e_last_valid_nodes_np.copy()
        env._e_velocities_np = env._e_last_valid_velocities_np.copy()

    def forbid_accept(*_args) -> None:
        raise AssertionError("invalid nodes must never become last-valid")

    env._basic_nodal_state_issue = validate
    env._reject_garment_readback = reject
    env._accept_garment_readback = forbid_accept
    trace = PhysxGroundTruthTrace(env)
    env._e_garment_view.positions[0, 0, 0] = np.nan
    velocities_before = env._e_garment_view.velocities.copy()

    with pytest.raises(RuntimeError, match="nonfinite_nodal_positions"):
        trace.zero_task_velocities()

    assert rejected == ["nonfinite_nodal_positions"]
    np.testing.assert_array_equal(env._e_nodes_np, last_valid_nodes)
    np.testing.assert_array_equal(
        env._e_velocities_np,
        last_valid_velocities,
    )
    np.testing.assert_array_equal(
        env._e_garment_view.velocities,
        velocities_before,
    )


def test_zero_task_velocities_syncs_rope_centerline_cache() -> None:
    env = _fake_env()
    del env._e_nodes_np
    del env._e_velocities_np
    live_positions = np.asarray(
        [[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]]],
        dtype=np.float64,
    )
    rope_view = _DeformableView()
    rope_view.positions = live_positions.copy()
    env._e02_nodes_np = np.zeros((2, 3), dtype=np.float64)
    env._e02_nodal_velocities_np = np.ones((2, 3), dtype=np.float64)
    env._e02_centers_np = np.zeros((2, 3), dtype=np.float64)
    env._e02_velocities_np = np.ones((2, 3), dtype=np.float64)
    env._e02_rope_view = rope_view
    env._e02_rope_indices = None
    env._centerline_state_from_nodes = lambda nodes, velocities: (
        np.asarray(nodes),
        np.asarray(velocities),
    )
    trace = PhysxGroundTruthTrace(env)

    zeroed = trace.zero_task_velocities()

    assert zeroed["deformable_nodes"] == 2
    np.testing.assert_array_equal(env._e02_nodes_np, live_positions[0])
    np.testing.assert_array_equal(env._e02_nodal_velocities_np, 0.0)
    np.testing.assert_array_equal(env._e02_centers_np, live_positions[0])
    np.testing.assert_array_equal(env._e02_velocities_np, 0.0)


def test_zero_task_velocities_passes_deformable_indices_to_setter() -> None:
    env = _fake_env()
    view = _IndexedDeformableView()
    env._e_garment_view = view
    env._e_garment_indices = np.asarray([1], dtype=np.int64)
    trace = PhysxGroundTruthTrace(env)

    zeroed = trace.zero_task_velocities()

    assert zeroed["deformable_nodes"] == 2
    assert len(view.velocity_writes) == 1
    written_velocities, written_indices = view.velocity_writes[0]
    np.testing.assert_array_equal(written_indices, [1])
    np.testing.assert_array_equal(written_velocities, 0.0)
    np.testing.assert_array_equal(view.velocities[0], 1.0)
    np.testing.assert_array_equal(view.velocities[1], 0.0)
    np.testing.assert_array_equal(env._e_nodes_np, view.positions[1])


def test_zero_task_velocities_uses_surface_backend_array_helper() -> None:
    env = _fake_env()
    calls: list[tuple[object, np.ndarray]] = []
    backend_payload = object()
    forwarded_payloads: list[object] = []

    def surface_array_like(template, values):
        calls.append((template, np.asarray(values).copy()))
        return backend_payload

    def capture_velocity_write(values) -> None:
        forwarded_payloads.append(values)

    env._surface_array_like = surface_array_like
    env._e_garment_view.set_simulation_nodal_velocities = (
        capture_velocity_write
    )
    trace = PhysxGroundTruthTrace(env)

    trace.zero_task_velocities()

    assert len(calls) == 1
    assert calls[0][0] is env._e_garment_view.velocities
    np.testing.assert_array_equal(calls[0][1], 0.0)
    assert forwarded_payloads == [backend_payload]


def test_zero_task_velocities_skips_scene_kinematic_rigid_object() -> None:
    env = _fake_env()
    env._task_reset_body_states = []
    env._task_reset_articulation_states = []
    del env._e_nodes_np
    del env._e_velocities_np
    dynamic_object = _SceneRigidObject(kinematic=False)
    kinematic_object = _SceneRigidObject(kinematic=True)
    env._scene = SimpleNamespace(
        rigid_objects={
            "dynamic_object": dynamic_object,
            "kinematic_table": kinematic_object,
        }
    )
    trace = PhysxGroundTruthTrace(env)

    zeroed = trace.zero_task_velocities()

    assert zeroed["rigid_bodies"] == 1
    assert len(dynamic_object.velocity_writes) == 1
    np.testing.assert_array_equal(dynamic_object.velocity_writes[0], 0.0)
    assert kinematic_object.velocity_writes == []


def test_zero_task_velocities_unwraps_isaaclab_v3_proxy_array() -> None:
    env = _fake_env()
    env._task_reset_body_states = []
    env._task_reset_articulation_states = []
    del env._e_nodes_np
    del env._e_velocities_np
    torch_module = _FakeTorchModule()
    env._torch = torch_module
    dynamic_object = _ProxySceneRigidObject()
    env._scene = SimpleNamespace(
        rigid_objects={"dynamic_object": dynamic_object}
    )
    trace = PhysxGroundTruthTrace(env)

    zeroed = trace.zero_task_velocities()

    assert zeroed["rigid_bodies"] == 1
    assert len(torch_module.calls) == 1
    values, dtype, device = torch_module.calls[0]
    np.testing.assert_array_equal(values, np.zeros((1, 6)))
    assert dtype is dynamic_object.data.root_state_w.torch.dtype
    assert device is dynamic_object.data.root_state_w.torch.device
    assert dtype is not dynamic_object.data.root_state_w.dtype
    assert dynamic_object.velocity_writes == [torch_module.payload]


def test_reset_contract_requires_one_exact_t0_restore() -> None:
    env = object.__new__(BimanualAssemblyIsaacLabEnv)
    env._reset_physics_steps = 120
    env._physics_dt = 1.0 / 120.0
    env._control_dt = 1.0 / 30.0
    env._n_substeps = 4
    env._isaac_device = "cuda:0"
    env._physical_recording_origin = {"stability_metric": "velocity"}

    contract = env.get_physical_replay_reset_contract()

    assert contract["allow_pre_action_state_overwrite"] is True
    assert contract["allows_single_recorded_t0_restore"] is True
    assert contract["forbids_state_writes_after_action_start"] is True
    assert contract["physics_steps_after_exact_recorded_t0_restore"] == 0
    assert contract["strategy"] == (
        "seeded_stable_reset_then_exact_recorded_physx_t0_restore"
    )


def test_schema_documents_alignment_and_physical_layout() -> None:
    schema = PhysxGroundTruthTrace(_fake_env()).schema()

    assert schema["name"] == "handson.physx_ground_truth"
    assert schema["version"] == 6
    assert "before action[t]" in schema["description"]
    assert "after action[t]" in schema["description"]
    assert schema["physx"]["robot_sides"] == ["right", "left"]
    assert schema["physx"]["robot_joint_names"]["left"] == ["left_0", "left_1"]
    assert schema["physx"]["rigid_body_names"] == ["/World/Task/cube"]
    assert schema["physx"]["certification_forbids_state_writes"] is False
    assert schema["physx"]["allows_single_recorded_t0_restore"] is True
    assert (
        schema["physx"][
            "certification_forbids_state_writes_after_action_start"
        ]
        is True
    )
    assert schema["physx"]["articulation_dof_names"][
        "/World/Task/drawer"
    ] == ["hinge", "slider"]
    assert schema["physx"]["deformable_name"] == "surface"
    assert schema["physx"]["physics_config"] == {
        "device": "cuda:0",
        "physics_dt": 1.0 / 120.0,
        "live_physics_dt": 1.0 / 120.0,
        "control_dt": 1.0 / 30.0,
        "n_substeps": 4,
        "simulation_dt": None,
        "render_interval": None,
        "solver": {},
    }
    assert schema["physx"]["requires_cuda_backend"] is True
    assert schema["fields"]["physx_tcp_state"]["shape"] == [2, 13]
    assert schema["fields"]["physx_deformable_position"]["shape"] == [2, 3]
    assert schema["fields"]["physx_substep_robot_joint_position"]["shape"] == [
        4,
        2,
        3,
    ]
    assert schema["fields"]["physx_substep_contact_point_data"]["shape"] == [
        4,
        512,
        10,
    ]
    assert schema["fields"]["physx_contact_point_data"][
        "physical_replay"
    ] is False
    assert schema["fields"]["physx_contact_signature_pair_metrics"][
        "physical_replay"
    ] is False
    assert schema["fields"]["physx_contact_signature_pair_metrics"][
        "replay_restore_supported"
    ] is False
    assert schema["fields"]["physx_substep_contact_signature_pair_metrics"][
        "physical_replay"
    ] is False
    assert schema["physx"]["contact_report"][
        "canonical_metric_replay_policy"
    ] == "diagnostic_only_solver_output"
    assert schema["fields"]["physx_robot_joint_effort"][
        "physical_replay"
    ] is True
    assert schema["fields"]["physx_robot_joint_effort"][
        "replay_restore_supported"
    ] is False
    assert schema["fields"]["physx_substep_robot_joint_effort"][
        "physical_replay"
    ] is True
    assert schema["fields"]["physx_substep_robot_joint_effort"][
        "physical_replay"
    ] is True
    assert "replay_restore_supported" not in schema["fields"][
        "physx_substep_robot_joint_effort"
    ]
    assert schema["physx"]["contact_report"] == {
        "available_in_this_runtime": False,
        "capture_api": (
            "IPhysxSimulation.subscribe_contact_report_events+"
            "get_full_contact_report, with "
            "RigidContactView.get_contact_data/get_friction_data fallback"
        ),
        "buffer_lifetime": "one_simulation_step",
        "post_step_callback_copy": True,
        "capture_frequency": "every_physics_substep",
        "overflow_policy": "raise_without_writing_partial_data",
        "max_headers": 128,
        "max_contact_points": 512,
        "max_friction_anchors": 512,
        "report_enabled_task_body_paths": [],
        "path_id_encoding": "PhysicsSchemaTools.sdfPathToInt",
        "path_table": [],
        "capture_source_encoding": {
            "0": "unavailable",
            "1": "full_contact_report",
            "2": "rigid_contact_tensor",
        },
        "tensor_fallback_available_in_this_runtime": False,
        "tensor_sensor_paths": [],
        "tensor_filter_paths": [],
        "tensor_event_type_sentinel": int(np.uint32(0xFFFFFFFE)),
        "tensor_source_actor_and_collider_ids_are_body_ids": True,
        "tensor_source_face_and_material_ids_available": False,
        "tensor_source_friction_rows_are_points_not_cache_anchors": True,
        "reported_contact_points_are_manifold_snapshot": True,
            "reported_friction_anchors_are_internal_cache": False,
            "reported_impulses_are_current_step_solver_output": True,
            "canonical_metric_replay_policy": (
                "diagnostic_only_solver_output"
            ),
            "solver_warm_start_state_available": False,
        "replay_restore_supported": False,
    }


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
def test_schema_rejects_cpu_or_noncanonical_backend(device: str) -> None:
    env = _fake_env()
    env._isaac_device = device

    with pytest.raises(RuntimeError, match="canonical CUDA device"):
        PhysxGroundTruthTrace(env).schema()


def test_contact_report_records_manifold_impulses_and_friction_anchors() -> None:
    env = _fake_env()
    env._physx_contact_report_interface = _ContactReportInterface(
        _contact_report()
    )
    trace = PhysxGroundTruthTrace(env)
    _configure_contact_signature_paths(trace)
    env._physx_contact_report_interface.emit()

    snapshot = trace.snapshot()

    np.testing.assert_array_equal(snapshot["physx_contact_report_valid"], [1])
    np.testing.assert_array_equal(snapshot["physx_contact_capture_source"], [1])
    np.testing.assert_array_equal(snapshot["physx_contact_header_count"], [1])
    np.testing.assert_array_equal(snapshot["physx_contact_point_count"], [1])
    np.testing.assert_array_equal(snapshot["physx_friction_anchor_count"], [1])
    np.testing.assert_array_equal(
        snapshot["physx_contact_header_ids"][0],
        [11, 12, 21, 22, 31],
    )
    np.testing.assert_array_equal(
        snapshot["physx_contact_header_meta"][0],
        [2, 0, 1, 0, 1, np.iinfo(np.uint32).max, 7],
    )
    np.testing.assert_allclose(
        snapshot["physx_contact_point_data"][0],
        [1, 2, 3, 0, 0, 1, 0.1, 0.2, 0.3, -0.004],
    )
    np.testing.assert_array_equal(
        snapshot["physx_contact_point_face_indices"][0],
        [np.iinfo(np.uint32).max, 4],
    )
    np.testing.assert_array_equal(
        snapshot["physx_contact_point_material_ids"][0],
        [101, 102],
    )
    np.testing.assert_allclose(
        snapshot["physx_friction_anchor_data"][0],
        [4, 5, 6, 0.4, 0.5, 0.6],
    )
    np.testing.assert_array_equal(
        snapshot["physx_contact_point_data"][1:],
        0,
    )
    np.testing.assert_array_equal(
        snapshot["physx_contact_signature_valid"],
        [1],
    )
    np.testing.assert_array_equal(
        snapshot["physx_contact_signature_pair_count"],
        [1],
    )
    np.testing.assert_allclose(
        snapshot["physx_contact_signature_pair_metrics"][0],
        [0.004, 0.3, np.linalg.norm([0.4, 0.5, 0.6])],
    )

    trace.begin_substep_capture()
    trace.capture_substep()
    substeps = trace.substep_snapshot()
    np.testing.assert_array_equal(
        substeps["physx_substep_contact_report_valid"][0],
        [1],
    )
    np.testing.assert_allclose(
        substeps["physx_substep_friction_anchor_data"][0, 0],
        [4, 5, 6, 0.4, 0.5, 0.6],
    )
    np.testing.assert_array_equal(
        substeps["physx_substep_contact_signature_valid"][0],
        [1],
    )


def test_contact_signature_is_independent_of_process_ids_and_buffer_order() -> None:
    first_env = _fake_env()
    first_env._physx_contact_report_interface = _ContactReportInterface(
        _two_pair_contact_report()
    )
    first = PhysxGroundTruthTrace(first_env)
    _configure_contact_signature_paths(first)

    second_env = _fake_env()
    second_env._physx_contact_report_interface = _ContactReportInterface(
        _two_pair_contact_report(reverse_headers=True, reverse_points=True)
    )
    second = PhysxGroundTruthTrace(second_env)
    _configure_contact_signature_paths(second)
    # The process-local ids change, while their resolved paths stay the same.
    second._contact_report_path_table = tuple(
        {"id": int(entry["id"]) + 1000, "path": entry["path"]}
        for entry in second._contact_report_path_table
    )
    headers, points, anchors = second_env._physx_contact_report_interface.report
    for header in headers:
        header.actor0 += 1000
        header.actor1 += 1000
        header.collider0 += 1000
        header.collider1 += 1000
    second_env._physx_contact_report_interface.report = (
        headers,
        points,
        anchors,
    )

    first_snapshot = first.snapshot()
    second_snapshot = second.snapshot()
    signature_fields = (
        "physx_contact_signature_valid",
        "physx_contact_signature_pair_count",
        "physx_contact_signature_pair_path_indices",
        "physx_contact_signature_pair_counts",
        "physx_contact_signature_pair_metrics",
    )
    for name in signature_fields:
        np.testing.assert_allclose(first_snapshot[name], second_snapshot[name])


@pytest.mark.parametrize("failure", ["empty_coverage", "missing_path"])
def test_contact_signature_fails_closed_without_path_coverage(failure: str) -> None:
    env = _fake_env()
    env._physx_contact_report_interface = _ContactReportInterface(
        _contact_report()
    )
    trace = PhysxGroundTruthTrace(env)
    _configure_contact_signature_paths(trace)
    if failure == "empty_coverage":
        trace._contact_report_paths = ()
    else:
        trace._contact_report_path_table = tuple(
            entry
            for entry in trace._contact_report_path_table
            if entry["id"] != 22
        )

    snapshot = trace.snapshot()

    np.testing.assert_array_equal(
        snapshot["physx_contact_signature_valid"],
        [0],
    )


@pytest.mark.parametrize("failure", ["orphan", "overlap"])
def test_contact_signature_requires_exact_header_buffer_coverage(
    failure: str,
) -> None:
    report = _contact_report(point_count=2)
    headers, points, anchors = report
    if failure == "orphan":
        headers[0].num_contact_data = 1
    else:
        headers.append(SimpleNamespace(**vars(headers[0])))
    env = _fake_env()
    env._physx_contact_report_interface = _ContactReportInterface(
        (headers, points, anchors)
    )
    trace = PhysxGroundTruthTrace(env)
    _configure_contact_signature_paths(trace)
    env._physx_contact_report_interface.emit()

    snapshot = trace.snapshot()

    np.testing.assert_array_equal(
        snapshot["physx_contact_signature_valid"],
        [0],
    )


def _tensor_contact_signature_fields(
    rows: list[list[float]],
) -> tuple[PhysxGroundTruthTrace, dict[str, np.ndarray]]:
    trace = object.__new__(PhysxGroundTruthTrace)
    trace.max_contact_headers = 4
    trace.max_contact_points = 8
    trace.max_friction_anchors = 8
    trace._contact_report_paths = ()
    trace._tensor_contact_sensor_paths = ("/World/Cube",)
    trace._tensor_contact_filter_paths = ("/World/Table",)
    trace._contact_report_path_table = (
        {"id": 41, "path": "/World/Cube"},
        {"id": 42, "path": "/World/Table"},
    )
    fields = trace._empty_contact_fields()
    fields["physx_contact_report_valid"][0] = 1
    fields["physx_contact_capture_source"][0] = 2
    fields["physx_contact_header_count"][0] = 1
    fields["physx_contact_header_ids"][0] = [41, 42, 41, 42, 99]
    fields["physx_contact_header_meta"][0, 1:5] = [0, len(rows), 0, 0]
    fields["physx_contact_point_count"][0] = len(rows)
    fields["physx_contact_point_data"][: len(rows)] = rows
    return trace, fields


def test_tensor_contact_signature_ignores_exact_zero_padding_row() -> None:
    trace, fields = _tensor_contact_signature_fields(
        [
            [1, 2, 3, 0, 0, 1, 0, 0, 2, -0.01],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    result = trace._canonicalize_contact_fields(fields)

    np.testing.assert_array_equal(
        result["physx_contact_signature_valid"],
        [1],
    )
    np.testing.assert_array_equal(
        result["physx_contact_signature_pair_counts"][0],
        [1, 0],
    )
    np.testing.assert_allclose(
        result["physx_contact_signature_pair_metrics"][0],
        [0.01, 2.0, 0.0],
    )
    # Canonicalization must not rewrite the raw diagnostic payload/count.
    np.testing.assert_array_equal(result["physx_contact_point_count"], [2])
    np.testing.assert_array_equal(result["physx_contact_point_data"][1], 0)


def test_tensor_contact_signature_omits_padding_only_pair() -> None:
    trace, fields = _tensor_contact_signature_fields(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )

    result = trace._canonicalize_contact_fields(fields)

    np.testing.assert_array_equal(
        result["physx_contact_signature_valid"],
        [1],
    )
    np.testing.assert_array_equal(
        result["physx_contact_signature_pair_count"],
        [0],
    )


def test_full_contact_report_does_not_filter_exact_zero_row() -> None:
    trace, fields = _tensor_contact_signature_fields(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )
    trace._contact_report_paths = ("/World/Cube", "/World/Table")
    fields["physx_contact_capture_source"][0] = 1

    result = trace._canonicalize_contact_fields(fields)

    np.testing.assert_array_equal(
        result["physx_contact_signature_valid"],
        [0],
    )


def test_tensor_contact_signature_rejects_nonzero_zero_normal_row() -> None:
    trace, fields = _tensor_contact_signature_fields(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )

    result = trace._canonicalize_contact_fields(fields)

    np.testing.assert_array_equal(
        result["physx_contact_signature_valid"],
        [0],
    )


def test_contact_report_overflow_fails_instead_of_truncating(monkeypatch) -> None:
    monkeypatch.setenv("HANDSON_PHYSX_MAX_CONTACT_POINTS", "1")
    env = _fake_env()
    env._physx_contact_report_interface = _ContactReportInterface(
        _contact_report(point_count=2)
    )
    trace = PhysxGroundTruthTrace(env)

    try:
        trace.snapshot()
    except RuntimeError as exc:
        assert "2 contact points" in str(exc)
        assert "HANDSON_PHYSX_MAX_CONTACT_POINTS" in str(exc)
    else:
        raise AssertionError("overflowing PhysX contact data was truncated")


def test_tensor_contact_fallback_encodes_pairs_and_impulses(monkeypatch) -> None:
    force = np.zeros((8, 1), dtype=np.float32)
    point = np.zeros((8, 3), dtype=np.float32)
    normal = np.zeros((8, 3), dtype=np.float32)
    separation = np.zeros((8, 1), dtype=np.float32)
    force[1:3, 0] = [2.0, 3.0]
    point[1:3] = [[1, 2, 3], [4, 5, 6]]
    normal[1:3] = [[0, 0, 1], [0, 1, 0]]
    separation[1:3, 0] = [-0.01, -0.02]
    friction_force = np.zeros((8, 3), dtype=np.float32)
    friction_point = np.zeros((8, 3), dtype=np.float32)
    friction_force[2] = [0.1, 0.2, 0.3]
    friction_point[2] = [7, 8, 9]
    view = SimpleNamespace(
        sensor_count=1,
        filter_count=1,
        max_contact_data_count=8,
        get_contact_data=lambda _dt: (
            force,
            point,
            normal,
            separation,
            np.asarray([[2]], dtype=np.uint32),
            np.asarray([[1]], dtype=np.uint32),
        ),
        get_friction_data=lambda _dt: (
            friction_force,
            friction_point,
            np.asarray([[1]], dtype=np.uint32),
            np.asarray([[2]], dtype=np.uint32),
        ),
    )
    path_ids = {"/World/Cube": 41, "/World/Table": 42}
    monkeypatch.setitem(
        sys.modules,
        "pxr",
        SimpleNamespace(
            PhysicsSchemaTools=SimpleNamespace(
                sdfPathToInt=lambda path: path_ids[str(path)]
            ),
            Sdf=SimpleNamespace(Path=lambda path: path),
        ),
    )
    trace = object.__new__(PhysxGroundTruthTrace)
    trace.max_contact_headers = 4
    trace.max_contact_points = 8
    trace.max_friction_anchors = 8
    trace._tensor_contact_views = (view,)
    trace._tensor_contact_view_paths = (
        (("/World/Cube",), ("/World/Table",)),
    )
    trace._contact_stage_id = 99
    trace._contact_tensor_dt = 1.0

    result = trace._sample_tensor_contact_fields()

    np.testing.assert_array_equal(result["physx_contact_capture_source"], [2])
    np.testing.assert_array_equal(result["physx_contact_header_count"], [1])
    np.testing.assert_array_equal(result["physx_contact_point_count"], [2])
    np.testing.assert_array_equal(result["physx_friction_anchor_count"], [1])
    np.testing.assert_array_equal(
        result["physx_contact_header_ids"][0],
        [41, 42, 41, 42, 99],
    )
    np.testing.assert_array_equal(
        result["physx_contact_header_meta"][0],
        [np.uint32(0xFFFFFFFE), 0, 2, 0, 1, 0xFFFFFFFF, 0xFFFFFFFF],
    )
    np.testing.assert_allclose(
        result["physx_contact_point_data"][:2],
        [
            [1, 2, 3, 0, 0, 1, 0, 0, 2, -0.01],
            [4, 5, 6, 0, 1, 0, 0, 3, 0, -0.02],
        ],
    )
    np.testing.assert_allclose(
        result["physx_friction_anchor_data"][0],
        [7, 8, 9, 0.1, 0.2, 0.3],
    )


def test_substep_snapshot_records_exact_robot_joint_path() -> None:
    env = _fake_env()
    trace = PhysxGroundTruthTrace(env)
    trace.begin_substep_capture()

    expected_right = []
    expected_left = []
    for index in range(env._n_substeps):
        right = np.asarray([[index + 0.1, index + 0.2, index + 0.3]])
        left = np.asarray([[index + 1.1, index + 1.2]])
        env._robots["right"].data.joint_pos = right
        env._robots["right"].data.joint_vel = right + 10.0
        env._robots["left"].data.joint_pos = left
        env._robots["left"].data.joint_vel = left + 10.0
        env._joint_targets["right"] = right + 20.0
        env._joint_targets["left"] = left + 20.0
        expected_right.append(right[0].copy())
        expected_left.append(left[0].copy())
        trace.capture_substep()

    snapshot = trace.substep_snapshot(consume=True)
    assert snapshot["physx_substep_count"].tolist() == [4.0]
    assert snapshot["physx_substep_robot_joint_position"].shape == (4, 2, 3)
    np.testing.assert_allclose(
        snapshot["physx_substep_robot_joint_position"][:, 0, :],
        expected_right,
    )
    np.testing.assert_allclose(
        snapshot["physx_substep_robot_joint_position"][:, 1, :2],
        expected_left,
    )
    np.testing.assert_allclose(
        snapshot["physx_substep_robot_joint_velocity"][:, 0, :],
        np.asarray(expected_right) + 10.0,
    )
    np.testing.assert_allclose(
        snapshot["physx_substep_robot_joint_target"][:, 0, :],
        np.asarray(expected_right) + 20.0,
    )
    np.testing.assert_array_equal(
        snapshot["physx_substep_robot_joint_valid"][:, 1, :],
        np.asarray([[1, 1, 0]] * 4),
    )


def test_fixed_articulations_omit_empty_dof_fields_during_episode_stack() -> None:
    env = _fake_env()
    del env._e_nodes_np
    del env._e_velocities_np
    fixed_view = _FixedArticulationView()
    env._task_reset_articulation_states = [
        (
            "/World/Task/fixed_object",
            fixed_view,
            None,
            None,
            None,
            None,
            None,
        )
    ]
    trace = PhysxGroundTruthTrace(env)
    snapshot = trace.snapshot()
    schema = trace.schema()

    assert trace.max_articulation_dofs == 0
    assert "physx_task_articulation_root_state" in schema["fields"]
    assert "physx_task_articulation_root_state" in snapshot
    restored = trace.restore(snapshot, schema, sync=False)
    assert restored["articulation_roots"] == 1
    assert restored["articulation_dofs"] == 0
    for name in (
        "physx_task_articulation_dof_position",
        "physx_task_articulation_dof_velocity",
        "physx_task_articulation_dof_valid",
    ):
        assert name not in schema["fields"]
        assert name not in snapshot

    snapshot.update(trace.substep_snapshot())
    trace.begin_substep_capture()
    for _ in range(env._n_substeps):
        trace.capture_substep()
    next_snapshot = trace.snapshot()
    next_snapshot.update(trace.substep_snapshot())
    trajectory = [
        {
            "observations": snapshot,
            "next_observations": next_snapshot,
        }
    ]
    stacked = stack_transition_numeric_observation_fields(
        trajectory,
        required_fields=schema["fields"],
        environment_schema=schema,
    )

    assert stacked["physx_task_articulation_root_state"].shape == (1, 1, 13)
    assert stacked["next_physx_task_articulation_root_state"].shape == (
        1,
        1,
        13,
    )
    assert all(value.size > 0 for value in stacked.values())


def test_zero_dof_articulation_duplicate_prefers_rigid_body_state() -> None:
    env = _fake_env()
    del env._e_nodes_np
    del env._e_velocities_np
    fixed_view = _FixedArticulationView()
    env._task_reset_articulation_states = [
        (
            "/World/Task/cube",
            fixed_view,
            None,
            None,
            None,
            None,
            None,
        )
    ]

    trace = PhysxGroundTruthTrace(env)
    snapshot = trace.snapshot()
    schema = trace.schema()

    assert [entry.name for entry in trace.rigid_bodies] == [
        "/World/Task/cube"
    ]
    assert trace.articulations == ()
    assert schema["physx"]["articulation_names"] == []
    assert "physx_task_articulation_root_state" not in snapshot
    restored = trace.restore(snapshot, schema, sync=False)
    assert restored["rigid_bodies"] == 1
    assert restored["articulation_roots"] == 0


def test_zero_dof_free_body_is_actually_sent_to_physx_sleep(
    monkeypatch,
) -> None:
    env = _fake_env()
    fixed_view = _FixedArticulationView()
    env._task_reset_articulation_states = [
        (
            "/World/Task/cube",
            fixed_view,
            None,
            None,
            None,
            None,
            None,
        )
    ]
    trace = PhysxGroundTruthTrace(env)
    calls: list[tuple[int, int]] = []
    simulation = SimpleNamespace(
        put_to_sleep=lambda stage_id, path_id: calls.append(
            (stage_id, path_id)
        )
    )

    omni = ModuleType("omni")
    omni_physx = ModuleType("omni.physx")
    omni_usd = ModuleType("omni.usd")
    omni_physx.get_physx_simulation_interface = lambda: simulation
    omni_usd.get_context = lambda: SimpleNamespace(get_stage_id=lambda: 17)
    omni.physx = omni_physx
    omni.usd = omni_usd
    pxr = ModuleType("pxr")
    pxr.PhysicsSchemaTools = SimpleNamespace(sdfPathToInt=lambda _path: 23)
    pxr.Sdf = SimpleNamespace(Path=lambda path: path)
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.physx", omni_physx)
    monkeypatch.setitem(sys.modules, "omni.usd", omni_usd)
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    result = trace.put_task_bodies_to_sleep()

    assert calls == [(17, 23)]
    assert result == {
        "rigid_bodies": 1,
        "body_paths": ["/World/Task/cube"],
        "skipped_non_dynamic_or_non_path_names": [],
    }


def test_restore_maps_named_entities_and_quaternion_layout() -> None:
    env = _fake_env()
    del env._e_nodes_np
    del env._e_velocities_np
    trace = PhysxGroundTruthTrace(env)
    schema = trace.schema()
    snapshot = trace.snapshot()

    # Exercise name-based mapping by reversing all recorded entity axes.
    schema["physx"]["robot_sides"].reverse()
    snapshot["physx_robot_joint_position"] = np.asarray(
        [[20.0, 21.0, 0.0], [10.0, 11.0, 12.0]]
    )
    snapshot["physx_robot_joint_velocity"] = np.asarray(
        [[2.0, 2.1, 0.0], [1.0, 1.1, 1.2]]
    )
    snapshot["physx_robot_joint_target"] = np.asarray(
        [[120.0, 121.0, 0.0], [110.0, 111.0, 112.0]]
    )
    snapshot["physx_robot_joint_valid"] = np.asarray(
        [[1, 1, 0], [1, 1, 1]], dtype=np.uint8
    )
    snapshot["physx_robot_joint_target_valid"] = (
        snapshot["physx_robot_joint_valid"].copy()
    )

    snapshot["physx_task_rigid_body_state"][0] = np.asarray(
        [9, 8, 7, 0.5, 0.1, 0.2, 0.3, 6, 5, 4, 3, 2, 1],
        dtype=np.float64,
    )
    snapshot["physx_task_articulation_root_state"][0] = np.asarray(
        [7, 8, 9, 0.6, 0.2, 0.3, 0.4, 1, 2, 3, 4, 5, 6],
        dtype=np.float64,
    )
    snapshot["physx_task_articulation_dof_position"][0] = [0.8, 0.9]
    snapshot["physx_task_articulation_dof_velocity"][0] = [1.8, 1.9]

    restored = trace.restore(snapshot, schema, sync=False)

    np.testing.assert_allclose(
        env._robots["right"].data.joint_pos,
        [[10.0, 11.0, 12.0]],
    )
    np.testing.assert_allclose(
        env._robots["left"].data.joint_pos,
        [[20.0, 21.0]],
    )
    np.testing.assert_allclose(
        env._joint_targets["right"],
        [[110.0, 111.0, 112.0]],
    )
    np.testing.assert_allclose(
        env._task_reset_body_states[0][1].get_transforms(),
        [[9, 8, 7, 0.1, 0.2, 0.3, 0.5]],
    )
    np.testing.assert_allclose(
        env._task_reset_body_states[0][1].get_velocities(),
        [[6, 5, 4, 3, 2, 1]],
    )
    articulation = env._task_reset_articulation_states[0][1]
    np.testing.assert_allclose(
        articulation.get_root_transforms(),
        [[7, 8, 9, 0.2, 0.3, 0.4, 0.6]],
    )
    np.testing.assert_allclose(articulation.get_dof_positions(), [[0.8, 0.9]])
    assert restored == {
        "robot_joints": 5,
        "rigid_bodies": 1,
        "articulation_roots": 1,
        "articulation_dofs": 2,
        "deformable_nodes": 0,
    }


def test_apply_robot_joint_targets_does_not_teleport_state() -> None:
    env = _fake_env()
    del env._e_nodes_np
    del env._e_velocities_np
    trace = PhysxGroundTruthTrace(env)
    schema = trace.schema()
    snapshot = trace.snapshot()
    initial_positions = {
        side: robot.data.joint_pos.copy()
        for side, robot in env._robots.items()
    }
    snapshot["physx_robot_joint_target"] = np.asarray(
        [[10.0, 11.0, 12.0], [20.0, 21.0, 0.0]]
    )

    applied = trace.apply_robot_joint_targets(snapshot, schema)

    assert applied == 5
    np.testing.assert_allclose(
        env._joint_targets["right"],
        [[10.0, 11.0, 12.0]],
    )
    np.testing.assert_allclose(
        env._joint_targets["left"],
        [[20.0, 21.0]],
    )
    for side, position in initial_positions.items():
        np.testing.assert_array_equal(
            env._robots[side].data.joint_pos,
            position,
        )


def test_restore_rejects_joint_layout_mismatch() -> None:
    env = _fake_env()
    del env._e_nodes_np
    del env._e_velocities_np
    trace = PhysxGroundTruthTrace(env)
    schema = trace.schema()
    schema["physx"]["robot_joint_names"]["right"].pop()

    try:
        trace.validate_restore_schema(schema)
    except ValueError as exc:
        assert "missing right joints" in str(exc)
    else:
        raise AssertionError("mismatched robot layout was accepted")


def test_rope_nodal_readback_uses_e02_runtime_fields() -> None:
    env = _fake_env()
    del env._e_nodes_np
    del env._e_velocities_np
    env._e02_nodes_np = np.asarray([[1.0, 2.0, 3.0]])
    env._e02_nodal_velocities_np = np.asarray([[0.1, 0.2, 0.3]])

    trace = PhysxGroundTruthTrace(env)
    snapshot = trace.snapshot()

    assert trace.schema()["physx"]["deformable_name"] == "rope"
    np.testing.assert_allclose(
        snapshot["physx_deformable_position"],
        [[1.0, 2.0, 3.0]],
    )
    np.testing.assert_array_equal(snapshot["physx_deformable_valid"], [1])


def test_interface_environment_exposes_trace_without_launching_isaac(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HANDSON_RECORD_PHYSX_GROUND_TRUTH", "1")
    env = BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
    )

    observation, _info = env.reset()
    schema = env.get_recording_schema()

    assert schema is not None
    assert "physx_robot_joint_position" in schema["fields"]
    assert "physx_robot_joint_position" in observation
    for name, value in observation.items():
        if name.startswith("physx_"):
            assert env.observation_space[name].contains(value)
    assert not np.any(observation["physx_robot_joint_valid"])
