from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from handson.sim.isaaclab import abc_collection_env
from handson.sim.isaaclab.abc_collection_env import (
    _B03_NEWTON_HAND_CONTACT_DIMENSION,
    _B03_NEWTON_HAND_CONTACT_GAP,
    _B03_NEWTON_HAND_CONTACT_PRIORITY,
    _B03_NEWTON_HAND_MATERIAL_PATH,
    _B03_NEWTON_HAND_ROLLING_FRICTION,
    _B03_NEWTON_HAND_SLIDING_FRICTION,
    _B03_NEWTON_HAND_SOLIMP,
    _B03_NEWTON_HAND_SOLREF,
    _B03_NEWTON_HAND_TORSIONAL_FRICTION,
    _B03_NEWTON_PEG_CONTACT_GAP,
    _B03_NEWTON_SOCKET_CONTACT_GAP,
    _B03_NEWTON_TABLE_CONTACT_GAP,
    _LIVE_MAIN_CAMERA_ONLY_ENV,
    HandsOnABCIsaacLabEnv,
)
from handson.sim.isaaclab.newton_ground_truth import NewtonGroundTruthTrace
from handson.sim.isaaclab.core_peg_env import HandsOnABCIsaacLabEnv as PaperPegEnv
from handson.tasks.t09_insert_peg.config import TaskConfig
from handson.tasks.sim_teleop import _target_pose_from_bridge_world_pose


def _unwrap(environment):
    current = environment
    for _ in range(8):
        if isinstance(current, (HandsOnABCIsaacLabEnv, PaperPegEnv)):
            return current
        current = getattr(current, "env", None)
        if current is None:
            break
    raise AssertionError("B03 did not construct the Isaac Lab environment")


@pytest.mark.parametrize(
    "backend",
    ["isaacsim_newton", "newton", "nvidia_newton", "isaac_newton"],
)
def test_b03_routes_newton_aliases_to_newton_physics(backend: str) -> None:
    environment = TaskConfig().get_environment(
        policy_mode=True,
        sim_backend=backend,
        launch_isaac=False,
        render_mode="rgb_array",
        randomize=False,
    )
    assert _unwrap(environment)._physics_backend == "newton"


def test_b03_isaac_backend_remains_physx() -> None:
    environment = TaskConfig().get_environment(
        policy_mode=True,
        sim_backend="isaacsim",
        launch_isaac=False,
        render_mode="rgb_array",
        randomize=False,
    )
    assert _unwrap(environment)._physics_backend == "physx"
    assert isinstance(_unwrap(environment), PaperPegEnv)


def test_b03_newton_direct_mode_removes_arms_ik_and_recording_cameras(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_DIRECT_HAND_CONTROL", "1")
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "1")
    monkeypatch.setenv("HANDSON_RECORD_NEWTON_GROUND_TRUTH", "1")
    monkeypatch.delenv("HANDSON_ISAACLAB_CONTROL_DT", raising=False)

    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
        hz=30,
    )

    assert env._direct_hand_control is True
    assert env._hand_target_control is True
    assert env._hands_only is True
    assert env._direct_tracker_passthrough is True
    assert env._deferred_camera_render is True
    assert env._control_dt == pytest.approx(1.0 / 30.0)
    assert env._n_substeps == 4
    assert env._arm_ik_backend == "dls"
    assert env.recorded_drive_target_replay_supported is False
    assert env._newton_ground_truth_enabled is False
    assert env._recording_ground_truth_enabled is False
    assert getattr(env, "_physx_ground_truth_trace", None) is None


def test_b03_newton_articulated_runtime_defaults_to_dataset_rate(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_DIRECT_HAND_CONTROL", "0")
    monkeypatch.delenv("HANDSON_ISAACLAB_CONTROL_DT", raising=False)

    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
        hz=30,
    )

    assert env._direct_hand_control is False
    assert env._control_dt == pytest.approx(1.0 / 30.0)
    assert env._physics_dt == pytest.approx(1.0 / 240.0)
    assert env._n_substeps == 8
    assert env.recorded_drive_target_replay_supported is True
    assert env._newton_use_cuda_graph is True


def test_b03_newton_articulated_runtime_can_defer_all_camera_rendering(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_DIRECT_HAND_CONTROL", "0")
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "1")

    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=True,
        render_mode="rgb_array",
    )

    assert env._direct_hand_control is False
    assert env._deferred_camera_render is True
    first = env._fallback_camera_frame(
        "ego",
        height=8,
        width=12,
    )
    second = env._fallback_camera_frame(
        "ego",
        height=8,
        width=12,
    )
    assert first is second
    assert first.flags.writeable is False
    env._sim = MagicMock()
    rendered = env.render()
    env._sim.render.assert_not_called()
    assert len(rendered) == 3


@pytest.mark.parametrize(
    ("task_name", "physics_backend", "defer_arm_render"),
    (
        ("t09_insert_peg", "newton", "1"),
        ("t09_insert_peg", "physx", "1"),
        ("t26_place_fruit_in_bucket", "newton", "1"),
    ),
)
def test_live_main_camera_only_rejects_non_b03_newton_deferred_modes(
    monkeypatch,
    task_name: str,
    physics_backend: str,
    defer_arm_render: str,
) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_DIRECT_HAND_CONTROL", "0")
    monkeypatch.setenv(
        "HANDSON_ISAACLAB_DEFER_ARM_RENDER",
        defer_arm_render,
    )
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "0")
    monkeypatch.setenv(_LIVE_MAIN_CAMERA_ONLY_ENV, "1")

    with pytest.raises(RuntimeError, match=_LIVE_MAIN_CAMERA_ONLY_ENV):
        HandsOnABCIsaacLabEnv(
            task_name,
            physics_backend=physics_backend,
            launch_isaac=False,
            image_obs=False,
        )


def test_live_main_camera_only_rejects_deferred_all_cameras(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_DIRECT_HAND_CONTROL", "0")
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_ARM_RENDER", "0")
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "1")
    monkeypatch.setenv(_LIVE_MAIN_CAMERA_ONLY_ENV, "1")

    with pytest.raises(RuntimeError, match="requires live camera rendering"):
        HandsOnABCIsaacLabEnv(
            "t09_insert_peg",
            physics_backend="newton",
            launch_isaac=False,
            image_obs=True,
        )


def test_live_main_camera_only_rejects_direct_hand_debug_pipeline(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_DIRECT_HAND_CONTROL", "1")
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_ARM_RENDER", "0")
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "0")
    monkeypatch.setenv(_LIVE_MAIN_CAMERA_ONLY_ENV, "1")

    with pytest.raises(RuntimeError, match="articulated t09_insert_peg"):
        HandsOnABCIsaacLabEnv(
            "t09_insert_peg",
            physics_backend="newton",
            launch_isaac=False,
            image_obs=False,
        )


def test_b03_newton_cuda_graph_can_be_disabled_for_diagnostics(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HANDSON_NEWTON_USE_CUDA_GRAPH", "0")

    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
    )

    assert env._newton_use_cuda_graph is False
    assert env._newton_cuda_graph_captured is False
    assert env._newton_cuda_graph_scope == "disabled"


def test_b03_formal_cross_gpu_profile_forbids_direct_state_writes(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_DIRECT_HAND_CONTROL", "1")
    monkeypatch.setattr(
        abc_collection_env,
        "configure_newton_determinism",
        lambda profile: {
            "profile": profile,
            "cross_gpu_ready": True,
            "warp_deterministic_mode": "gpu_to_gpu",
        },
    )

    with pytest.raises(RuntimeError, match="forbids direct hand state writes"):
        HandsOnABCIsaacLabEnv(
            "t09_insert_peg",
            physics_backend="newton",
            newton_replay_profile="cross_gpu_v1",
            launch_isaac=False,
            image_obs=False,
        )


def test_direct_hand_pose_velocity_is_tracker_world_velocity() -> None:
    previous = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    current = previous.copy()
    current[0] = 0.002

    velocity = HandsOnABCIsaacLabEnv._direct_pose_velocity(
        previous,
        current,
        1.0 / 120.0,
    )

    np.testing.assert_allclose(velocity[:3], [0.24, 0.0, 0.0])
    np.testing.assert_allclose(velocity[3:], np.zeros(3), atol=1.0e-12)


def test_b03_newton_keeps_established_dls_without_changing_a01_or_physx(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_ARM_IK_BACKEND", "dls")

    b03_newton = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
    )
    a01_newton = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
    )
    b03_physx = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="physx",
        launch_isaac=False,
        image_obs=False,
    )

    assert b03_newton._arm_ik_backend == "dls"
    assert a01_newton._arm_ik_backend == "dls"
    assert b03_physx._arm_ik_backend == "dls"


def test_b03_newton_reuses_existing_b03_teleop_contract() -> None:
    config = TaskConfig.teleop

    assert config.pose_scale == 1.0
    assert config.command_on_tracker_motion_only is False
    assert config.rebase_on_stationary_drift is False
    assert config.max_tracker_delta_m is None


def test_b03_newton_tracker_mapping_matches_existing_b03_mapping() -> None:
    start = np.eye(4, dtype=np.float64)
    inside = start.copy()
    inside[0, 3] = 0.20
    outside = start.copy()
    outside[0, 3] = 0.80

    inside_target = _target_pose_from_bridge_world_pose(
        start,
        start,
        inside,
        1.0,
        max_tracker_delta_m=TaskConfig.teleop.max_tracker_delta_m,
    )
    outside_target = _target_pose_from_bridge_world_pose(
        start,
        start,
        outside,
        1.0,
        max_tracker_delta_m=TaskConfig.teleop.max_tracker_delta_m,
    )

    np.testing.assert_allclose(inside_target[:3, 3], [0.20, 0.0, 0.0])
    # None selects the shared legacy teleop cap, exactly as B03 did before
    # Newton support was added.  Newton must not introduce a second contract.
    np.testing.assert_allclose(outside_target[:3, 3], [0.34, 0.0, 0.0])


def test_b03_newton_robot_cfg_reuses_a01_newton_robot_without_task_branches() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "handson"
        / "sim"
        / "isaaclab"
        / "bimanual_assembly_scene.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "complete_external_hand_collisions" not in source
    assert "b03_newton" not in source
    assert "effort_limit_sim=25.0 if newton else 6.0" in source
    assert "stiffness=40.0 if newton else 8.0" in source
    assert "damping=1.2 if newton else 0.12" in source


def test_b03_newton_uses_shipped_wuji_hand_drives_for_thin_peg_contact() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
    )
    hand_actuator = types.SimpleNamespace(
        stiffness=40.0,
        damping=1.2,
        effort_limit_sim=25.0,
        velocity_limit_sim=None,
        joint_names_expr=["right_finger[1-5]_joint[1-4]"],
        armature=None,
        friction=1.0,
        dynamic_friction=None,
        viscous_friction=None,
    )
    robot_cfg = types.SimpleNamespace(actuators={"hand": hand_actuator})

    assert env._apply_b03_newton_source_hand_actuators(
        robot_cfg,
        "right",
        ideal_actuator_cfg_cls=lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    explicit_hand = robot_cfg.actuators["hand"]
    assert explicit_hand.stiffness["right_finger1_joint1"] == pytest.approx(
        0.40844710645048043
    )
    assert explicit_hand.damping["right_finger1_joint1"] == pytest.approx(
        0.020882010257063675 * 2.0
    )
    assert explicit_hand.effort_limit[
        "right_finger1_joint1"
    ] == pytest.approx(0.8904)
    assert explicit_hand.effort_limit_sim == explicit_hand.effort_limit
    assert explicit_hand.velocity_limit == 4.0
    assert explicit_hand.velocity_limit_sim == 4.0
    assert explicit_hand.saturation_effort == pytest.approx(1.2988)
    assert explicit_hand.friction == 0.0


def test_b03_newton_arm_restores_physical_velocity_limits_only() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
    )
    arm_actuator = types.SimpleNamespace(
        effort_limit={"shared": 80.0},
        stiffness={"shared": 40.0},
        damping={"shared": 8.0},
        velocity_limit_sim=None,
        armature=None,
    )
    robot_cfg = types.SimpleNamespace(actuators={"arm": arm_actuator})

    assert env._apply_b03_newton_velocity_limits(robot_cfg, "right")
    assert arm_actuator.velocity_limit_sim == {
        "joint[1-4]_right(_1)?": 2.0,
        "joint[5-7]_right(_1)?": 3.0,
    }
    assert arm_actuator.armature == {
        "joint[1-4]_right(_1)?": 0.5,
        "joint[5-7]_right(_1)?": 0.2,
    }
    assert arm_actuator.effort_limit == {"shared": 80.0}
    assert arm_actuator.stiffness == {"shared": 40.0}
    assert arm_actuator.damping == {"shared": 8.0}


def test_b03_zero_tracker_pose_action_does_not_command_arm_motion() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
    )
    right_before = env._right_tcp_pose.copy()
    left_before = env._left_tcp_pose.copy()
    hand = np.zeros(env._hand_action_dim, dtype=np.float64)

    env._apply_action(
        {
            "right": np.concatenate([np.zeros(7), hand]),
            "left": np.concatenate([np.zeros(7), hand]),
        }
    )

    assert env._right_pose_commanded is False
    assert env._left_pose_commanded is False
    np.testing.assert_array_equal(env._right_tcp_pose, right_before)
    np.testing.assert_array_equal(env._left_tcp_pose, left_before)


def test_b03_newton_stationary_tracker_preserves_active_physical_servo() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
    )
    env._right_pose_commanded = True
    env._left_pose_commanded = True
    right_before = env._right_tcp_pose.copy()
    left_before = env._left_tcp_pose.copy()
    hand = np.zeros(env._hand_action_dim, dtype=np.float64)

    env._apply_action(
        {
            "right": np.concatenate([np.zeros(7), hand]),
            "left": np.concatenate([np.zeros(7), hand]),
        }
    )

    assert env._right_pose_commanded is True
    assert env._left_pose_commanded is True
    np.testing.assert_array_equal(env._right_tcp_pose, right_before)
    np.testing.assert_array_equal(env._left_tcp_pose, left_before)


def test_b03_newton_uses_shared_newton_joint_target_accumulator(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "HANDSON_NEWTON_ARM_MAX_JOINT_STEP",
        raising=False,
    )
    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
    )
    env._torch = types.SimpleNamespace(
        clamp=lambda value, low, high: np.clip(value, low, high)
    )
    env._arm_joint_ids = {"right": [0, 2]}
    env._joint_targets = {
        "right": np.asarray([[1.0, 20.0, 3.0]], dtype=np.float64),
    }
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_pos=np.asarray([[10.0, 20.0, 30.0]], dtype=np.float64),
        )
    )

    limited = env._limit_newton_arm_joint_target(
        "right",
        robot,
        np.asarray([[1.20, 2.80]], dtype=np.float64),
    )

    # The shared Newton path accumulates from the previous finite command,
    # exactly as A01 does; B03 must not substitute the lagging physical q.
    np.testing.assert_allclose(limited, [[1.08, 2.92]])


def test_b03_physx_keeps_existing_pose_command_semantics() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="physx",
        launch_isaac=False,
        image_obs=False,
    )
    env._right_pose_commanded = True
    hand = np.zeros(env._hand_action_dim, dtype=np.float64)

    env._apply_action(np.concatenate([np.zeros(7), hand]))

    assert env._right_pose_commanded is True


def test_b03_newton_contact_gaps_preserve_round_socket_clearance() -> None:
    radial_clearance = 0.02025 - 0.01785
    assert (
        _B03_NEWTON_PEG_CONTACT_GAP
        + _B03_NEWTON_SOCKET_CONTACT_GAP
        < radial_clearance
    )
    assert _B03_NEWTON_HAND_CONTACT_GAP > _B03_NEWTON_PEG_CONTACT_GAP
    assert _B03_NEWTON_TABLE_CONTACT_GAP >= _B03_NEWTON_PEG_CONTACT_GAP


def test_b03_newton_restores_source_hand_contact_material(monkeypatch) -> None:
    class FakeAttribute:
        def __init__(self, value=None) -> None:
            self.value = value

        def __bool__(self) -> bool:
            return True

        def IsValid(self) -> bool:
            return True

        def Get(self):
            return self.value

        def Set(self, value) -> None:
            self.value = value

    class FakePrim:
        def __init__(
            self,
            path: str,
            *,
            collision: bool = False,
            descendants=(),
        ) -> None:
            self.path = path
            self.collision = collision
            self.descendants = list(descendants)
            self.attributes = {}
            self.material_targets = []

        def IsValid(self) -> bool:
            return True

        def GetPath(self) -> str:
            return self.path

        def GetName(self) -> str:
            return self.path.rsplit("/", 1)[-1]

        def ApplyAPI(self, _schema_name: str) -> bool:
            return True

        def GetAttribute(self, name: str):
            return self.attributes.get(name)

        def CreateAttribute(self, name: str, _value_type):
            attribute = FakeAttribute()
            self.attributes[name] = attribute
            return attribute

    class FakeCollisionAPI:
        def __init__(self, prim: FakePrim) -> None:
            self.prim = prim

        def __bool__(self) -> bool:
            return self.prim.collision

        def GetCollisionEnabledAttr(self) -> FakeAttribute:
            return FakeAttribute(True)

    class FakeBindingRel:
        def __init__(self, prim: FakePrim) -> None:
            self.prim = prim

        def GetTargets(self):
            return list(self.prim.material_targets)

    class FakeMaterialBindingAPI:
        def __init__(self, prim: FakePrim) -> None:
            self.prim = prim

        def GetDirectBindingRel(self, purpose: str) -> FakeBindingRel:
            assert purpose == "physics"
            return FakeBindingRel(self.prim)

    hand_names = {
        f"{side}_palm_link_col2"
        for side in ("right", "left")
    }
    hand_names.update(
        f"{side}_finger{finger}_link{link}_col2"
        for side in ("right", "left")
        for finger in range(1, 6)
        for link in range(1, 5)
    )
    hand_names.update(
        f"{side}_finger{finger}_tip_link_col2"
        for side in ("right", "left")
        for finger in range(1, 6)
    )
    hand_colliders = {
        side: [
            FakePrim(
                f"/World/HandsOnRobots/{side.capitalize()}Robot/{name}",
                collision=True,
            )
            for name in sorted(hand_names)
            if name.startswith(f"{side}_")
        ]
        for side in ("right", "left")
    }
    roots = {
        "/World/HandsOnRobots/RightRobot": FakePrim(
            "/World/HandsOnRobots/RightRobot",
            descendants=hand_colliders["right"],
        ),
        "/World/HandsOnRobots/LeftRobot": FakePrim(
            "/World/HandsOnRobots/LeftRobot",
            descendants=hand_colliders["left"],
        ),
    }

    class FakeMaterialCfg:
        def __init__(self, **kwargs) -> None:
            self.values = kwargs
            self.func = self.spawn

        def spawn(self, path: str, cfg) -> None:
            material = FakePrim(path)
            material.attributes["physics:staticFriction"] = FakeAttribute(
                cfg.values["static_friction"]
            )
            material.attributes["physics:dynamicFriction"] = FakeAttribute(
                cfg.values["dynamic_friction"]
            )
            roots[path] = material

    def bind_physics_material(
        prim_path: str,
        material_path: str,
        **_kwargs,
    ) -> bool:
        roots_by_path[prim_path].material_targets = [material_path]
        return True

    roots_by_path = {
        str(prim.GetPath()): prim
        for side_colliders in hand_colliders.values()
        for prim in side_colliders
    }
    sim_utils = types.SimpleNamespace(
        RigidBodyMaterialCfg=FakeMaterialCfg,
        bind_physics_material=bind_physics_material,
    )
    pxr = types.ModuleType("pxr")
    pxr.Sdf = types.SimpleNamespace(
        ValueTypeNames=types.SimpleNamespace(
            Float="Float",
            Int="Int",
            DoubleArray="DoubleArray",
        )
    )
    pxr.Usd = types.SimpleNamespace(PrimRange=lambda prim: prim.descendants)
    pxr.UsdPhysics = types.SimpleNamespace(CollisionAPI=FakeCollisionAPI)
    pxr.UsdShade = types.SimpleNamespace(
        MaterialBindingAPI=FakeMaterialBindingAPI
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
    )
    env._stage = types.SimpleNamespace(GetPrimAtPath=roots.get)

    bound = env._configure_b03_newton_hand_grasp_physics(sim_utils)

    assert bound == 52
    material = roots[_B03_NEWTON_HAND_MATERIAL_PATH]
    assert material.attributes["physics:staticFriction"].value == pytest.approx(
        _B03_NEWTON_HAND_SLIDING_FRICTION
    )
    assert material.attributes["physics:dynamicFriction"].value == pytest.approx(
        _B03_NEWTON_HAND_SLIDING_FRICTION
    )
    assert material.attributes["newton:torsionalFriction"].value == pytest.approx(
        _B03_NEWTON_HAND_TORSIONAL_FRICTION
    )
    assert material.attributes["newton:rollingFriction"].value == pytest.approx(
        _B03_NEWTON_HAND_ROLLING_FRICTION
    )
    for prim in sum(hand_colliders.values(), []):
        assert prim.material_targets == [_B03_NEWTON_HAND_MATERIAL_PATH]
        assert prim.attributes["mjc:condim"].value == (
            _B03_NEWTON_HAND_CONTACT_DIMENSION
        )
        assert prim.attributes["mjc:priority"].value == (
            _B03_NEWTON_HAND_CONTACT_PRIORITY
        )
        assert prim.attributes["mjc:solref"].value == list(
            _B03_NEWTON_HAND_SOLREF
        )
        assert prim.attributes["mjc:solimp"].value == list(
            _B03_NEWTON_HAND_SOLIMP
        )


def test_b03_newton_physics_schema_binds_contact_topology_and_gaps() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
    )
    trace = object.__new__(NewtonGroundTruthTrace)
    trace.env = env
    env._newton_determinism_contract = {
        "solver_deterministic_mode": "gpu_to_gpu"
    }
    env._newton_velocity_projection_contract = {
        "format": "handson.newton_qd_projection",
        "version": 1,
        "algorithm": "candidate_qd_clamp_before_semi_implicit_qpos",
        "integrator": "implicitfast",
    }

    physics_config = trace._newton_physics_config()
    contact = physics_config["contact_model"]

    assert contact == {
        "representation": "articulated_wuji_primitive_v1",
        "task_collider_count": 9,
        "hand_collider_count": 52,
        "hand_contact_gap_m": _B03_NEWTON_HAND_CONTACT_GAP,
        "peg_contact_gap_m": _B03_NEWTON_PEG_CONTACT_GAP,
        "socket_contact_gap_m": _B03_NEWTON_SOCKET_CONTACT_GAP,
        "table_contact_gap_m": _B03_NEWTON_TABLE_CONTACT_GAP,
        "hand_sliding_friction": _B03_NEWTON_HAND_SLIDING_FRICTION,
        "hand_torsional_friction": _B03_NEWTON_HAND_TORSIONAL_FRICTION,
        "hand_rolling_friction": _B03_NEWTON_HAND_ROLLING_FRICTION,
        "hand_contact_dimension": _B03_NEWTON_HAND_CONTACT_DIMENSION,
        "hand_contact_priority": _B03_NEWTON_HAND_CONTACT_PRIORITY,
        "hand_solref": list(_B03_NEWTON_HAND_SOLREF),
        "hand_solimp": list(_B03_NEWTON_HAND_SOLIMP),
    }
    assert physics_config["solver"]["deterministic_mode"] == "gpu_to_gpu"
    assert physics_config["solver"]["njmax"] == 512
    assert physics_config["solver"]["nconmax"] == 512
    assert physics_config["solver"]["capacity_policy"] == "fixed_explicit"
    assert physics_config["solver"]["mujoco_read_only_sensor_pipeline"] == (
        "disabled"
    )
    assert physics_config["joint_velocity_projection"] == (
        env._newton_velocity_projection_contract
    )
    assert physics_config["execution"] == {
        "cuda_graph_requested": True,
        "use_cuda_graph": False,
        "cuda_graph_captured": False,
        "cuda_graph_scope": "disabled",
        "cuda_graph_capture_mode": "disabled",
        "capture_completed_before_episode_reset": False,
        "newton_manager_num_substeps": 1,
        "newton_manager_decimation": 1,
    }

    env._newton_ground_truth_enabled = True
    reset_contract = env.get_physical_replay_reset_contract()["newton"]
    assert reset_contract["njmax"] == 512
    assert reset_contract["nconmax"] == 512
    assert reset_contract["execution"] == physics_config["execution"]
    assert reset_contract["solver_history_reset"] == {
        "strategy": "solver_mujoco_internal_buffers_flags_zero",
        "flags": 0,
        "preserves_authored_joint_state": True,
        "performed_on_last_reset": False,
    }


def test_b03_newton_contact_gap_covers_peg_socket_table_and_complete_hands(
    monkeypatch,
) -> None:
    class FakeAttribute:
        def __init__(self, value=None) -> None:
            self.value = value

        def __bool__(self) -> bool:
            return True

        def IsValid(self) -> bool:
            return True

        def Get(self):
            return self.value

        def Set(self, value) -> None:
            self.value = value

    class FakePrim:
        def __init__(
            self,
            path: str,
            *,
            collision: bool = False,
            descendants=(),
        ) -> None:
            self.path = path
            self.collision = collision
            self.descendants = list(descendants)
            self.attributes = {}

        def IsValid(self) -> bool:
            return True

        def GetPath(self) -> str:
            return self.path

        def GetName(self) -> str:
            return self.path.rsplit("/", 1)[-1]

        def ApplyAPI(self, schema_name: str) -> bool:
            assert schema_name == "NewtonCollisionAPI"
            return True

        def GetAttribute(self, name: str):
            return self.attributes.get(name)

        def CreateAttribute(self, name: str, _value_type):
            attribute = FakeAttribute()
            self.attributes[name] = attribute
            return attribute

    class FakeCollisionAPI:
        def __init__(self, prim: FakePrim) -> None:
            self.prim = prim

        def __bool__(self) -> bool:
            return self.prim.collision

        def GetCollisionEnabledAttr(self) -> FakeAttribute:
            return FakeAttribute(True)

    peg_colliders = [
        FakePrim(
            "/World/Task/Geometry/industreal_round_peg_8mm/"
            f"peg_collision_{index}",
            collision=True,
        )
        for index in range(2)
    ]
    socket_colliders = [
        FakePrim(
            "/World/Task/Geometry/industreal_tray_insert_round_peg_8mm/"
            f"socket_collision_{index}",
            collision=True,
        )
        for index in range(6)
    ]
    table_colliders = [
        FakePrim(
            "/World/Task/Geometry/table/table_collision",
            collision=True,
        )
    ]
    hand_names = {
        f"{side}_palm_link_col2"
        for side in ("right", "left")
    }
    hand_names.update(
        f"{side}_finger{finger}_link{link}_col2"
        for side in ("right", "left")
        for finger in range(1, 6)
        for link in range(1, 5)
    )
    hand_names.update(
        f"{side}_finger{finger}_tip_link_col2"
        for side in ("right", "left")
        for finger in range(1, 6)
    )
    hand_colliders = {
        side: [
            FakePrim(
                f"/World/HandsOnRobots/{side.capitalize()}Robot/{name}",
                collision=True,
            )
            for name in sorted(hand_names)
            if name.startswith(f"{side}_")
        ]
        for side in ("right", "left")
    }
    roots = {
        "/World/Task": FakePrim(
            "/World/Task",
            descendants=[
                *peg_colliders,
                *socket_colliders,
                *table_colliders,
            ],
        ),
        "/World/HandsOnRobots/RightRobot": FakePrim(
            "/World/HandsOnRobots/RightRobot",
            descendants=hand_colliders["right"],
        ),
        "/World/HandsOnRobots/LeftRobot": FakePrim(
            "/World/HandsOnRobots/LeftRobot",
            descendants=hand_colliders["left"],
        ),
    }
    pxr = types.ModuleType("pxr")
    pxr.Sdf = types.SimpleNamespace(
        ValueTypeNames=types.SimpleNamespace(Float="Float")
    )
    pxr.Usd = types.SimpleNamespace(PrimRange=lambda prim: prim.descendants)
    pxr.UsdPhysics = types.SimpleNamespace(CollisionAPI=FakeCollisionAPI)
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
    )
    env._stage = types.SimpleNamespace(GetPrimAtPath=roots.get)

    task_count, hand_count = env._configure_b03_newton_contacts()

    assert task_count == 9
    assert hand_count == 52
    for prim in peg_colliders:
        assert prim.attributes["newton:contactGap"].value == pytest.approx(
            _B03_NEWTON_PEG_CONTACT_GAP
        )
    for prim in socket_colliders:
        assert prim.attributes["newton:contactGap"].value == pytest.approx(
            _B03_NEWTON_SOCKET_CONTACT_GAP
        )
    for prim in table_colliders:
        assert prim.attributes["newton:contactGap"].value == pytest.approx(
            _B03_NEWTON_TABLE_CONTACT_GAP
        )
    for prim in sum(hand_colliders.values(), []):
        assert prim.attributes["newton:contactGap"].value == pytest.approx(
            _B03_NEWTON_HAND_CONTACT_GAP
        )


def test_b03_newton_trace_measures_real_hand_peg_solid_penetration(
    monkeypatch,
) -> None:
    class Buffer:
        def __init__(self, value) -> None:
            self.value = np.asarray(value)
            self.shape = self.value.shape

        def numpy(self) -> np.ndarray:
            return self.value.copy()

    contacts = types.SimpleNamespace(
        rigid_contact_count=Buffer([1]),
        rigid_contact_shape0=Buffer([0]),
        rigid_contact_shape1=Buffer([1]),
        rigid_contact_point0=Buffer([[0.0, 0.0, 0.0]]),
        rigid_contact_point1=Buffer([[-0.0004, 0.0, 0.0]]),
        rigid_contact_normal=Buffer([[1.0, 0.0, 0.0]]),
        rigid_contact_margin0=Buffer([0.003]),
        rigid_contact_margin1=Buffer([0.0005]),
    )
    model = types.SimpleNamespace(
        shape_label=[
            "/World/HandsOnRobots/RightRobot/right_finger1_link1_col2",
            "/World/Task/Geometry/industreal_round_peg_8mm/collision",
        ],
        shape_body=Buffer([0, 1]),
        shape_margin=Buffer([0.003, 0.0005]),
    )
    state = types.SimpleNamespace(
        body_q=Buffer(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
    )

    class Manager:
        get_contacts = staticmethod(lambda: contacts)
        get_model = staticmethod(lambda: model)
        get_state_0 = staticmethod(lambda: state)

    physics = types.ModuleType("isaaclab_newton.physics")
    physics.NewtonManager = Manager
    package = types.ModuleType("isaaclab_newton")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "isaaclab_newton", package)
    monkeypatch.setitem(sys.modules, "isaaclab_newton.physics", physics)

    trace = object.__new__(NewtonGroundTruthTrace)
    trace.env = types.SimpleNamespace(task_name="t09_insert_peg")
    fields = trace._sample_contact_fields()

    assert fields["physx_hand_peg_contact_valid"].tolist() == [1]
    assert fields["physx_hand_peg_contact_count"].tolist() == [1]
    assert fields["physx_hand_peg_min_solid_separation"][0] == pytest.approx(
        -0.0004
    )
    assert fields["physx_hand_peg_max_solid_penetration"][0] == pytest.approx(
        0.0004
    )
