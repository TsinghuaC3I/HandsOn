from __future__ import annotations

import ast
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from handson.sim.isaaclab import compat
from handson.sim.isaaclab.abc_collection_env import (
    _A01_CLEAR_ARM_HOME,
    _A01_CLEAR_RIGHT_START_TCP,
    _A01_NEWTON_CONTACT_GAP,
    _TASK_TABLE_HEIGHT_BOUNDS,
    HandsOnABCIsaacLabEnv,
)
from handson.tasks.t26_place_fruit_in_bucket.config import TaskConfig


_SCENE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "handson"
    / "sim"
    / "isaaclab"
    / "bimanual_assembly_scene.py"
)
_ABC_ENV_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "handson"
    / "sim"
    / "isaaclab"
    / "abc_collection_env.py"
)


class _FakeTensor(np.ndarray):
    def clone(self):
        return self.copy().view(_FakeTensor)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


class _FakeTorch:
    @staticmethod
    def clamp(value, minimum, maximum):
        return np.clip(value, minimum, maximum).view(_FakeTensor)


def _tensor(value) -> _FakeTensor:
    return np.asarray(value, dtype=np.float32).view(_FakeTensor)


def _unwrap(environment):
    current = environment
    for _ in range(8):
        if isinstance(current, HandsOnABCIsaacLabEnv):
            return current
        current = getattr(current, "env", None)
        if current is None:
            break
    raise AssertionError("A01 did not construct the Isaac Lab environment")


@pytest.mark.parametrize(
    "backend",
    ["isaacsim_newton", "newton", "nvidia_newton", "isaac_newton"],
)
def test_a01_routes_newton_aliases_to_newton_physics(backend: str) -> None:
    environment = TaskConfig().get_environment(
        policy_mode=True,
        sim_backend=backend,
        launch_isaac=False,
        render_mode="rgb_array",
        randomize=False,
    )
    assert _unwrap(environment)._physics_backend == "newton"


def test_newton_backend_is_rejected_for_other_abc_tasks() -> None:
    with pytest.raises(ValueError, match="only for t26_place_fruit_in_bucket"):
        HandsOnABCIsaacLabEnv(
            "t25_place_cup_on_plate",
            physics_backend="newton",
            launch_isaac=False,
        )


def test_newton_cfg_enables_solver_native_target_driven_actuators(
    monkeypatch,
) -> None:
    class FakeNewtonCfg:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class FakeMJWarpSolverCfg:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

        def to_dict(self):
            return dict(self.__dict__)

    physics_module = types.SimpleNamespace(
        NewtonCfg=FakeNewtonCfg,
        MJWarpSolverCfg=FakeMJWarpSolverCfg,
    )
    gpu_to_gpu = object()
    warp_module = types.SimpleNamespace(
        DeterministicMode=types.SimpleNamespace(GPU_TO_GPU=gpu_to_gpu),
        config=types.SimpleNamespace(deterministic=gpu_to_gpu),
    )

    class FakeSimulationCfg:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    monkeypatch.setattr(compat, "_load_lab3_physx_cfg", lambda: object)
    monkeypatch.setattr(compat.importlib.metadata, "version", lambda _name: "3.8.0")
    monkeypatch.setattr(
        compat.importlib,
        "import_module",
        lambda name: physics_module
        if name == "isaaclab_newton.physics"
        else warp_module
        if name == "warp"
        else pytest.fail(f"unexpected import: {name}"),
    )

    cfg = compat.make_simulation_cfg(
        types.SimpleNamespace(SimulationCfg=FakeSimulationCfg),
        physics_backend="newton",
        newton_gpu_to_gpu=True,
        newton_use_cuda_graph=True,
        device="cuda:0",
        dt=1.0 / 120.0,
    )

    assert cfg.use_newton_actuators is True
    assert cfg.physics.use_cuda_graph is True
    assert cfg.physics.solver_cfg.use_mujoco_contacts is False
    assert cfg.physics.solver_cfg.njmax is None
    assert cfg.physics.solver_cfg.nconmax is None
    assert cfg.physics.solver_cfg.deterministic is gpu_to_gpu


def test_single_environment_metadata_is_registered_for_native_newton_actuators(
    monkeypatch,
) -> None:
    class FakeNewtonManager:
        _num_envs = None
        _builder = types.SimpleNamespace(
            body_count=1,
            custom_attributes={"mujoco:gravcomp": object()},
        )

        @classmethod
        def get_num_envs(cls):
            return cls._num_envs

    original_import = compat.importlib.import_module
    monkeypatch.setattr(
        compat.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(NewtonManager=FakeNewtonManager)
        if name == "isaaclab_newton.physics.newton_manager"
        else original_import(name),
    )

    compat.configure_newton_single_environment()

    assert FakeNewtonManager._num_envs == 1


def test_single_environment_builder_registers_mujoco_attributes_before_usd_import(
    monkeypatch,
) -> None:
    stage = object()

    class FakeBuilder:
        body_count = 0

        def __init__(self) -> None:
            self.custom_attributes = {}
            self.added_stage = None
            self.schema_resolvers = None

        def add_usd(self, added_stage, *, schema_resolvers) -> None:
            self.added_stage = added_stage
            self.schema_resolvers = schema_resolvers
            self.body_count = 1

    builder = FakeBuilder()

    class FakeNewtonManager:
        _num_envs = None
        _builder = None
        _world_xforms = None

        @classmethod
        def get_num_envs(cls):
            return cls._num_envs

        @classmethod
        def create_builder(cls):
            return builder

        @classmethod
        def set_builder(cls, value) -> None:
            cls._builder = value

    class FakeSolverMuJoCo:
        @staticmethod
        def register_custom_attributes(value) -> None:
            value.custom_attributes["mujoco:gravcomp"] = object()

    class FakeNewtonResolver:
        pass

    class FakePhysxResolver:
        pass

    modules = {
        "isaaclab_newton.physics.newton_manager": types.SimpleNamespace(
            NewtonManager=FakeNewtonManager
        ),
        "newton": types.SimpleNamespace(
            solvers=types.SimpleNamespace(SolverMuJoCo=FakeSolverMuJoCo)
        ),
        "isaaclab.sim": types.SimpleNamespace(get_current_stage=lambda: stage),
        "newton._src.usd.schemas": types.SimpleNamespace(
            SchemaResolverNewton=FakeNewtonResolver,
            SchemaResolverPhysx=FakePhysxResolver,
        ),
        "warp": types.SimpleNamespace(transform=lambda: "identity"),
    }
    original_import = compat.importlib.import_module
    monkeypatch.setattr(
        compat.importlib,
        "import_module",
        lambda name: modules[name] if name in modules else original_import(name),
    )

    compat.configure_newton_single_environment()

    assert FakeNewtonManager._builder is builder
    assert builder.added_stage is stage
    assert len(builder.schema_resolvers) == 2
    assert FakeNewtonManager._world_xforms == ["identity"]


def test_gpu_to_gpu_solver_uses_validated_sparse_joint_actuator_bound(
    monkeypatch,
) -> None:
    model = types.SimpleNamespace(
        nu=2,
        njnt=2,
        nv=2,
        actuator_trntype=np.asarray([0, 0], dtype=np.int32),
        actuator_trnid=np.asarray([[0, 0], [1, 0]], dtype=np.int32),
        jnt_type=np.asarray([3, 1], dtype=np.int32),
        nbody=3,
        body_weldid=np.asarray([0, 1, 2], dtype=np.int32),
        body_dofadr=np.asarray([0, 0, 1], dtype=np.int32),
        body_dofnum=np.asarray([0, 1, 1], dtype=np.int32),
        dof_parentid=np.asarray([-1, -1], dtype=np.int32),
        ten_J_rownnz=np.asarray([], dtype=np.int32),
        flexedge_J_rownnz=np.asarray([], dtype=np.int32),
        tendon_num=np.asarray([], dtype=np.int32),
        nsensor=5,
        opt=types.SimpleNamespace(disableflags=0),
    )

    class FakeSolver:
        mj_model = model
        mjw_data = types.SimpleNamespace(njmax=512)
        mjw_model = types.SimpleNamespace(
            opt=types.SimpleNamespace(disableflags=0)
        )
        module_options_set = False
        generated_kernels_prepared = False

        def _set_mujoco_warp_module_options(self) -> None:
            self.module_options_set = True

        def _prepare_generated_kernels(self) -> None:
            self.generated_kernels_prepared = True

    solver = FakeSolver()
    modules = {
        "isaaclab_newton.physics.newton_manager": types.SimpleNamespace(
            NewtonManager=types.SimpleNamespace(_solver=solver)
        ),
        "mujoco": types.SimpleNamespace(
            mjtTrn=types.SimpleNamespace(
                mjTRN_JOINT=0,
                mjTRN_JOINTINPARENT=1,
            ),
            mjtJoint=types.SimpleNamespace(
                mjJNT_FREE=0,
                mjJNT_BALL=1,
                mjJNT_SLIDE=2,
                mjJNT_HINGE=3,
            ),
            mjtDisableBit=types.SimpleNamespace(mjDSBL_SENSOR=8192),
        ),
    }
    monkeypatch.setattr(
        compat.importlib,
        "import_module",
        lambda name: modules[name],
    )

    contract = compat.qualify_newton_mujoco_gpu_to_gpu_solver()

    assert contract == {
        "solver_deterministic_max_records": 512,
        "solver_constraint_capacity": 512,
        "solver_sparse_row_width": 2,
        "solver_actuator_row_width": 3,
        "solver_disabled_read_only_sensors": 5,
    }
    assert solver._deterministic_max_records == 512
    assert solver.module_options_set is True
    assert solver.generated_kernels_prepared is True
    assert model.opt.disableflags == 8192
    assert solver.mjw_model.opt.disableflags == 8192


def test_newton_mujoco_history_reset_preserves_authored_state(monkeypatch) -> None:
    state = object()
    calls = []

    class Solver:
        def reset(self, reset_state, *, world_mask, flags) -> None:
            calls.append((reset_state, world_mask, flags))

    physics = types.SimpleNamespace(
        NewtonManager=types.SimpleNamespace(
            _solver=Solver(),
            _state_0=state,
        )
    )
    monkeypatch.setattr(
        compat.importlib,
        "import_module",
        lambda name: physics
        if name == "isaaclab_newton.physics"
        else pytest.fail(f"unexpected import: {name}"),
    )

    contract = compat.reset_newton_mujoco_solver_history()

    assert calls == [(state, None, 0)]
    assert contract == {
        "strategy": "solver_mujoco_internal_buffers_flags_zero",
        "flags": 0,
        "preserves_authored_joint_state": True,
        "performed_on_last_reset": True,
    }


def test_newton_world_is_built_before_recording_render_products() -> None:
    source = _ABC_ENV_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_ABC_ENV_SOURCE))
    environment_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "HandsOnABCIsaacLabEnv"
    )
    setup_method = next(
        node
        for node in environment_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_setup_isaaclab"
    )
    calls = {
        "newton": next(
            node
            for node in ast.walk(setup_method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "configure_newton_single_environment"
        ),
        "cameras": next(
            node
            for node in ast.walk(setup_method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_init_recording_cameras"
        ),
    }

    assert calls["newton"].lineno < calls["cameras"].lineno


def test_newton_robot_arm_uses_explicit_torque_pd_without_state_restore() -> None:
    source = _SCENE_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_SCENE_SOURCE))
    robot_cfg = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_robot_cfg"
    )
    names = {
        node.id
        for node in ast.walk(robot_cfg)
        if isinstance(node, ast.Name)
    }
    robot_source = ast.get_source_segment(source, robot_cfg) or ""

    assert "IdealPDActuatorCfg" in names
    assert "ImplicitActuatorCfg" in names
    assert "MujocoJointDrivePropertiesCfg" in names
    assert 'f"joint[1-4]_{side}(_1)?": 40.0' in robot_source
    assert "actuatorgravcomp=True" in robot_source
    assert "write_joint_state_to_sim" not in robot_source


def test_newton_robot_derivative_adds_each_proximal_finger_collider() -> None:
    source = _SCENE_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_SCENE_SOURCE))
    prepare_robot = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_prepare_robot_mjcf"
    )
    prepare_source = ast.get_source_segment(source, prepare_robot) or ""
    robot_cfg = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_robot_cfg"
    )
    robot_source = ast.get_source_segment(source, robot_cfg) or ""

    assert 'f"{side}_finger{finger}_link1"' in prepare_source
    assert '"name": f"{name}_col2"' in prepare_source
    assert '"contype": "1"' in prepare_source
    assert '"conaffinity": "1"' in prepare_source
    assert "newton or bool(ensure_proximal_finger_collisions)" in robot_source


def test_a01_newton_contact_gap_covers_task_and_complete_hands(
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

    task_colliders = [
        FakePrim(
            f"/World/Task/Geometry/{name}/{name}_collision",
            collision=True,
        )
        for name in ("apple", "banana", "orange", "pear", "bucket", "table")
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
            descendants=task_colliders,
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
    pxr.Usd = types.SimpleNamespace(
        PrimRange=lambda prim: prim.descendants
    )
    pxr.UsdPhysics = types.SimpleNamespace(CollisionAPI=FakeCollisionAPI)
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
    )
    env._stage = types.SimpleNamespace(GetPrimAtPath=roots.get)

    task_count, hand_count = env._configure_a01_newton_contacts()

    assert task_count == 6
    assert hand_count == 52
    for prim in task_colliders + sum(hand_colliders.values(), []):
        assert prim.attributes["newton:contactGap"].value == pytest.approx(
            _A01_NEWTON_CONTACT_GAP
        )


def test_newton_robot_gravity_compensation_covers_each_imported_link() -> None:
    source = _SCENE_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_SCENE_SOURCE))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "configure_newton_robot_gravity_compensation"
    )
    helper_source = ast.get_source_segment(source, helper) or ""

    assert "Usd.PrimRange" in helper_source
    assert 'prim.CreateAttribute(\n                    "mjc:gravcomp"' in helper_source


def test_a01_table_randomization_keeps_the_newton_home_collision_free() -> None:
    assert _TASK_TABLE_HEIGHT_BOUNDS["t26_place_fruit_in_bucket"] == (0.0, 0.010)


def test_a01_uses_a_collision_safe_control_aligned_robot_home() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
    )
    env._sample_reset_state(clear_manual_request=True)

    np.testing.assert_allclose(env._right_tcp_pose, _A01_CLEAR_RIGHT_START_TCP)
    assert env._right_tcp_pose[2] > 1.07
    assert _A01_CLEAR_ARM_HOME["right"][3] > np.pi / 2.0
    np.testing.assert_allclose(
        _A01_CLEAR_ARM_HOME["left"],
        [-0.441741, 0.721729, -0.239304, 1.982058, 0.346778, -0.925025, -0.925025],
    )


def test_newton_arm_target_step_has_a_backend_specific_safety_bound() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
    )
    env._torch = _FakeTorch()
    env._arm_joint_ids = {"right": [0, 2]}
    env._joint_targets = {
        "right": _tensor([[1.0, 20.0, 3.0]]),
    }
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_pos=_tensor([[10.0, 20.0, 30.0]]),
        )
    )

    limited = env._limit_newton_arm_joint_target(
        "right",
        robot,
        _tensor([[2.0, 1.0]]),
    )

    np.testing.assert_allclose(
        limited,
        _tensor([[1.08, 2.92]]),
    )


def test_newton_live_step_only_writes_targets_and_never_restores_joint_state() -> None:
    class FakeRobot:
        def __init__(self) -> None:
            self.write_count = 0
            self.update_dt = None

        def write_joint_state_to_sim(self, *_args, **_kwargs) -> None:
            raise AssertionError("live Newton stepping must not restore joint state")

        def write_data_to_sim(self) -> None:
            self.write_count += 1

        def update(self, dt) -> None:
            self.update_dt = dt

    class FakeSim:
        def __init__(self) -> None:
            self.steps = []

        def step(self, *, render) -> None:
            self.steps.append(render)

    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
    )
    robot = FakeRobot()
    env._robots = {"right": robot}
    env._sim = FakeSim()
    env._sim_dt = 1.0 / 120.0
    env._resume_timeline_if_paused = lambda **_kwargs: None
    env._update_b01_release_assist = lambda: None
    env._update_c04_trigger_state = lambda: None
    env._update_c04_spray_visual = lambda: None
    env._after_task_physics_substep = lambda: None

    env._step_isaaclab()

    assert robot.write_count == 1
    assert robot.update_dt == pytest.approx(1.0 / 120.0)
    assert env._sim.steps == [False]


def test_newton_end_effector_pose_converts_lab3_xyzw_to_wxyz() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
    )
    env._isaaclab_uses_xyzw = True
    env._robots = {
        "right": types.SimpleNamespace(
            data=types.SimpleNamespace(
                body_pose_w=_tensor(
                    [[[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]]
                )
            )
        )
    }
    env._ee_body_indices = {"right": 0}

    matrix = env._native_end_effector_pose_matrix(
        "right",
        np.asarray([9.0, 9.0, 9.0, 1.0, 0.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(matrix[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(matrix[:3, :3], np.eye(3), atol=1.0e-6)
