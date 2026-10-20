from __future__ import annotations

from pathlib import Path
import gc
import sys
import types
import weakref

import numpy as np
import pytest

from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
from handson.sim.isaaclab.bimanual_assembly_env import (
    BimanualAssemblyIsaacLabEnv,
)
from handson.sim.isaaclab.d_collection_env import HandsOnDIsaacLabEnv
from handson.sim.isaaclab.d_collection_scene import (
    D_TASK_FIRST_PHYSX_POSITION_OVERRIDES,
    D_TASK_BODY_POSITION_OVERRIDES,
    D_TASK_BODY_QUATERNION_OVERRIDES,
)
from handson.sim.isaaclab.physics_config import (
    MULTI_ENV_GPU_MAX_RIGID_CONTACT_COUNT,
    MULTI_ENV_GPU_MAX_RIGID_PATCH_COUNT,
    SINGLE_ENV_GPU_MAX_RIGID_CONTACT_COUNT,
    SINGLE_ENV_GPU_MAX_RIGID_PATCH_COUNT,
    gpu_rigid_buffer_capacities,
)


class _FakeSimulationApp:
    def __init__(self, events: list[object] | None = None) -> None:
        self.events = events if events is not None else []
        self.close_calls = 0

    def update(self) -> None:
        self.events.append("app_update")

    def close(
        self,
        *,
        wait_for_replicator: bool = True,
        skip_cleanup: bool = False,
    ) -> None:
        self.close_calls += 1
        self.events.append(("app_close", wait_for_replicator, skip_cleanup))


class _FakeRigidBodyView:
    def __init__(self) -> None:
        self.transform_writes: list[tuple[object, object]] = []
        self.velocity_writes: list[tuple[object, object]] = []

    def set_transforms(self, transforms: object, indices: object) -> None:
        self.transform_writes.append((transforms, indices))

    def set_velocities(self, velocities: object, indices: object) -> None:
        self.velocity_writes.append((velocities, indices))


def test_rigid_reset_writes_transform_only_for_kinematic_paths() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket", launch_isaac=False, image_obs=False
    )
    env._task_kinematic_body_paths = {"/World/Task/table"}
    transforms = object()
    velocities = object()
    indices = object()
    table_view = _FakeRigidBodyView()
    dynamic_view = _FakeRigidBodyView()

    env._write_rigid_reset_state(
        "/World/Task/table", table_view, indices, transforms, velocities
    )
    env._write_rigid_reset_state(
        "/World/Task/object", dynamic_view, indices, transforms, velocities
    )

    assert table_view.transform_writes == [(transforms, indices)]
    assert table_view.velocity_writes == []
    assert dynamic_view.transform_writes == [(transforms, indices)]
    assert dynamic_view.velocity_writes == [(velocities, indices)]


def test_single_environment_physx_buffers_avoid_multi_env_overallocation() -> None:
    assert SINGLE_ENV_GPU_MAX_RIGID_CONTACT_COUNT == 2**20
    assert SINGLE_ENV_GPU_MAX_RIGID_PATCH_COUNT == 5 * 2**15
    assert gpu_rigid_buffer_capacities(1) == (
        SINGLE_ENV_GPU_MAX_RIGID_CONTACT_COUNT,
        SINGLE_ENV_GPU_MAX_RIGID_PATCH_COUNT,
    )
    assert gpu_rigid_buffer_capacities(2) == (
        MULTI_ENV_GPU_MAX_RIGID_CONTACT_COUNT,
        MULTI_ENV_GPU_MAX_RIGID_PATCH_COUNT,
    )
    with pytest.raises(ValueError, match="num_envs"):
        gpu_rigid_buffer_capacities(0)


def test_collection_and_task_preview_use_single_environment_physx_profile() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sources = (
        repo_root
        / "handson/handson/sim/isaaclab/bimanual_assembly_env.py",
        repo_root / "handson/handson/sim/isaaclab/abc_collection_env.py",
    )
    for source in sources:
        text = source.read_text(encoding="utf-8")
        for setting, constant in (
            (
                "gpu_max_rigid_contact_count",
                "SINGLE_ENV_GPU_MAX_RIGID_CONTACT_COUNT",
            ),
            (
                "gpu_max_rigid_patch_count",
                "SINGLE_ENV_GPU_MAX_RIGID_PATCH_COUNT",
            ),
        ):
            assert constant in text
            assert f"{setting}=" in text or f'"{setting}":' in text


def test_standalone_task_preview_keeps_the_imported_robot_pair() -> None:
    """The task-only MJCF is reserved for envs that spawn native arms."""

    source = (
        Path(__file__).resolve().parents[2]
        / "scripts/preview_isaaclab_task.py"
    ).read_text(encoding="utf-8")
    assert "first_physx_safe=False" in source
    assert "does not construct the native collection environment" in source


def test_standalone_task_preview_initializes_cuda_modules_lazily() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "preview_isaaclab_task.py"
    ).read_text(encoding="utf-8")
    assert 'os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")' in source


def test_d02_inactive_food_is_parked_before_first_physx_reset() -> None:
    positions = D_TASK_BODY_POSITION_OVERRIDES["t12_load_and_start_microwave"]
    quaternions = D_TASK_BODY_QUATERNION_OVERRIDES[
        "t12_load_and_start_microwave"
    ]
    assert positions["003_cracker_box"][:2] == positions["d02_plate"][:2]
    assert positions["008_pudding_box"] == (-2.0, 0.0, 0.018804)
    assert positions["008_pudding_box"][:2] != positions["003_cracker_box"][:2]
    expected_broad_side = pytest.approx((2**-0.5, 2**-0.5, 0.0, 0.0))
    assert quaternions["003_cracker_box"] == expected_broad_side
    assert quaternions["008_pudding_box"] == expected_broad_side


def test_d02_first_physx_reset_keeps_fixed_microwave_at_final_anchor() -> None:
    positions = D_TASK_FIRST_PHYSX_POSITION_OVERRIDES[
        "t12_load_and_start_microwave"
    ]
    final_positions = D_TASK_BODY_POSITION_OVERRIDES[
        "t12_load_and_start_microwave"
    ]

    # Both packages are rotated +90 degrees around X, so their local Y half
    # extent becomes vertical.  Keep a 1 mm construction gap above the plate
    # and ground instead of starting GPU PhysX with coplanar broad faces.
    plate_support_z = positions["d02_plate"][2] + 0.012
    cracker_bottom_z = positions["003_cracker_box"][2] - 0.0299465
    pudding_bottom_z = positions["008_pudding_box"][2] - 0.017804
    assert cracker_bottom_z - plate_support_z == pytest.approx(0.001)
    assert pudding_bottom_z == pytest.approx(0.001)
    assert positions["microwave_object"] == final_positions["microwave_object"]
    assert positions["d02_plate"][0] < -1.5


def test_d02_uses_task_scoped_complete_hand_colliders() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "handson/sim/isaaclab/d_collection_env.py"
    ).read_text(encoding="utf-8")

    assert "d02_hand_collision_roots" not in source
    assert '"hand_colliders=instanced"' in source
    assert "D02 plate grasp physics found no hand rigid links" in source

    shared_source = (
        Path(__file__).resolve().parents[1]
        / "handson/sim/isaaclab/abc_collection_env.py"
    ).read_text(encoding="utf-8")
    assert '"t12_load_and_start_microwave"' in shared_source
    assert "ensure_proximal_finger_collisions=" in shared_source
    robot_config = shared_source[
        shared_source.index("robot_cfg = _robot_cfg(") :
        shared_source.index("robot_cfg.prim_path", shared_source.index("robot_cfg = _robot_cfg("))
    ]
    assert "_COMPLETE_EXTERNAL_HAND_COLLISION_TASKS" in robot_config
    assert "hand_contact_margin=" in robot_config
    task_preview_source = (
        Path(__file__).resolve().parents[1]
        / "handson/sim/isaaclab/task_preview.py"
    ).read_text(encoding="utf-8")
    task_only_set = task_preview_source[
        task_preview_source.index("COLLECTION_TASK_ONLY_ROBOT_TASKS") :
        task_preview_source.index(
            "_TASK_MJCF_CACHE_VERSION",
            task_preview_source.index("COLLECTION_TASK_ONLY_ROBOT_TASKS"),
        )
    ]
    assert '"t12_load_and_start_microwave"' in task_only_set

    robot_source = (
        Path(__file__).resolve().parents[1]
        / "handson/sim/isaaclab/bimanual_assembly_scene.py"
    ).read_text(encoding="utf-8")
    assert 'collision.attrib["contype"] = "1"' in robot_source
    assert 'collision.attrib["conaffinity"] = "1"' in robot_source


@pytest.mark.parametrize(
    ("active_food", "inactive_food", "active_z", "inactive_z"),
    (
        ("003_cracker_box", "008_pudding_box", 0.9449465, 0.018804),
        ("008_pudding_box", "003_cracker_box", 0.932804, 0.0309465),
    ),
)
def test_d02_reset_can_activate_either_food_after_initial_parking(
    active_food: str,
    inactive_food: str,
    active_z: float,
    inactive_z: float,
) -> None:
    env = HandsOnDIsaacLabEnv(
        "t12_load_and_start_microwave",
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    env._d02_active_food = active_food
    env._task_reset_body_states = [object()]
    writes: dict[str, dict[str, object]] = {}

    def capture(name: str, **kwargs: object) -> bool:
        writes[name] = kwargs
        return True

    env._write_reset_pose = capture
    env._set_task_dof = lambda *_args, **_kwargs: True
    env._apply_task_pose_randomization()

    assert tuple(writes[active_food]["xy"]) == (-0.34, -0.54)
    assert writes[active_food]["z"] == pytest.approx(active_z)
    assert writes[inactive_food]["xy"] == (-2.0, 0.0)
    assert writes[inactive_food]["z"] == pytest.approx(inactive_z)


@pytest.mark.parametrize(
    ("environment_cls", "kwargs"),
    (
        (BimanualAssemblyIsaacLabEnv, {}),
        (HandsOnABCIsaacLabEnv, {"task_name": "t26_place_fruit_in_bucket"}),
    ),
)
def test_setup_failure_closes_half_constructed_simulation_app(
    monkeypatch: pytest.MonkeyPatch,
    environment_cls: type,
    kwargs: dict[str, str],
) -> None:
    app = _FakeSimulationApp()

    def fail_setup(environment: object) -> None:
        environment._simulation_app = app
        environment._app_launcher = object()
        raise RuntimeError("setup failed")

    monkeypatch.setattr(environment_cls, "_setup_isaaclab", fail_setup)
    with pytest.raises(RuntimeError, match="setup failed"):
        environment_cls(
            **kwargs,
            launch_isaac=True,
            image_obs=False,
        )

    assert app.close_calls == 1
    assert app.events == [("app_close", False, False)]


def test_abc_close_stops_sim_before_app_and_is_idempotent() -> None:
    events: list[object] = []

    class FakeTimeline:
        def stop(self) -> None:
            events.append("timeline_stop")

    class FakeSim:
        _timeline = FakeTimeline()

        def clear_all_callbacks(self) -> None:
            events.append("clear_callbacks")

        def stop(self) -> None:
            events.append("sim_stop")

    app = _FakeSimulationApp(events)
    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        launch_isaac=False,
        image_obs=False,
    )
    env._sim = FakeSim()
    env._simulation_app = app
    env._app_launcher = object()

    env.close()
    env.close()

    assert events == [
        "timeline_stop",
        "clear_callbacks",
        "sim_stop",
        ("app_close", False, False),
    ]
    assert app.close_calls == 1
    assert env._sim is None
    assert env._simulation_app is None
    assert env._app_launcher is None


def test_fast_shutdown_skips_blocking_isaac_cleanup() -> None:
    events: list[object] = []
    app = _FakeSimulationApp(events)
    env = BimanualAssemblyIsaacLabEnv(launch_isaac=False, image_obs=False)
    env._simulation_app = app
    env._app_launcher = object()
    env._fast_shutdown_on_close = True

    env.close()

    assert events == [("app_close", False, True)]


def test_bimanual_interface_only_mode_supports_full_gym_transition() -> None:
    env = BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
        hz=0,
    )

    observation, reset_info = env.reset(seed=7)
    next_observation, reward, terminated, truncated, step_info = env.step(
        np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    )

    assert observation["state"]["tcp_pose"].shape == (14,)
    assert next_observation["state"]["tcp_pose"].shape == (14,)
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert reset_info["succeed"] is False
    assert step_info["succeed"] is False
    assert env.env_step == 1


@pytest.mark.parametrize(
    ("is_fixed_base", "expected_jacobi_index"),
    ((True, 6), (False, 7)),
)
def test_joint_target_cache_uses_articulation_jacobian_body_contract(
    is_fixed_base: bool,
    expected_jacobi_index: int,
) -> None:
    class StopAfterJacobianCache(RuntimeError):
        pass

    class FakeTarget:
        def clone(self):
            return self

    class FakeRobot:
        data = types.SimpleNamespace(default_joint_pos=FakeTarget())
        device = "cpu"

        def __init__(self) -> None:
            self.is_fixed_base = is_fixed_base

        def find_joints(self, _names):
            return list(range(7)), None

        def find_bodies(self, _name):
            return [7], None

    env = object.__new__(BimanualAssemblyIsaacLabEnv)
    env._haorui_ik_enabled = False
    env._control_robot = lambda _side: FakeRobot()

    def stop_after_cache(**_kwargs):
        raise StopAfterJacobianCache

    env._ik_controller_cfg_cls = stop_after_cache
    env._ik_controller_cls = object

    with pytest.raises(StopAfterJacobianCache):
        env._cache_joint_targets()

    assert env._ee_body_indices["right"] == 7
    assert env._ee_jacobi_indices["right"] == expected_jacobi_index


@pytest.mark.parametrize("uses_xyzw", (True, False))
def test_dls_uses_cached_jacobian_index_for_lab3_and_legacy(
    monkeypatch: pytest.MonkeyPatch,
    uses_xyzw: bool,
) -> None:
    calls: list[tuple[int, tuple[int, ...], bool]] = []

    def stop_at_jacobian(
        _robot,
        ee_jacobi_idx,
        arm_joint_ids,
        *,
        lab3,
    ):
        calls.append((ee_jacobi_idx, tuple(arm_joint_ids), lab3))
        raise RuntimeError("stop after Jacobian selection")

    monkeypatch.setattr(
        "handson.sim.isaaclab.bimanual_assembly_env._end_effector_link_jacobian",
        stop_at_jacobian,
    )
    env = object.__new__(BimanualAssemblyIsaacLabEnv)
    env._haorui_ik_enabled = False
    env._physics_backend = "newton"
    env._arm_ik_backend = "dls"
    env._isaaclab_uses_xyzw = uses_xyzw
    env._ik_controllers = {"right": object()}
    env._arm_joint_ids = {"right": list(range(7))}
    env._ee_body_indices = {"right": 7}
    env._ee_jacobi_indices = {"right": 6}
    env._isaac_warnings = []

    target, status = env._compute_arm_joint_target(
        "right",
        object(),
        np.asarray([0.1, -0.2, 0.8, 1.0, 0.0, 0.0, 0.0]),
        commanded=True,
    )

    assert target is None
    assert status == "ik_error"
    assert calls == [(6, tuple(range(7)), uses_xyzw)]


def test_scene_rebuild_cleanup_releases_visual_resources_once() -> None:
    events: list[object] = []

    class FakeRenderer:
        def cleanup(self, render_data: object) -> None:
            events.append(("renderer_cleanup", render_data))

    class FakeCamera:
        def __init__(self) -> None:
            self._renderer = FakeRenderer()
            self._render_data = object()
            self._data = object()
            self._view = object()
            self._sensor_prims = [object()]

        def _clear_callbacks(self) -> None:
            events.append("camera_callbacks_cleared")

    class FakeFrame:
        visible = True

        def clear(self) -> None:
            events.append("overlay_frame_cleared")

    class FakeWindow:
        visible = True

        def destroy(self) -> None:
            events.append("overlay_window_destroyed")

    camera = FakeCamera()
    render_data = camera._render_data
    env = BimanualAssemblyIsaacLabEnv(launch_isaac=False, image_obs=False)
    # The same Camera can be exposed both through the standalone mapping and
    # InteractiveScene.sensors.  Cleanup must deduplicate it by identity.
    env._cameras = {"head": camera}
    env._scene = types.SimpleNamespace(sensors={"head": camera})
    env._wrist_obstruction_visibility = (("/World/wrist", "inherited"),)
    env._wrist_obstructions_hidden = True
    env._set_wrist_obstructions_hidden = lambda hidden: events.append(
        ("wrist_hidden", hidden)
    )
    env._deferred_robot_body_visibility = (("/World/arm", "inherited"),)
    env._set_deferred_robot_body_hidden = lambda hidden: events.append(
        ("arms_hidden", hidden)
    )
    env._success_condition_frame = FakeFrame()
    env._success_condition_window = FakeWindow()
    env._success_condition_labels = {"condition": object()}
    env._success_condition_text_labels = {"condition": object()}
    env._task_status_label = object()

    report = env._prepare_scene_for_in_process_rebuild()

    assert report == {
        "wrist_visibility_restored": True,
        "overlay_released": True,
        "camera_renderers_released": 1,
    }
    assert events == [
        ("wrist_hidden", False),
        ("arms_hidden", False),
        "overlay_frame_cleared",
        "overlay_window_destroyed",
        ("renderer_cleanup", render_data),
        "camera_callbacks_cleared",
    ]
    assert camera._renderer is None
    assert camera._render_data is None
    assert camera._data is None
    assert camera._view is None
    assert camera._sensor_prims == []
    assert env._success_condition_frame is None
    assert env._success_condition_window is None
    assert env._success_condition_labels == {}
    assert env._success_condition_text_labels == {}
    assert env._task_status_label is None


def test_viewport_restore_selects_camera_before_lab3_eye_target_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCameraType:
        pass

    class FakeAttribute:
        def __init__(self) -> None:
            self.values: list[object] = []

        def IsValid(self) -> bool:
            return True

        def Set(self, value: object) -> None:
            self.values.append(value)

    focal_length = FakeAttribute()
    transform = FakeAttribute()

    class FakePrim:
        def IsValid(self) -> bool:
            return True

        def IsA(self, prim_type: type) -> bool:
            return prim_type is FakeCameraType

        def GetAttribute(self, name: str) -> FakeAttribute:
            return {
                "focalLength": focal_length,
                "xformOp:transform": transform,
            }[name]

    class FakeStage:
        def GetPrimAtPath(self, _path: str) -> FakePrim:
            return FakePrim()

    class FakeViewport:
        camera_path = "/OmniverseKit_Persp"

    viewport = FakeViewport()
    camera_calls: list[tuple[tuple[float, ...], tuple[float, ...]]] = []

    class FakeSim:
        def render(self):
            app.events.append("render")

        # Deliberately matches Lab 3: a camera_prim_path keyword would fail.
        def set_camera_view(self, eye: tuple, target: tuple) -> None:
            assert viewport.camera_path == "/World/OperatorCamera"
            camera_calls.append((tuple(eye), tuple(target)))

    omni = types.ModuleType("omni")
    omni.__path__ = []
    omni_kit = types.ModuleType("omni.kit")
    omni_kit.__path__ = []
    omni_viewport = types.ModuleType("omni.kit.viewport")
    omni_viewport.__path__ = []
    viewport_utility = types.ModuleType("omni.kit.viewport.utility")
    viewport_utility.get_active_viewport = lambda: viewport
    fake_pxr = types.ModuleType("pxr")
    fake_pxr.Sdf = types.SimpleNamespace(Path=lambda value: str(value))
    fake_pxr.UsdGeom = types.SimpleNamespace(Camera=FakeCameraType)
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.kit", omni_kit)
    monkeypatch.setitem(sys.modules, "omni.kit.viewport", omni_viewport)
    monkeypatch.setitem(
        sys.modules,
        "omni.kit.viewport.utility",
        viewport_utility,
    )
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)

    app = _FakeSimulationApp()
    env = BimanualAssemblyIsaacLabEnv(launch_isaac=False, image_obs=False)
    env._headless = False
    env._stage = FakeStage()
    env._sim = FakeSim()
    env._simulation_app = app

    restored = env._restore_viewport_camera_state(
        {
            "camera_path": "/World/OperatorCamera",
            "eye": (1.0, 2.0, 3.0),
            "target": (0.0, 0.0, 1.0),
            "attributes": {
                "focalLength": 42.0,
                "xformOp:transform": "old transform",
            },
        }
    )

    assert restored is True
    assert viewport.camera_path == "/World/OperatorCamera"
    assert camera_calls == [
        ((1.0, 2.0, 3.0), (0.0, 0.0, 1.0)),
        ((1.0, 2.0, 3.0), (0.0, 0.0, 1.0)),
    ]
    assert focal_length.values == [42.0, 42.0]
    assert transform.values == []
    assert app.events == ["render", "render", "render"]


def test_hard_reset_rebuilds_scene_in_same_app_and_python_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = FakeCuda()

    class FakeSimulationContext:
        @staticmethod
        def clear_instance() -> None:
            events.append("clear_simulation_context")

    fake_sim = types.ModuleType("isaaclab.sim")
    fake_sim.SimulationContext = FakeSimulationContext
    fake_sim.create_new_stage = lambda: events.append("create_new_stage")
    fake_isaaclab = types.ModuleType("isaaclab")
    fake_isaaclab.__path__ = []
    fake_isaaclab.sim = fake_sim
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", fake_sim)

    fake_omni = types.ModuleType("omni")
    fake_omni.__path__ = []
    fake_physx = types.ModuleType("omni.physx")
    fake_physx.get_physx_simulation_interface = lambda: types.SimpleNamespace(
        detach_stage=lambda: events.append("detach_physx_stage"))
    fake_omni.physx = fake_physx
    monkeypatch.setitem(sys.modules, "omni", fake_omni)
    monkeypatch.setitem(sys.modules, "omni.physx", fake_physx)

    class SceneResource:
        pass

    class FakeContext:
        def stop(self):
            events.append("stop")
        def clear(self):
            events.append("clear")
        def render(self):
            events.append("render")
        def clear_all_callbacks(self) -> None:
            events.append("clear_scene_callbacks")

    class Harness(BimanualAssemblyIsaacLabEnv):
        @staticmethod
        def _close_usd_stage_for_scene_rebuild(app):
            events.append("close_usd_stage")
        @staticmethod
        def _release_native_memory_after_scene_close(torch):
            events.append("release_memory")
            return {}
        def __init__(
            self,
            seed: int,
            *,
            existing_simulation_app: object,
            existing_app_launcher: object,
        ) -> None:
            self._seed = int(seed)
            self._simulation_app = existing_simulation_app
            self._app_launcher = existing_app_launcher
            self._sim = FakeContext()
            self._scene = SceneResource()
            self._stage = SceneResource()
            self._physx_scene_generation = 0
            self._robots = {"right": SceneResource()}
            self._task_body_views = {"fruit": SceneResource()}
            self._cameras = {"head": SceneResource()}
            self._physx_ground_truth_trace = SceneResource()
            self._keyboard_subscription = None
            self._keyboard_input_interface = None
            self._keyboard = None
            self._timeline_pause_subscription = None
            self._timeline = None
            self._restored_viewport_camera_state = None

        def _in_process_scene_rebuild_kwargs(
            self,
            seed: int,
        ) -> dict[str, object]:
            return {"seed": int(seed)}

        def _capture_viewport_camera_state(self) -> dict[str, object]:
            return {"camera_path": "/OmniverseKit_Persp", "attributes": {"pose": 17}}

        def _restore_viewport_camera_state(self, state: dict[str, object]) -> bool:
            self._restored_viewport_camera_state = state
            return True

    app = _FakeSimulationApp(events)
    launcher = object()
    env = Harness(
        17,
        existing_simulation_app=app,
        existing_app_launcher=launcher,
    )
    old_identity = id(env)
    old_scene = env._scene
    old_robot_ref = weakref.ref(env._robots["right"])
    old_view_ref = weakref.ref(env._task_body_views["fruit"])
    frozen_camera_state = {
        "camera_path": "/OmniverseKit_Persp",
        "eye": (-1.0, 0.2, 1.5),
        "target": (0.0, 0.0, 1.0),
    }
    env._manual_reset_viewport_camera_state = frozen_camera_state

    report = env.hard_reset_physx_scene(seed=18)
    gc.collect()

    assert id(env) == old_identity
    assert env._simulation_app is app
    assert env._app_launcher is launcher
    assert env._seed == 18
    assert env._scene is not old_scene
    assert env._physx_scene_generation == 1
    assert env._restored_viewport_camera_state is frozen_camera_state
    assert old_robot_ref() is None
    assert old_view_ref() is None
    assert report.pop("pre_rebuild_cleanup")["camera_renderers_released"] == 0
    assert report == {
        "scene_rebuilt": True,
        "scene_generation": 1,
        "seed": 18,
        "process_restarted": False,
        "simulation_app_reused": True,
    }
    assert events == [
        "app_update",
        "detach_physx_stage",
        "stop",
        "clear",
        "clear_scene_callbacks",
        "clear_simulation_context",
        "close_usd_stage",
        "release_memory",
        "create_new_stage",
        "app_update",
        "render",
        "render",
    ]
