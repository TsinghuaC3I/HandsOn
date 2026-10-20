from __future__ import annotations

import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from handson.sim.isaaclab.lw_native_articulation_contract import (
    LWNativeArticulationState,
    TASK_SPECS as CONTRACT_SPECS,
    sample_reset,
)
from handson.sim.isaaclab.lw_native_articulation_scene import (
    G05_TASK_NAME,
    G08_TASK_NAME,
    G10_TASK_NAME,
    LW_TABLE_COLLIDER_PATH,
    LW_TABLE_TOP_Z,
    LW_TASK_SPECS,
    _author_fixed_root,
    _configure_native_physics,
    _count_asset_topology,
    _spawn_fixture,
    _spawn_grip_sleeves,
    spawn_lw_task_assets,
)


@pytest.fixture()
def runtime_module(monkeypatch: pytest.MonkeyPatch):
    """Load the LW subclass without importing Kit or the shared ABC runtime."""

    base_module = types.ModuleType(
        "handson.sim.isaaclab.abc_collection_env"
    )

    class DummyBase:
        def _capture_task_scene_state(self, _task_mjcf) -> None:
            return None

        def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
            del clear_manual_request

        def reset(self, seed=None, **kwargs):
            del seed, kwargs
            return "dummy-reset"

        def get_physical_replay_reset_contract(self):
            return {"base": True}

        def _task_info(self):
            return {}

    base_module.HandsOnABCIsaacLabEnv = DummyBase
    monkeypatch.setitem(
        sys.modules,
        "handson.sim.isaaclab.abc_collection_env",
        base_module,
    )

    camera_module = types.ModuleType(
        "handson.sim.isaaclab.recording_cameras"
    )
    camera_module.WRIST_RECORDING_CAMERA_OUTPUT_SIZE = (640, 480)
    monkeypatch.setitem(
        sys.modules,
        "handson.sim.isaaclab.recording_cameras",
        camera_module,
    )

    class RecordingOriginStabilityError(RuntimeError):
        def __init__(self, message: str, *, diagnostics=None) -> None:
            super().__init__(message)
            self.diagnostics = dict(diagnostics or {})

    recording_module = types.ModuleType(
        "handson.sim.isaaclab.recording_origin"
    )
    recording_module.RecordingOriginStabilityError = (
        RecordingOriginStabilityError
    )
    monkeypatch.setitem(
        sys.modules,
        "handson.sim.isaaclab.recording_origin",
        recording_module,
    )

    if "gymnasium" not in sys.modules:
        gymnasium_module = types.ModuleType("gymnasium")
        gymnasium_module.spaces = SimpleNamespace()
        monkeypatch.setitem(sys.modules, "gymnasium", gymnasium_module)

    path = (
        Path(__file__).resolve().parents[1]
        / "handson"
        / "sim"
        / "isaaclab"
        / "lw_native_articulation_env.py"
    )
    module_name = "handson.sim.isaaclab._lw_runtime_review_under_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, RecordingOriginStabilityError


def _bare_runtime(runtime_module, task_name: str = G10_TASK_NAME):
    module, _error = runtime_module
    env = module.HandsOnLWNativeArticulationIsaacLabEnv.__new__(
        module.HandsOnLWNativeArticulationIsaacLabEnv
    )
    env.task_name = task_name
    env._lw_scene_spec = LW_TASK_SPECS[task_name]
    env._lw_contract_spec = CONTRACT_SPECS[task_name]
    env._lw_reset_sample = sample_reset(task_name, seed=17, randomize=False)
    env._lw_state = LWNativeArticulationState(
        env._lw_contract_spec,
        env._lw_reset_sample,
    )
    env._lw_contact_force_n = {"left": 0.0, "right": 0.0}
    env._lw_contact_readable = {"left": False, "right": False}
    env._lw_contact_views = {}
    env._lw_support_contact_view = None
    env._lw_support_force_n = 0.0
    env._lw_support_readable = False
    env._lw_support_contact_count = 0
    env._lw_support_min_separation_m = float("inf")
    env._lw_root_baseline_pose = None
    env._lw_origin_accepted = False
    env._lw_truth_primed = False
    env._lw_truth_wait_substeps = 0
    env._lw_last_terminal_reason_reported = ""
    env.env_step = 0
    env._reset_post_accept_substep_audits()
    return env


def _capture_spawned_cuboids(spawn, spec):
    """Exercise a scene authoring helper without importing Kit or pxr."""

    cuboids: dict[str, SimpleNamespace] = {}

    prims: dict[str, object] = {}

    class CuboidCfg:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

        def func(self, path, cfg, *, translation, **kwargs) -> None:
            assert cfg is self
            cuboids[path] = SimpleNamespace(
                size=tuple(float(value) for value in self.size),
                translation=tuple(float(value) for value in translation),
                collision_props=self.collision_props,
                spawn_kwargs=kwargs,
                visibility="inherited",
                custom_data={},
            )
            prims.setdefault(path, Prim(path))

    class Prim:
        def __init__(self, path: str) -> None:
            self.path = path
            self.visibility = "inherited"
            self.custom_data: dict[str, object] = {}

        def SetCustomDataByKey(self, key, value) -> None:
            self.custom_data[str(key)] = value

    class Imageable:
        def __init__(self, prim) -> None:
            self.prim = prim

        def __bool__(self) -> bool:
            return self.prim is not None

        def MakeInvisible(self) -> None:
            self.prim.visibility = "invisible"

        def ComputeVisibility(self) -> str:
            return self.prim.visibility

    def config(**kwargs):
        return SimpleNamespace(**kwargs)

    stage = SimpleNamespace(
        GetPrimAtPath=lambda path: prims.setdefault(path, Prim(path))
    )
    sim_utils = SimpleNamespace(
        CuboidCfg=CuboidCfg,
        CollisionPropertiesCfg=config,
        RigidBodyMaterialCfg=config,
        PreviewSurfaceCfg=config,
    )
    pxr = types.ModuleType("pxr")
    pxr.UsdGeom = SimpleNamespace(
        Imageable=Imageable,
        Tokens=SimpleNamespace(invisible="invisible"),
    )
    previous_pxr = sys.modules.get("pxr")
    sys.modules["pxr"] = pxr
    try:
        paths = spawn(stage, sim_utils, spec)
    finally:
        if previous_pxr is None:
            sys.modules.pop("pxr", None)
        else:
            sys.modules["pxr"] = previous_pxr
    assert set(paths) == set(cuboids)
    for path, cuboid in cuboids.items():
        cuboid.visibility = prims[path].visibility
        cuboid.custom_data = dict(prims[path].custom_data)
    return cuboids


def test_file_scale_is_configured_on_usd_file_cfg_not_ignored_spawn_kwarg() -> None:
    source = inspect.getsource(spawn_lw_task_assets)
    assert "scale=spec.scale" in source
    spawn_call = source[source.index("asset_cfg.func(") : source.index(
        "asset_prim =", source.index("asset_cfg.func(")
    )]
    assert "scale=" not in spawn_call


def test_fixed_source_is_converted_to_physx_inbound_world_joint_root() -> None:
    source = inspect.getsource(_author_fixed_root)
    assert "asset_root.RemoveAPI(UsdPhysics.ArticulationRootAPI)" in source
    assert "UsdPhysics.ArticulationRootAPI.Apply(fixed_root)" in source
    assert "CreateEnabledSelfCollisionsAttr().Set(spec.task_name == GEN3_TASK_NAME)" in source
    assert "CreateJointEnabledAttr().Set(True)" in source
    assert "GetLocalToWorldTransform(root_body)" in source
    assert "world_matrix.ExtractTranslation()" in source
    assert "CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))" in source


def test_visible_fixed_fixtures_keep_the_audited_initial_clearance() -> None:
    fan_spec = LW_TASK_SPECS[G05_TASK_NAME]
    fan = _capture_spawned_cuboids(_spawn_fixture, fan_spec)
    pedestal = fan[f"{fan_spec.asset_path.rsplit('/', 1)[0]}/fan_pedestal"]

    assert fan_spec.translation == (-0.300, 0.0, 0.960)
    assert fan_spec.orientation_wxyz == (0.5, -0.5, -0.5, 0.5)
    assert pedestal.size == pytest.approx((0.075, 0.085, 0.040))
    assert pedestal.translation == pytest.approx((-0.300, 0.0, 0.910))
    assert pedestal.translation[2] - pedestal.size[2] / 2 == pytest.approx(
        LW_TABLE_TOP_Z
    )
    assert pedestal.collision_props.contact_offset == pytest.approx(0.001)

    # CPU-audited collision envelopes across every declared G05 reset choice.
    # Each source collider is configured with a 1 mm contact offset, as is its
    # fixture, so a geometric gap must exceed their combined 2 mm shell.
    audited_fan_min_z_m = 0.9376203429
    pedestal_top_z = pedestal.translation[2] + pedestal.size[2] / 2
    combined_contact_offset_m = 0.002
    assert audited_fan_min_z_m - pedestal_top_z > combined_contact_offset_m

    ruler_spec = LW_TASK_SPECS[G10_TASK_NAME]
    ruler = _capture_spawned_cuboids(_spawn_fixture, ruler_spec)
    # The published ruler is free on the table, with no fixed cradle.
    assert ruler == {}
    assert ruler_spec.fixed_root is False


def test_ruler_roles_follow_the_authored_serial_joint_tree() -> None:
    spec = LW_TASK_SPECS[G10_TASK_NAME]
    assert spec.root_body_name == "Ruler035_Prop001"
    assert spec.left_body_name == "Ruler035"
    assert spec.right_body_name == "Ruler035_Prop002"
    assert spec.translation == (-0.300, 0.0, 0.8958)
    assert spec.orientation_wxyz == (1.0, 0.0, 0.0, 0.0)

    sleeves = _capture_spawned_cuboids(_spawn_grip_sleeves, spec)
    left = sleeves[f"{spec.left_body_path}/left_grip"]
    right = sleeves[f"{spec.right_body_path}/right_grip"]
    assert left.translation == pytest.approx((0.090, 0.010, 0.010))
    assert left.size == pytest.approx((0.050, 0.030, 0.018))
    assert right.translation == pytest.approx((-0.170, -0.010, 0.010))
    assert right.size == pytest.approx((0.050, 0.030, 0.018))


def test_ruler_root_body_position_matches_the_source_offset() -> None:
    spec = LW_TASK_SPECS[G10_TASK_NAME]
    # The source is a binary USD crate and the host CPU test environment has
    # no pxr module.  Its manifest hash is pinned elsewhere, so retain the
    # independently audited Prop001 local offset here and regress the numeric
    # composed body position at the published table height.
    audited_prop001_source_offset_m = np.asarray(
        (-0.095010802, -0.000009654, -0.000471499), dtype=np.float64
    )
    expected_composed_anchor_m = np.asarray(
        (-0.395010802, -0.000009654, 0.895328501), dtype=np.float64
    )
    actual_composed_anchor_m = (
        np.asarray(spec.translation, dtype=np.float64)
        + audited_prop001_source_offset_m
    )
    assert spec.root_body_name == "Ruler035_Prop001"
    assert actual_composed_anchor_m == pytest.approx(
        expected_composed_anchor_m, abs=1.0e-12
    )


def test_fan_grip_proxies_are_invisible_and_minimally_overlap_outer_ribs() -> None:
    spec = LW_TASK_SPECS[G05_TASK_NAME]
    sleeves = _capture_spawned_cuboids(_spawn_grip_sleeves, spec)
    left = sleeves[f"{spec.left_body_path}/left_grip"]
    right = sleeves[f"{spec.right_body_path}/right_grip"]
    # The collision-only sleeves now cover the usable outer-rib length so a
    # physical hand does not have to hit a small 50 mm patch to establish
    # assigned contact.  They remain invisible and keep the same thin inward
    # envelope around each rib.
    assert left.translation == pytest.approx((0.0, -0.020, 0.105))
    assert right.translation == pytest.approx((0.0, 0.020, 0.105))
    assert left.size == pytest.approx((0.030, 0.004, 0.180))
    assert right.size == pytest.approx((0.030, 0.004, 0.180))
    assert left.visibility == right.visibility == "invisible"
    assert left.custom_data["handson:renderClassification"] == (
        "collision_only_grasp_proxy"
    )
    assert right.custom_data["handson:renderClassification"] == (
        "collision_only_grasp_proxy"
    )

    # At the nominal 2.25 degree coherent reset, the tabletop mapping gives
    # uncrossed +/-Y targets and at least 40 mm clearance under the lower pad.
    nominal_left_world = np.asarray((-0.44541966, 0.05364814, 0.984))
    nominal_right_world = np.asarray((-0.44541966, -0.05364814, 0.936))
    assert nominal_left_world[1] > 0.0
    assert nominal_right_world[1] < 0.0
    assert nominal_right_world[2] - right.size[1] / 2 - LW_TABLE_TOP_Z > 0.03

    def y_interval(cuboid):
        half_width = cuboid.size[1] / 2
        return (
            cuboid.translation[1] - half_width,
            cuboid.translation[1] + half_width,
        )

    def overlap(a, b):
        return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

    # Audited source-USD local-Y collider bounds at the closed configuration.
    # A positive gap to the immediate inward neighbor also separates the
    # sleeve from every link farther inside the ordered rib stack.
    left_assigned = (-0.018859, -0.017061)
    left_neighbor = (-0.016864, -0.015066)
    right_neighbor = (0.015051, 0.016848)
    right_assigned = (0.017046, 0.018843)
    left_sleeve = y_interval(left)
    right_sleeve = y_interval(right)

    assert overlap(left_sleeve, left_assigned) > 0.0
    assert overlap(left_sleeve, left_neighbor) == 0.0
    assert left_neighbor[0] - left_sleeve[1] > 0.0
    assert overlap(right_sleeve, right_assigned) > 0.0
    assert overlap(right_sleeve, right_neighbor) == 0.0
    assert right_sleeve[0] - right_neighbor[1] > 0.0
    # Hidden collision aid may overlap its assigned rib by less than 1 mm,
    # but its outward expansion is bounded to roughly 3.2 mm rather than the
    # former 11.1 mm off-surface shell.
    assert left_assigned[0] - left_sleeve[0] < 0.0032
    assert right_sleeve[1] - right_assigned[1] < 0.0032


def test_composed_topology_counts_integration_colliders() -> None:
    source = inspect.getsource(spawn_lw_task_assets)
    assert "body_count, collider_count = _count_asset_topology(stage, spec)" in source
    assert "source_collider_count + len(grip_paths)" in source
    assert "source_body_count" in source
    assert "source_collider_count" in source
    assert callable(_count_asset_topology)


def test_capture_requires_exact_floating_ruler_articulation_and_filters_links(
    runtime_module,
) -> None:
    env = _bare_runtime(runtime_module)
    expected = env._lw_scene_spec.dof_names

    class View:
        count = 1

        def get_metatype(self, _index):
            return SimpleNamespace(
                dof_indices={expected[0]: 1, expected[1]: 0},
                fixed_base=False,
            )

        def get_dof_positions(self):
            return np.zeros((1, 2), dtype=np.float64)

        def get_dof_velocities(self):
            return np.zeros((1, 2), dtype=np.float64)

    view = View()
    root_path = env._lw_scene_spec.asset_path
    env._task_reset_articulation_states = [
        (root_path, view, np.asarray([0]), None, None, None, None)
    ]
    env._task_reset_body_states = [
        (env._lw_scene_spec.root_body_path, object()),
        (env._lw_scene_spec.left_body_path, object()),
        ("/World/Task/unrelated", object()),
    ]
    env._lw_last_q = np.zeros(2, dtype=np.float64)
    env._lw_last_qd = np.zeros(2, dtype=np.float64)

    env._capture_task_scene_state(None)

    assert env._lw_articulation_path == root_path
    assert env._lw_dof_indices == (1, 0)
    assert [entry[0] for entry in env._task_reset_body_states] == [
        "/World/Task/unrelated"
    ]


def test_capture_rejects_fixed_classification_for_floating_task(
    runtime_module,
) -> None:
    env = _bare_runtime(runtime_module)
    expected = env._lw_scene_spec.dof_names

    class View:
        count = 1

        def get_metatype(self, _index):
            return SimpleNamespace(
                dof_indices={name: index for index, name in enumerate(expected)},
                fixed_base=True,
            )

    env._task_reset_articulation_states = [
        (
            env._lw_scene_spec.asset_path,
            View(),
            np.asarray([0]),
            None,
            None,
            None,
            None,
        )
    ]
    env._task_reset_body_states = []
    with pytest.raises(RuntimeError, match="fixed-base classification changed"):
        env._capture_task_scene_state(None)


def test_zero_contact_tensor_is_readable_but_empty_tensor_fails_closed(
    runtime_module,
) -> None:
    env = _bare_runtime(runtime_module)
    env._sim_dt = 1.0 / 120.0

    class View:
        def __init__(
            self,
            values,
            *,
            distances=(0.0,),
            counts=(1,),
            starts=(0,),
        ) -> None:
            self.values = values
            self.distances = np.asarray(distances, dtype=np.float32)
            self.counts = np.asarray(counts, dtype=np.int64)
            self.starts = np.asarray(starts, dtype=np.int64)

        def get_contact_force_matrix(self, _dt):
            return self.values

        def get_contact_data(self, _dt):
            return (
                np.zeros((1,), dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32),
                self.distances,
                self.counts,
                self.starts,
            )

    env._lw_contact_views["left"] = View(
        np.zeros((1, 3, 3), dtype=np.float32)
    )
    assert env._read_assigned_contact("left") == (0.0, True)
    assert env._lw_contact_readable["left"] is True

    env._lw_contact_views["left"] = View(np.zeros((0, 3), dtype=np.float32))
    assert env._read_assigned_contact("left") == (0.0, False)
    assert env._lw_contact_readable["left"] is False


def test_g08_support_tensor_is_separate_and_fails_closed_when_empty(
    runtime_module,
) -> None:
    env = _bare_runtime(runtime_module, G08_TASK_NAME)
    env._sim_dt = 1.0 / 120.0

    class View:
        def __init__(
            self,
            values,
            *,
            distances=(0.0,),
            counts=(1,),
            starts=(0,),
        ) -> None:
            self.values = values
            self.distances = np.asarray(distances, dtype=np.float32)
            self.counts = np.asarray(counts, dtype=np.int64)
            self.starts = np.asarray(starts, dtype=np.int64)

        def get_contact_force_matrix(self, _dt):
            return self.values

        def get_contact_data(self, _dt):
            return (
                np.zeros((1,), dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32),
                self.distances,
                self.counts,
                self.starts,
            )

    env._lw_support_contact_view = View(
        np.asarray([[[0.0, 0.0, 9.81]]], dtype=np.float32),
        distances=(-0.1, -0.2, 0.0001, 0.0002),
        counts=(1,),
        starts=(2,),
    )
    force, readable = env._read_support_contact()
    assert readable is True
    assert force == pytest.approx(9.81, abs=1.0e-5)
    assert env._lw_support_readable is True
    assert env._lw_support_contact_count == 1
    assert env._lw_support_min_separation_m == pytest.approx(0.0001)

    env._lw_support_contact_view = View(np.zeros((0, 3), dtype=np.float32))
    assert env._read_support_contact() == (0.0, False)
    assert env._lw_support_readable is False


def test_g08_support_view_filters_the_actual_static_collision_prim(
    runtime_module,
) -> None:
    module, _error = runtime_module
    source = inspect.getsource(
        module.HandsOnLWNativeArticulationIsaacLabEnv._init_task_specific_runtime
    )

    assert LW_TABLE_COLLIDER_PATH.endswith("/table/geometry/mesh")
    assert "filter_patterns=[LW_TABLE_COLLIDER_PATH]" in source
    assert "filter_count) != 1" in source


def test_live_validator_writes_strict_json_without_infinity(tmp_path: Path) -> None:
    validator_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "validate_lw_native_articulation_live.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_lw_live_validator_under_test",
        validator_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = tmp_path / "evidence.json"
    module._write_evidence(
        output,
        {
            "finite": 1.25,
            "positive_infinity": math.inf,
            "negative_infinity": -math.inf,
            "not_a_number": math.nan,
            "nested": np.asarray([0.0, math.inf]),
        },
    )

    raw = output.read_text(encoding="utf-8")
    assert "Infinity" not in raw
    assert "NaN" not in raw
    assert json.loads(raw) == {
        "finite": 1.25,
        "negative_infinity": None,
        "nested": [0.0, None],
        "not_a_number": None,
        "positive_infinity": None,
    }


def test_live_validator_records_the_actual_origin_key_and_observations() -> None:
    validator_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "validate_lw_native_articulation_live.py"
    )
    source = validator_path.read_text(encoding="utf-8")

    assert '"task_recording_origin_acceptance"' in source
    assert '"task_origin_acceptance"' not in source
    assert "env.observation_space.contains(observation)" in source
    assert "raw._lw_articulation_view.get_metatype(0)" in source
    assert '"physx_link_count"' in source
    assert '"physx_dof_count"' in source
    assert '"handson:renderClassification"' in source
    assert "ComputeVisibility()" in source
    assert '"renderable_collision_only_paths"' in source


@pytest.mark.parametrize(
    "task_name",
    (G05_TASK_NAME, G08_TASK_NAME, G10_TASK_NAME),
)
def test_post_accept_audit_latches_every_physics_frame(
    runtime_module,
    task_name: str,
) -> None:
    env = _bare_runtime(runtime_module, task_name)
    env._lw_origin_accepted = True
    accepted_q = np.asarray(
        env._lw_reset_sample.initial_q_rad,
        dtype=np.float64,
    )
    for index in range(240):
        truth = {
            "q": accepted_q + (index + 1) * 1.0e-6,
            "qd": np.full_like(accepted_q, (index + 1) * 1.0e-5),
            "dof_available": True,
            "contact_available": True,
            "root_pose": np.asarray(
                [
                    *env._lw_scene_spec.translation,
                    *env._lw_scene_spec.orientation_wxyz,
                ],
                dtype=np.float64,
            ),
            "root_translation_error_m": index * 1.0e-6,
            "root_rotation_error_deg": index * 1.0e-4,
        }
        if task_name == G08_TASK_NAME:
            truth.update(
                {
                    "support_available": True,
                    "support_force": 9.81 - index * 1.0e-3,
                    "support_contact_count": 1 + int(index % 2 == 0),
                    "support_clearance_available": True,
                    "support_clearance_m": (
                        -0.00020 if index == 17 else 0.00040
                    ),
                    "support_min_separation_m": (
                        -0.00010 if index == 29 else 0.00030
                    ),
                }
            )
        env._accumulate_post_accept_substep_audits(truth)

    audit = env._post_accept_substep_audit_snapshot()
    assert audit["sample_count"] == 240
    assert audit["unreadable_count"] == 0
    assert audit["truth_unreadable"] is False
    assert audit["max_abs_q_error_rad"] == pytest.approx(0.000240)
    assert audit["max_abs_qd_rad_s"] == pytest.approx(0.00240)
    assert audit["max_root_translation_error_m"] == pytest.approx(0.000239)
    assert audit["max_root_rotation_error_deg"] == pytest.approx(0.0239)

    support = env._g08_support_substep_audit_snapshot()
    if task_name == G08_TASK_NAME:
        assert support["sample_count"] == 240
        assert support["unreadable_count"] == 0
        assert support["support_unreadable"] is False
        assert support["min_support_force_n"] == pytest.approx(9.571)
        assert support["min_support_contact_count"] == 1
        assert support["support_clearance_range_m"] == pytest.approx(
            [-0.00020, 0.00040]
        )
        assert support["support_separation_range_m"] == pytest.approx(
            [-0.00010, 0.00030]
        )
    else:
        assert support["sample_count"] == 0


def test_post_accept_audit_marks_any_unreadable_frame_and_callback_wires_it(
    runtime_module,
) -> None:
    module, _error = runtime_module
    env = _bare_runtime(runtime_module, G08_TASK_NAME)
    env._lw_origin_accepted = True
    q = np.asarray(env._lw_reset_sample.initial_q_rad, dtype=np.float64)
    env._accumulate_post_accept_substep_audits(
        {
            "q": q,
            "qd": np.zeros_like(q),
            "dof_available": False,
            "contact_available": True,
            "root_pose": None,
            "root_translation_error_m": math.inf,
            "root_rotation_error_deg": math.inf,
            "support_available": False,
            "support_force": 9.81,
            "support_contact_count": 2,
            "support_clearance_available": False,
            "support_clearance_m": math.inf,
            "support_min_separation_m": math.inf,
        }
    )
    assert env._post_accept_substep_audit_snapshot()["truth_unreadable"] is True
    support = env._g08_support_substep_audit_snapshot()
    assert support["support_unreadable"] is True
    assert support["min_support_force_n"] == 0.0
    assert support["min_support_contact_count"] == 0

    source = inspect.getsource(
        module.HandsOnLWNativeArticulationIsaacLabEnv
        ._handson_post_physics_substep_callback
    )
    read_index = source.index("truth = self._read_lw_truth()")
    latch_index = source.index(
        "self._accumulate_post_accept_substep_audits(truth)"
    )
    update_index = source.index("self._lw_state.update(")
    assert read_index < latch_index < update_index


def test_live_validator_requires_exact_substep_counts_and_extrema() -> None:
    validator_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "validate_lw_native_articulation_live.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_lw_live_validator_substep_under_test",
        validator_path,
    )
    assert spec is not None and spec.loader is not None
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)

    class Raw:
        task_name = G08_TASK_NAME

        def _post_accept_substep_audit_snapshot(self):
            return {
                "sample_count": 240,
                "unreadable_count": 0,
                "truth_unreadable": False,
                "max_abs_q_error_rad": 0.001,
                "max_abs_qd_rad_s": 0.02,
                "max_root_translation_error_m": 0.0004,
                "max_root_rotation_error_deg": 0.1,
            }

        def _g08_support_substep_audit_snapshot(self):
            return {
                "sample_count": 240,
                "unreadable_count": 0,
                "support_unreadable": False,
                "min_support_force_n": 9.5,
                "min_support_contact_count": 1,
                "support_clearance_range_m": [-0.00020, 0.00040],
                "support_separation_range_m": [-0.00010, 0.00030],
            }

    raw = Raw()
    audit, support = validator._validate_post_accept_substep_audits(raw, 240)
    assert audit["sample_count"] == 240
    assert support is not None and support["sample_count"] == 240

    raw._post_accept_substep_audit_snapshot = lambda: {
        **audit,
        "sample_count": 239,
    }
    with pytest.raises(RuntimeError, match="count mismatch"):
        validator._validate_post_accept_substep_audits(raw, 240)


def test_g08_authored_bottom_clearance_uses_collision_not_visual_bounds(
    runtime_module,
) -> None:
    module, _error = runtime_module
    spec = LW_TASK_SPECS[G08_TASK_NAME]
    pose = np.asarray(
        [*spec.translation, *spec.orientation_wxyz],
        dtype=np.float64,
    )
    clearance = module._g08_support_clearance_m(pose)

    assert clearance == pytest.approx(0.00025, abs=2.0e-9)
    assert 0.0 <= clearance <= 0.002


def test_g08_composed_stage_makes_the_true_bottom_gpu_compatible() -> None:
    source = inspect.getsource(_configure_native_physics)

    assert 'prim.GetName() == "MicropipetteBox001_Collider1"' in source
    assert "UsdPhysics.Tokens.boundingCube" in source


def test_source_collision_geometry_is_explicitly_non_rendering() -> None:
    source = inspect.getsource(_configure_native_physics)

    assert '"/Collisions/" in prim_path' in source
    assert "imageable.MakeInvisible()" in source
    assert '"collision_only_source"' in source
    assert "purpose=guide" in source


def test_fixed_lid_origin_records_table_support_without_requiring_it(
    runtime_module,
) -> None:
    env = _bare_runtime(runtime_module, G08_TASK_NAME)
    env._physics_body_velocity = lambda _name: np.zeros(6, dtype=np.float64)
    q = np.asarray(env._lw_contract_spec.authored_q0_rad, dtype=np.float64)
    truth = {
        "q": q,
        "qd": np.zeros_like(q),
        "dof_available": True,
        "left_force": 0.0,
        "right_force": 0.0,
        "contact_available": True,
        "support_force": 0.0,
        "support_available": True,
        "support_active": False,
        "support_contact_count": 0,
        "support_min_separation_m": 0.0,
        "support_separation_valid": False,
        "support_clearance_m": 0.0001,
        "support_clearance_available": True,
        "support_clearance_valid": True,
        "root_pose": np.asarray(
            [*env._lw_scene_spec.translation, *env._lw_scene_spec.orientation_wxyz],
            dtype=np.float64,
        ),
        "root_translation_error_m": 0.0,
        "root_rotation_error_deg": 0.0,
    }
    env._read_lw_truth = lambda: dict(truth)

    report = env.stable_recording_origin_task_readiness()
    assert report["ready"] is True
    assert report["support_truth_required"] is False
    assert report["support_truth_available"] is True
    assert report["support_contact_active"] is False

    truth["support_force"] = 9.81
    truth["support_active"] = True
    truth["support_contact_count"] = 1
    truth["support_min_separation_m"] = 0.0001
    truth["support_separation_valid"] = True
    report = env.stable_recording_origin_task_readiness()
    assert report["ready"] is True
    assert report["support_contact_active"] is True


@pytest.mark.parametrize(
    "task_name",
    (G05_TASK_NAME, G08_TASK_NAME, G10_TASK_NAME),
)
def test_unavailable_origin_truth_is_strict_json_evidence(
    runtime_module,
    task_name: str,
) -> None:
    env = _bare_runtime(runtime_module, task_name)
    count = env._lw_contract_spec.dof_count
    env._physics_body_velocity = lambda _name: None
    env._read_lw_truth = lambda: {
        "q": np.full(count, np.nan, dtype=np.float64),
        "qd": np.full(count, np.inf, dtype=np.float64),
        "dof_available": False,
        "left_force": 0.0,
        "right_force": 0.0,
        "contact_available": False,
        "support_force": 0.0,
        "support_available": False,
        "support_active": False,
        "support_contact_count": 0,
        "support_min_separation_m": math.inf,
        "support_separation_valid": False,
        "support_clearance_m": math.inf,
        "support_clearance_available": False,
        "support_clearance_valid": False,
        "root_pose": None,
        "root_translation_error_m": math.inf,
        "root_rotation_error_deg": math.inf,
    }

    report = env.stable_recording_origin_task_readiness()

    assert report["ready"] is False
    assert report["q_rad"] == [None] * count
    assert report["qd_rad_s"] == [None] * count
    assert report["support_clearance_m"] is None
    assert report["support_min_separation_m"] is None
    assert report["root_translation_error_m"] is None
    assert report["root_rotation_error_deg"] is None
    assert report["root_linear_speed_mps"] is None
    assert report["root_angular_speed_radps"] is None
    assert report["max_dof_speed_radps"] is None
    assert report["max_initial_dof_drift_rad"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "task_name",
    (G05_TASK_NAME, G08_TASK_NAME, G10_TASK_NAME),
)
def test_task_info_never_exports_nonfinite_support_sentinels(
    runtime_module,
    task_name: str,
) -> None:
    env = _bare_runtime(runtime_module, task_name)

    info = env._task_info()

    assert info["lw_support_min_separation_m"] is None
    json.dumps(info, allow_nan=False)


def test_recording_origin_readiness_rejects_target_and_fixture_drift(
    runtime_module,
) -> None:
    env = _bare_runtime(runtime_module, G08_TASK_NAME)
    env._physics_body_velocity = lambda _name: np.zeros(6, dtype=np.float64)
    target = np.asarray(env._lw_contract_spec.target_q_rad, dtype=np.float64)
    truth = {
        "q": target,
        "qd": np.zeros_like(target),
        "dof_available": True,
        "left_force": 0.0,
        "right_force": 0.0,
        "contact_available": True,
        "root_pose": np.asarray(
            [*env._lw_scene_spec.translation, *env._lw_scene_spec.orientation_wxyz],
            dtype=np.float64,
        ),
        "root_translation_error_m": 0.0,
        "root_rotation_error_deg": 0.0,
    }
    env._read_lw_truth = lambda: dict(truth)

    report = env.stable_recording_origin_task_readiness()
    assert report["ready"] is False
    assert report["initial_coordinates_valid"] is False
    assert "target window" in report["initial_coordinates_error"]

    truth["q"] = np.asarray(
        env._lw_contract_spec.authored_q0_rad,
        dtype=np.float64,
    )
    truth["root_translation_error_m"] = (
        env._lw_contract_spec.root_translation_limit_m + 1.0e-6
    )
    report = env.stable_recording_origin_task_readiness()
    assert report["initial_coordinates_valid"] is True
    assert report["root_within_task_contract"] is False
    assert report["ready"] is False


def test_origin_acceptance_requires_final_proof_action_zero_and_exact_qd(
    runtime_module,
) -> None:
    _module, error_type = runtime_module
    env = _bare_runtime(runtime_module)
    q = np.asarray(env._lw_contract_spec.authored_q0_rad, dtype=np.float64)
    truth = {
        "q": q,
        "qd": np.zeros_like(q),
        "dof_available": True,
        "left_force": 0.0,
        "right_force": 0.0,
        "contact_available": True,
        "root_pose": np.asarray(
            [*env._lw_scene_spec.translation, *env._lw_scene_spec.orientation_wxyz],
            dtype=np.float64,
        ),
        "root_translation_error_m": 0.0,
        "root_rotation_error_deg": 0.0,
    }
    env._read_lw_truth = lambda: dict(truth)
    valid_report = {
        "task_readiness_required": True,
        "final_task_readiness": {
            "ready": True,
            "task_name": env.task_name,
        },
    }

    with pytest.raises(error_type, match="final continuous"):
        env.accept_stable_recording_origin({})

    env.env_step = 1
    with pytest.raises(error_type, match="after action zero"):
        env.accept_stable_recording_origin(valid_report)
    env.env_step = 0

    truth["qd"] = np.asarray([0.0, 1.0e-12], dtype=np.float64)
    with pytest.raises(error_type, match="exact-zero"):
        env.accept_stable_recording_origin(valid_report)
    truth["qd"] = np.zeros_like(q)

    env._lw_origin_accepted = True
    env._accumulate_post_accept_substep_audits(truth)
    env._lw_origin_accepted = False
    assert env._post_accept_substep_audit_snapshot()["sample_count"] == 1
    acceptance = env.accept_stable_recording_origin(valid_report)
    assert acceptance["accepted"] is True
    json.dumps(acceptance, allow_nan=False)
    assert env._lw_origin_accepted is True
    assert env._lw_truth_primed is True
    assert env._lw_state.sim_time_s == 0.0
    assert not np.any(env._lw_state.contact_seen)
    assert env._post_accept_substep_audit_snapshot()["sample_count"] == 0


def test_consecutive_reset_samples_use_distinct_deterministic_episode_seeds(
    runtime_module,
) -> None:
    env = _bare_runtime(runtime_module)
    env._seed = 23
    env.randomize = True
    env.delta_h = np.ones(1, dtype=np.float64)
    env._lw_reset_serial = -1

    env._lw_origin_accepted = True
    q = np.asarray(env._lw_reset_sample.initial_q_rad, dtype=np.float64)
    env._accumulate_post_accept_substep_audits(
        {
            "q": q,
            "qd": np.zeros_like(q),
            "dof_available": True,
            "contact_available": True,
            "root_pose": np.asarray(
                [*env._lw_scene_spec.translation, *env._lw_scene_spec.orientation_wxyz],
                dtype=np.float64,
            ),
            "root_translation_error_m": 0.0,
            "root_rotation_error_deg": 0.0,
        }
    )
    assert env._post_accept_substep_audit_snapshot()["sample_count"] == 1
    env._sample_reset_state(clear_manual_request=True)
    first = env._lw_reset_sample
    assert env._post_accept_substep_audit_snapshot()["sample_count"] == 0
    env._sample_reset_state(clear_manual_request=True)
    second = env._lw_reset_sample

    assert first.seed == 23
    assert second.seed == 23 + 104729
    assert first != second
    assert env.delta_h.tolist() == [0.0]
    assert env._lw_state.failure_reason == ""


def test_replay_contract_names_the_true_fixed_or_floating_articulation_root(
    runtime_module,
) -> None:
    fixed = _bare_runtime(runtime_module, G08_TASK_NAME)
    floating = _bare_runtime(runtime_module, G10_TASK_NAME)

    assert fixed.get_physical_replay_reset_contract()[
        "lw_articulation_root"
    ] == f"{fixed._lw_scene_spec.asset_path}/world_fixed_joint"
    assert floating.get_physical_replay_reset_contract()[
        "lw_articulation_root"
    ] == floating._lw_scene_spec.asset_path


def test_fixed_lid_callback_arms_after_origin_without_table_support_gate(
    runtime_module,
) -> None:
    env = _bare_runtime(runtime_module, G08_TASK_NAME)
    env._launch_isaac = True
    env._lw_truth_primed = True
    env._sim_dt = 1.0 / 120.0
    env._accumulate_post_accept_substep_audits = lambda _truth: None
    truth = {
        "q": np.asarray(env._lw_contract_spec.authored_q0_rad),
        "qd": np.zeros(env._lw_contract_spec.dof_count),
        "dof_available": True,
        "left_force": 0.2,
        "right_force": 0.2,
        "contact_available": True,
        "support_available": True,
        "support_active": False,
        "support_clearance_valid": True,
        "support_separation_valid": True,
        "root_pose": np.zeros(7),
        "root_translation_error_m": 0.0,
        "root_rotation_error_deg": 0.0,
    }
    env._read_lw_truth = lambda: dict(truth)
    captured: list[tuple[object, ...]] = []
    env._lw_state = SimpleNamespace(
        update=lambda *args: captured.append(args),
    )

    # Truth priming during stable-origin settling is observational only.  Task
    # progress and terminal predicates arm at the accepted action-zero boundary.
    env._handson_post_physics_substep_callback(0, 4)
    assert captured == []

    env._lw_origin_accepted = True
    env._handson_post_physics_substep_callback(0, 4)
    assert len(captured) == 1
    assert captured[0][-2] is True
    assert captured[0][-1] is True


def test_first_lw_terminal_reason_is_structured_before_reset(
    runtime_module,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = _bare_runtime(runtime_module, G08_TASK_NAME)
    env._lw_state.root_translation_error_m = 0.201
    env._lw_state.root_translation_escape_s = 0.5
    env._lw_state._fail("root_translation_escape")

    assert env._compute_failure() is True
    first = capsys.readouterr().out.strip()
    assert first.startswith("[HandsOn/LW terminal] ")
    payload = json.loads(first.split("] ", 1)[1])
    assert payload["event"] == "lw_first_terminal"
    assert payload["failure_reason"] == "root_translation_escape"
    assert payload["root_translation_error_m"] == pytest.approx(0.201)
    assert payload["root_escape_translation_limit_m"] == pytest.approx(0.2)
    assert payload["root_escape_hold_s"] == pytest.approx(0.5)
    assert payload["arming_stage"] == "post_recording_origin_action_substeps"
    json.dumps(payload, allow_nan=False)

    # The first-cause latch is emitted once; repeated failure polling before
    # the reset transaction cannot spam or replace it.
    assert env._compute_failure() is True
    assert capsys.readouterr().out == ""
    info = env._task_info()
    assert info["failure_reason"] == "root_translation_escape"
    assert info["lw_failure_reason"] == "root_translation_escape"
    json.dumps(info, allow_nan=False)
