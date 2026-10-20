from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
from handson.sim.isaaclab.bimanual_assembly_env import (
    BimanualAssemblyIsaacLabEnv,
)
from handson.sim.isaaclab.fu015_contract import (
    ACTION_DIM,
    BASE_BODY_PATH,
    DISPLAY_BODY_PATH,
    FORBIDDEN_CONTACT_FORCE_N,
    FU015PhysicalReadback,
    FU015ResetState,
    FU015ScriptedPolicy,
    FU015SuccessTracker,
    HAND_ACTION_DIM,
    HINGE_TRAVEL_DEG,
    LEFT_HOME,
    NOMINAL_ROOT_YAW_DEG,
    REQUIRED_CONTACT_FORCE_N,
    RENDER_RESOLUTION_HW,
    RIGHT_HOME,
    SUCCESS_HOLD_STEPS,
    TARGET_OPEN_DEG,
    TASK_ID,
    TASK_NAME,
    TIMEOUT_STEPS,
    WUJI_POWER_GRIP,
    _NOMINAL_HINGE_PIVOT_XZ_M,
    _RIGHT_DISPLAY_TRACK_RADIUS_M,
    _RIGHT_COMMAND_OPEN_DEG,
    _RIGHT_REACHABLE_QUAT_WXYZ,
    _RIGHT_REFERENCE_TCP_XZ_M,
    _RIGHT_SHALLOW_CLEARANCE_MAX_M,
    _RIGHT_SHALLOW_LIVE_LEAD_MAX_DEG,
    _RIGHT_TCP_ARC_CENTER_XZ_M,
    _display_edge_radial_xz,
    _right_shallow_attachment_tilt_pose,
    _right_shallow_clearance_lift_m,
    _right_tcp_hinge_arc_pose,
    _right_tcp_pose_for_opening,
    evaluate_physical_readback,
    sample_reset_state,
)
from handson.sim.isaaclab.fu015_collection_env import (
    HandsOnFU015IsaacLabEnv,
    _ASSIST_DEGREES_PER_PHYSICS_STEP,
    _BASE_FIXED_JOINT_PATH,
    _BASE_FRONT_PROXY_CENTER_M,
    _BASE_FRONT_PROXY_LOWER_M,
    _BASE_FRONT_PROXY_SIZE_M,
    _BASE_FRONT_PROXY_UPPER_M,
    _HINGE_CLOSED_LOCK_WINDOW_DEG,
    _HINGE_DRIVE_MAX_FORCE,
    _HINGE_DRIVE_STIFFNESS,
    _KITCHEN_PREVIEW_ASSET_RELATIVE_PATH,
    _KITCHEN_PREVIEW_ASSET_SHA256,
    _KITCHEN_PREVIEW_ENV,
    _KITCHEN_PREVIEW_ISLAND_EXPECTED_MAX_M,
    _KITCHEN_PREVIEW_ISLAND_EXPECTED_MIN_M,
    _KITCHEN_PREVIEW_ROOT,
    _KITCHEN_PREVIEW_SCALE,
    _KITCHEN_PREVIEW_TRANSLATE_M,
    _LEFT_ROBOT_BASE_X_OFFSET_M,
    _RIGHT_ROBOT_BASE_X_OFFSET_M,
    _RIGHT_OPENING_INTENT_MIN_DELTA_M,
    _author_fu015_fixed_base_articulation,
    _configure_fu015_robot_cfg,
    _fu015_env_flag,
    _fu015_is_articulation_link_path,
    _hinge_motion_allowed,
    _next_hinge_drive_target_deg,
    _right_tcp_opening_intent,
    points_in_body_box,
    _transform_composed_root_pose,
    summarize_contact_forces_by_filter,
    summarize_contact_manifold_points_by_filter,
    summarize_contact_points_by_filter,
    validate_fu015_kitchen_preview_asset,
)
from handson.sim.isaaclab.realistic_kitchen_scene import (
    compose_realistic_kitchen_visual,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _successful_readback(**overrides) -> FU015PhysicalReadback:
    values = {
        "hinge_angle_rad": math.radians(-90.0),
        "base_position_m": (0.02, 0.0, 0.891),
        "reset_base_position_m": (0.02, 0.0, 0.891),
        "base_linear_speed_mps": 0.0,
        "base_angular_speed_radps": 0.0,
        "display_linear_speed_mps": 0.0,
        "display_angular_speed_radps": 0.0,
        "left_base_contact_force_n": REQUIRED_CONTACT_FORCE_N + 0.1,
        "right_display_contact_force_n": REQUIRED_CONTACT_FORCE_N + 0.1,
        "left_touchpad_contact_force_n": 0.0,
        "right_base_contact_force_n": 0.0,
        "robot_table_contact_force_n": 0.0,
    }
    values.update(overrides)
    return FU015PhysicalReadback(**values)


def test_b320_fu015_identity_ui_and_collection_registration() -> None:
    """Audit-facing identity/UI coverage for the formal collection route."""
    from handson.runtime.collection_support import ISAACSIM_COLLECTION_TASKS
    from handson.tasks.mappings import CONFIG_MAPPING

    assert TASK_ID == "B320-FU015"
    assert TASK_NAME == "t16_open_laptop"
    assert TASK_NAME in CONFIG_MAPPING
    assert TASK_NAME in ISAACSIM_COLLECTION_TASKS


def test_b320_fu015_main_camera_resolution_uses_height_width_runtime_order() -> None:
    """Task YAML's 640x480 width/height becomes runtime (height, width)."""
    assert RENDER_RESOLUTION_HW == (480, 640)


def test_b320_fu015_kitchen_preview_is_explicitly_opt_in(monkeypatch) -> None:
    monkeypatch.delenv(_KITCHEN_PREVIEW_ENV, raising=False)
    assert not _fu015_env_flag(_KITCHEN_PREVIEW_ENV)

    monkeypatch.setenv(_KITCHEN_PREVIEW_ENV, "1")
    assert _fu015_env_flag(_KITCHEN_PREVIEW_ENV)
    monkeypatch.setenv(_KITCHEN_PREVIEW_ENV, "0")
    assert not _fu015_env_flag(_KITCHEN_PREVIEW_ENV)

    for invalid in ("true", "false", "yes", "no", "on", "off", "", " 1"):
        monkeypatch.setenv(_KITCHEN_PREVIEW_ENV, invalid)
        with pytest.raises(ValueError, match="exactly '1' or '0'"):
            _fu015_env_flag(_KITCHEN_PREVIEW_ENV)


def test_b320_fu015_kitchen_asset_is_attributed_and_hash_locked() -> None:
    asset_path = _REPOSITORY_ROOT / "handson/handson" / _KITCHEN_PREVIEW_ASSET_RELATIVE_PATH
    source_path = asset_path.parent / "SOURCE.md"
    license_path = asset_path.parent / "LICENSE.txt"
    conversion_path = asset_path.parent / "modern_kitchen_visual.conversion.json"
    textures_dir = asset_path.parent / "textures"

    assert validate_fu015_kitchen_preview_asset(_REPOSITORY_ROOT / "handson/handson") == asset_path
    digest_builder = hashlib.sha256()
    with asset_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest_builder.update(block)
    digest = digest_builder.hexdigest()
    assert digest == _KITCHEN_PREVIEW_ASSET_SHA256

    license_text = license_path.read_text(encoding="utf-8")
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    assert textures_dir.is_dir()
    textures = sorted(
        path
        for path in textures_dir.iterdir()
        if path.is_file()
    )
    assert "Visthétique" in license_text
    assert "CC-BY-4.0" in license_text
    assert conversion["output"]["sha256"] == digest
    assert len(textures) == conversion["output"]["external_texture_count"] == 59
    assert sum(path.stat().st_size for path in textures) == conversion["output"][
        "external_texture_bytes"
    ] == 98_093_376
    assert {path.suffix.lower() for path in textures} <= {".png", ".jpeg"}
    assert conversion["sanitization"] == {
        "render_only": True,
        "physics_schemas": 0,
        "camera_prims": 0,
        "light_prims": 0,
        "mesh_count_before_pruning": 155,
        "active_mesh_count_after_pruning": 144,
        "deactivated_brand_or_label_groups": 8,
    }


def test_b320_fu015_kitchen_preview_reuses_only_canonical_colliders() -> None:
    task_source = inspect.getsource(
        HandsOnABCIsaacLabEnv._spawn_realistic_kitchen_preview
    )
    composer_source = inspect.getsource(compose_realistic_kitchen_visual)

    assert _KITCHEN_PREVIEW_ROOT.startswith("/World/VisualEnvironment/")
    assert 'for support_name in ("table", "floor")' in task_source
    assert "compose_realistic_kitchen_visual" in task_source
    assert "any IsaacLab task" in task_source
    assert "UsdPhysics.CollisionAPI" in composer_source
    assert "UsdPhysics.RigidBodyAPI" in composer_source
    assert "UsdPhysics.ArticulationRootAPI" in composer_source
    assert "MakeInvisible()" in composer_source
    assert "AddReference(asset_path.as_posix())" in composer_source
    assert "CollisionAPI.Apply" not in composer_source

    standalone_source = inspect.getsource(
        BimanualAssemblyIsaacLabEnv._spawn_realistic_kitchen_preview
    )
    assert "compose_realistic_kitchen_visual" in standalone_source
    assert 'table_prim_path="/World/envs/env_0/Table"' in standalone_source
    assert "align_to_table=True" in standalone_source


def test_b320_fu015_kitchen_island_matches_canonical_table() -> None:
    source_min = np.asarray([-1.035248, 0.152960, -1.720574])
    source_max = np.asarray([0.759922, 1.087552, -0.827508])
    corners = np.asarray(
        [
            [x, y, z]
            for x in (source_min[0], source_max[0])
            for y in (source_min[1], source_max[1])
            for z in (source_min[2], source_max[2])
        ],
        dtype=np.float64,
    )
    scale = _KITCHEN_PREVIEW_SCALE
    translated = np.column_stack(
        (-scale * corners[:, 2], -scale * corners[:, 0], scale * corners[:, 1])
    ) + np.asarray(_KITCHEN_PREVIEW_TRANSLATE_M)
    island_min = translated.min(axis=0)
    island_max = translated.max(axis=0)

    assert np.allclose(island_min, _KITCHEN_PREVIEW_ISLAND_EXPECTED_MIN_M, atol=2e-6)
    assert np.allclose(island_max, _KITCHEN_PREVIEW_ISLAND_EXPECTED_MAX_M, atol=2e-6)
    assert island_max[2] == pytest.approx(0.89, abs=2e-6)
    assert island_min[0] <= -0.55 and island_max[0] >= 0.25
    assert island_min[1] <= -0.8 and island_max[1] >= 0.8


def test_b320_fu015_has_no_auxiliary_front_edge_collider() -> None:
    """Closed-lid contact must use the display shell without an added strip."""
    source = inspect.getsource(
        HandsOnFU015IsaacLabEnv._spawn_task_specific_assets
    )

    assert "MacBookFrontEdgeGripCollision" not in source
    assert "analytic_boxes=2" in source


def test_b320_fu015_uses_native_fixed_base_articulation() -> None:
    """The laptop root must be fixed to world, not to a movable anchor body."""
    source = inspect.getsource(_author_fu015_fixed_base_articulation)

    assert _BASE_FIXED_JOINT_PATH == f"{BASE_BODY_PATH}/world_fixed_joint"
    assert "CreateBody1Rel().SetTargets([base.GetPath()])" in source
    assert "UsdPhysics.ArticulationRootAPI.Apply(fixed_root)" in source
    assert "CreateBody0Rel" not in source
    assert "KinematicEnabledAttr().Set(True)" not in source


def test_b320_fu015_reset_cache_excludes_articulation_links() -> None:
    assert _fu015_is_articulation_link_path(BASE_BODY_PATH)
    assert _fu015_is_articulation_link_path(DISPLAY_BODY_PATH)
    assert not _fu015_is_articulation_link_path("/World/Task/table")

    source = inspect.getsource(HandsOnFU015IsaacLabEnv._capture_task_scene_state)
    assert "_task_reset_body_states" in source
    assert "_fu015_is_articulation_link_path" in source


def test_b320_fu015_hinge_drive_never_retracts_overshoot() -> None:
    assert _next_hinge_drive_target_deg(
        current_angle_deg=-100.0,
        drive_target_deg=-90.0,
        task_target_deg=90.0,
        assist_increment_deg=0.0,
    ) == pytest.approx(-100.0)
    assert _next_hinge_drive_target_deg(
        current_angle_deg=-100.0,
        drive_target_deg=-90.0,
        task_target_deg=90.0,
        assist_increment_deg=0.4,
    ) == pytest.approx(-100.0)
    assert _next_hinge_drive_target_deg(
        current_angle_deg=-40.0,
        drive_target_deg=-40.0,
        task_target_deg=90.0,
        assist_increment_deg=0.4,
    ) == pytest.approx(-40.4)
    assert _next_hinge_drive_target_deg(
        current_angle_deg=-10.0,
        drive_target_deg=-20.0,
        task_target_deg=90.0,
        assist_increment_deg=0.0,
    ) == pytest.approx(-20.0)


def test_b320_fu015_downward_contact_cannot_request_opening_assist() -> None:
    threshold = _RIGHT_OPENING_INTENT_MIN_DELTA_M

    assert not _right_tcp_opening_intent(None, 1.0)
    assert not _right_tcp_opening_intent(1.0, 1.0)
    assert not _right_tcp_opening_intent(1.0, 0.999)
    assert not _right_tcp_opening_intent(1.0, 1.0 + threshold * 0.99)
    assert _right_tcp_opening_intent(1.0, 1.0 + threshold)
    assert not _right_tcp_opening_intent(math.nan, 1.0)

    source = inspect.getsource(
        HandsOnFU015IsaacLabEnv._after_task_physics_substep
    )
    assert "owner_ready and self._fu015_right_opening_intent" in source
    assert "if self._fu015_assist_active" in source


def test_b320_fu015_hinge_motion_requires_display_contact_and_safe_intent() -> None:
    assert _HINGE_CLOSED_LOCK_WINDOW_DEG == pytest.approx(0.25)
    assert not _hinge_motion_allowed(
        right_display_contact=False,
        hinge_angle_deg=0.0,
        opening_intent=True,
    )
    assert not _hinge_motion_allowed(
        right_display_contact=True,
        hinge_angle_deg=0.0,
        opening_intent=False,
    )
    assert _hinge_motion_allowed(
        right_display_contact=True,
        hinge_angle_deg=0.0,
        opening_intent=True,
    )
    assert _hinge_motion_allowed(
        right_display_contact=True,
        hinge_angle_deg=-5.0,
        opening_intent=False,
    )
    assert not _hinge_motion_allowed(
        right_display_contact=False,
        hinge_angle_deg=-5.0,
        opening_intent=False,
    )
    assert not _hinge_motion_allowed(
        right_display_contact=True,
        hinge_angle_deg=math.nan,
        opening_intent=True,
    )


def test_b320_fu015_contact_lock_is_a_pre_physics_joint_constraint() -> None:
    shared_step_source = inspect.getsource(
        HandsOnFU015IsaacLabEnv.__mro__[1]._step_isaaclab
    )
    assert shared_step_source.index("_before_task_physics_substep") < (
        shared_step_source.index("_sim.step")
    )

    lock_source = inspect.getsource(HandsOnFU015IsaacLabEnv._set_hinge_contact_lock)
    assert "set_dof_limits" in lock_source
    assert "set_dof_positions" not in lock_source
    assert "limits[:, 0, 0]" in lock_source
    assert "limits[:, 0, 1] = hold_angle_rad" in lock_source
    assert "_set_hinge_drive_target(desired_hold_deg)" in lock_source

    pre_step_source = inspect.getsource(
        HandsOnFU015IsaacLabEnv._before_task_physics_substep
    )
    assert "_hinge_motion_allowed" in pre_step_source
    assert "_set_hinge_contact_lock" in pre_step_source
    assert 'get("right_display_contact", False)' in pre_step_source


def test_b320_fu015_contact_lock_holds_live_angle_and_clears_spring() -> None:
    class FakeTensor:
        def __init__(self, values) -> None:
            self.values = np.asarray(values, dtype=np.float64)

        @property
        def ndim(self) -> int:
            return self.values.ndim

        @property
        def shape(self) -> tuple[int, ...]:
            return self.values.shape

        def clone(self):
            return FakeTensor(self.values.copy())

        def cpu(self):
            return self

        def __setitem__(self, key, value) -> None:
            self.values[key] = value

    class FakeView:
        count = 1

        def __init__(self) -> None:
            self.last_limits = np.zeros((1, 1, 2), dtype=np.float64)

        def get_dof_limits(self):
            return FakeTensor([[[-math.pi, 0.0]]])

        def set_dof_limits(self, limits, _indices) -> None:
            self.last_limits = limits.values.copy()

    fake_view = FakeView()
    drive_targets: list[float] = []
    fake_env = SimpleNamespace(
        _fu015_hinge_lock_angle_deg=0.0,
        _fu015_hinge_contact_lock_active=False,
        _fu015_hinge_limit_signature=None,
        _fu015_articulation_view=fake_view,
        _fu015_drive_target_deg=-30.0,
        _set_hinge_drive_target=drive_targets.append,
        _torch=SimpleNamespace(
            long=np.int64,
            arange=lambda count, **_kwargs: np.arange(count, dtype=np.int64),
        ),
    )

    HandsOnFU015IsaacLabEnv._set_hinge_contact_lock(
        fake_env,
        True,
        hold_angle_deg=-12.5,
    )
    expected_hold_rad = math.radians(-12.5)
    assert fake_view.last_limits[0, 0, 0] == pytest.approx(expected_hold_rad)
    assert fake_view.last_limits[0, 0, 1] == pytest.approx(expected_hold_rad)
    assert drive_targets == [-12.5]
    assert fake_env._fu015_hinge_contact_lock_active

    HandsOnFU015IsaacLabEnv._set_hinge_contact_lock(fake_env, False)
    assert fake_view.last_limits[0, 0, 0] == pytest.approx(
        math.radians(-HINGE_TRAVEL_DEG)
    )
    assert fake_view.last_limits[0, 0, 1] == pytest.approx(0.0)
    assert not fake_env._fu015_hinge_contact_lock_active


def test_b320_fu015_seeded_reset_is_repeatable() -> None:
    """Reset/random coverage: a seed must reproduce the hidden task state."""
    assert sample_reset_state(15015) == sample_reset_state(15015)
    assert sample_reset_state(15015) != sample_reset_state(15016)


def test_b320_fu015_places_both_arms_inside_randomized_workspace() -> None:
    right_cfg = SimpleNamespace(
        init_state=SimpleNamespace(pos=(-0.75, -0.065, 1.34))
    )
    left_cfg = SimpleNamespace(
        init_state=SimpleNamespace(pos=(-0.75, 0.065, 1.34))
    )

    configured_right = _configure_fu015_robot_cfg(right_cfg, "right")
    configured_left = _configure_fu015_robot_cfg(left_cfg, "left")

    assert configured_right is right_cfg
    assert configured_left is left_cfg
    assert right_cfg.init_state.pos == (-0.75, -0.065, 1.34)
    assert left_cfg.init_state.pos == (-0.75, 0.065, 1.34)
    assert _RIGHT_ROBOT_BASE_X_OFFSET_M == 0.0
    assert _LEFT_ROBOT_BASE_X_OFFSET_M == 0.0
    np.testing.assert_allclose(
        RIGHT_HOME[:3], [-0.22150017, -0.23950003, 0.99750000]
    )
    np.testing.assert_allclose(
        LEFT_HOME,
        [-0.32149994, 0.23950008, 0.99749994, *_RIGHT_REACHABLE_QUAT_WXYZ],
    )


def test_b320_fu015_observation_and_action_policy_contract() -> None:
    """Observation/proprio/action/space coverage for the raw dual-arm policy."""
    policy = FU015ScriptedPolicy(sample_reset_state(15015))
    action = policy.action(180)
    assert action.shape == (ACTION_DIM,)
    assert np.all(np.isfinite(action))


def test_b320_fu015_transit_keeps_wrists_high_and_fingers_preshaped() -> None:
    policy = FU015ScriptedPolicy(sample_reset_state(15015))
    start = policy.action(0)
    transit_high = policy.action(44)
    transit_end = policy.action(59)
    assert transit_end[9] >= start[9]
    assert np.linalg.norm(start[14:]) > 0.0
    assert np.allclose(
        start[14 : 14 + HAND_ACTION_DIM],
        transit_end[14 : 14 + HAND_ACTION_DIM],
    )
    assert np.linalg.norm(transit_end[14 + HAND_ACTION_DIM :]) > np.linalg.norm(
        start[14 + HAND_ACTION_DIM :]
    )
    assert transit_end[2] > start[2]
    assert np.isclose(transit_high[2], transit_end[2])
    assert np.isclose(transit_high[9], transit_end[9])
    assert transit_high[9] > start[9]
    assert (
        abs(float(np.dot(transit_high[10:14], _RIGHT_REACHABLE_QUAT_WXYZ)))
        > 0.999
    )
    assert (
        abs(float(np.dot(transit_end[10:14], _RIGHT_REACHABLE_QUAT_WXYZ)))
        < 0.8
    )
    assert np.allclose(start[3:7], _RIGHT_REACHABLE_QUAT_WXYZ)
    assert abs(float(np.dot(transit_end[3:7], _RIGHT_REACHABLE_QUAT_WXYZ))) > 0.999


def test_b320_fu015_policy_quaternions_remain_normalized() -> None:
    policy = FU015ScriptedPolicy(sample_reset_state(15015))
    for step in range(0, 451, 5):
        action = policy.action(step)
        assert np.isclose(np.linalg.norm(action[3:7]), 1.0)
        assert np.isclose(np.linalg.norm(action[10:14]), 1.0)


def test_b320_fu015_right_tcp_tracks_the_audited_hinge_arc() -> None:
    reset = FU015ResetState(
        seed=15015,
        root_x_m=-0.08,
        root_y_m=0.0,
        root_yaw_deg=-90.0,
        hinge_angle_deg=-18.011459850775623,
        friction=0.6,
        mass_scale=1.0,
        target_open_deg=120.0,
    )
    start = _right_tcp_pose_for_opening(reset.hinge_angle_deg, reset)
    midpoint = _right_tcp_hinge_arc_pose(reset, 0.5)
    end = _right_tcp_pose_for_opening(_RIGHT_COMMAND_OPEN_DEG, reset)

    assert np.allclose(start[[0, 2]], _RIGHT_REFERENCE_TCP_XZ_M)
    for pose in (start, midpoint, end):
        assert np.isclose(
            np.linalg.norm(pose[[0, 2]] - _RIGHT_TCP_ARC_CENTER_XZ_M),
            _RIGHT_DISPLAY_TRACK_RADIUS_M,
        )

    chord_midpoint = 0.5 * (start[[0, 2]] + end[[0, 2]])
    assert midpoint[2] > chord_midpoint[1] + 0.02
    assert midpoint[0] < chord_midpoint[0] - 0.01
    assert np.allclose(
        FU015ScriptedPolicy(reset).action(480)[:7],
        end,
    )


def test_b320_fu015_shallow_right_contact_keeps_attachment_clear() -> None:
    reset = sample_reset_state(15506)
    audited_contact = _right_tcp_pose_for_opening(
        reset.hinge_angle_deg, reset
    )
    policy_contact = FU015ScriptedPolicy(reset).action(269)[:7]
    lift_m = _right_shallow_clearance_lift_m(reset.hinge_angle_deg)

    expected = audited_contact.copy()
    expected[2] += lift_m
    expected = _right_shallow_attachment_tilt_pose(
        expected, reset.hinge_angle_deg, reset
    )

    assert lift_m > 0.006
    assert np.allclose(policy_contact, expected)
    assert not np.allclose(policy_contact[3:7], audited_contact[3:7])
    assert np.linalg.norm(policy_contact[3:7]) == pytest.approx(1.0)
    assert _right_shallow_clearance_lift_m(22.0) == 0.0
    assert _right_shallow_clearance_lift_m(11.0) == pytest.approx(
        _RIGHT_SHALLOW_CLEARANCE_MAX_M / 2
    )
    unmodified = _right_tcp_pose_for_opening(22.0, reset)
    assert np.allclose(
        _right_shallow_attachment_tilt_pose(unmodified, 22.0, reset),
        unmodified,
    )

    measured_deg = abs(reset.hinge_angle_deg)
    readback = _successful_readback(
        hinge_angle_rad=math.radians(-measured_deg)
    )
    expected_opening_deg = measured_deg + _RIGHT_SHALLOW_LIVE_LEAD_MAX_DEG
    expected = _right_tcp_pose_for_opening(expected_opening_deg, reset)
    expected[2] += _right_shallow_clearance_lift_m(expected_opening_deg)
    expected = _right_shallow_attachment_tilt_pose(
        expected, expected_opening_deg, reset
    )
    assert np.allclose(
        FU015ScriptedPolicy(reset).action(360, readback=readback)[:7],
        expected,
    )


def test_b320_fu015_shallow_left_support_unloads_table_edge() -> None:
    shallow_reset = sample_reset_state(15506)
    ordinary_reset = replace(shallow_reset, hinge_angle_deg=-30.0)
    restored_reset = replace(shallow_reset, hinge_angle_deg=-16.46)

    shallow_pose = FU015ScriptedPolicy(shallow_reset).action(269)[7:14]
    ordinary_pose = FU015ScriptedPolicy(ordinary_reset).action(269)[7:14]
    restored_pose = FU015ScriptedPolicy(restored_reset).action(269)[7:14]

    assert shallow_pose[2] > ordinary_pose[2] + 0.0025
    assert not np.allclose(shallow_pose[3:7], ordinary_pose[3:7])
    assert np.linalg.norm(shallow_pose[3:7]) == pytest.approx(1.0)
    assert np.allclose(restored_pose, ordinary_pose)


def test_b320_fu015_right_arc_adapts_to_randomized_initial_hinge() -> None:
    first = sample_reset_state(15015)
    second = sample_reset_state(15016)
    first_pose = _right_tcp_pose_for_opening(first.hinge_angle_deg, first)
    second_pose = _right_tcp_pose_for_opening(second.hinge_angle_deg, second)

    assert not np.allclose(first_pose[:3], second_pose[:3])
    assert np.allclose(
        _NOMINAL_HINGE_PIVOT_XZ_M
        + _display_edge_radial_xz(TARGET_OPEN_DEG),
        [0.1048212, 1.0927043],
        atol=1.0e-9,
    )
    for reset, audited_pose in ((first, first_pose), (second, second_pose)):
        expected = audited_pose.copy()
        expected[2] += _right_shallow_clearance_lift_m(
            reset.hinge_angle_deg
        )
        expected = _right_shallow_attachment_tilt_pose(
            expected, reset.hinge_angle_deg, reset
        )
        assert np.allclose(
            FU015ScriptedPolicy(reset).action(269)[:7], expected
        )


def test_b320_fu015_right_arc_tracks_live_hinge_readback() -> None:
    reset = sample_reset_state(15302)
    hinge_angle_deg = -27.5
    readback = _successful_readback(
        hinge_angle_rad=math.radians(hinge_angle_deg)
    )
    expected = _right_tcp_pose_for_opening(hinge_angle_deg, reset)

    assert np.allclose(
        FU015ScriptedPolicy(reset).action(360, readback=readback)[:7],
        expected,
    )
    # Continue following a lagging physical display after the nominal opening
    # horizon instead of jumping to the scripted endpoint.
    assert np.allclose(
        FU015ScriptedPolicy(reset).action(495, readback=readback)[:7],
        expected,
    )
    assert not np.allclose(
        FU015ScriptedPolicy(reset).action(360)[:7],
        expected,
    )

    # The measured successful replay needs a bounded geometric lead only near
    # the high-angle static-brace region.
    high_angle_readback = _successful_readback(
        hinge_angle_rad=math.radians(-90.0)
    )
    high_angle_expected = _right_tcp_pose_for_opening(102.0, reset)
    assert np.allclose(
        FU015ScriptedPolicy(reset).action(
            449, readback=high_angle_readback
        )[:7],
        high_angle_expected,
    )
    ramp_readback = _successful_readback(
        hinge_angle_rad=math.radians(-81.0)
    )
    assert np.allclose(
        FU015ScriptedPolicy(reset).action(
            420, readback=ramp_readback
        )[:7],
        _right_tcp_hinge_arc_pose(replace(reset, target_open_deg=120.0), (420 - 300) / 179.0),
    )


def test_b320_fu015_right_edge_grip_tightens_only_after_contact() -> None:
    policy = FU015ScriptedPolicy(sample_reset_state(15015))
    hand = slice(14, 14 + HAND_ACTION_DIM)
    approach = policy.action(269)[hand]
    contact = policy.action(270)[hand]
    settled = policy.action(299)[hand]
    opening = policy.action(300)[hand]
    tightened = policy.action(345)[hand]
    held = policy.action(495)[hand]

    assert np.allclose(approach, WUJI_POWER_GRIP * 0.25)
    assert np.allclose(contact, approach)
    assert np.allclose(settled, approach)
    assert np.allclose(opening, approach)
    changed = np.flatnonzero(~np.isclose(tightened, approach))
    assert np.array_equal(changed, [14, 15, 18, 19])
    assert np.all(tightened[changed] > approach[changed])
    assert np.allclose(held, tightened)
    assert np.max(tightened - approach) < 0.1


def test_b320_fu015_tucks_only_left_finger4_during_opening() -> None:
    policy = FU015ScriptedPolicy(sample_reset_state(15302))
    left_hand = slice(14 + HAND_ACTION_DIM, ACTION_DIM)
    approach = policy.action(269)[left_hand]
    contact = policy.action(270)[left_hand]
    tuck_mid = policy.action(279)[left_hand]
    settled = policy.action(289)[left_hand]
    opening = policy.action(420)[left_hand]

    assert np.allclose(approach, contact)
    assert np.allclose(approach[12:16], [1.40, 0.00, 1.35, 1.35])
    assert np.allclose(approach[16:20], [1.48, 0.00, 1.49, 1.50])
    assert np.allclose(settled[:12], approach[:12])
    assert np.allclose(settled[16:20], approach[16:20])
    assert np.allclose(settled[12:16], [1.46, 0.00, 1.39, 1.40])
    assert np.all(tuck_mid[12:16] >= approach[12:16])
    assert np.all(tuck_mid[12:16] <= settled[12:16])
    assert np.allclose(opening, settled)


def test_b320_fu015_assist_ramp_fits_task_timeout() -> None:
    physics_substeps_per_action = 4
    opening_action_steps = TIMEOUT_STEPS - SUCCESS_HOLD_STEPS - 300
    degrees_per_action = (
        _ASSIST_DEGREES_PER_PHYSICS_STEP * physics_substeps_per_action
    )
    assert degrees_per_action <= 0.40
    assert degrees_per_action * opening_action_steps >= TARGET_OPEN_DEG - 11.0
    assert 18.0 < _HINGE_DRIVE_STIFFNESS <= 36.0
    assert 3.0 < _HINGE_DRIVE_MAX_FORCE <= 4.0


def test_b320_fu015_left_contact_feedback_holds_and_relieves_force() -> None:
    reset = sample_reset_state(15301)
    policy = FU015ScriptedPolicy(reset)
    landing = policy.action(194)
    loaded = _successful_readback(left_base_contact_force_n=25.0)
    relieved = policy.action(195, readback=loaded)

    relief_delta = relieved[7:10] - landing[7:10]
    yaw_delta = math.radians(reset.root_yaw_deg - NOMINAL_ROOT_YAW_DEG)
    outward_xy = np.array([-math.cos(yaw_delta), -math.sin(yaw_delta)])
    assert np.allclose(relief_delta[:2], 0.001 * outward_xy)
    assert np.allclose(relieved[10:14], landing[10:14])
    assert relief_delta[2] == pytest.approx(0.001)

    # The live 8 N overload requests 3 mm total relief and reaches it through
    # the bounded 1 mm-per-control-step slew rate.
    moderately_loaded = _successful_readback(left_base_contact_force_n=8.0)
    relieved_twice = policy.action(196, readback=moderately_loaded)
    relieved_thrice = policy.action(197, readback=moderately_loaded)
    assert np.allclose(relieved_twice[7:9] - landing[7:9], 0.002 * outward_xy)
    assert np.allclose(relieved_thrice[7:9] - landing[7:9], 0.003 * outward_xy)
    assert relieved_twice[9] - landing[9] == pytest.approx(0.002)
    assert relieved_thrice[9] - landing[9] == pytest.approx(0.003)

    marginal = _successful_readback(left_base_contact_force_n=0.5)
    marginally_reengaged = policy.action(198, readback=marginal)
    assert marginally_reengaged[9] - landing[9] == pytest.approx(0.0028)

    unloaded = _successful_readback(left_base_contact_force_n=0.0)
    reengaged = policy.action(199, readback=unloaded)
    reengaged_delta = reengaged[7:10] - landing[7:10]
    assert np.allclose(reengaged_delta[:2], 0.005 * outward_xy)
    assert 0.0 < reengaged_delta[2] < relieved_thrice[9] - landing[9]


def test_b320_fu015_left_control_reacts_before_contact_enters_role_band() -> None:
    reset = sample_reset_state(15501)
    policy = FU015ScriptedPolicy(reset)
    landing = policy.action(194)
    interior_contact = _successful_readback(
        left_base_contact_force_n=0.0,
        left_any_base_contact_force_n=29.7,
    )

    # A base contact outside the front-edge role band must not count as task
    # success, but it must stop the controller's blind downward press.
    assert not evaluate_physical_readback(interior_contact)["left_base_contact"]
    reacted = policy.action(195, readback=interior_contact)
    delta = reacted[7:10] - landing[7:10]
    yaw_delta = math.radians(reset.root_yaw_deg - NOMINAL_ROOT_YAW_DEG)
    outward_xy = np.array([-math.cos(yaw_delta), -math.sin(yaw_delta)])
    assert np.allclose(delta[:2], 0.001 * outward_xy)
    assert delta[2] == pytest.approx(0.001)

def test_b320_fu015_left_contact_retreat_tracks_reset_yaw() -> None:
    reset = sample_reset_state(15301)
    reset = FU015ResetState(
        seed=reset.seed,
        root_x_m=reset.root_x_m,
        root_y_m=reset.root_y_m,
        root_yaw_deg=NOMINAL_ROOT_YAW_DEG + 5.0,
        hinge_angle_deg=reset.hinge_angle_deg,
        friction=reset.friction,
        mass_scale=reset.mass_scale,
        target_open_deg=reset.target_open_deg,
    )
    policy = FU015ScriptedPolicy(reset)
    landing = policy.action(194)
    loaded = _successful_readback(left_base_contact_force_n=8.0)
    retreated = policy.action(195, readback=loaded)
    delta = retreated[7:9] - landing[7:9]
    angle = math.radians(5.0)
    assert delta[0] == pytest.approx(-0.001 * math.cos(angle))
    assert delta[1] == pytest.approx(-0.001 * math.sin(angle))


def test_b320_fu015_left_stabilizer_clears_display_before_opening() -> None:
    reset = sample_reset_state(15302)
    policy = FU015ScriptedPolicy(reset)
    landing = policy.action(194)
    loaded = _successful_readback(left_base_contact_force_n=8.0)
    for step in range(195, 204):
        policy.action(step, readback=loaded)

    yaw_delta = math.radians(reset.root_yaw_deg - NOMINAL_ROOT_YAW_DEG)
    outward_xy = np.array([-math.cos(yaw_delta), -math.sin(yaw_delta)])
    settled = policy.action(269, readback=loaded)
    opening_clearance = None
    for step in range(270, 275):
        opening_clearance = policy.action(step, readback=loaded)

    assert np.allclose(settled[7:9] - landing[7:9], 0.009 * outward_xy)
    assert opening_clearance is not None
    assert np.allclose(
        opening_clearance[7:9] - landing[7:9],
        0.009 * outward_xy,
    )


def test_b320_fu015_front_proxy_contact_uses_body_local_coordinates() -> None:
    yaw = math.radians(-92.0)
    pose = np.asarray(
        [
            0.044,
            -0.022,
            0.896,
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        ]
    )
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    local_points = np.asarray(
        [
            [-0.1006, -0.00195, 0.00420],
            [-0.1006, 0.05000, 0.00420],
        ]
    )
    world_points = pose[:3] + local_points @ rotation.T

    mask = points_in_body_box(
        world_points,
        pose,
        _BASE_FRONT_PROXY_LOWER_M,
        _BASE_FRONT_PROXY_UPPER_M,
        margin_m=0.001,
    )
    assert np.array_equal(mask, [True, False])


def test_b320_fu015_success_terminal_is_contact_gated_and_stable() -> None:
    """Success/succeed/terminated/truncated semantics require a stable hold."""
    tracker = FU015SuccessTracker()
    successful = _successful_readback()
    assert evaluate_physical_readback(successful)["success_now"] is True
    for _ in range(SUCCESS_HOLD_STEPS - 1):
        assert tracker.update(successful) == (False, False)
    assert tracker.update(successful) == (True, False)

    forbidden = _successful_readback(
        left_touchpad_contact_force_n=FORBIDDEN_CONTACT_FORCE_N + 0.01
    )
    # Wrong-hand contacts remain diagnostics for the published fixed-base task.
    assert evaluate_physical_readback(forbidden)["forbidden_contact"] is True
    assert evaluate_physical_readback(forbidden)["failure"] is False
    for bad in (_successful_readback(right_display_contact_force_n=0),
                _successful_readback(hinge_angle_rad=math.radians(-30)),
                _successful_readback(display_linear_speed_mps=0.1)):
        assert not evaluate_physical_readback(bad)["success_now"]
    assert evaluate_physical_readback(_successful_readback(hinge_angle_rad=float("nan")))["failure"]


def test_b320_fu015_robot_table_diagnostic_preserves_all_filter_forces() -> None:
    paths = ("/left/base", "/left/palm", "/right/finger")
    summary = summarize_contact_forces_by_filter(
        paths,
        np.asarray([1, 2, 1], dtype=np.int64),
        np.asarray([10.0, 0.25, 8.5], dtype=np.float64),
    )
    assert summary == {"/left/palm": 18.5, "/right/finger": 0.25}
    assert sum(summary.values()) == 18.75


def test_b320_fu015_robot_table_diagnostic_preserves_bounded_contact_points() -> None:
    paths = ("/left/base", "/left/palm", "/right/finger")
    points = summarize_contact_points_by_filter(
        paths,
        np.asarray([1, 2, 1], dtype=np.int64),
        np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
        np.asarray([10.0, 0.0, 8.5], dtype=np.float64),
        max_points_per_path=1,
    )
    assert points == {"/left/palm": [[0.1, 0.2, 0.3]]}


def test_b320_fu015_diagnostic_attributes_zero_force_manifold_points() -> None:
    paths = ("/left/finger2", "/left/finger5")
    points = summarize_contact_manifold_points_by_filter(
        paths,
        np.asarray([1, 0, 1], dtype=np.int64),
        np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
        max_points_per_path=1,
    )
    assert points == {
        "/left/finger5": [[0.1, 0.2, 0.3]],
        "/left/finger2": [[0.4, 0.5, 0.6]],
    }


def test_b320_fu015_composed_root_pose_preserves_asset_body_offset() -> None:
    reference_reset = FU015ResetState(
        seed=1,
        root_x_m=-0.0062,
        root_y_m=-0.0057,
        root_yaw_deg=-92.0,
        hinge_angle_deg=-20.0,
        friction=0.6,
        mass_scale=1.0,
        target_open_deg=90.0,
    )
    target_reset = FU015ResetState(
        seed=2,
        root_x_m=0.031,
        root_y_m=0.014,
        root_yaw_deg=-86.0,
        hinge_angle_deg=-25.0,
        friction=0.7,
        mass_scale=1.1,
        target_open_deg=90.0,
    )
    body_offset = np.asarray([0.0000559, -0.1408278])

    def rotate(yaw_deg: float, vector: np.ndarray) -> np.ndarray:
        yaw = math.radians(yaw_deg)
        return np.asarray(
            [
                math.cos(yaw) * vector[0] - math.sin(yaw) * vector[1],
                math.sin(yaw) * vector[0] + math.cos(yaw) * vector[1],
            ]
        )

    reference_pose = np.asarray(
        [
            *(
                np.asarray([reference_reset.root_x_m, reference_reset.root_y_m])
                + rotate(reference_reset.root_yaw_deg, body_offset)
            ),
            0.8973279,
            0.0,
            0.0,
            math.sin(math.radians(reference_reset.root_yaw_deg) / 2.0),
            math.cos(math.radians(reference_reset.root_yaw_deg) / 2.0),
        ],
        dtype=np.float64,
    )

    identity = _transform_composed_root_pose(
        reference_pose, reference_reset, reference_reset
    )
    transformed = _transform_composed_root_pose(
        reference_pose, reference_reset, target_reset
    )
    expected_xy = np.asarray([target_reset.root_x_m, target_reset.root_y_m]) + rotate(
        target_reset.root_yaw_deg, body_offset
    )

    assert np.array_equal(identity, reference_pose)
    assert np.allclose(transformed[:2], expected_xy)
    assert transformed[2] == reference_pose[2]
    assert np.isclose(np.linalg.norm(transformed[3:7]), 1.0)
    assert np.allclose(
        transformed[3:7],
        [
            0.0,
            0.0,
            math.sin(math.radians(target_reset.root_yaw_deg) / 2.0),
            math.cos(math.radians(target_reset.root_yaw_deg) / 2.0),
        ],
    )


def test_b320_fu015_front_contact_proxy_matches_body_local_front_shell_band() -> None:
    lower = np.asarray(_BASE_FRONT_PROXY_LOWER_M, dtype=np.float64)
    upper = np.asarray(_BASE_FRONT_PROXY_UPPER_M, dtype=np.float64)
    center = np.asarray(_BASE_FRONT_PROXY_CENTER_M, dtype=np.float64)
    size = np.asarray(_BASE_FRONT_PROXY_SIZE_M, dtype=np.float64)

    assert np.all(size > 0.0)
    assert np.allclose(center - 0.5 * size, lower)
    assert np.allclose(center + 0.5 * size, upper)
    assert np.isclose(size[1], 0.045)
    assert np.isclose(size[2], 0.0065)
    assert np.allclose(lower, [-0.1641000062, -0.0020000001, -0.0023000001])
    assert np.allclose(upper, [0.1641000062, 0.0429999999, 0.0041999999])
