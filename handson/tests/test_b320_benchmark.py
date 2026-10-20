from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from handson.runtime.collection_support import (
    ISAACSIM_COLLECTION_TASKS,
    ISAACSIM_COLLECTION_UNVERIFIED_TASKS,
    ISAACSIM_FORMAL_COLLECTION_TASKS,
)
from handson.sim.isaaclab import b320_bimanual_env as b320_env_module
from handson.sim.isaaclab import b320_bimanual_scene as b320_scene_module
from handson.sim.isaaclab import compat as isaaclab_compat
from handson.sim.isaaclab.abc_collection_env import (
    _BC004_CARTON_HAND_CONTACT_OFFSET,
    _BC004_CARTON_HAND_DYNAMIC_FRICTION,
    _BC004_CARTON_HAND_REST_OFFSET,
    _BC004_CARTON_HAND_STATIC_FRICTION,
    _BC021_TABLETOP_ARM_HOME,
    _BC021_TABLETOP_LEFT_START_TCP,
    _BC021_TABLETOP_RIGHT_START_TCP,
    _B320_BLOCK_MATING_START_TASKS,
    _B320_COLLISION_CLEAR_START_TASKS,
    _B320_SPREAD_START_TASKS,
    _C_TOOL_SPREAD_ARM_HOME,
    _C_TOOL_SPREAD_LEFT_START_TCP,
    _C_TOOL_SPREAD_RIGHT_START_TCP,
    _KA012_CLEAR_ARM_HOME,
    _KA012_CLEAR_LEFT_START_TCP,
    _KA012_CLEAR_RIGHT_START_TCP,
    _KA012_TASK_NAME,
    _KA021_CLEAR_ARM_HOME,
    _KA021_CLEAR_LEFT_START_TCP,
    _KA021_CLEAR_RIGHT_START_TCP,
    _KA021_TASK_NAME,
    _TASK_TABLE_HEIGHT_BOUNDS,
    _TASK_SPECS,
    _disable_physx_gravity_for_robot_links,
    _physx_gpu_buffer_kwargs,
    supported_native_collection_tasks,
)
from handson.sim.envs.rokae_wuji_ik_replay import rokae as rokae08
from handson.sim.isaaclab.b320_bimanual_env import (
    HandsOnB320BimanualIsaacLabEnv,
)
from handson.sim.isaaclab.b320_bc017_scene import (
    BC017_DOF_NAMES,
    BC017_DRAWER_ROLES_BY_SIDE,
)
from handson.sim.isaaclab.b320_bc021_env import (
    HandsOnB320BC021IsaacLabEnv,
)
from handson.sim.isaaclab.b320_bimanual_scene import (
    BC011_PEN_SCALE,
    BC011_PEN_TABLE_CLEARANCE_M,
    BC011_PEN_TIP_AXIS_LOCAL,
    BC011_PEN_VARIANTS,
    BC011_SOURCE_PEN_LENGTH_M,
    BC011_TARGET_PEN_LENGTH_M,
    BC011_WUJI1_INDEX_REFERENCE_LENGTH_M,
    BC014_ASSET_ARCHIVE_SHA256,
    BC014_ASSET_BUNDLE_KEY,
    BC014_ASSET_LICENSE,
    BC014_ASSET_SOURCE_URL,
    BC014_FLOW_PROXY_PATHS,
    BC014_GOAL_CENTERS_XY,
    BC014_GOAL_SPACING_M,
    BC014_GOAL_ZONE_SIZE_M,
    BC014_JAR_TABLE_CLEARANCE_M,
    BC014_JAR_TARGET_HEIGHT_M,
    B320_BC014_TASK_NAME,
    B320_BC017_TASK_NAME,
    B320_BC004_SCENE_XML,
    B320_CONTROL_HZ,
    B320_RIGID_TASK_NAMES,
    B320_SOURCE_BUNDLE_SHA256,
    B320_TASK_NAMES,
    B320_TASK_SCENES,
    B320_TASK_SPECS,
    B320_TOWEL_TASK_NAME,
    validate_b320_asset_files,
)
from handson.sim.isaaclab.bimanual_assembly_env import (
    _ARM_WORKSPACE_MAX_RADIUS,
)
from handson.sim.isaaclab.realistic_kitchen_scene import (
    REALISTIC_KITCHEN_ASSET_RELATIVE_PATH,
    REALISTIC_KITCHEN_ASSET_SHA256,
    REALISTIC_KITCHEN_ISLAND_MAX_M,
    REALISTIC_KITCHEN_ISLAND_MIN_M,
    REALISTIC_KITCHEN_LICENSE,
    RealisticKitchenReport,
    compose_realistic_kitchen_visual,
    realistic_kitchen_metadata,
)
from handson.sim.isaaclab.b320_towel_env import HandsOnB320TowelIsaacLabEnv
from handson.sim.isaaclab.b320_towel_env import (
    _detect_bc025_surface_contact_response,
)
from handson.sim.isaaclab.b320_towel_scene import (
    BC025_TOWEL_NODE_COUNT,
    BC025_TOWEL_TRANSLATION_BOUNDS_M,
    BC025_TOWEL_TRIANGLE_COUNT,
    B320_TOWEL_PLACEMENT,
    b320_towel_surface_metadata,
)
from handson.sim.isaaclab.b320_bc025_assets import (
    BC025_COLLISION_SOURCE_SHA256,
    BC025_MATERIAL_ARCHIVE_SHA256,
    resolve_bc025_assets,
)
from handson.tasks.mappings import CONFIG_MAPPING


EXPECTED_TASKS = {
    "t10_lift_carton": ("10", 22),
    "t20_pass_and_orient_pen": ("20", 32),
    "t30_pour_seasonings": ("30", 16),
    B320_BC017_TASK_NAME: ("15", 30),
    "t01_align_blocks_in_midair": ("01", 60),
    "b320_towel_corner_synchronized_spread": ("B320-BC025", 45),
    "t45_stir_pot": ("45", 40),
    "t32_press_soap_pump": ("32", 90),
    "t05_fill_cup": ("05", 22),
    "b320_synchronized_double_cabinet_open": ("B320-FU001", 32),
    "t38_set_dual_knobs": ("38", 40),
    "eval_comb05_scrub_bottle_then_fill_cup": ("XY4", 90),
    "eval_comb06_season_and_stir": ("XY6", 90),
    "eval_gen05_set_mixer_knobs": ("GEN6", 40),
}
INTEGRATED_RIGID_TASK_NAMES = B320_RIGID_TASK_NAMES - {
    B320_BC017_TASK_NAME
}
BC017_REQUIREMENT = (
    "Use each assigned hand to pull the highlighted drawer on its side beyond "
    "60 percent travel; each side independently selects one of three drawers "
    "per episode, and both thresholds must be crossed within 1.2 seconds."
)


def _rigid_env(task_name: str) -> HandsOnB320BimanualIsaacLabEnv:
    if task_name in CONFIG_MAPPING:
        return CONFIG_MAPPING[task_name]().get_native_environment(
            policy_mode=True, launch_isaac=False, image_obs=False, randomize=False
        ).unwrapped
    env_class = (
        HandsOnB320BC021IsaacLabEnv
        if task_name == "t01_align_blocks_in_midair"
        else HandsOnB320BimanualIsaacLabEnv
    )
    return env_class(
        task_name,
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )


def _pen_tip_up_pose(
    error_degrees: float = 0.0,
    *,
    tip_axis_x: float = -1.0,
) -> np.ndarray:
    """Return a pose whose selected authored pen-tip axis is near world +Z."""

    rotation_y = np.deg2rad(-90.0 * tip_axis_x + float(error_degrees))
    half_angle = 0.5 * rotation_y
    return np.asarray(
        [
            0.0,
            0.0,
            1.0,
            np.cos(half_angle),
            0.0,
            np.sin(half_angle),
            0.0,
        ],
        dtype=np.float64,
    )


def _prime_contacts(env: HandsOnB320BimanualIsaacLabEnv) -> None:
    env._b320_contact_available = {"left": True, "right": True}
    env._b320_contact_readable = {"left": True, "right": True}
    env._b320_contact_seen = {"left": True, "right": True}
    env._b320_contact_force = {"left": 2.0, "right": 2.0}
    env._b320_contact_peak = {"left": 2.0, "right": 2.0}
    env._b320_carton_contact_detail_available = {"left": True, "right": True}
    env._b320_carton_zone_valid = {"left": True, "right": True}
    env._b320_carton_table_contact_available = True
    env._b320_carton_table_contact_readable = True
    env._b320_carton_table_contact_force = 0.0
    env._b320_pen_table_contact_available = True
    env._b320_pen_table_contact_readable = True
    env._b320_pen_table_contact_force = 0.0
    env._b320_carton_transport_started = True


def _lift_carton(
    env: HandsOnB320BimanualIsaacLabEnv,
    *,
    clearance: float = 0.020,
    x: float = -0.30,
) -> None:
    env._poses["carton"] = np.asarray(
        [x, 0.0, 0.89 + clearance, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )


def _freeze_injected_contacts(env: HandsOnB320BimanualIsaacLabEnv) -> None:
    env._read_contacts = lambda *, update_latches: None


class _CartonContactView:
    sensor_count = 1
    filter_count = 1

    def __init__(self, force: float, points: list[tuple[float, float, float]]):
        self.force = float(force)
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 3)

    def get_contact_force_matrix(self, _dt: float) -> np.ndarray:
        return np.asarray([[[self.force, 0.0, 0.0]]], dtype=np.float64)

    def get_contact_data(self, _dt: float) -> tuple[np.ndarray, ...]:
        count = len(self.points)
        return (
            np.full(count, self.force / max(count, 1), dtype=np.float64),
            self.points.copy(),
            np.tile(np.asarray([[0.0, 1.0, 0.0]]), (count, 1)),
            np.zeros(count, dtype=np.float64),
            np.asarray([[count]], dtype=np.int64),
            np.asarray([[0]], dtype=np.int64),
        )


class _InvalidContactView:
    sensor_count = 1
    filter_count = 1

    def get_contact_force_matrix(self, _dt: float) -> np.ndarray:
        return np.asarray([[[np.nan, 0.0, 0.0]]], dtype=np.float64)


def _set_carton_contact_views(
    env: HandsOnB320BimanualIsaacLabEnv,
    *,
    left_force: float = 2.0,
    right_force: float = 2.0,
    left_y: float = 0.235,
    right_y: float = -0.235,
) -> None:
    carton = env._pose("carton")
    env._b320_contact_views = {
        "left": _CartonContactView(
            left_force,
            [(carton[0], carton[1] + left_y, carton[2] + 0.10)],
        ),
        "right": _CartonContactView(
            right_force,
            [(carton[0], carton[1] + right_y, carton[2] + 0.10)],
        ),
    }
    env._b320_carton_table_contact_view = _CartonContactView(0.0, [])


def test_all_rigid_b320_tasks_bypass_arm_target_limits_like_d08() -> None:
    env = _rigid_env("t10_lift_carton")
    assert env._arm_ik_backend == "dls"
    assert env._physics_backend == "physx"

    for task_name in B320_RIGID_TASK_NAMES:
        env.task_name = task_name
        assert env._physx_contact_arm_target_limits_enabled() is False

    env.task_name = "bimanual_assembly"
    assert env._physx_contact_arm_target_limits_enabled() is False

    env.task_name = "t21_pass_block"
    assert env._physx_contact_arm_target_limits_enabled() is True

    env._physics_backend = "newton"
    assert env._physx_contact_arm_target_limits_enabled() is False


def test_b320_preserves_the_d08_default_robot_follower_profile() -> None:
    env = _rigid_env("t10_lift_carton")
    source_stiffness = {
        "joint[1-4]_right(_1)?": 321.0,
        "joint[5-7]_right(_1)?": 123.0,
    }
    arm = SimpleNamespace(stiffness=source_stiffness.copy(), damping=17.0)
    articulation_props = SimpleNamespace(solver_velocity_iteration_count=4)
    robot_cfg = SimpleNamespace(
        actuators={"arm": arm},
        spawn=SimpleNamespace(articulation_props=articulation_props),
    )

    configured = env._configure_task_robot_cfg(robot_cfg, "right")

    assert configured is robot_cfg
    assert configured.actuators["arm"] is arm
    assert arm.stiffness == source_stiffness
    assert arm.damping == pytest.approx(17.0)
    assert articulation_props.solver_velocity_iteration_count == 4


def test_carton_uses_bounded_gpu_buffers_without_changing_contact_partitions() -> None:
    assert _physx_gpu_buffer_kwargs("t10_lift_carton") == {
        "gpu_max_rigid_contact_count": 2**18,
        "gpu_max_rigid_patch_count": 2**14,
        "gpu_found_lost_pairs_capacity": 2**16,
        "gpu_found_lost_aggregate_pairs_capacity": 2**16,
        "gpu_total_aggregate_pairs_capacity": 2**16,
        "gpu_collision_stack_size": 2**22,
        "gpu_heap_capacity": 2**22,
        "gpu_temp_buffer_capacity": 2**20,
        "gpu_max_num_partitions": 8,
        "gpu_max_particle_contacts": 2**16,
        "gpu_max_soft_body_contacts": 2**20,
    }


def test_carton_uses_collision_clear_spread_home_and_staging_tcp_targets() -> None:
    task_name = "t10_lift_carton"
    assert _B320_SPREAD_START_TASKS == {task_name}
    env = _rigid_env(task_name)
    env._sample_reset_state(clear_manual_request=True)

    np.testing.assert_allclose(env._right_tcp_pose, _C_TOOL_SPREAD_RIGHT_START_TCP)
    np.testing.assert_allclose(env._left_tcp_pose, _C_TOOL_SPREAD_LEFT_START_TCP)
    assert env._right_tcp_pose[1] == pytest.approx(-0.36)
    assert env._left_tcp_pose[1] == pytest.approx(0.36)

    for side in ("right", "left"):
        joint_pos = {
            f"joint{index}_{side}(_1)?": -99.0
            for index in range(1, 8)
        }
        robot_cfg = SimpleNamespace(
            init_state=SimpleNamespace(joint_pos=joint_pos),
        )
        assert env._apply_raised_hand_home(robot_cfg, side) is True
        np.testing.assert_allclose(
            [
                robot_cfg.init_state.joint_pos[f"joint{index}_{side}(_1)?"]
                for index in range(1, 8)
            ],
            _C_TOOL_SPREAD_ARM_HOME[side],
        )


@pytest.mark.usefixtures("rokae_assets")
def test_bc021_uses_parallel_tabletop_palms_from_a_dedicated_home() -> None:
    task_name = "t01_align_blocks_in_midair"
    assert _B320_BLOCK_MATING_START_TASKS == {task_name}
    env = _rigid_env(task_name)
    env._sample_reset_state(clear_manual_request=True)

    np.testing.assert_allclose(
        env._right_tcp_pose,
        _BC021_TABLETOP_RIGHT_START_TCP,
    )
    np.testing.assert_allclose(
        env._left_tcp_pose,
        _BC021_TABLETOP_LEFT_START_TCP,
    )
    np.testing.assert_allclose(
        env._right_tcp_pose[3:7],
        env._left_tcp_pose[3:7],
    )

    expected_palm_rotation = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    palm_rotations = {}
    for side in ("right", "left"):
        joint_pos = {
            f"joint{index}_{side}(_1)?": -99.0
            for index in range(1, 8)
        }
        robot_cfg = SimpleNamespace(
            init_state=SimpleNamespace(joint_pos=joint_pos),
        )
        assert env._apply_raised_hand_home(robot_cfg, side) is True
        actual_home = np.asarray(
            [
                robot_cfg.init_state.joint_pos[
                    f"joint{index}_{side}(_1)?"
                ]
                for index in range(1, 8)
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(actual_home, _BC021_TABLETOP_ARM_HOME[side])

        base_in_world = rokae08.base_in_world_matrix(side)
        base_in_world[:3, 3] = np.asarray(
            [-0.75, 0.065 if side == "left" else -0.065, 1.34],
            dtype=np.float64,
        )
        flange_in_world = base_in_world @ rokae08.load_model(side).fk(
            actual_home
        )
        mount_rotation = np.asarray(
            [
                [0.0, -1.0 if side == "left" else 1.0, 0.0],
                [1.0 if side == "left" else -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        palm_rotation = flange_in_world[:3, :3] @ mount_rotation
        palm_rotations[side] = palm_rotation
        np.testing.assert_allclose(
            flange_in_world[:3, 3],
            (
                _BC021_TABLETOP_LEFT_START_TCP[:3]
                if side == "left"
                else _BC021_TABLETOP_RIGHT_START_TCP[:3]
            ),
            atol=2.0e-5,
        )
        np.testing.assert_allclose(
            palm_rotation,
            expected_palm_rotation,
            atol=2.0e-4,
        )
        assert rokae08.wrist_limit_ratio(actual_home) < 0.60

    np.testing.assert_allclose(
        palm_rotations["left"],
        palm_rotations["right"],
        atol=2.0e-4,
    )


def test_bc014_uses_attributed_textured_assets_and_collision_clear_start() -> None:
    task_name = B320_BC014_TASK_NAME
    assert task_name in _B320_COLLISION_CLEAR_START_TASKS

    task_spec = B320_TASK_SPECS[task_name]
    specs = task_spec.object_specs
    assert [spec.name for spec in specs] == ["salt_jar", "pepper_jar"]
    assert [spec.relative_usd for spec in specs] == [
        f"{BC014_ASSET_BUNDLE_KEY}/salt_jar.usda",
        f"{BC014_ASSET_BUNDLE_KEY}/pepper_jar.usda",
    ]
    assert all(spec.source_path.is_file() for spec in specs)
    assert all(spec.scale == (1.0, 1.0, 1.0) for spec in specs)
    assert all(spec.expected_colliders == 2 for spec in specs)
    assert all(spec.contact_offset == pytest.approx(0.0015) for spec in specs)
    assert all(
        spec.home_translation[2]
        == pytest.approx(
            b320_scene_module.B320_TABLE_TOP_Z + BC014_JAR_TABLE_CLEARANCE_M
        )
        for spec in specs
    )
    assert task_spec.source_asset_keys == (
        BC014_ASSET_BUNDLE_KEY,
        "target_marker",
        "water_flow_proxy",
    )

    asset_dir = b320_scene_module.B320_ASSET_DIRECTORY / BC014_ASSET_BUNDLE_KEY
    conversion = json.loads((asset_dir / "conversion.json").read_text())
    assert conversion["source"]["archive_sha256"] == BC014_ASSET_ARCHIVE_SHA256
    assert conversion["source"]["source_url"] == BC014_ASSET_SOURCE_URL
    assert conversion["source"]["license"] == BC014_ASSET_LICENSE
    assert conversion["normalization"]["target_height_m"] == pytest.approx(
        BC014_JAR_TARGET_HEIGHT_M
    )
    assert conversion["collision"] == {
        "approximation": "convexHull",
        "colliders_per_jar": 2,
        "content_mesh_collision_enabled": False,
        "old_placeholder_collision_reused": False,
    }
    assert "Anthony Yanez" in (asset_dir / "LICENSE.txt").read_text()

    for relative_path, expected_hash in conversion["source"]["files"].items():
        source_file = asset_dir / relative_path
        assert source_file.is_file() and source_file.stat().st_size > 0
        assert hashlib.sha256(source_file.read_bytes()).hexdigest() == expected_hash

    for filename, metadata in conversion["outputs"].items():
        output_file = asset_dir / filename
        assert hashlib.sha256(output_file.read_bytes()).hexdigest() == metadata["sha256"]
        assert metadata["active_mesh_count"] == 3
        assert metadata["collider_count"] == 2
        bounds_min = np.asarray(metadata["bounds_min_m"], dtype=np.float64)
        bounds_max = np.asarray(metadata["bounds_max_m"], dtype=np.float64)
        dimensions = bounds_max - bounds_min
        assert dimensions[0] < BC014_GOAL_ZONE_SIZE_M[0]
        assert dimensions[1] < BC014_GOAL_ZONE_SIZE_M[1]
        assert bounds_min[2] == pytest.approx(0.0, abs=1.0e-8)
        assert dimensions[2] == pytest.approx(
            BC014_JAR_TARGET_HEIGHT_M, abs=2.0e-8
        )

    env = _rigid_env(task_name)
    env._sample_reset_state(clear_manual_request=True)
    np.testing.assert_allclose(env._right_tcp_pose, _C_TOOL_SPREAD_RIGHT_START_TCP)
    np.testing.assert_allclose(env._left_tcp_pose, _C_TOOL_SPREAD_LEFT_START_TCP)

    for side in ("right", "left"):
        joint_pos = {
            f"joint{index}_{side}(_1)?": -99.0
            for index in range(1, 8)
        }
        robot_cfg = SimpleNamespace(
            init_state=SimpleNamespace(joint_pos=joint_pos),
        )
        assert env._apply_raised_hand_home(robot_cfg, side) is True
        np.testing.assert_allclose(
            [
                robot_cfg.init_state.joint_pos[f"joint{index}_{side}(_1)?"]
                for index in range(1, 8)
            ],
            _C_TOOL_SPREAD_ARM_HOME[side],
        )


def test_ka012_uses_cc0_open_pot_with_separate_lid() -> None:
    task_spec = B320_TASK_SPECS["t45_stir_pot"]
    specs = {spec.name: spec for spec in task_spec.object_specs}

    assert tuple(specs) == ("dutch_oven", "dutch_oven_lid", "spoon")
    assert specs["dutch_oven"].relative_usd == (
        "polyhaven_pot_enamel_01/pot_base.usd"
    )
    assert specs["dutch_oven_lid"].relative_usd == (
        "polyhaven_pot_enamel_01/pot_lid.usd"
    )
    assert specs["dutch_oven"].source_path.is_file()
    assert specs["dutch_oven_lid"].source_path.is_file()
    assert specs["dutch_oven"].scale == pytest.approx((0.78, 0.78, 0.78))
    assert specs["dutch_oven_lid"].scale == pytest.approx((0.78, 0.78, 0.78))
    assert specs["dutch_oven"].mass_kg == pytest.approx(2.80)
    assert specs["dutch_oven_lid"].mass_kg == pytest.approx(0.55)
    assert "removable lid set beside" in task_spec.prompt
    assert "lid is already set beside" in CONFIG_MAPPING[task_spec.task_name].task_prompt
    assert task_spec.source_asset_keys == (
        "polyhaven_pot_enamel_01",
        "dexverse_authored_spoon",
    )

    asset_dir = (
        b320_scene_module.B320_ASSET_DIRECTORY / "polyhaven_pot_enamel_01"
    )
    manifest = json.loads((asset_dir / "asset_manifest.json").read_text())
    assert manifest["provenance"]["license"] == "CC0 1.0"
    assert manifest["provenance"]["source_url"] == (
        "https://polyhaven.com/a/pot_enamel_01"
    )
    assert manifest["source_structure"] == {
        "fully_modeled_interior": True,
        "independent_nodes": ["pot_enamel_01_base", "pot_enamel_01_lid"],
        "lid_removable": True,
        "source_up_axis": "Y",
    }
    components = {
        entry["path"]: entry for entry in manifest["runtime"]["components"]
    }
    assert components["pot_base.usd"]["collision_approximation"] == (
        "convexDecomposition"
    )
    assert components["pot_lid.usd"]["collision_approximation"] == "convexHull"


def test_ka012_uses_bilateral_home_above_open_pot_rim() -> None:
    env = _rigid_env(_KA012_TASK_NAME)
    env._sample_reset_state(clear_manual_request=True)

    assert _KA012_CLEAR_RIGHT_START_TCP[2] == pytest.approx(1.15)
    assert _KA012_CLEAR_LEFT_START_TCP[2] == pytest.approx(1.15)
    np.testing.assert_allclose(
        env._right_tcp_pose,
        _KA012_CLEAR_RIGHT_START_TCP,
    )
    np.testing.assert_allclose(
        env._left_tcp_pose,
        _KA012_CLEAR_LEFT_START_TCP,
    )

    # Current collection pot: z=0.891 m plus its 0.1315546 m source height at
    # the configured 0.78 scale. The FK-measured lowest open fingertip at this
    # home is z=1.059598 m, leaving more than 6 cm of initial clearance.
    pot_rim_z = 0.891 + 0.13155460357666016 * 0.78
    assert 1.059598 - pot_rim_z > 0.06

    for side in ("right", "left"):
        joint_pos = {
            f"joint{index}_{side}(_1)?": -99.0
            for index in range(1, 8)
        }
        robot_cfg = SimpleNamespace(
            init_state=SimpleNamespace(joint_pos=joint_pos),
        )
        assert env._apply_raised_hand_home(robot_cfg, side) is True
        np.testing.assert_allclose(
            [
                robot_cfg.init_state.joint_pos[f"joint{index}_{side}(_1)?"]
                for index in range(1, 8)
            ],
            _KA012_CLEAR_ARM_HOME[side],
        )


def test_ka021_reuses_hs005_realistic_sink_and_source_handle_geometry() -> None:
    task_spec = B320_TASK_SPECS["t05_fill_cup"]
    assert task_spec.fixture_kind == "ka021_hs005_realistic_faucet"
    assert task_spec.source_asset_keys == (
        "cc_ka021_hs005_kitchen_anilai_white_mug",
        b320_scene_module.KA021_FAUCET_ASSET_BUNDLE_KEY,
    )
    assert len(task_spec.object_specs) == 1
    mug = task_spec.object_specs[0]
    assert mug.name == "ceramic_mug"
    assert mug.relative_usd == (
        "cc_ka021_hs005_kitchen_anilai_white_mug/ceramic_white_mug.usd"
    )
    assert mug.home_translation == pytest.approx((-0.15, 0.02, 0.895))
    assert mug.home_orientation_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert mug.scale == pytest.approx((0.006, 0.006, 0.006))
    assert mug.linear_damping == pytest.approx(0.60)
    assert mug.angular_damping == pytest.approx(12.00)
    assert mug.sleep_threshold == pytest.approx(0.025)
    assert mug.source_path.is_file()

    asset_dir = (
        b320_scene_module.B320_ASSET_DIRECTORY
        / "cc_ka021_hs005_kitchen_anilai_white_mug"
    )
    manifest = json.loads((asset_dir / "asset_manifest.json").read_text())
    assert manifest["schema_version"] == 8
    assert manifest["license_summary"] == ["CC BY 4.0", "CC BY 4.0"]
    sources = {entry["role"]: entry for entry in manifest["sources"]}
    faucet = sources["kitchen_sink_and_faucet_visual"]
    assert faucet["creator"] == "Visthétique"
    assert faucet["shared_from_task"] == "B320-HS005"
    assert faucet["source_url"] == (
        "https://sketchfab.com/3d-models/"
        "modern-apartment-1fbb649cd6624f2bb7b7d6e30c6533a5"
    )
    assert faucet["license"] == {
        "name": "CC BY 4.0",
        "url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution_required": True,
    }
    assert sources["manipulated_cup"]["creator"] == "Ani 3D (anilai)"
    assert sources["manipulated_cup"]["source_url"] == (
        "https://sketchfab.com/3d-models/"
        "ceramic-white-mug-fdf735d1f8f6469d892a989b509cd245"
    )
    boundary = manifest["authorship_boundary"]
    assert boundary["source_faucet_articulated"] is False
    assert boundary["runtime_handle_visual_uses_source_geometry"] is True
    assert boundary["reuses_hs005_realistic_kitchen"] is True
    assert boundary["source_open_sink_collision_runtime_enabled"] is True
    assert boundary["runtime_safe_open_basin_floor"] is False
    assert (
        "fixed-base articulation and revolute joint for the source side lever"
        in boundary["handson_authored_runtime"]
    )
    assert "analytic mug body collider and recessed flat cube foot" in boundary[
        "handson_authored_runtime"
    ]
    assert "verification and reuse of the shared HS005 scene collision" in boundary[
        "handson_authored_runtime"
    ]
    assert manifest["hs005_scene"]["runtime_disabled_collision_prims"] == []
    safe_floor = manifest["runtime"]["safe_open_basin_floor"]
    assert safe_floor["path"] == b320_scene_module.KA021_SAFE_BASIN_FLOOR_PATH
    assert safe_floor["center_w_m"] == pytest.approx(
        b320_scene_module.KA021_SAFE_BASIN_FLOOR_CENTER_W
    )
    assert safe_floor["size_m"] == pytest.approx(
        b320_scene_module.KA021_SAFE_BASIN_FLOOR_SIZE_M
    )
    assert safe_floor["top_z_m"] == pytest.approx(
        b320_scene_module.B320_TABLE_TOP_Z + 0.005
    )
    assert safe_floor["top_z_m"] < 1.07
    spawn_source = inspect.getsource(
        b320_scene_module._spawn_ka021_hs005_faucet
    )
    assert "source_sink_collider" in spawn_source
    assert "KA021 shared basin floor is disabled" in spawn_source
    assert "safe_basin_floor_path = source_sink_collider_path" in spawn_source
    assert 'name="cup_park_marker"' in spawn_source
    assert "color=(0.95, 0.68, 0.08)" in spawn_source
    assert "post_fill_mug_placement" in spawn_source
    components = {
        entry["path"]: entry for entry in manifest["runtime"]["components"]
    }
    assert components["faucet_static.usd"]["mesh_count"] == 1
    assert components["faucet_handle.usd"]["mesh_count"] == 1
    split = manifest["runtime"]["source_mesh_split"]
    assert split["source_face_count"] == 10568
    assert split["static_face_count"] == 8664
    assert split["handle_face_count"] == 1904
    assert split["preserves_source_vertices"] is True
    assert split["preserves_source_uvs"] is True
    assert manifest["hs005_scene"]["sha256"] == (
        "ea706ecab4785bab4b7f7aa96a6c83b488602b1b47a8866674063f03e3679cba"
    )
    assert manifest["hs005_scene"]["deactivated_props"] == [
        "Bottle",
        "Brush",
        "DishSoap",
        "Sponge",
    ]
    mug_collision = components["ceramic_white_mug.usd"]["collision"]
    assert mug_collision["type"] == "cylinder_with_recessed_cube_foot"
    assert mug_collision["runtime_axis"] == "Z"
    assert mug_collision["runtime_radius_m"] == pytest.approx(0.0405)
    assert mug_collision["runtime_height_m"] == pytest.approx(0.0864)
    assert mug_collision["runtime_foot_xyz_m"] == pytest.approx(
        [0.054, 0.054, 0.006]
    )
    assert mug_collision["runtime_support_min_z_m"] == pytest.approx(0.0)
    assert (
        mug.home_translation[2] + mug_collision["runtime_support_min_z_m"]
    ) == pytest.approx(safe_floor["top_z_m"])
    coordinate_contract = components["ceramic_white_mug.usd"][
        "coordinate_contract"
    ]
    assert coordinate_contract["source_up_axis"] == "Y"
    assert coordinate_contract["runtime_up_axis"] == "Z"
    assert coordinate_contract["runtime_home_orientation_wxyz"] == pytest.approx(
        [1.0, 0.0, 0.0, 0.0]
    )
    assert coordinate_contract["runtime_visual_bottom_m"] > 0.0
    assert (
        mug.home_translation[2]
        + coordinate_contract["runtime_visual_bottom_m"]
    ) > safe_floor["top_z_m"]
    assert coordinate_contract["runtime_visual_rim_z_m"] == pytest.approx(
        b320_scene_module.KA021_CUP_RIM_HEIGHT_M,
        abs=5.0e-5,
    )
    assert manifest["runtime"]["mug_up_axis_local"] == pytest.approx(
        b320_scene_module.KA021_CUP_UP_AXIS_LOCAL
    )
    assert manifest["runtime"]["mug_home_orientation_wxyz"] == pytest.approx(
        [1.0, 0.0, 0.0, 0.0]
    )
    repo_root = Path(__file__).resolve().parents[2]
    for source in sources.values():
        source_file = (
            asset_dir / source["source_file"]
            if "source_file" in source
            else repo_root / "handson/handson/sim/envs/usd/hs005_sink" / source["source_file_repo_relative"].split("bench320_hs005_sink_bottle_stabilized_scrub/", 1)[1]
        )
        assert hashlib.sha256(source_file.read_bytes()).hexdigest() == source[
            "source_sha256"
        ]


def test_ka021_raises_only_hands_and_keeps_scene_height_randomization() -> None:
    env = _rigid_env(_KA021_TASK_NAME)
    env._sample_reset_state(clear_manual_request=True)

    assert _KA021_CLEAR_RIGHT_START_TCP[2] == pytest.approx(1.15)
    assert _KA021_CLEAR_LEFT_START_TCP[2] == pytest.approx(1.15)
    np.testing.assert_allclose(env._right_tcp_pose, _KA021_CLEAR_RIGHT_START_TCP)
    np.testing.assert_allclose(env._left_tcp_pose, _KA021_CLEAR_LEFT_START_TCP)

    for side in ("right", "left"):
        joint_pos = {
            f"joint{index}_{side}(_1)?": -99.0
            for index in range(1, 8)
        }
        robot_cfg = SimpleNamespace(
            init_state=SimpleNamespace(joint_pos=joint_pos),
        )
        assert env._apply_raised_hand_home(robot_cfg, side) is True
        np.testing.assert_allclose(
            [
                robot_cfg.init_state.joint_pos[f"joint{index}_{side}(_1)?"]
                for index in range(1, 8)
            ],
            _KA021_CLEAR_ARM_HOME[side],
        )

    assert _TASK_TABLE_HEIGHT_BOUNDS.get(
        _KA021_TASK_NAME,
        (0.0, 0.05),
    ) == (0.0, 0.05)
    mug = B320_TASK_SPECS[_KA021_TASK_NAME].object_specs[0]
    assert mug.home_translation[2] == pytest.approx(
        b320_scene_module.B320_TABLE_TOP_Z + 0.005
    )


def test_ka021_flow_requires_upright_aligned_mug_and_open_joint() -> None:
    success_base_z = (
        b320_scene_module.KA021_OUTLET_POSITION_W[2]
        - b320_scene_module.KA021_CUP_RIM_HEIGHT_M
        - 0.12
    )
    upright = np.asarray(
        [
            *b320_scene_module.KA021_OUTLET_POSITION_W[:2],
            success_base_z,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    closed = b320_env_module._ka021_flow_projection(upright, 0.0)
    assert closed["opening_aligned"] is True
    assert closed["cup_upright"] is True
    assert closed["height_valid"] is True
    assert closed["emitting"] is False
    assert closed["hits_cup"] is False

    open_flow = b320_env_module._ka021_flow_projection(upright, 0.80)
    assert open_flow["emitting"] is True
    assert open_flow["hits_cup"] is True
    assert open_flow["opening_xy_error_m"] == pytest.approx(0.0)
    assert open_flow["outlet_rim_gap_m"] == pytest.approx(0.12)

    misaligned = upright.copy()
    misaligned[0] += b320_scene_module.KA021_CUP_OPENING_RADIUS_M + 0.001
    assert b320_env_module._ka021_flow_projection(misaligned, 0.80)[
        "hits_cup"
    ] is False
    tipped = upright.copy()
    tipped[3:7] = (np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0)
    assert b320_env_module._ka021_flow_projection(tipped, 0.80)[
        "hits_cup"
    ] is False


@pytest.mark.parametrize(
    ("tilt_deg", "expected_upright"),
    (
        (-180.0, False),
        (-90.0, False),
        (-15.1, False),
        (-15.0, True),
        (-14.9, True),
        (0.0, True),
        (14.9, True),
        (15.0, True),
        (15.1, False),
        (90.0, False),
        (180.0, False),
    ),
)
def test_ka021_upright_gate_uses_local_z_and_inclusive_15_degree_boundary(
    tilt_deg: float,
    expected_upright: bool,
) -> None:
    half_angle = np.deg2rad(tilt_deg) / 2.0
    pose = np.asarray(
        [
            *b320_scene_module.KA021_OUTLET_POSITION_W[:2],
            b320_scene_module.KA021_OUTLET_POSITION_W[2]
            - b320_scene_module.KA021_CUP_RIM_HEIGHT_M
            - 0.12,
            np.cos(half_angle),
            np.sin(half_angle),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    flow = b320_env_module._ka021_flow_projection(pose, 0.80)
    rim_offset = np.asarray(
        [
            0.0,
            -np.sin(np.deg2rad(tilt_deg))
            * b320_scene_module.KA021_CUP_RIM_HEIGHT_M,
            np.cos(np.deg2rad(tilt_deg))
            * b320_scene_module.KA021_CUP_RIM_HEIGHT_M,
        ],
        dtype=np.float64,
    )
    expected_rim_center = pose[:3] + rim_offset
    np.testing.assert_allclose(
        flow["cup_rim_position"],
        expected_rim_center,
        atol=1.0e-9,
    )
    assert flow["opening_xy_error_m"] == pytest.approx(
        np.linalg.norm(
            expected_rim_center[:2]
            - np.asarray(b320_scene_module.KA021_OUTLET_POSITION_W[:2])
        ),
        abs=1.0e-9,
    )
    assert flow["cup_rim_z_m"] == pytest.approx(
        expected_rim_center[2],
        abs=1.0e-9,
    )
    assert flow["cup_tilt_deg"] == pytest.approx(abs(tilt_deg), abs=1.0e-6)
    assert flow["cup_upright"] is expected_upright
    assert flow["hits_cup"] is expected_upright


def test_ka021_faucet_flow_joint_pulls_outward_and_starts_closed() -> None:
    assert b320_scene_module.KA021_FAUCET_OPEN_AXIS == "Y"
    assert b320_scene_module.KA021_FAUCET_OPEN_LIMIT_DEG == pytest.approx(-45.0)
    assert b320_scene_module.KA021_FAUCET_RESET_POSITION_RAD == pytest.approx(0.0)
    assert b320_scene_module.KA021_FAUCET_OUTWARD_DIRECTION_W == pytest.approx(
        (-1.0, 0.0, 0.0)
    )
    # The axis token remains Y for the source lever, but the joint frame is
    # yawed with the downloaded model so it stays coaxial with HandleBase.
    np.testing.assert_allclose(
        b320_scene_module.KA021_FAUCET_JOINT_AXIS_WORLD_W,
        (0.6533108024, 0.7570898199, 0.0),
        atol=1.0e-9,
    )

    closed_grip = np.asarray(
        b320_scene_module.KA021_FAUCET_GRIP_COLLIDER_CENTER_LOCAL_M,
        dtype=np.float64,
    )
    open_angle = -b320_scene_module.KA021_FAUCET_OPEN_TRAVEL_RAD
    axis = np.asarray(
        b320_scene_module.KA021_FAUCET_JOINT_AXIS_WORLD_W,
        dtype=np.float64,
    )
    # Rodrigues rotation in the handle-root frame.  The source visual wrapper
    # carries the same yaw, so this is the actual closed->open lever motion.
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    rotation = (
        np.eye(3) * np.cos(open_angle)
        + (1.0 - np.cos(open_angle)) * np.outer(axis, axis)
        + np.sin(open_angle) * skew
    )
    open_grip = rotation @ closed_grip
    assert open_grip[0] < closed_grip[0]
    assert np.dot(
        open_grip - closed_grip,
        np.asarray(b320_scene_module.KA021_FAUCET_INITIAL_OPEN_DIRECTION_W),
    ) > 0.0

    spawn_source = inspect.getsource(b320_scene_module._spawn_ka021_hs005_faucet)
    assert "axis=KA021_FAUCET_OPEN_AXIS" in spawn_source
    assert "lower=KA021_FAUCET_OPEN_LIMIT_DEG" in spawn_source
    assert "local_rot0_wxyz=KA021_FAUCET_JOINT_FRAME_ORIENTATION_WXYZ" in spawn_source
    assert "source_lever_grip_collision" in spawn_source
    assert "size=(0.060, 0.075, 0.120)" not in spawn_source


@pytest.mark.parametrize(
    ("joint_angle_deg", "expected_fraction"),
    (
        (0.0, 0.0),
        (22.5, 0.0),
        (45.0, 0.0),
        (-22.5, 0.5),
        (-31.5, 0.7),
        (-45.0, 1.0),
        (-90.0, 1.0),
    ),
)
def test_ka021_faucet_fraction_uses_signed_45_degree_open_travel(
    joint_angle_deg: float,
    expected_fraction: float,
) -> None:
    env = _rigid_env(_KA021_TASK_NAME)
    env._physics_dof_position = lambda _name: (
        float(np.deg2rad(joint_angle_deg)),
        0.0,
    )
    fraction, available = env._dof_fraction("RevoluteJoint_faucet_3_right")
    assert available is True
    assert fraction == pytest.approx(expected_fraction)


def test_ka021_faucet_reset_writes_closed_position_velocity_and_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    joint_name = "RevoluteJoint_faucet_3_right"

    class FakeArticulationView:
        def __init__(self) -> None:
            self.root = np.asarray(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            self.root_velocity = np.zeros((1, 6), dtype=np.float32)
            self.q = np.full((1, 1), -0.5, dtype=np.float32)
            self.qd = np.full((1, 1), 2.0, dtype=np.float32)
            self.targets = np.full((1, 1), -0.5, dtype=np.float32)

        def get_root_transforms(self):
            return self.root.copy()

        def get_root_velocities(self):
            return self.root_velocity.copy()

        def get_dof_positions(self):
            return self.q.copy()

        def get_dof_velocities(self):
            return self.qd.copy()

        def get_metatype(self, _index: int):
            return SimpleNamespace(dof_indices={joint_name: 0})

        def set_dof_positions(self, values, _indices) -> None:
            self.q = np.asarray(values, dtype=np.float32).copy()

        def set_dof_velocities(self, values, _indices) -> None:
            self.qd = np.asarray(values, dtype=np.float32).copy()

        def set_dof_position_targets(self, values, _indices) -> None:
            self.targets = np.asarray(values, dtype=np.float32).copy()

    monkeypatch.setattr(
        b320_env_module,
        "_array_like",
        lambda _reference, values, _torch: np.asarray(
            values,
            dtype=np.float32,
        ).copy(),
    )
    env = _rigid_env(_KA021_TASK_NAME)
    view = FakeArticulationView()
    indices = np.asarray([0], dtype=np.int32)
    for randomize in (False, True):
        env.randomize = randomize
        view.q.fill(-b320_scene_module.KA021_FAUCET_OPEN_TRAVEL_RAD)
        view.qd.fill(2.0)
        view.targets.fill(-b320_scene_module.KA021_FAUCET_OPEN_TRAVEL_RAD)

        env._set_articulation_reset(view, indices, fixture=True)

        np.testing.assert_array_equal(view.q, 0.0)
        np.testing.assert_array_equal(view.qd, 0.0)
        np.testing.assert_array_equal(view.targets, 0.0)


def test_ka021_faucet_colliders_leave_grip_and_arch_clearance() -> None:
    spawn_source = inspect.getsource(b320_scene_module._spawn_ka021_hs005_faucet)
    assert 'path=f"{base_path}/spout_collision"' not in spawn_source
    assert "size=(0.135, 0.155, 0.055)" not in spawn_source

    source_grip_min = np.asarray(
        (-0.0104, -0.0104, -0.0161),
        dtype=np.float64,
    )
    source_grip_max = np.asarray(
        (0.0098, 0.0098, 0.0581),
        dtype=np.float64,
    )
    grip_center = np.asarray(
        b320_scene_module.KA021_FAUCET_GRIP_COLLIDER_CENTER_LOCAL_M,
        dtype=np.float64,
    )
    grip_half_extents = 0.5 * np.asarray(
        b320_scene_module.KA021_FAUCET_GRIP_COLLIDER_SIZE_M,
        dtype=np.float64,
    )
    grip_min = grip_center - grip_half_extents
    grip_max = grip_center + grip_half_extents
    # The lower two millimetres of the authored lever nest into its fixed
    # HandleBase; the runtime grip intentionally covers the graspable upper
    # lever while leaving that base out of the dynamic collision body.
    assert np.all(grip_min[:2] <= source_grip_min[:2])
    assert np.all(grip_max[:2] >= source_grip_max[:2])
    assert grip_min[2] < 0.0 < grip_max[2]
    assert grip_max[2] >= source_grip_max[2] - 0.01

    pivot_xy = np.asarray(
        b320_scene_module.KA021_FAUCET_HANDLE_PIVOT_W[:2],
        dtype=np.float64,
    )
    column_center_world = np.asarray(
        b320_scene_module._ka021_model_world_point((0.0, 0.0, 0.065)),
        dtype=np.float64,
    )
    column_center_local_xy = column_center_world[:2] - pivot_xy
    closest_grip_xy = np.clip(
        column_center_local_xy,
        grip_min[:2],
        grip_max[:2],
    )
    column_to_grip_surface_m = float(
        np.linalg.norm(column_center_local_xy - closest_grip_xy)
        - b320_scene_module.KA021_FAUCET_COLUMN_COLLIDER_RADIUS_M
    )
    assert column_to_grip_surface_m >= 0.002 + 0.0015

    probe_center = np.asarray(
        b320_scene_module.KA021_FAUCET_ARCH_CLEARANCE_PROBE_CENTER_W,
        dtype=np.float64,
    )
    probe_radius = float(
        np.linalg.norm(
            b320_scene_module.KA021_FAUCET_ARCH_CLEARANCE_PROBE_HALF_EXTENTS_M
        )
    )
    assert len(b320_scene_module.KA021_FAUCET_ARCH_COLLIDER_SEGMENTS_W) == 4
    assert b320_scene_module.KA021_FAUCET_ARCH_COLLIDER_RADIUS_M == pytest.approx(
        0.023
    )
    for start_w, end_w in b320_scene_module.KA021_FAUCET_ARCH_COLLIDER_SEGMENTS_W:
        start = np.asarray(start_w, dtype=np.float64)
        segment = np.asarray(end_w, dtype=np.float64) - start
        projection = float(
            np.clip(
                np.dot(probe_center - start, segment)
                / np.dot(segment, segment),
                0.0,
                1.0,
            )
        )
        distance_to_centerline = float(
            np.linalg.norm(probe_center - (start + projection * segment))
        )
        assert distance_to_centerline >= (
            b320_scene_module.KA021_FAUCET_ARCH_COLLIDER_RADIUS_M
            + probe_radius
        )


def test_bc014_goal_zone_is_inside_both_physical_arm_workspaces() -> None:
    contract = B320_TASK_SPECS[B320_BC014_TASK_NAME]
    assert contract.goal_pose_wxyz[:2] == pytest.approx((-0.22, 0.0))
    assert BC014_GOAL_CENTERS_XY == {
        "salt_jar": (-0.23, 0.29),
        "pepper_jar": (-0.23, -0.29),
    }
    assert BC014_GOAL_SPACING_M == pytest.approx(0.58)

    assert b320_env_module._BC014_COMFORT_MAX_RADIUS_M == pytest.approx(
        _ARM_WORKSPACE_MAX_RADIUS - 0.07
    )
    for role, expected_side in (("salt_jar", 1), ("pepper_jar", -1)):
        low, high = (
            np.asarray(value, dtype=np.float64)
            for value in b320_env_module._BC014_COMFORT_CENTER_BOUNDS_XY[role]
        )
        assert expected_side * low[1] > 0.0
        assert expected_side * high[1] > 0.0
        base = np.asarray(
            b320_env_module._BC014_CANONICAL_ARM_BASE_POSITIONS[role],
            dtype=np.float64,
        )
        for x in (low[0], high[0]):
            for y in (low[1], high[1]):
                reach_point = np.asarray(
                    [x, y, b320_env_module._BC014_COMFORT_REACH_Z_M]
                )
                assert np.linalg.norm(reach_point - base) <= (
                    b320_env_module._BC014_COMFORT_MAX_RADIUS_M
                )


def test_bc014_goal_markers_use_high_contrast_visual_only_outlines() -> None:
    class FakePrim:
        def __init__(self) -> None:
            self.custom_data: dict[str, object] = {}

        def SetCustomDataByKey(self, key: str, value: object) -> None:
            self.custom_data[key] = value

    class FakeStage:
        def __init__(self) -> None:
            self.prims: dict[str, FakePrim] = {}

        def GetPrimAtPath(self, path: str) -> FakePrim:
            return self.prims.setdefault(path, FakePrim())

    class FakePreviewSurfaceCfg:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeCuboidCfg:
        calls: list[tuple[str, tuple[float, ...], tuple[float, ...], object]] = []

        def __init__(self, *, size: tuple[float, ...], visual_material: object) -> None:
            self.size = size
            self.visual_material = visual_material

        def func(
            self,
            path: str,
            _cfg: object,
            *,
            translation: tuple[float, ...],
        ) -> None:
            self.calls.append((path, self.size, translation, self.visual_material))

    sim_utils = SimpleNamespace(
        CuboidCfg=FakeCuboidCfg,
        PreviewSurfaceCfg=FakePreviewSurfaceCfg,
    )
    stage = FakeStage()
    translation = (-0.22, 0.05, b320_scene_module.B320_TABLE_TOP_Z + 0.003)
    color = (0.0, 0.72, 1.0)

    marker_path = b320_scene_module._spawn_goal_marker(
        stage,
        sim_utils,
        name="salt_goal",
        translation=translation,
        size=BC014_GOAL_ZONE_SIZE_M,
        color=color,
        highlight=True,
    )

    assert marker_path.endswith("/salt_goal")
    assert len(FakeCuboidCfg.calls) == 5
    fill_call, *edge_calls = FakeCuboidCfg.calls
    assert fill_call[1] == BC014_GOAL_ZONE_SIZE_M
    assert fill_call[3].kwargs["opacity"] == pytest.approx(0.72)
    assert fill_call[3].kwargs["emissive_color"] == pytest.approx(
        (0.0, 0.3024, 0.42)
    )
    assert {call[0].rsplit("_", 1)[-1] for call in edge_calls} == {
        "front",
        "rear",
        "left",
        "right",
    }
    assert all(call[3].kwargs["opacity"] == 1.0 for call in edge_calls)
    # Every outline lies inside its predicate zone with the outside face on
    # the exact boundary, so the adjacent salt/pepper outlines cannot overlap.
    for edge_path, edge_size, edge_translation, _ in edge_calls:
        assert edge_path.startswith(f"{marker_path}_highlight_")
        assert abs(edge_translation[0] - translation[0]) + 0.5 * edge_size[0] <= (
            0.5 * BC014_GOAL_ZONE_SIZE_M[0] + 1.0e-12
        )
        assert abs(edge_translation[1] - translation[1]) + 0.5 * edge_size[1] <= (
            0.5 * BC014_GOAL_ZONE_SIZE_M[1] + 1.0e-12
        )
    assert stage.prims[marker_path].custom_data == {
        "handson:goalMarker": True,
        "handson:highlighted": True,
        "handson:goalZoneSizeM": BC014_GOAL_ZONE_SIZE_M,
    }
    assert all(
        stage.prims[call[0]].custom_data
        == {"handson:goalHighlight": True, "handson:visualOnly": True}
        for call in edge_calls
    )


def test_bc014_goal_marker_fill_and_outlines_follow_per_reset_centers(
    monkeypatch,
) -> None:
    class FakeTranslateOp:
        def __init__(self) -> None:
            self.value: tuple[float, ...] | None = None

        def GetOpType(self) -> str:
            return "translate"

        def GetPrecision(self) -> str:
            return "double"

        def Set(self, value: tuple[float, ...]) -> None:
            self.value = tuple(value)

    class FakePrim:
        def __init__(self) -> None:
            self.op = FakeTranslateOp()
            self.custom_data: dict[str, object] = {}

        def IsValid(self) -> bool:
            return True

        def SetCustomDataByKey(self, key: str, value: object) -> None:
            self.custom_data[key] = value

    class FakeXformable:
        def __init__(self, prim: FakePrim) -> None:
            self.prim = prim

        def GetOrderedXformOps(self) -> list[FakeTranslateOp]:
            return [self.prim.op]

    prims = {}
    for path in b320_scene_module.BC014_GOAL_MARKER_PATHS.values():
        prims[path] = FakePrim()
        for edge in ("front", "rear", "left", "right"):
            prims[f"{path}_highlight_{edge}"] = FakePrim()
    stage = SimpleNamespace(GetPrimAtPath=lambda path: prims[path])
    fake_xform_op = SimpleNamespace(
        TypeTranslate="translate",
        PrecisionFloat="float",
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "pxr",
        SimpleNamespace(
            Gf=SimpleNamespace(
                Vec3d=lambda *values: tuple(values),
                Vec3f=lambda *values: tuple(values),
            ),
            UsdGeom=SimpleNamespace(
                Xformable=FakeXformable,
                XformOp=fake_xform_op,
            ),
        ),
    )
    env = object.__new__(HandsOnB320BimanualIsaacLabEnv)
    env.task_name = B320_BC014_TASK_NAME
    env._stage = stage
    env._launch_isaac = False
    env._bc014_goal_visual_warning_emitted = False
    env._bc014_goal_centers_xy = {
        "salt_jar": np.asarray([-0.40, 0.14]),
        "pepper_jar": np.asarray([-0.25, -0.19]),
    }

    assert env._set_bc014_goal_marker_visuals() is True
    for role, path in b320_scene_module.BC014_GOAL_MARKER_PATHS.items():
        center = env._bc014_goal_centers_xy[role]
        assert prims[path].op.value == pytest.approx(
            (*center, b320_scene_module.B320_TABLE_TOP_Z + 0.003)
        )
        assert prims[path].custom_data["handson:goalCenterXY"] == pytest.approx(
            center
        )
        edge_positions = [
            prims[f"{path}_highlight_{edge}"].op.value
            for edge in ("front", "rear", "left", "right")
        ]
        assert all(position is not None for position in edge_positions)


def test_bc014_water_flow_proxy_is_hidden_and_render_only(monkeypatch) -> None:
    class FakePrim:
        def __init__(self) -> None:
            self.custom_data: dict[str, object] = {}
            self.invisible = False

        def SetCustomDataByKey(self, key: str, value: object) -> None:
            self.custom_data[key] = value

    class FakeStage:
        def __init__(self) -> None:
            self.prims: dict[str, FakePrim] = {}

        def GetPrimAtPath(self, path: str) -> FakePrim:
            return self.prims.setdefault(path, FakePrim())

    class FakePreviewSurfaceCfg:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeCylinderCfg:
        calls: list[tuple[str, dict[str, object], tuple[float, ...]]] = []

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def func(
            self,
            path: str,
            _cfg: object,
            *,
            translation: tuple[float, ...],
        ) -> None:
            self.calls.append((path, self.kwargs, translation))

    class FakeImageable:
        def __init__(self, prim: FakePrim) -> None:
            self.prim = prim

        def MakeInvisible(self) -> None:
            self.prim.invisible = True

    monkeypatch.setitem(
        __import__("sys").modules,
        "pxr",
        SimpleNamespace(UsdGeom=SimpleNamespace(Imageable=FakeImageable)),
    )
    stage = FakeStage()
    sim_utils = SimpleNamespace(
        CylinderCfg=FakeCylinderCfg,
        PreviewSurfaceCfg=FakePreviewSurfaceCfg,
    )

    path = b320_scene_module._spawn_bc014_flow_proxy(
        stage,
        sim_utils,
        role="salt_jar",
    )

    assert path == BC014_FLOW_PROXY_PATHS["salt_jar"]
    _, cfg_kwargs, _ = FakeCylinderCfg.calls[-1]
    assert cfg_kwargs["radius"] == pytest.approx(0.006)
    assert cfg_kwargs["height"] == pytest.approx(1.0)
    assert cfg_kwargs["axis"] == "Z"
    assert "collision_props" not in cfg_kwargs
    assert "rigid_props" not in cfg_kwargs
    assert stage.prims[path].invisible is True
    assert stage.prims[path].custom_data == {
        "handson:flowProxy": "water_stream",
        "handson:seasoningRole": "salt_jar",
        "handson:visualOnly": True,
    }


def test_bc014_randomized_reset_is_seeded_and_collision_clear() -> None:
    env = HandsOnB320BimanualIsaacLabEnv(
        B320_BC014_TASK_NAME,
        launch_isaac=False,
        image_obs=False,
        randomize=True,
    )
    try:
        env._random.seed(31415)
        env._sample_reset_state(clear_manual_request=True)
        first = {
            name: pose.copy() for name, pose in env._bc014_reset_poses.items()
        }
        first_goals = {
            name: center.copy()
            for name, center in env._bc014_goal_centers_xy.items()
        }
        env._random.seed(31415)
        env._sample_reset_state(clear_manual_request=True)
        second = {
            name: pose.copy() for name, pose in env._bc014_reset_poses.items()
        }
        second_goals = {
            name: center.copy()
            for name, center in env._bc014_goal_centers_xy.items()
        }
        env._random.seed(31416)
        env._sample_reset_state(clear_manual_request=True)
        third = {
            name: pose.copy() for name, pose in env._bc014_reset_poses.items()
        }
        third_goals = {
            name: center.copy()
            for name, center in env._bc014_goal_centers_xy.items()
        }

        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
            np.testing.assert_array_equal(first_goals[name], second_goals[name])
        assert any(
            not np.array_equal(first[name], third[name]) for name in first
        ) or any(
            not np.array_equal(first_goals[name], third_goals[name])
            for name in first_goals
        )
        assert env._bc014_reset_attempts >= 1

        multi_seed = []
        multi_seed_goals = []
        for seed in range(8):
            env._random.seed(seed)
            env._sample_reset_state(clear_manual_request=True)
            sampled = {
                name: pose.copy()
                for name, pose in env._bc014_reset_poses.items()
            }
            goals = {
                name: center.copy()
                for name, center in env._bc014_goal_centers_xy.items()
            }
            multi_seed.append(sampled)
            multi_seed_goals.append(goals)
            for role, pose in sampled.items():
                assert -0.57 + 0.020 + 0.029 <= pose[0] <= 0.27 - 0.020 - 0.029
                assert -0.64 + 0.020 + 0.029 <= pose[1] <= 0.64 - 0.020 - 0.029
                assert pose[2] == pytest.approx(
                    b320_scene_module.B320_TABLE_TOP_Z
                    + BC014_JAR_TABLE_CLEARANCE_M
                )
                low, high = (
                    np.asarray(value, dtype=np.float64)
                    for value in b320_env_module._BC014_COMFORT_CENTER_BOUNDS_XY[
                        role
                    ]
                )
                assert np.all(pose[:2] >= low)
                assert np.all(pose[:2] <= high)
                assert np.all(goals[role] >= low)
                assert np.all(goals[role] <= high)
                assert np.linalg.norm(pose[:2] - goals[role]) >= (
                    b320_env_module._BC014_MIN_JAR_GOAL_SEPARATION_M
                )
                base = np.asarray(
                    b320_env_module._BC014_CANONICAL_ARM_BASE_POSITIONS[role]
                )
                for point_xy in (pose[:2], goals[role]):
                    reach_point = np.asarray(
                        [
                            point_xy[0],
                            point_xy[1],
                            b320_env_module._BC014_COMFORT_REACH_Z_M,
                        ]
                    )
                    assert np.linalg.norm(reach_point - base) <= (
                        b320_env_module._BC014_COMFORT_MAX_RADIUS_M
                    )
            assert np.linalg.norm(
                sampled["salt_jar"][:2] - sampled["pepper_jar"][:2]
            ) >= 0.085
        assert len(
            {
                tuple(np.round(sample["salt_jar"][:2], decimals=8))
                for sample in multi_seed
            }
        ) >= 2
        assert len(
            {
                tuple(np.round(goals["salt_jar"], decimals=8))
                for goals in multi_seed_goals
            }
        ) >= 2
        assert len(
            {
                tuple(np.round(sample["salt_jar"][3:7], decimals=8))
                for sample in multi_seed
            }
        ) >= 2
        observation, info = env.reset(seed=20260825)
        recorded_goals = info["bc014_goal_centers_xy"]
        replay = env.get_physical_replay_reset_contract()
        assert recorded_goals == info["bc014_reset"]["goal_centers_xy"]
        assert recorded_goals == replay["bc014_goal_centers_xy"]
        np.testing.assert_allclose(
            observation["state"]["goal_pose"][:2],
            np.mean(np.asarray(list(recorded_goals.values())), axis=0),
        )
    finally:
        env.close()


def test_bc014_cap_rays_use_per_reset_goal_zones_and_timed_sprinkling() -> None:
    down = (0.0, 1.0, 0.0, 0.0)
    salt_x, salt_y = BC014_GOAL_CENTERS_XY["salt_jar"]
    pepper_x, pepper_y = BC014_GOAL_CENTERS_XY["pepper_jar"]
    salt_pose = np.asarray([salt_x, salt_y, 1.10, *down])
    pepper_pose = np.asarray([pepper_x, pepper_y, 1.10, *down])

    salt_projection = b320_env_module._bc014_flow_projection(
        salt_pose,
        BC014_GOAL_CENTERS_XY["salt_jar"],
    )
    assert salt_projection["ray_valid"] is True
    assert salt_projection["projects_into_zone"] is True
    np.testing.assert_allclose(
        salt_projection["impact_position"][:2],
        BC014_GOAL_CENTERS_XY["salt_jar"],
        atol=1.0e-12,
    )
    wrong_projection = b320_env_module._bc014_flow_projection(
        pepper_pose,
        BC014_GOAL_CENTERS_XY["salt_jar"],
    )
    assert wrong_projection["projects_into_zone"] is False

    env = _rigid_env(B320_BC014_TASK_NAME)
    try:
        _prime_contacts(env)
        # Formal truth follows the per-reset targets rather than the static
        # scene fallback centers.
        env._bc014_goal_centers_xy = {
            "salt_jar": np.asarray([-0.39, 0.13]),
            "pepper_jar": np.asarray([-0.26, -0.17]),
        }
        env._poses["salt_jar"] = np.asarray([-0.39, 0.13, 1.10, *down])
        env._poses["pepper_jar"] = np.asarray([-0.26, -0.17, 1.10, *down])
        required_steps = int(
            np.ceil(
                b320_env_module._BC014_FLOW_REQUIRED_SECONDS
                / env._control_dt
            )
        )
        for _ in range(required_steps):
            metrics, flags = env._seasoning_metrics(update=True)

        assert metrics["salt_stream_hits_target"] is True
        assert metrics["pepper_stream_hits_target"] is True
        assert metrics["salt_hit_seconds"] == pytest.approx(0.6)
        assert metrics["pepper_hit_seconds"] == pytest.approx(0.6)
        assert metrics["overlap_hit_seconds"] == pytest.approx(0.6)
        assert metrics["salt_goal_x_m"] == pytest.approx(-0.39)
        assert metrics["pepper_goal_y_m"] == pytest.approx(-0.17)
        assert all(flags)
        assert env._compute_failure() is False
    finally:
        env.close()

    # Merely placing both jars upright in the old target zones no longer
    # satisfies BC014. The cap-directed proxy streams must hit the zones.
    env = _rigid_env(B320_BC014_TASK_NAME)
    try:
        _prime_contacts(env)
        salt_x, salt_y = BC014_GOAL_CENTERS_XY["salt_jar"]
        pepper_x, pepper_y = BC014_GOAL_CENTERS_XY["pepper_jar"]
        env._poses["salt_jar"] = np.asarray(
            [salt_x, salt_y, 0.95, 1.0, 0.0, 0.0, 0.0]
        )
        env._poses["pepper_jar"] = np.asarray(
            [pepper_x, pepper_y, 0.95, 1.0, 0.0, 0.0, 0.0]
        )
        metrics, flags = env._seasoning_metrics()
        assert metrics["salt_stream_emitting"] is False
        assert metrics["pepper_stream_emitting"] is False
        assert flags == (True, False, False, False)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("active_role", "expected_reason"),
    (
        ("salt_jar", "bc014_salt_pour_timeout"),
        ("pepper_jar", "bc014_pepper_pour_timeout"),
    ),
)
def test_bc014_single_jar_pour_over_four_seconds_resets_episode(
    active_role: str,
    expected_reason: str,
) -> None:
    env = _rigid_env(B320_BC014_TASK_NAME)
    try:
        _prime_contacts(env)
        env.hz = 0
        env._read_contacts = lambda *, update_latches: None
        down = (0.0, 1.0, 0.0, 0.0)
        upright = (1.0, 0.0, 0.0, 0.0)
        for role in ("salt_jar", "pepper_jar"):
            x, y = BC014_GOAL_CENTERS_XY[role]
            orientation = down if role == active_role else upright
            env._poses[role] = np.asarray([x, y, 1.10, *orientation])

        boundary_steps = int(
            round(
                b320_env_module._BC014_FLOW_MAX_POUR_SECONDS
                / env._control_dt
            )
        )
        for _ in range(boundary_steps):
            _observation, _reward, terminated, truncated, info = (
                env.step_with_current_joint_targets()
            )
            assert terminated is False
            assert truncated is False

        active_side = "left" if active_role == "salt_jar" else "right"
        assert env._bc014_flow_pour_seconds[active_side] == pytest.approx(4.0)
        assert info["b320_metrics"]["maximum_pour_seconds"] == pytest.approx(
            4.0
        )

        _observation, _reward, terminated, truncated, info = (
            env.step_with_current_joint_targets()
        )
        assert terminated is True
        assert truncated is False
        assert info["failed"] is True
        assert env._b320_failure_reason == expected_reason
        assert env._bc014_flow_pour_seconds[active_side] > 4.0
        assert env._success_counter == 0
        assert info["b320_failure_reason"] == expected_reason

        env._sample_reset_state(clear_manual_request=True)
        assert env._bc014_flow_pour_seconds == {"left": 0.0, "right": 0.0}
        assert env._b320_failure_reason is None
    finally:
        env.close()


def test_bc014_reuses_attributed_render_only_kitchen_and_records_it() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    assert (repository_root / "handson/handson" / REALISTIC_KITCHEN_ASSET_RELATIVE_PATH).is_file()
    assert REALISTIC_KITCHEN_ASSET_SHA256 == (
        "babf31ddcaa34b81dd669df0b99fe1cdbddccb15bb29f7c10f0a266d3d8a5cb3"
    )
    assert REALISTIC_KITCHEN_LICENSE == "CC BY 4.0"

    # The visual island covers the canonical 0.84 x 1.28 m physical table,
    # while both retain the same 0.89 m support height.
    physical_table_min = np.asarray([-0.57, -0.64, 0.83])
    physical_table_max = np.asarray([0.27, 0.64, 0.89])
    island_min = np.asarray(REALISTIC_KITCHEN_ISLAND_MIN_M)
    island_max = np.asarray(REALISTIC_KITCHEN_ISLAND_MAX_M)
    assert np.all(island_min[:2] <= physical_table_min[:2])
    assert np.all(island_max[:2] >= physical_table_max[:2])
    assert island_max[2] == pytest.approx(physical_table_max[2], abs=0.002)

    source = inspect.getsource(compose_realistic_kitchen_visual)
    assert "UsdPhysics.RigidBodyAPI" in source
    assert "UsdPhysics.CollisionAPI" in source
    assert "UsdPhysics.ArticulationRootAPI" in source
    assert "CollisionAPI.Apply" not in source
    assert "MakeInvisible()" in source
    assert "AddReference(asset_path.as_posix())" in source

    init_source = inspect.getsource(HandsOnB320BimanualIsaacLabEnv.__init__)
    assert init_source.index("super().__init__") < init_source.index(
        "_spawn_bc014_kitchen_visual"
    )
    spawn_source = inspect.getsource(
        HandsOnB320BimanualIsaacLabEnv._spawn_bc014_kitchen_visual
    )
    assert "table_prim_path=B320_TABLE_PATH" in spawn_source
    assert 'floor = self._find_prim("floor")' in spawn_source

    env = _rigid_env(B320_BC014_TASK_NAME)
    _, info = env.reset(seed=7)
    metadata = info["bc014_realistic_kitchen"]
    assert metadata["asset_sha256"] == REALISTIC_KITCHEN_ASSET_SHA256
    assert metadata["license"] == "CC BY 4.0"
    assert metadata["render_only"] is True
    assert metadata["collision_mode"] == "canonical_table_and_floor_only"
    flow_metadata = info["bc014_flow_proxy"]
    assert flow_metadata["paths"] == BC014_FLOW_PROXY_PATHS
    assert flow_metadata["visual_model"] == "d09_water_stream_cylinder"
    assert flow_metadata["particle_model"] is False
    assert flow_metadata["jar_placement_required"] is False
    assert flow_metadata["jar_release_required"] is False
    assert flow_metadata["maximum_pour_seconds_before_reset"] == pytest.approx(
        4.0
    )
    assert flow_metadata["pour_timeout_accumulation"] == "cumulative_per_jar"

    replay = env.get_physical_replay_reset_contract()
    assert replay["bc014_goal_centers_xy"] == {
        role: list(center) for role, center in BC014_GOAL_CENTERS_XY.items()
    }
    assert replay["bc014_flow_proxy"] == flow_metadata
    assert replay["bc014_realistic_kitchen"] == metadata

    live_report = RealisticKitchenReport(
        asset_path=repository_root / "handson/handson" / REALISTIC_KITCHEN_ASSET_RELATIVE_PATH,
        root_path=metadata["root_path"],
        mesh_count=144,
        island_min_m=tuple(REALISTIC_KITCHEN_ISLAND_MIN_M),
        island_max_m=tuple(REALISTIC_KITCHEN_ISLAND_MAX_M),
        physical_table_min_m=(-0.57, -0.64, 0.83),
        physical_table_max_m=(0.27, 0.64, 0.89),
        hidden_support_paths=("/World/Task/table", "/World/Task/floor"),
    )
    assert set(live_report.replay_metadata()) == set(
        realistic_kitchen_metadata(root_path=metadata["root_path"])
    )


def test_carton_disables_gravity_on_every_physx_robot_link(monkeypatch) -> None:
    class FakeAttribute:
        def __init__(self) -> None:
            self.value = None

        def Set(self, value) -> bool:
            self.value = value
            return True

        def Get(self):
            return self.value

    class FakePrim:
        def __init__(self, path: str, *, rigid: bool) -> None:
            self.path = path
            self.rigid = rigid
            self.disable_gravity = FakeAttribute()

        def IsValid(self) -> bool:
            return True

        def HasAPI(self, api) -> bool:
            return self.rigid and api is FakeUsdPhysics.RigidBodyAPI

        def GetPath(self) -> str:
            return self.path

    class FakeUsdPhysics:
        class RigidBodyAPI:
            pass

    class FakePhysxRigidBodyAPI:
        def __init__(self, prim: FakePrim) -> None:
            self.prim = prim

        @classmethod
        def Apply(cls, prim: FakePrim):
            return cls(prim)

        def CreateDisableGravityAttr(self) -> FakeAttribute:
            return self.prim.disable_gravity

    roots = {}
    all_prims = []
    for side in ("Right", "Left"):
        path = f"/World/HandsOnRobots/{side}Robot"
        root = FakePrim(path, rigid=True)
        root.prims = [
            root,
            FakePrim(f"{path}/arm", rigid=True),
            FakePrim(f"{path}/visual", rigid=False),
        ]
        roots[path] = root
        all_prims.extend(root.prims)
    fake_pxr = SimpleNamespace(
        PhysxSchema=SimpleNamespace(PhysxRigidBodyAPI=FakePhysxRigidBodyAPI),
        Usd=SimpleNamespace(PrimRange=lambda root: root.prims),
        UsdPhysics=FakeUsdPhysics,
    )
    monkeypatch.setitem(__import__("sys").modules, "pxr", fake_pxr)
    stage = SimpleNamespace(GetPrimAtPath=lambda path: roots[path])

    count = _disable_physx_gravity_for_robot_links(stage, tuple(roots))

    assert count == 4
    for prim in all_prims:
        assert prim.disable_gravity.Get() is (True if prim.rigid else None)


def test_carton_reset_tcp_commands_sync_to_physical_palms(monkeypatch) -> None:
    env = _rigid_env("t10_lift_carton")

    class FakeRobot:
        def update(self, _dt: float) -> None:
            pass

    robots = {"right": FakeRobot(), "left": FakeRobot()}
    env._robots = robots
    env._ee_body_indices = {"right": 3, "left": 4}
    matrices = {}
    for side, y in (("right", -0.4041345), ("left", 0.4041345)):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = (-0.4091808, y, 0.9975)
        matrices[id(robots[side])] = matrix
    monkeypatch.setattr(
        env,
        "_robot_body_pose_matrix",
        lambda robot, _index: matrices[id(robot)].copy(),
    )

    assert env._sync_b320_reset_tcp_commands_to_physics() == 2
    np.testing.assert_allclose(
        env._right_tcp_pose,
        (-0.4091808, -0.4041345, 0.9975, 1.0, 0.0, 0.0, 0.0),
    )
    np.testing.assert_allclose(
        env._left_tcp_pose,
        (-0.4091808, 0.4041345, 0.9975, 1.0, 0.0, 0.0, 0.0),
    )
    assert env._right_pose_commanded is False
    assert env._left_pose_commanded is False


def test_b320_direct_asset_spawn_converts_wxyz_to_runtime_quaternion_layout(
    monkeypatch,
) -> None:
    captured: list[tuple[float, ...]] = []

    class FakeUsdFileCfg:
        def __init__(self, **_kwargs) -> None:
            pass

        def func(self, _prim_path, _cfg, *, translation, orientation) -> None:
            captured.append(tuple(orientation))

    class FakePrim:
        def SetCustomDataByKey(self, _key, _value) -> None:
            pass

    fake_stage = SimpleNamespace(GetPrimAtPath=lambda _path: FakePrim())
    fake_sim_utils = SimpleNamespace(
        CollisionPropertiesCfg=lambda **kwargs: kwargs,
        UsdFileCfg=FakeUsdFileCfg,
        make_uninstanceable=lambda *_args, **_kwargs: None,
        update_stage=lambda: None,
    )
    monkeypatch.setattr(isaaclab_compat, "isaaclab_uses_xyzw", lambda: True)
    monkeypatch.setattr(
        b320_scene_module,
        "_configure_dynamic_source",
        lambda _stage, _sim_utils, object_spec: (object_spec.prim_path, 1),
    )
    carton_spec = B320_TASK_SPECS["t10_lift_carton"].object_specs[0]
    b320_scene_module._spawn_direct_object(fake_stage, fake_sim_utils, carton_spec)

    np.testing.assert_allclose(captured[0], (0.0, 0.0, 0.0, 1.0))


def test_bc011_both_pen_variants_match_realistic_wuji_hand_scale() -> None:
    pen_spec = B320_TASK_SPECS["t20_pass_and_orient_pen"].object_specs[0]

    assert len(b320_scene_module._BC011_PEN_VARIANTS) == 2
    assert set(BC011_PEN_TIP_AXIS_LOCAL) == set(BC011_PEN_VARIANTS)
    assert BC011_PEN_TIP_AXIS_LOCAL[BC011_PEN_VARIANTS[0]] == (-1.0, 0.0, 0.0)
    assert BC011_PEN_TIP_AXIS_LOCAL[BC011_PEN_VARIANTS[1]] == (1.0, 0.0, 0.0)
    assert BC011_SOURCE_PEN_LENGTH_M == pytest.approx(0.160)
    assert BC011_TARGET_PEN_LENGTH_M == pytest.approx(0.200)
    assert BC011_PEN_SCALE == pytest.approx((1.25, 1.25, 1.25))
    assert pen_spec.scale == pytest.approx(BC011_PEN_SCALE)
    assert pen_spec.home_translation[2] == pytest.approx(
        b320_scene_module.B320_TABLE_TOP_Z + BC011_PEN_TABLE_CLEARANCE_M
    )
    assert (
        BC011_TARGET_PEN_LENGTH_M / BC011_WUJI1_INDEX_REFERENCE_LENGTH_M
    ) == pytest.approx(1.78, abs=0.01)


def test_bc011_pen_variant_changes_on_each_randomized_fresh_scene() -> None:
    first_env = HandsOnB320BimanualIsaacLabEnv(
        "t20_pass_and_orient_pen",
        launch_isaac=False,
        image_obs=False,
        randomize=True,
        seed=13,
    )
    _, first_info = first_env.reset(seed=13)
    first = first_info["bc011_pen_variant"]
    rebuild_kwargs = first_env._in_process_scene_rebuild_kwargs(14)

    second_env = HandsOnB320BimanualIsaacLabEnv(
        "t20_pass_and_orient_pen",
        launch_isaac=False,
        image_obs=False,
        randomize=True,
        seed=14,
        bc011_previous_pen_variant_index=rebuild_kwargs[
            "bc011_previous_pen_variant_index"
        ],
    )
    _, second_info = second_env.reset(seed=14)
    second = second_info["bc011_pen_variant"]

    assert first in b320_scene_module.BC011_PEN_VARIANTS
    assert second in b320_scene_module.BC011_PEN_VARIANTS
    assert second != first
    assert first_info["bc011_pen_tip_axis_local"] == list(
        BC011_PEN_TIP_AXIS_LOCAL[first]
    )
    assert second_info["bc011_pen_tip_axis_local"] == list(
        BC011_PEN_TIP_AXIS_LOCAL[second]
    )
    replay = second_env.get_physical_replay_reset_contract()
    assert replay["bc011_pen_variant"] == second
    assert replay["bc011_pen_variant_index"] == second_env._bc011_pen_variant_index
    assert replay["bc011_pen_tip_axis_local"] == list(
        BC011_PEN_TIP_AXIS_LOCAL[second]
    )


def test_bc011_pen_variant_is_fixed_when_randomization_is_disabled() -> None:
    env = _rigid_env("t20_pass_and_orient_pen")

    _, first_info = env.reset(seed=13)
    _, second_info = env.reset()

    assert first_info["bc011_pen_variant"] == second_info["bc011_pen_variant"]
    assert first_info["bc011_pen_variant"] == b320_scene_module.BC011_PEN_VARIANTS[0]


def test_b320_articulated_asset_spawn_converts_wxyz_to_runtime_quaternion_layout(
    monkeypatch,
) -> None:
    captured: list[tuple[float, ...]] = []

    class FakeUsdFileCfg:
        def __init__(self, **_kwargs) -> None:
            pass

        def func(self, _prim_path, _cfg, *, translation, orientation) -> None:
            captured.append(tuple(orientation))

    class FakePrim:
        def __init__(self, *, rigid: bool = False, collider: bool = False) -> None:
            self.rigid = rigid
            self.collider = collider

        def HasAPI(self, api) -> bool:
            if api is FakeUsdPhysics.RigidBodyAPI:
                return self.rigid
            if api is FakeUsdPhysics.CollisionAPI:
                return self.collider
            return False

        def SetCustomDataByKey(self, _key, _value) -> None:
            pass

    class FakeUsdPhysics:
        class RigidBodyAPI:
            pass

        class CollisionAPI:
            pass

    root = FakePrim()
    descendants = [
        FakePrim(rigid=index < 2, collider=index < 8)
        for index in range(8)
    ]
    fake_stage = SimpleNamespace(GetPrimAtPath=lambda _path: root)
    fake_sim_utils = SimpleNamespace(
        CollisionPropertiesCfg=lambda **kwargs: kwargs,
        UsdFileCfg=FakeUsdFileCfg,
        make_uninstanceable=lambda *_args, **_kwargs: None,
        update_stage=lambda: None,
    )
    monkeypatch.setattr(isaaclab_compat, "isaaclab_uses_xyzw", lambda: True)
    monkeypatch.setattr(
        b320_scene_module,
        "_descendants",
        lambda _stage, _path: descendants,
    )
    monkeypatch.setattr(
        b320_scene_module,
        "_bind_material",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "pxr",
        SimpleNamespace(UsdPhysics=FakeUsdPhysics),
    )
    source_spec = B320_TASK_SPECS[
        "t05_fill_cup"
    ].object_specs[0]

    path, collider_count = b320_scene_module._spawn_articulated_source(
        fake_stage,
        fake_sim_utils,
        source_spec,
    )

    assert path == source_spec.prim_path
    assert collider_count == 8
    np.testing.assert_allclose(
        captured,
        [(0.0, 0.0, 0.0, 1.0)],
    )


def test_rigid_b320_uses_thirty_hz_control_with_four_physics_substeps() -> None:
    env = _rigid_env("t10_lift_carton")

    assert env.hz == B320_CONTROL_HZ
    assert env.control_dt == pytest.approx(1.0 / B320_CONTROL_HZ)
    assert env.physics_dt == pytest.approx(1.0 / 120.0)
    assert env._n_substeps == 4


def test_bc004_rejects_environment_control_dt_override(monkeypatch) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_CONTROL_DT", str(1.0 / 60.0))

    with pytest.raises(RuntimeError, match="30/120 Hz"):
        _rigid_env("t10_lift_carton")


def test_b320_final_rigid_reset_pose_is_written_to_authored_reset_journal(
    monkeypatch,
) -> None:
    class FakeView:
        count = 1

        def __init__(self) -> None:
            self.transforms = np.asarray(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            self.velocities = np.ones((1, 6), dtype=np.float32)

        def get_transforms(self):
            return self.transforms.copy()

        def get_velocities(self):
            return self.velocities.copy()

        def set_transforms(self, values, _indices) -> None:
            self.transforms = np.asarray(values, dtype=np.float32).copy()

        def set_velocities(self, values, _indices) -> None:
            self.velocities = np.asarray(values, dtype=np.float32).copy()

    monkeypatch.setattr(
        b320_env_module,
        "_array_like",
        lambda _reference, values, _torch: np.asarray(
            values,
            dtype=np.float32,
        ).copy(),
    )
    monkeypatch.setattr(
        b320_env_module,
        "_indices_like",
        lambda _reference, count, _torch: np.arange(count, dtype=np.int32),
    )

    env = _rigid_env("t10_lift_carton")
    view = FakeView()
    path = "/World/B320/carton"
    env._torch = object()
    env._task_body_views = {"carton": view}
    env._task_reset_body_states = [
        (path, view, np.asarray([0]), view.transforms.copy(), view.velocities.copy())
    ]
    env._recording_origin_reset_write_capture_active = True
    env._recording_origin_authored_task_reset_writes = {}
    requested_pose_wxyz = np.asarray(
        [-0.30, 0.0, 0.89, 0.5, 0.1, 0.2, 0.3],
        dtype=np.float64,
    )

    assert env._set_task_body_world_pose(view, requested_pose_wxyz) is True

    expected_transform_xyzw = np.asarray(
        [-0.30, 0.0, 0.89, 0.1, 0.2, 0.3, 0.5],
        dtype=np.float64,
    )
    np.testing.assert_allclose(view.transforms[0], expected_transform_xyzw)
    np.testing.assert_allclose(view.velocities, 0.0)
    journal = env._recording_origin_authored_task_reset_writes[path]
    np.testing.assert_allclose(journal["transform_xyzw"], expected_transform_xyzw)
    np.testing.assert_allclose(journal["velocity"], 0.0)


def test_b320_catalog_assets_and_exact_furniture_slots_are_complete() -> None:
    assert set(B320_TASK_NAMES) == set(EXPECTED_TASKS)
    assert B320_RIGID_TASK_NAMES == set(EXPECTED_TASKS) - {B320_TOWEL_TASK_NAME}
    assert set(EXPECTED_TASKS) & set(CONFIG_MAPPING) <= set(ISAACSIM_COLLECTION_TASKS)
    assert ISAACSIM_COLLECTION_UNVERIFIED_TASKS <= ISAACSIM_COLLECTION_TASKS
    assert set(EXPECTED_TASKS) <= set(supported_native_collection_tasks())

    required_assets = validate_b320_asset_files()
    assert all(path.is_file() and path.stat().st_size > 0 for path in required_assets)
    required_relative = {
        path.relative_to(b320_scene_module.B320_ASSET_DIRECTORY).as_posix()
        for path in required_assets
    }
    retired_bundle_name = "dv_storage_drawers_" + "40417"
    assert not any(
        relative.startswith(f"{retired_bundle_name}/")
        for relative in required_relative
    )
    assert {
        "dv_double_cabinet_44781/mobility.usd",
        "dv_double_cabinet_44781/meta.json",
    } <= required_relative
    assert retired_bundle_name not in B320_SOURCE_BUNDLE_SHA256
    assert "dv_double_cabinet_44781" in B320_SOURCE_BUNDLE_SHA256
    assert B320_SOURCE_BUNDLE_SHA256[BC014_ASSET_BUNDLE_KEY] == (
        "efba30ea08e24bbed9b5d638a62ad274c6cba33ef418c6d1a71f18a7c83a1412"
    )
    assert "dexverse_spice_salt" not in B320_SOURCE_BUNDLE_SHA256
    assert "dexverse_spice_pepper" not in B320_SOURCE_BUNDLE_SHA256

    for task_name, (task_id, time_limit) in EXPECTED_TASKS.items():
        spec = B320_TASK_SPECS[task_name]
        assert (spec.task_id, spec.time_limit_seconds) == (task_id, time_limit)
        if task_name in CONFIG_MAPPING:
            config = CONFIG_MAPPING[task_name]
            assert config.task_name == task_name
            assert config.time_limit_seconds == time_limit
        expected_label_count = (
            10 if task_name.startswith("eval_comb") else 6 if task_name.startswith("eval_gen") else
            8
            if task_name == "t05_fill_cup"
            else 6
            if task_name == "t38_set_dual_knobs"
            else 4
            if task_name
            in {
                "t20_pass_and_orient_pen",
                B320_BC014_TASK_NAME,
                "t01_align_blocks_in_midair",
            }
            else 6
            if task_name == "b320_synchronized_double_cabinet_open"
            else 5
        )
        assert len(spec.success_labels) == expected_label_count

    pending = B320_TASK_SPECS[B320_BC017_TASK_NAME]
    pending_config = CONFIG_MAPPING[B320_BC017_TASK_NAME]
    assert pending.prompt == pending_config.task_prompt == BC017_REQUIREMENT
    assert pending.fixture_kind == "bc017_dedicated_runtime"
    assert (pending.left_role, pending.right_role) == (
        "left_drawer",
        "right_drawer",
    )
    assert pending.stable_frames == 1
    assert pending.object_specs == ()
    assert pending.source_asset_keys == ("cc0_polyhaven_metal_office_desk",)
    assert pending.dof_names == BC017_DOF_NAMES
    assert pending.success_labels[0] == (
        "All six drawer DOFs, twelve physical contact views, four selected "
        "semantic contact channels, and fixed-fixture truth are readable"
    )
    assert B320_BC017_TASK_NAME not in ISAACSIM_COLLECTION_UNVERIFIED_TASKS
    assert B320_BC017_TASK_NAME in ISAACSIM_FORMAL_COLLECTION_TASKS

    metadata = b320_towel_surface_metadata()
    assert metadata["sim_vertex_count"] == BC025_TOWEL_NODE_COUNT == 1681
    assert metadata["sim_triangle_count"] == BC025_TOWEL_TRIANGLE_COUNT == 3200


def test_bc017_shared_native_spec_uses_only_spawned_physical_roles() -> None:
    contract = B320_TASK_SPECS[B320_BC017_TASK_NAME]
    assert contract.dof_names == BC017_DOF_NAMES
    assert len(contract.dof_names) == 6

    native = _TASK_SPECS[B320_BC017_TASK_NAME]
    assert native.target_objects == (
        *BC017_DRAWER_ROLES_BY_SIDE["left"],
        *BC017_DRAWER_ROLES_BY_SIDE["right"],
    )
    assert native.receptacle == "fixture"
    assert not {"left_drawer", "right_drawer", "fixture_base"}.intersection(
        {*native.target_objects, native.receptacle}
    )


def test_cabinet_visual_dependency_closure_fails_closed(tmp_path, monkeypatch) -> None:
    paths = b320_scene_module.validate_b320_visual_dependencies()
    assert paths and all(path.is_file() for path in paths)
    monkeypatch.setattr(b320_scene_module, "_furniture_texture_paths", lambda _: (tmp_path / "missing.png",))
    with pytest.raises(FileNotFoundError, match="visual dependency closure"):
        b320_scene_module.validate_b320_visual_dependencies()


@pytest.mark.parametrize("task_name", sorted(INTEGRATED_RIGID_TASK_NAMES))
def test_rigid_b320_interface_contract_is_fixed_and_fail_closed(task_name: str) -> None:
    env = _rigid_env(task_name)
    observation, info = env.reset(seed=7)

    assert env.action_space.shape == (54,)
    assert env.observation_space.contains(observation)
    assert env._compute_success_now() is False
    expected_rows = (
        11 if task_name.startswith("eval_comb") else 7 if task_name.startswith("eval_gen") else
        4
        if task_name == "t01_align_blocks_in_midair"
        else 9
        if task_name == "t05_fill_cup"
        else 5
        if task_name
        in {
            "t10_lift_carton",
            "t20_pass_and_orient_pen",
            B320_BC014_TASK_NAME,
        }
        else 7
        if task_name == "t38_set_dual_knobs"
        else 7
        if task_name == "b320_synchronized_double_cabinet_open"
        else 6
    )
    assert len(env._success_condition_rows()) == expected_rows
    assert info["task_name"] == task_name
    assert env.get_physical_replay_reset_contract()[
        "b320_physical_replay_certified"
    ] is False

    with pytest.raises(ValueError, match="hand_type='wuji1'|XY6 requires"):
        HandsOnB320BimanualIsaacLabEnv(
            task_name,
            hand_type="wuji2",
            launch_isaac=False,
            image_obs=False,
        )


def _assert_nested_finite(value: object) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_nested_finite(child)
        return
    if isinstance(value, (tuple, list)):
        for child in value:
            _assert_nested_finite(child)
        return
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        # Metadata may contain opaque objects; only numeric leaves are part
        # of the finite observation/diagnostic contract.
        return
    if array.dtype.kind not in "biufc":
        return
    if array.size:
        assert np.all(np.isfinite(array)), f"non-finite B320 observation: {array!r}"


@pytest.mark.parametrize("task_name", sorted(INTEGRATED_RIGID_TASK_NAMES))
def test_rigid_b320_reset_and_zero_action_are_finite_and_not_spuriously_terminal(
    task_name: str,
) -> None:
    """The interface-only contract must not emit inf sentinels or false failure."""

    env = _rigid_env(task_name)
    observation, _ = env.reset(seed=7)
    _assert_nested_finite(observation)
    next_observation, _reward, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float64)
    )
    _assert_nested_finite(next_observation)
    _assert_nested_finite(info)
    assert terminated is False
    assert truncated is False
    assert info["succeed"] is False


def test_b320_towel_reset_and_zero_action_are_finite_and_not_terminal() -> None:
    env = HandsOnB320TowelIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    observation, _ = env.reset(seed=7)
    _assert_nested_finite(observation)
    next_observation, _reward, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float64)
    )
    _assert_nested_finite(next_observation)
    _assert_nested_finite(info)
    assert terminated is False
    assert truncated is False
    assert info["succeed"] is False


@pytest.mark.parametrize("field", ["_e_nodes_np", "_e_base_nodes_np", "_e_velocities_np"])
def test_b320_towel_invalid_nodal_readback_keeps_observation_finite(
    field: str,
) -> None:
    """Malformed deformable tensors must fail closed without NaN leakage."""

    env = HandsOnB320TowelIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    observation, _ = env.reset(seed=7)
    value = getattr(env, field).copy()
    value.flat[0] = np.nan
    setattr(env, field, value)

    next_observation = env._compute_observation()
    _assert_nested_finite(next_observation)
    assert env.observation_space.contains(next_observation)
    assert next_observation["state"]["spread_metrics"][6] == 0.0


def test_b320_towel_extreme_finite_nodes_cannot_overflow_pose_observation() -> None:
    env = HandsOnB320TowelIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    env.reset(seed=7)
    env._e_nodes_np = np.full_like(env._e_nodes_np, 1.0e308)
    observation = env._compute_observation()
    _assert_nested_finite(observation)
    assert env.observation_space.contains(observation)
    assert observation["state"]["towel_pose"][3] == 1.0
    assert observation["state"]["spread_metrics"][6] == 0.0


def test_bc025_contact_witness_requires_local_physx_response() -> None:
    """Proximity alone cannot arm the BC025 cloth assist."""

    env = HandsOnB320TowelIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    points = env._e_nodes_np.copy()
    assigned = points[1640].copy()
    proxy = np.asarray(
        [assigned[0], assigned[1], assigned[2] + 0.034, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )

    no_response, no_response_diagnostics = (
        _detect_bc025_surface_contact_response(
            points,
            points.copy(),
            np.zeros_like(points),
            proxy,
            proxy.copy(),
            proxy_radius=0.032,
            hand_closed=True,
        )
    )
    assert no_response is False
    assert no_response_diagnostics["reason"] == "awaiting_surface_response"

    response_nodes = points.copy()
    response_nodes[1640, 2] += 0.001
    response_velocities = np.zeros_like(points)
    response_velocities[1640, 2] = 0.12
    response_proxy = proxy.copy()
    response_proxy[2] -= 0.004
    response, response_diagnostics = _detect_bc025_surface_contact_response(
        response_nodes,
        points,
        response_velocities,
        response_proxy,
        proxy,
        proxy_radius=0.032,
        hand_closed=True,
    )
    assert response is True
    assert response_diagnostics["reason"] == "physical_surface_response"


def test_bc025_wrong_corner_response_is_not_an_assigned_success_event() -> None:
    env = HandsOnB320TowelIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    nodes = env._e_nodes_np.copy()
    previous = nodes.copy()
    velocities = np.zeros_like(nodes)
    # Simulate a physically responding left proxy at the lower-right corner.
    wrong_corner = nodes[40].copy()
    response_nodes = nodes.copy()
    response_nodes[40, 2] += 0.001
    velocities[40, 2] = 0.12
    proxy = np.asarray(
        [wrong_corner[0], wrong_corner[1], wrong_corner[2] + 0.030, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    env._contact_proxy_last_pose["left"] = proxy.copy()
    env._cloth_contact_assist_last_proxy_pose["left"] = proxy.copy()
    env._detached_motion_target_hand["left"] = env._detached_sim_hand[
        "left"
    ].copy()
    env._detached_motion_target_hand["left"][0] += 0.12
    env._bc025_previous_physx_nodes = previous

    env._prepare_cloth_contact_assist(response_nodes, velocities)

    assert env._bc025_physical_contact_observed["left"] is False
    assert env._bc025_physical_contact_armed["left"] is False
    assert env._b320_wrong_corner_contact_seen["left"] is True


def test_rigid_b320_success_predicates_have_reachable_true_branches() -> None:
    # BC004 carton.
    env = _rigid_env("t10_lift_carton")
    _prime_contacts(env)
    _lift_carton(env)
    env._b320_carton_airborne_substeps = (
        b320_env_module._CARTON_AIRBORNE_REQUIRED_SUBSTEPS
    )
    assert all(env._carton_metrics()[1])

    # BC011 handover ordering and load transition.
    env = _rigid_env("t20_pass_and_orient_pen")
    _prime_contacts(env)
    env._poses["pen"] = _pen_tip_up_pose()
    env._velocity = lambda _name: np.zeros(6)
    env._b320_handover_right_first = True
    env._b320_handover_overlap = True
    env._b320_handover_left_only = True
    env._b320_contact_force = {"left": 2.0, "right": 0.0}
    assert all(env._handover_metrics()[1])

    # BC014 synchronized seasoning streams.
    env = _rigid_env(B320_BC014_TASK_NAME)
    _prime_contacts(env)
    salt_x, salt_y = BC014_GOAL_CENTERS_XY["salt_jar"]
    pepper_x, pepper_y = BC014_GOAL_CENTERS_XY["pepper_jar"]
    env._poses["salt_jar"] = np.asarray(
        [salt_x, salt_y, 1.10, 0.0, 1.0, 0.0, 0.0]
    )
    env._poses["pepper_jar"] = np.asarray(
        [pepper_x, pepper_y, 1.10, 0.0, 1.0, 0.0, 0.0]
    )
    for _ in range(
        int(
            np.ceil(
                b320_env_module._BC014_FLOW_REQUIRED_SECONDS
                / env._control_dt
            )
        )
    ):
        _, flags = env._seasoning_metrics(update=True)
    assert all(flags)

    # FU001 cabinet retains its exact source joint contract.
    env = _rigid_env("b320_synchronized_double_cabinet_open")
    _prime_contacts(env)
    env._dof_fraction = lambda _name: (0.90, True)
    env._b320_threshold_frame = {"left": 0, "right": 0}
    env._b320_release_frame = {"left": 1, "right": 2}
    assert all(env._double_cabinet_metrics()[1])

    # KA012 full stirring history with a stationary pot.
    env = _rigid_env("t45_stir_pot")
    _prime_contacts(env)
    oven = np.asarray([-0.15, 0.0, 0.94, 1.0, 0.0, 0.0, 0.0])
    env._poses["dutch_oven"] = oven
    env._poses["spoon"] = np.asarray([-0.10, 0.0, 1.04, 1.0, 0.0, 0.0, 0.0])
    env._b320_reset_poses["dutch_oven"] = oven.copy()
    env._b320_stir_sectors = set(range(6))
    env._b320_stir_accumulated_rad = np.deg2rad(700.0)
    assert all(env._stir_metrics()[1])

    # KA020 physical pump and KA021 real faucet joint.
    env = _rigid_env("t32_press_soap_pump")
    _prime_contacts(env)
    sponge = np.asarray([0.0, 0.0, 0.90, 1.0, 0.0, 0.0, 0.0])
    env._poses["sponge"] = sponge
    env._poses["soap_dispense_pad"] = np.asarray(
        [0.0, 0.0, 0.95, 1.0, 0.0, 0.0, 0.0]
    )
    env._b320_reset_poses["sponge"] = sponge.copy()
    # Sponge thickness is local Y; place the nozzle 50 mm above that surface.
    env._poses["soap_dispense_pad"][:3] = sponge[:3] + np.array(
        [0.0, b320_scene_module.KA020_SPONGE_THICKNESS_M + 0.05, 0.0]
    ) - np.asarray(b320_scene_module.KA020_PUMP_OUTLET_OFFSET_M)
    env._ka020_pump_contact_readable = True
    env._ka020_pump_contact_force_n = 2.0
    env._dof_fraction = lambda _name: (0.80, True)
    assert all(env._soap_metrics()[1])

    env = _rigid_env("t05_fill_cup")
    _prime_contacts(env)
    env._poses["ceramic_mug"] = np.asarray(
        [
            *b320_scene_module.KA021_OUTLET_POSITION_W[:2],
            b320_scene_module.KA021_OUTLET_POSITION_W[2]
            - b320_scene_module.KA021_CUP_RIM_HEIGHT_M
            - 0.12,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    env._dof_fraction = lambda _name: (0.80, True)
    for _ in range(env._success_steps_required()):
        fill_flags = env._faucet_metrics()[1]
    assert all(fill_flags[:6])
    assert fill_flags[6:] == (False, False)
    env._poses["ceramic_mug"] = np.asarray(
        [*b320_scene_module.KA021_CUP_PARK_POSITION_W, 1.0, 0.0, 0.0, 0.0]
    )
    env._dof_fraction = lambda _name: (0.0, True)
    terminal_flags = env._faucet_metrics()[1]
    assert terminal_flags[6:] == (True, True)
    assert all(terminal_flags)


def test_bc011_anywhere_contact_and_single_frame_tip_up_success_contract() -> None:
    contract = B320_TASK_SPECS["t20_pass_and_orient_pen"]
    pen = contract.object_specs[0]
    salt = B320_TASK_SPECS[
        "t30_pour_seasonings"
    ].object_specs[0]

    assert (pen.static_friction, pen.dynamic_friction) == (2.5, 2.0)
    assert (salt.static_friction, salt.dynamic_friction) == (1.2, 1.0)
    assert contract.stable_frames == 1
    np.testing.assert_allclose(
        b320_env_module._axis_world(
            np.asarray(contract.goal_pose_wxyz, dtype=np.float64),
            BC011_PEN_TIP_AXIS_LOCAL[BC011_PEN_VARIANTS[0]],
        ),
        (0.0, 0.0, 1.0),
        atol=1.0e-12,
    )

    env = _rigid_env(contract.task_name)
    _prime_contacts(env)
    _freeze_injected_contacts(env)
    env._poses["pen"] = _pen_tip_up_pose(error_degrees=1.99)
    # Contact location and object speed are not final success gates. Arrival at
    # the complete ordered tip-up state is an immediate latch.
    env._velocity = lambda _name: np.asarray(
        [3.0, 0.0, 0.0, 0.0, 8.0, 0.0]
    )
    env._b320_handover_right_first = True
    env._b320_handover_overlap = True
    env._b320_handover_left_only = True
    env._b320_contact_force = {"left": 1.0e-12, "right": 0.0}

    assert all(env._handover_metrics()[1])
    env._poses["pen"] = _pen_tip_up_pose(error_degrees=2.0)
    assert all(env._handover_metrics()[1])
    assert env._success_steps_required() == 1
    assert env._compute_success() is True
    assert env._success_counter == 1
    assert env._b320_handover_completed is True

    # The explicit terminal latch remains true if a caller samples again after
    # the successful terminal frame; reset clears it for the next episode.
    env._poses["pen"] = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    env._b320_contact_force = {"left": 0.0, "right": 2.0}
    assert env._compute_success() is True
    assert env._b320_handover_completed is True
    env._sample_reset_state(clear_manual_request=True)
    assert env._b320_handover_completed is False


def test_bc011_live_angle_guidance_is_collection_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        b320_env_module.BC011_COLLECTION_ANGLE_GUIDANCE_ENV,
        "1",
    )
    env = _rigid_env("t20_pass_and_orient_pen")
    env._bc011_pen_variant = BC011_PEN_VARIANTS[0]
    env._poses["pen"] = _pen_tip_up_pose(error_degrees=1.5)

    metrics, _flags = env._handover_metrics()
    assert metrics["tip_elevation_deg"] == pytest.approx(88.5)
    assert env._task_status_text() == "笔尖仰角 88.5°  |  目标 90° ± 2°"

    monkeypatch.delenv(
        b320_env_module.BC011_COLLECTION_ANGLE_GUIDANCE_ENV,
        raising=False,
    )
    replay_env = _rigid_env("t20_pass_and_orient_pen")
    replay_env._bc011_pen_variant = BC011_PEN_VARIANTS[0]
    replay_env._poses["pen"] = _pen_tip_up_pose(error_degrees=1.5)
    assert replay_env._task_status_text() == ""


@pytest.mark.parametrize(
    ("variant", "tip_axis_x"),
    [
        (BC011_PEN_VARIANTS[0], -1.0),
        (BC011_PEN_VARIANTS[1], 1.0),
    ],
)
def test_bc011_tip_up_predicate_tracks_randomized_asset_tip_axis(
    variant: str,
    tip_axis_x: float,
) -> None:
    env = _rigid_env("t20_pass_and_orient_pen")
    _prime_contacts(env)
    _freeze_injected_contacts(env)
    env._bc011_pen_variant = variant
    env._b320_handover_right_first = True
    env._b320_handover_overlap = True
    env._b320_handover_left_only = True
    env._b320_contact_force = {"left": 1.0e-12, "right": 0.0}
    env._velocity = lambda _name: np.zeros(6, dtype=np.float64)

    env._poses["pen"] = _pen_tip_up_pose(tip_axis_x=tip_axis_x)
    metrics, flags = env._handover_metrics()
    assert metrics["tip_axis_local"] == (tip_axis_x, 0.0, 0.0)
    assert metrics["tip_axis_readable"] is True
    assert all(flags)

    env._poses["pen"] = _pen_tip_up_pose(tip_axis_x=-tip_axis_x)
    metrics, flags = env._handover_metrics()
    assert metrics["tip_up_error_deg"] == pytest.approx(180.0)
    assert flags[-1] is False

    env._bc011_pen_variant = "unknown-pen.usda"
    metrics, flags = env._handover_metrics()
    assert metrics["tip_axis_readable"] is False
    assert flags[-1] is False


def test_bc011_grip_material_is_bound_with_max_combine() -> None:
    created: list[tuple[str, dict[str, object]]] = []
    bindings: list[tuple[str, str, bool]] = []

    class FakeMaterialCfg:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def func(self, path: str, cfg) -> None:
            created.append((path, dict(cfg.kwargs)))

    fake_sim_utils = SimpleNamespace(
        RigidBodyMaterialCfg=FakeMaterialCfg,
        bind_physics_material=lambda prim, material, **kwargs: bindings.append(
            (prim, material, kwargs["stronger_than_descendants"])
        ),
    )

    b320_scene_module._bind_material(
        "/World/Task/B320Benchmark/pen",
        sim_utils=fake_sim_utils,
        static_friction=2.5,
        dynamic_friction=2.0,
    )

    assert created == [
        (
            "/World/Task/B320Benchmark/pen/physicsMaterial",
            {
                "static_friction": 2.5,
                "dynamic_friction": 2.0,
                "restitution": 0.0,
                "friction_combine_mode": "max",
                "restitution_combine_mode": "min",
            },
        )
    ]
    assert bindings == [
        (
            "/World/Task/B320Benchmark/pen",
            "/World/Task/B320Benchmark/pen/physicsMaterial",
            True,
        )
    ]


def test_bc011_step_latches_terminal_success_without_writing_pen_pose() -> None:
    env = _rigid_env("t20_pass_and_orient_pen")
    env.hz = 0
    _prime_contacts(env)
    _freeze_injected_contacts(env)
    env._b320_reset_poses["pen"] = env._fallback_b320_pose("pen")
    env._poses["pen"] = _pen_tip_up_pose(error_degrees=1.0)
    env._velocity = lambda _name: np.zeros(6)
    env._b320_handover_right_first = True
    env._b320_handover_overlap = True
    env._b320_handover_left_only = True
    env._b320_contact_force = {"left": 2.0, "right": 0.0}
    task_body_writes: list[tuple[object, object]] = []
    env._set_task_body_world_pose = lambda view, pose: (
        task_body_writes.append((view, pose)),
        True,
    )[1]

    _, reward, terminated, truncated, info = env.step(
        np.zeros(54, dtype=np.float64)
    )

    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info["succeed"] is True
    assert info["b320_handover_completed"] is True
    assert task_body_writes == []


def test_bc011_rejects_orientation_contact_and_release_near_misses() -> None:
    env = _rigid_env("t20_pass_and_orient_pen")
    _prime_contacts(env)
    env._b320_handover_right_first = True
    env._b320_handover_overlap = True
    env._b320_handover_left_only = True
    env._b320_contact_force = {"left": 2.0, "right": 0.0}

    env._poses["pen"] = _pen_tip_up_pose(error_degrees=2.01)
    assert env._handover_metrics()[1][3] is False

    env._poses["pen"] = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert env._handover_metrics()[1][3] is False

    env._poses["pen"] = _pen_tip_up_pose(error_degrees=1.99)
    env._b320_contact_force["right"] = 1.0e-12
    assert env._handover_metrics()[1][3] is False

    env._b320_contact_force = {"left": 0.0, "right": 0.0}
    assert env._handover_metrics()[1][3] is False

    env._b320_contact_force = {"left": 1.0e-12, "right": 0.0}
    assert env._handover_metrics()[1][3] is True

    env._b320_contact_force = {"left": 2.0, "right": 0.0}
    env._poses["pen"] = _pen_tip_up_pose(error_degrees=1.99)
    dropped_pose = _pen_tip_up_pose(error_degrees=0.0)
    dropped_pose[2] = env._fallback_b320_pose("pen")[2]
    env._poses["pen"] = dropped_pose
    assert env._handover_metrics()[1][3] is False

    env._poses["pen"] = _pen_tip_up_pose(error_degrees=0.0)
    env._b320_pen_table_contact_force = (
        b320_env_module._BC011_TABLE_CONTACT_MAX_N + 1.0e-6
    )
    assert env._handover_metrics()[1][3] is False


def test_bc011_velocity_readback_is_diagnostic_not_a_success_gate() -> None:
    env = _rigid_env("t20_pass_and_orient_pen")
    _prime_contacts(env)
    env._poses["pen"] = _pen_tip_up_pose()
    env._b320_handover_right_first = True
    env._b320_handover_overlap = True
    env._b320_handover_left_only = True
    env._b320_contact_force = {"left": 2.0, "right": 0.0}
    env._launch_isaac = True

    values, flags = env._handover_metrics()

    assert values["velocity_readable"] is False
    assert all(flags)


def test_bc011_pen_table_contact_channel_is_read_and_fail_closed() -> None:
    env = _rigid_env("t20_pass_and_orient_pen")
    env._b320_contact_views = {
        "left": _CartonContactView(2.0, []),
        "right": _CartonContactView(0.0, []),
    }
    env._b320_pen_table_contact_view = _CartonContactView(0.03, [])

    env._read_contacts(update_latches=False)

    assert env._b320_pen_table_contact_available is True
    assert env._b320_pen_table_contact_readable is True
    assert env._b320_pen_table_contact_force == pytest.approx(0.03)

    env._b320_pen_table_contact_view = _InvalidContactView()
    env._read_contacts(update_latches=False)
    assert env._b320_pen_table_contact_readable is False


def test_bc011_requires_right_only_then_both_then_left_only_contact() -> None:
    env = _rigid_env("t20_pass_and_orient_pen")
    env._velocity = lambda _name: np.zeros(6)
    left = _CartonContactView(0.2, [])
    right = _CartonContactView(0.2, [])
    env._b320_contact_views = {"left": left, "right": right}
    env._poses["pen"] = _pen_tip_up_pose()

    env._read_contacts(update_latches=True)
    assert env._b320_handover_right_first is False
    assert env._b320_handover_overlap is False

    # Once left contact has occurred first or simultaneously, later right-only
    # contact cannot repair the wrong-order episode.
    left.force = 0.0
    env._read_contacts(update_latches=True)
    assert env._b320_handover_right_first is False

    env = _rigid_env("t20_pass_and_orient_pen")
    env._velocity = lambda _name: np.zeros(6)
    # Arbitrary contact points model the pen tip and pen body. BC011 consumes
    # only whole-pen aggregate force, so neither location is filtered out.
    left = _CartonContactView(0.0, [(9.0, 8.0, 7.0)])
    right = _CartonContactView(1.0e-12, [(-9.0, -8.0, -7.0)])
    env._b320_contact_views = {"left": left, "right": right}
    env._poses["pen"] = env._fallback_b320_pose("pen").copy()
    env._read_contacts(update_latches=True)
    assert env._b320_handover_right_first is True

    env._velocity = lambda _name: np.asarray(
        [0.0, 0.0, 0.0, 0.0, 8.0, 0.0]
    )
    left.force = 1.0e-12
    env._read_contacts(update_latches=True)
    assert env._b320_handover_overlap is True
    assert env._b320_handover_left_only is False

    left.force = 1.0e-12
    right.force = 1.0e-12
    env._read_contacts(update_latches=True)
    assert env._b320_handover_left_only is False

    right.force = 0.0
    env._read_contacts(update_latches=True)
    assert env._b320_handover_left_only is True


def test_bc011_post_overlap_ground_contact_latches_drop_and_resets_state() -> None:
    env = _rigid_env("t20_pass_and_orient_pen")
    _prime_contacts(env)
    env._b320_handover_right_first = True
    env._b320_handover_overlap = True
    env._poses["pen"] = _pen_tip_up_pose()
    env._b320_contact_force = {"left": 0.0, "right": 0.2}

    env._update_bc011_drop_latches()
    assert env._bc011_pen_airborne_latched is True
    assert env._bc011_pen_drop_detected is False

    # Releasing both hands in midair is no longer the drop predicate.
    env._b320_contact_force = {"left": 0.0, "right": 0.0}
    for _ in range(3):
        env._update_bc011_drop_latches()
    assert env._bc011_pen_ground_contact_substeps == 0
    assert env._bc011_pen_drop_detected is False

    # Any positive readable pen-table contact after arming counts as ground
    # contact; two solver substeps debounce a one-sample contact glitch.
    env._b320_pen_table_contact_force = 1.0e-12
    env._update_bc011_drop_latches()
    assert env._bc011_pen_ground_contact_substeps == 1
    assert env._bc011_pen_drop_detected is False
    env._update_bc011_drop_latches()
    assert env._bc011_pen_drop_detected is True
    assert env._compute_failure() is True
    assert env._b320_failure_reason == "bc011_pen_dropped"

    env._sample_reset_state(clear_manual_request=True)
    assert env._bc011_pen_airborne_latched is False
    assert env._bc011_pen_ground_contact_substeps == 0
    assert env._bc011_pen_drop_detected is False
    assert env._b320_failure_reason is None


def test_bc011_drop_monitor_arms_only_after_overlap_and_airborne() -> None:
    env = _rigid_env("t20_pass_and_orient_pen")
    _prime_contacts(env)
    env._b320_handover_right_first = True
    env._b320_contact_force = {"left": 0.0, "right": 0.0}

    # Even an unsupported airborne pen does not reset before the required
    # simultaneous two-hand contact stage has happened.
    env._poses["pen"] = _pen_tip_up_pose()
    for _ in range(3):
        env._update_bc011_drop_latches()

    assert env._bc011_pen_airborne_latched is False
    assert env._bc011_pen_drop_detected is False

    # Overlap alone still does not arm the monitor while the pen remains on
    # its initial table support.
    env._b320_handover_overlap = True
    env._poses["pen"] = env._fallback_b320_pose("pen").copy()
    for _ in range(3):
        env._update_bc011_drop_latches()

    assert env._bc011_pen_airborne_latched is False
    assert env._bc011_pen_drop_detected is False


def test_bc011_unreadable_left_contact_cannot_count_as_right_only_touch() -> None:
    env = _rigid_env("t20_pass_and_orient_pen")
    env._poses["pen"] = _pen_tip_up_pose()
    env._b320_contact_views = {
        "left": _InvalidContactView(),
        "right": _CartonContactView(0.2, []),
    }

    env._read_contacts(update_latches=True)

    assert env._b320_contact_readable == {"left": False, "right": True}
    assert env._b320_handover_right_first is False


def test_carton_rejects_same_side_contacts_and_resets_airborne_hold() -> None:
    env = _rigid_env("t10_lift_carton")
    env._b320_reset_poses["carton"] = env._fallback_b320_pose("carton")
    _lift_carton(env)
    _set_carton_contact_views(env, left_y=0.235, right_y=0.180)

    env._read_contacts(update_latches=True)

    assert env._b320_carton_transport_started is True
    assert env._b320_carton_zone_valid == {"left": True, "right": False}
    assert env._b320_carton_transport_contact_lost is True
    assert env._carton_metrics()[1][2] is False
    assert env._compute_failure() is False
    assert env._b320_failure_reason is None


def test_carton_requires_uninterrupted_two_hand_support_during_transport() -> None:
    env = _rigid_env("t10_lift_carton")
    home = env._fallback_b320_pose("carton")
    env._b320_reset_poses["carton"] = home.copy()
    _lift_carton(env)
    _set_carton_contact_views(env)

    env._read_contacts(update_latches=True)
    assert env._b320_carton_transport_started is True
    assert env._b320_carton_transport_contact_lost is False

    right = env._b320_contact_views["right"]
    right.force = 0.0
    env._handson_post_physics_substep_callback(0, 4)

    assert env._b320_carton_transport_contact_lost is True
    assert env._b320_carton_airborne_substeps == 0
    assert env._compute_failure() is False
    assert env._b320_failure_reason is None


def test_bc004_on_table_horizontal_motion_and_old_goal_are_not_success_rules() -> None:
    env = _rigid_env("t10_lift_carton")
    _prime_contacts(env)
    home = env._fallback_b320_pose("carton")
    env._poses["carton"] = home.copy()
    env._poses["carton"][0] += 0.20

    metrics, flags = env._carton_metrics()
    assert metrics["bottom_clearance_m"] == pytest.approx(0.0)
    assert flags[1] is False

    # The removed green target position has no bearing on the new predicate.
    _lift_carton(env, x=-0.48)
    env._b320_carton_airborne_substeps = (
        b320_env_module._CARTON_AIRBORNE_REQUIRED_SUBSTEPS
    )
    assert all(env._carton_metrics()[1])


def test_bc004_airborne_threshold_uses_lowest_transformed_corner() -> None:
    env = _rigid_env("t10_lift_carton")
    _prime_contacts(env)
    threshold = b320_env_module._CARTON_AIRBORNE_CLEARANCE_M

    _lift_carton(env, clearance=threshold - 1.0e-6)
    assert env._carton_instantaneous_state()["airborne"] is False
    _lift_carton(env, clearance=threshold)
    assert env._carton_instantaneous_state()["airborne"] is True

    # A high root is not enough when a rotated lower corner still reaches the
    # support plane.
    angle = np.deg2rad(20.0)
    env._poses["carton"] = np.asarray(
        [
            -0.30,
            0.0,
            0.89 + threshold,
            np.cos(angle / 2.0),
            np.sin(angle / 2.0),
            0.0,
            0.0,
        ]
    )
    state = env._carton_instantaneous_state()
    assert state["bottom_clearance_m"] < threshold
    assert state["airborne"] is False


def test_bc004_table_contact_truth_fails_closed_and_resets_substep_timer() -> None:
    env = _rigid_env("t10_lift_carton")
    _prime_contacts(env)
    _lift_carton(env)
    _freeze_injected_contacts(env)

    env._handson_post_physics_substep_callback(0, 4)
    assert env._b320_carton_airborne_substeps == 1

    env._b320_carton_table_contact_force = 0.03
    env._handson_post_physics_substep_callback(1, 4)
    assert env._b320_carton_airborne_substeps == 0
    assert env._carton_instantaneous_state()["airborne"] is False

    env._b320_carton_table_contact_force = 0.0
    env._b320_carton_table_contact_readable = False
    env._handson_post_physics_substep_callback(2, 4)
    assert env._b320_carton_airborne_substeps == 0


@pytest.mark.parametrize(
    "bad_quaternion",
    (
        np.asarray([0.0, 0.0, 0.0, 0.0]),
        np.asarray([0.5, 0.0, 0.0, 0.0]),
        np.asarray([np.nan, 0.0, 0.0, 0.0]),
    ),
)
def test_bc004_invalid_pose_quaternion_resets_airborne_timer(
    bad_quaternion: np.ndarray,
) -> None:
    env = _rigid_env("t10_lift_carton")
    _prime_contacts(env)
    _lift_carton(env)
    _freeze_injected_contacts(env)
    physics_pose = [env._poses["carton"].copy()]
    env._physics_body_pose = (
        lambda name: physics_pose[0].copy() if name == "carton" else None
    )
    env._handson_post_physics_substep_callback(0, 4)
    assert env._b320_carton_airborne_substeps == 1

    physics_pose[0][3:7] = bad_quaternion
    env._handson_post_physics_substep_callback(1, 4)

    state = env._carton_instantaneous_state()
    assert state["pose_readable"] is False
    assert state["airborne"] is False
    assert env._b320_carton_airborne_substeps == 0


def test_bc004_succeeds_on_90th_complete_control_frame() -> None:
    env = _rigid_env("t10_lift_carton")
    env.hz = 0
    _prime_contacts(env)
    _lift_carton(env)
    _freeze_injected_contacts(env)
    action = np.zeros(env.action_space.shape, dtype=np.float64)

    for frame in range(1, 90):
        _obs, reward, terminated, truncated, info = env.step(action)
        assert reward == 0.0
        assert terminated is False
        assert truncated is False
        assert info["succeed"] is False
        assert env._b320_carton_airborne_substeps == 4 * frame

    _obs, reward, terminated, truncated, info = env.step(action)
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info["succeed"] is True
    assert info["b320_carton_airborne_substeps"] == 360
    assert info["b320_carton_airborne_hold_seconds"] == pytest.approx(3.0)


def test_bc004_touchdown_restarts_the_full_three_second_timer() -> None:
    env = _rigid_env("t10_lift_carton")
    env.hz = 0
    _prime_contacts(env)
    _lift_carton(env)
    _freeze_injected_contacts(env)
    action = np.zeros(env.action_space.shape, dtype=np.float64)

    for _ in range(45):
        assert env.step(action)[2] is False
    assert env._b320_carton_airborne_substeps == 180

    _lift_carton(env, clearance=0.0)
    assert env.step(action)[2] is False
    assert env._b320_carton_airborne_substeps == 0

    _lift_carton(env)
    for _ in range(89):
        assert env.step(action)[2] is False
    assert env._b320_carton_airborne_substeps == 356
    assert env.step(action)[2] is True


def test_bc004_reset_clears_all_airborne_episode_state() -> None:
    env = _rigid_env("t10_lift_carton")
    env._b320_carton_airborne_substeps = (
        b320_env_module._CARTON_AIRBORNE_REQUIRED_SUBSTEPS
    )
    env._b320_carton_airborne_now = True
    env._b320_carton_table_contact_available = True
    env._b320_carton_table_contact_readable = True
    env._b320_carton_table_contact_force = 7.0

    _observation, info = env.reset(seed=13)

    assert env._b320_carton_airborne_substeps == 0
    assert env._b320_carton_airborne_now is False
    assert env._b320_carton_table_contact_available is False
    assert env._b320_carton_table_contact_readable is False
    assert env._b320_carton_table_contact_force == 0.0
    assert info["b320_carton_airborne_substeps"] == 0
    assert info["b320_carton_airborne_now"] is False


def test_bc004_global_ik_replay_preserves_substep_success_observer() -> None:
    from handson.sim.isaaclab.global_ik_trajectory import (
        _install_isaac_substep_dls,
    )

    env = _rigid_env("t10_lift_carton")
    _prime_contacts(env)
    _lift_carton(env)
    _freeze_injected_contacts(env)

    _install_isaac_substep_dls(env)

    assert callable(env._handson_pre_physics_substep_callback)
    assert callable(env._handson_post_physics_substep_callback)
    env._handson_post_physics_substep_callback(0, 4)
    assert env._b320_carton_airborne_substeps == 1


def test_bc004_recorded_drive_replay_preserves_substep_success_observer() -> None:
    from handson.sim.isaaclab.articulated_replay import (
        run_articulated_drive_control_step,
    )

    class Trace:
        @staticmethod
        def apply_robot_joint_targets(_snapshot, _schema) -> int:
            return 1

    env = _rigid_env("t10_lift_carton")
    env.hz = 0
    _prime_contacts(env)
    _lift_carton(env)
    _freeze_injected_contacts(env)
    drive_step = SimpleNamespace(
        substep_capacity=4,
        substep_count=4,
        expected_joint_count=1,
        max_intra_step_target_delta=0.0,
        target_snapshot=lambda _index: {},
    )

    _result, report = run_articulated_drive_control_step(
        env,
        Trace(),
        {},
        drive_step,
    )

    assert report["recorded_substep_count"] == 4
    assert env._b320_carton_airborne_substeps == 4
    assert callable(env._handson_post_physics_substep_callback)
    assert not hasattr(env, "_handson_pre_physics_substep_callback")


def test_bc004_recorded_drive_replay_rejects_replaced_post_callback() -> None:
    from handson.sim.isaaclab.articulated_replay import (
        run_articulated_drive_control_step,
    )

    env = _rigid_env("t10_lift_carton")
    env._handson_post_physics_substep_callback = lambda *_: None
    drive_step = SimpleNamespace(
        substep_capacity=4,
        substep_count=4,
        expected_joint_count=1,
        max_intra_step_target_delta=0.0,
        target_snapshot=lambda _index: {},
    )

    with pytest.raises(RuntimeError, match="cannot share"):
        run_articulated_drive_control_step(
            env,
            SimpleNamespace(apply_robot_joint_targets=lambda *_: 1),
            {},
            drive_step,
        )


@pytest.mark.parametrize(
    ("timing_key", "timing_value"),
    (("control_dt", 1.0 / 60.0), ("physics_dt", 1.0 / 60.0)),
)
def test_bc004_rejects_timing_that_breaks_d08_alignment(
    timing_key: str,
    timing_value: float,
) -> None:
    with pytest.raises(ValueError, match="30 Hz control / 120 Hz physics"):
        HandsOnB320BimanualIsaacLabEnv(
            "t10_lift_carton",
            launch_isaac=False,
            image_obs=False,
            **{timing_key: timing_value},
        )


def test_bc004_exact_12_newton_limit_is_allowed() -> None:
    env = _rigid_env("t10_lift_carton")
    _prime_contacts(env)
    _lift_carton(env)
    env._b320_contact_peak = {"left": 12.0, "right": 12.0}
    env._b320_carton_airborne_substeps = (
        b320_env_module._CARTON_AIRBORNE_REQUIRED_SUBSTEPS
    )
    assert all(env._carton_metrics()[1])
    assert env._compute_failure() is False

    env._b320_contact_peak["left"] = 12.0 + 1.0e-6
    assert env._carton_metrics()[1][3] is False
    assert env._compute_failure() is True
    assert env._b320_failure_reason == "carton_side_force_exceeded"


def test_ka012_randomized_pot_reset_stays_inside_both_arm_workspaces() -> None:
    env = HandsOnB320BimanualIsaacLabEnv(
        "t45_stir_pot",
        launch_isaac=False,
        image_obs=False,
        randomize=True,
    )

    samples = []
    for seed in range(64):
        env._random.seed(seed)
        env._sample_reset_state(clear_manual_request=True)
        sample = env._ka012_reset_pose
        assert sample is not None
        samples.append(sample.copy())

        low, high = (
            np.asarray(value, dtype=np.float64)
            for value in b320_env_module._KA012_POT_CENTER_BOUNDS_XY
        )
        assert np.all(sample[:2] >= low)
        assert np.all(sample[:2] <= high)
        assert sample[2] == pytest.approx(env._fallback_b320_pose("dutch_oven")[2])
        assert np.linalg.norm(sample[3:7]) == pytest.approx(1.0)
        assert env._ka012_pot_reset_is_reachable(sample)
        metrics = env._ka012_pot_reach_metrics(sample)
        assert metrics["left_handle_reach_m"] <= (
            b320_env_module._KA012_COMFORT_MAX_RADIUS_M
        )
        assert metrics["right_max_stir_reach_m"] <= (
            b320_env_module._KA012_COMFORT_MAX_RADIUS_M
        )
        assert b320_env_module._KA012_COMFORT_MAX_RADIUS_M < (
            _ARM_WORKSPACE_MAX_RADIUS
        )
        np.testing.assert_allclose(env._poses["dutch_oven"], sample)
        np.testing.assert_allclose(env._b320_reset_poses["dutch_oven"], sample)

    samples_array = np.stack(samples)
    assert np.ptp(samples_array[:, 0]) > 0.04
    assert np.ptp(samples_array[:, 1]) > 0.06
    assert np.ptp(samples_array[:, 3:7], axis=0).max() > 0.05

    env._random.seed(2408)
    env._sample_reset_state(clear_manual_request=True)
    first = env._ka012_reset_pose.copy()
    metadata = env._ka012_reset_metadata()
    assert metadata["pot_pose_wxyz"] == pytest.approx(first.tolist())
    assert metadata["reachable"] is True
    assert metadata["right_role"] == "complete_stir_path"
    env._random.seed(2408)
    env._sample_reset_state(clear_manual_request=True)
    np.testing.assert_allclose(env._ka012_reset_pose, first)

    # The detached lid and spoon retain their existing small tabletop jitter.
    for object_name in ("dutch_oven_lid", "spoon"):
        home = env._fallback_b320_pose(object_name)
        object_samples = np.stack(
            [
                env._randomized_pose(
                    home,
                    fixture=False,
                    object_name=object_name,
                )
                for _ in range(64)
            ]
        )

        assert np.all(np.abs(object_samples[:, :2] - home[:2]) <= 0.035)
        np.testing.assert_allclose(object_samples[:, 2], home[2])
        np.testing.assert_allclose(
            np.linalg.norm(object_samples[:, 3:7], axis=1),
            np.ones(64),
        )
        assert np.ptp(object_samples[:, 0]) > 0.01
        assert np.ptp(object_samples[:, 1]) > 0.01
        assert np.ptp(object_samples[:, 3:7], axis=0).max() > 0.01


def test_ka012_nonrandomized_pot_home_is_reachable() -> None:
    env = _rigid_env("t45_stir_pot")
    env._sample_reset_state(clear_manual_request=True)
    home = env._fallback_b320_pose("dutch_oven")

    assert home[:2] == pytest.approx((-0.195, 0.08))
    assert env._ka012_pot_reset_is_reachable(home)
    np.testing.assert_allclose(env._ka012_reset_pose, home)


def test_ka012_pot_drift_anchors_when_valid_stirring_begins() -> None:
    env = _rigid_env("t45_stir_pot")
    _prime_contacts(env)
    reset = np.asarray([-0.30, 0.0, 0.94, 1.0, 0.0, 0.0, 0.0])
    grasped = reset.copy()
    grasped[0] += 0.20
    env._b320_reset_poses["dutch_oven"] = reset
    env._poses["dutch_oven"] = grasped.copy()
    env._poses["spoon"] = np.asarray(
        [grasped[0] + 0.05, grasped[1], grasped[2] + 0.10, 1.0, 0.0, 0.0, 0.0]
    )
    env._b320_stir_sectors = set(range(6))
    env._b320_stir_accumulated_rad = np.deg2rad(700.0)

    metrics, flags = env._stir_metrics(update=True)

    assert all(flags)
    assert metrics["stir_anchor_set"] is True
    assert metrics["pot_drift_m"] == pytest.approx(0.0)
    assert metrics["pot_reset_drift_m"] == pytest.approx(0.20)
    np.testing.assert_allclose(env._b320_stir_anchor_pose, grasped)

    env._poses["dutch_oven"][0] += 0.026
    metrics, flags = env._stir_metrics(update=False)
    assert metrics["pot_drift_m"] == pytest.approx(0.026)
    assert flags[-1] is False

    env._poses["dutch_oven"][0] -= 0.002
    metrics, flags = env._stir_metrics(update=False)
    assert metrics["pot_drift_m"] == pytest.approx(0.024)
    assert flags[-1] is True

    metadata = env._ka012_stir_state_metadata()
    assert metadata["anchor_pose_wxyz"] == pytest.approx(grasped.tolist())
    assert metadata["translation_limit_m"] == pytest.approx(0.025)

    env._sample_reset_state(clear_manual_request=True)
    assert env._b320_stir_anchor_pose is None
    assert env._b320_stir_anchor_frame is None


def test_ka021_randomizes_complete_mug_inside_basin_and_rejects_historical_contact() -> None:
    env = HandsOnB320BimanualIsaacLabEnv(
        "t05_fill_cup",
        launch_isaac=False,
        image_obs=False,
        randomize=True,
    )
    home = env._fallback_b320_pose("ceramic_mug")
    samples = []
    for seed in range(256):
        env._random = np.random.RandomState(seed)
        sampled_pose = env._sample_ka021_reset_pose()
        assert sampled_pose is not None
        assert b320_env_module._ka021_cup_reset_fits_basin(sampled_pose[:2])
        samples.append(sampled_pose)
    samples = np.stack(samples)

    reset_low, reset_high = (
        np.asarray(value, dtype=np.float64)
        for value in b320_env_module._KA021_CUP_RESET_CENTER_BOUNDS_XY
    )
    assert np.all(samples[:, :2] >= reset_low)
    assert np.all(samples[:, :2] <= reset_high)
    for x_value in (reset_low[0], reset_high[0]):
        for y_value in (reset_low[1], reset_high[1]):
            assert b320_env_module._ka021_cup_reset_fits_basin(
                np.asarray([x_value, y_value], dtype=np.float64)
            )
    basin_center = np.asarray(
        b320_scene_module.KA021_SAFE_BASIN_FLOOR_CENTER_W[:2],
        dtype=np.float64,
    )
    basin_half_extent = 0.5 * np.asarray(
        b320_scene_module.KA021_SAFE_BASIN_FLOOR_SIZE_M[:2],
        dtype=np.float64,
    )
    reset_envelope_radius = (
        b320_env_module._KA021_CUP_FOOTPRINT_RADIUS_M
        + b320_env_module._KA021_CUP_RESET_EDGE_CLEARANCE_M
    )
    visible_low, visible_high = (
        np.asarray(value, dtype=np.float64)
        for value in b320_scene_module.KA021_VISIBLE_BASIN_OPENING_BOUNDS_XY
    )
    assert np.all(reset_low - reset_envelope_radius >= visible_low)
    assert np.all(reset_high + reset_envelope_radius <= visible_high)
    assert np.all(
        reset_low - reset_envelope_radius
        >= basin_center - basin_half_extent
    )
    assert np.all(
        reset_high + reset_envelope_radius
        <= basin_center + basin_half_extent
    )
    # The hidden support floor continues beneath the ribbed drainboard.  A
    # center there used to pass the floor-only check and caused the mug to
    # appear under or through that board; it must not count as in-basin.
    under_drainboard_xy = np.asarray([-0.15, -0.30], dtype=np.float64)
    assert np.all(under_drainboard_xy >= basin_center - basin_half_extent)
    assert np.all(under_drainboard_xy <= basin_center + basin_half_extent)
    assert not b320_env_module._ka021_cup_reset_fits_basin(
        under_drainboard_xy
    )
    mug_scale = B320_TASK_SPECS[
        "t05_fill_cup"
    ].object_specs[0].scale[0]
    handle_radius = max(
        np.linalg.norm(np.asarray(point[:2], dtype=np.float64) * mug_scale)
        for segment in b320_scene_module.KA021_MUG_HANDLE_CENTERLINE_SOURCE_M
        for point in segment
    ) + b320_scene_module.KA021_MUG_HANDLE_COLLIDER_RADIUS_M
    assert b320_env_module._KA021_CUP_FOOTPRINT_RADIUS_M >= handle_radius
    assert b320_env_module._KA021_CUP_FOOTPRINT_RADIUS_M >= 0.073379
    basin_floor_top_z = (
        b320_scene_module.KA021_SAFE_BASIN_FLOOR_CENTER_W[2]
        + 0.5 * b320_scene_module.KA021_SAFE_BASIN_FLOOR_SIZE_M[2]
    )
    assert samples[:, 2] == pytest.approx(basin_floor_top_z)
    assert home[2] == pytest.approx(basin_floor_top_z)
    assert b320_env_module._ka021_cup_reset_pose_fits_basin(home)
    unsupported_home = home.copy()
    unsupported_home[2] += 0.001
    assert not b320_env_module._ka021_cup_reset_pose_fits_basin(
        unsupported_home
    )
    assert np.ptp(samples[:, 0]) > 0.01
    assert np.ptp(samples[:, 1]) > 0.01

    contract = B320_TASK_SPECS["t05_fill_cup"]
    assert contract.success_labels[-2:] == (
        "After filling, mug is parked on the marked counter pad outside the sink",
        "Faucet is closed after the mug is parked",
    )
    metadata = env._ka021_faucet_metadata()
    assert metadata["post_fill_requires_cup_park"] is True
    assert metadata["post_fill_requires_faucet_closed"] is True
    assert metadata["cup_reset_requires_complete_mug_inside_basin"] is True
    assert metadata["cup_reset_complete_visual_footprint_radius_m"] == pytest.approx(
        b320_env_module._KA021_CUP_FOOTPRINT_RADIUS_M
    )
    assert metadata["cup_reset_basin_edge_clearance_m"] == pytest.approx(
        b320_env_module._KA021_CUP_RESET_EDGE_CLEARANCE_M
    )
    assert metadata["cup_reset_basin_floor_top_z_m"] == pytest.approx(
        basin_floor_top_z
    )
    assert metadata["cup_reset_visible_basin_opening_bounds_xy_m"] == [
        list(values)
        for values in b320_scene_module.KA021_VISIBLE_BASIN_OPENING_BOUNDS_XY
    ]
    assert metadata["cup_reset_excludes_drainboard"] is True

    env._poses["ceramic_mug"] = np.asarray(
        [
            *b320_scene_module.KA021_OUTLET_POSITION_W[:2],
            b320_scene_module.KA021_OUTLET_POSITION_W[2]
            - b320_scene_module.KA021_CUP_RIM_HEIGHT_M
            - 0.12,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    env._dof_fraction = lambda _name: (0.80, True)
    _prime_contacts(env)
    fill_flags = env._faucet_metrics()[1]
    assert all(fill_flags[:5])
    assert fill_flags[5] is False
    assert fill_flags[6:] == (False, False)

    env._b320_contact_seen["left"] = True
    env._b320_contact_force["left"] = 0.0
    flags = env._faucet_metrics()[1]
    assert flags[0] is False
    assert flags[-1] is False


def test_bc004_d08_alignment_contract_is_explicit_and_geometry_safe() -> None:
    contract = B320_TASK_SPECS["t10_lift_carton"]
    carton = contract.object_specs[0]
    assert B320_TASK_SCENES[contract.task_name] == B320_BC004_SCENE_XML
    assert (carton.linear_damping, carton.angular_damping) == (0.40, 2.00)
    assert carton.sleep_threshold == pytest.approx(0.020)
    assert (carton.static_friction, carton.dynamic_friction) == (1.25, 1.0)
    assert carton.contact_offset == pytest.approx(0.001)
    assert _BC004_CARTON_HAND_STATIC_FRICTION == pytest.approx(3.0)
    assert _BC004_CARTON_HAND_DYNAMIC_FRICTION == pytest.approx(3.0)
    assert _BC004_CARTON_HAND_CONTACT_OFFSET == pytest.approx(0.002)
    assert _BC004_CARTON_HAND_REST_OFFSET == pytest.approx(0.0)
    assert (
        2.0 * 10.0 * _BC004_CARTON_HAND_DYNAMIC_FRICTION
        > carton.mass_kg * 9.81
    )

    bc004_spec = _TASK_SPECS[contract.task_name]
    d08_spec = _TASK_SPECS["t12_load_and_start_microwave"]
    assert bc004_spec.head_eye == d08_spec.head_eye
    assert bc004_spec.head_target == d08_spec.head_target

    teleop = CONFIG_MAPPING[contract.task_name].teleop
    assert teleop == CONFIG_MAPPING["t12_load_and_start_microwave"].teleop
    assert teleop.filter_alpha == pytest.approx(0.55)

    env = HandsOnB320BimanualIsaacLabEnv(
        contract.task_name,
        launch_isaac=False,
        image_obs=False,
        randomize=True,
    )
    home = env._fallback_b320_pose("carton")
    for _ in range(64):
        sampled = env._randomized_pose(home, fixture=False, object_name="carton")
        assert abs(sampled[0] - home[0]) <= 0.025
        assert abs(sampled[1] - home[1]) <= 0.020
        np.testing.assert_allclose(sampled[3:7], home[3:7])

    xml_root = Path(b320_scene_module.__file__).resolve().parent.parent / "envs" / "xmls"
    bc004_xml = ET.parse(xml_root / B320_BC004_SCENE_XML).getroot()
    d08_xml = ET.parse(
        xml_root / "arena_arm_hand_d_long_horizon_rokae_wuji1.xml"
    ).getroot()
    for camera_name in ("front", "back", "left", "right", "top0"):
        bc004_camera = bc004_xml.find(f".//camera[@name='{camera_name}']")
        d08_camera = d08_xml.find(f".//camera[@name='{camera_name}']")
        assert bc004_camera is not None and d08_camera is not None
        for attribute in ("pos", "quat", "fovy"):
            assert bc004_camera.get(attribute) == d08_camera.get(attribute)
    assert bc004_xml.find(".//light").get("diffuse") == d08_xml.find(
        ".//light"
    ).get("diffuse")
    table_rgb = tuple(
        map(
            float,
            d08_xml.find(".//material[@name='dlong_table_material']")
            .get("rgba")
            .split()[:3],
        )
    )
    leg_rgb = tuple(
        map(
            float,
            d08_xml.find(".//material[@name='dlong_leg_material']")
            .get("rgba")
            .split()[:3],
        )
    )
    assert b320_scene_module.B320_D08_TABLE_DIFFUSE == table_rgb
    assert b320_scene_module.B320_D08_LEG_DIFFUSE == leg_rgb


def test_every_b320_task_uses_the_d08_teleop_profile() -> None:
    d08_teleop = CONFIG_MAPPING["t12_load_and_start_microwave"].teleop

    for task_name in set(EXPECTED_TASKS) & set(CONFIG_MAPPING):
        assert CONFIG_MAPPING[task_name].teleop == d08_teleop


def test_invalid_contact_samples_cannot_latch_handover_or_success() -> None:
    env = _rigid_env("t20_pass_and_orient_pen")
    env._b320_contact_views = {
        "left": _InvalidContactView(),
        "right": _InvalidContactView(),
    }
    env._read_contacts(update_latches=True)

    assert env._b320_contact_readable == {"left": False, "right": False}
    assert env._b320_contact_seen == {"left": False, "right": False}
    assert env._b320_handover_right_first is False
    assert env._b320_handover_overlap is False
    assert env._b320_handover_left_only is False
    handover_flags = env._handover_metrics()[1]
    assert handover_flags == (False, False, False, False)
    assert env._metrics(update=False)["success_now"] is False


def test_bc004_physx_force_and_peak_saturate_at_10n() -> None:
    env = _rigid_env("t10_lift_carton")
    _set_carton_contact_views(env, left_force=12.5, right_force=2.0)

    env._handson_post_physics_substep_callback(0, 4)
    assert env._b320_contact_force["left"] == pytest.approx(10.0)
    assert env._b320_contact_peak["left"] == pytest.approx(10.0)
    env._b320_contact_views["left"].force = 2.0
    env._handson_post_physics_substep_callback(1, 4)

    assert env._b320_contact_force["left"] == pytest.approx(2.0)
    assert env._b320_contact_peak["left"] == pytest.approx(10.0)
    assert env._compute_failure() is False
    assert env._b320_failure_reason is None


def test_bc004_recorded_side_forces_saturate_at_10n() -> None:
    env = _rigid_env("t10_lift_carton")
    env._b320_contact_available = {"left": True, "right": True}
    env._b320_contact_readable = {"left": True, "right": True}
    env._b320_contact_force = {"left": 37.0, "right": 8.0}
    env._b320_contact_peak = {"left": 42.0, "right": 9.0}
    env._b320_carton_zone_force = {"left": 31.0, "right": 7.0}
    env._b320_carton_force_guard_predicted_n = {
        "left": 55.0,
        "right": 6.0,
    }

    observation = env._compute_observation()
    contact_metrics = observation["state"]["contact_metrics"]
    metrics = env._carton_metrics()[0]
    info = env._task_info()

    np.testing.assert_allclose(
        contact_metrics[2:6],
        np.asarray([10.0, 8.0, 10.0, 9.0]),
    )
    assert metrics["left_side_force_peak_n"] == pytest.approx(10.0)
    assert metrics["right_side_force_peak_n"] == pytest.approx(9.0)
    assert metrics["side_force_peak_n"] == pytest.approx(10.0)
    assert metrics["left_lower_side_force_n"] == pytest.approx(10.0)
    assert metrics["force_velocity_guard_predicted_left_n"] == pytest.approx(
        10.0
    )
    assert info["b320_contact_force_n"] == {"left": 10.0, "right": 8.0}
    assert info["b320_contact_peak_n"] == {"left": 10.0, "right": 9.0}
    assert info["b320_carton_zone_force_n"] == {
        "left": 10.0,
        "right": 7.0,
    }

def _configure_bc004_force_guard(
    env: HandsOnB320BimanualIsaacLabEnv,
    *,
    left_force: float,
    right_force: float,
) -> tuple[np.ndarray, np.ndarray]:
    env._b320_contact_available = {"left": True, "right": True}
    env._b320_contact_readable = {"left": True, "right": True}
    env._b320_contact_force = {
        "left": float(left_force),
        "right": float(right_force),
    }
    env._poses["carton"] = np.asarray(
        [-0.30, 0.0, 0.89, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    right = np.asarray(
        [-0.30, -0.20, 1.00, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    left = np.asarray(
        [-0.30, 0.20, 1.00, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    env._b320_carton_force_guard_targets = {
        "right": right.copy(),
        "left": left.copy(),
    }
    env._b320_carton_force_guard_hand_targets = {
        "right": env._right_hand.copy(),
        "left": env._left_hand.copy(),
    }
    return right, left


def test_bc004_prelock_guard_blocks_only_upward_motion() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _configure_bc004_force_guard(
        env,
        left_force=0.0,
        right_force=0.0,
    )
    env._right_tcp_pose = right.copy()
    env._left_tcp_pose = left.copy()
    env._right_tcp_pose[1] += 0.010
    env._left_tcp_pose[1] -= 0.010
    env._right_tcp_pose[2] += 0.020
    env._left_tcp_pose[2] += 0.020
    requested_quaternion = np.asarray(
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        dtype=np.float64,
    )
    env._right_tcp_pose[3:7] = requested_quaternion
    env._left_tcp_pose[3:7] = requested_quaternion
    env._right_hand += 1.0
    env._left_hand += 1.0

    requested_right_hand = env._right_hand.copy()
    requested_left_hand = env._left_hand.copy()
    assert env._apply_carton_prelock_lift_guard() is True

    assert env._right_tcp_pose[1] == pytest.approx(right[1] + 0.010)
    assert env._left_tcp_pose[1] == pytest.approx(left[1] - 0.010)
    assert env._right_tcp_pose[2] == pytest.approx(right[2])
    assert env._left_tcp_pose[2] == pytest.approx(left[2])
    np.testing.assert_allclose(
        env._right_tcp_pose[3:7],
        requested_quaternion,
    )
    np.testing.assert_allclose(
        env._left_tcp_pose[3:7],
        requested_quaternion,
    )
    np.testing.assert_allclose(env._right_hand, requested_right_hand)
    np.testing.assert_allclose(env._left_hand, requested_left_hand)


def test_bc004_prelock_guard_releases_lift_at_simultaneous_10n() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _configure_bc004_force_guard(
        env,
        left_force=10.0,
        right_force=10.0,
    )
    env._right_tcp_pose = right.copy()
    env._left_tcp_pose = left.copy()
    env._right_tcp_pose[2] += 0.020
    env._left_tcp_pose[2] += 0.015

    assert env._apply_carton_prelock_lift_guard() is False

    assert env._right_tcp_pose[2] == pytest.approx(right[2] + 0.020)
    assert env._left_tcp_pose[2] == pytest.approx(left[2] + 0.015)


def test_bc004_each_hand_independently_rejects_predicted_over_10n_velocity() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _configure_bc004_force_guard(
        env,
        left_force=2.0,
        right_force=0.0,
    )
    env._right_tcp_pose = right.copy()
    env._left_tcp_pose = left.copy()
    env._right_tcp_pose[1] += 0.010
    env._left_tcp_pose[1] -= 0.010
    env._right_tcp_pose[2] += 0.050
    env._left_tcp_pose[2] += 0.050

    env._apply_carton_force_velocity_guard()

    assert env._b320_carton_force_guard_active is True
    assert env._b320_carton_force_guard_active_by_side == {
        "left": True,
        "right": False,
    }
    assert env._b320_carton_force_guard_clamped == {
        "left": True,
        "right": False,
    }
    assert env._left_tcp_pose[1] == pytest.approx(left[1])
    assert env._right_tcp_pose[1] == pytest.approx(right[1] + 0.010)
    assert env._left_tcp_pose[2] == pytest.approx(left[2] + 0.050)
    assert env._right_tcp_pose[2] == pytest.approx(right[2] + 0.050)


def test_bc004_force_velocity_guard_keeps_outward_displacement_free() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _configure_bc004_force_guard(
        env,
        left_force=2.0,
        right_force=2.0,
    )
    env._right_tcp_pose = right.copy()
    env._left_tcp_pose = left.copy()
    env._right_tcp_pose[1] -= 0.010
    env._left_tcp_pose[1] += 0.010

    env._apply_carton_force_velocity_guard()

    assert env._b320_carton_force_guard_active is True
    assert env._b320_carton_force_guard_clamped == {
        "left": False,
        "right": False,
    }
    assert env._left_tcp_pose[1] == pytest.approx(left[1] + 0.010)
    assert env._right_tcp_pose[1] == pytest.approx(right[1] - 0.010)


def test_bc004_force_velocity_prediction_accepts_or_rejects_whole_component() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _configure_bc004_force_guard(
        env,
        left_force=8.0,
        right_force=0.0,
    )
    env._right_tcp_pose = right.copy()
    env._left_tcp_pose = left.copy()
    env._left_tcp_pose[1] -= 0.0001

    env._apply_carton_force_velocity_guard()

    assert env._b320_carton_force_guard_predicted_n["left"] == pytest.approx(9.0)
    assert env._b320_carton_force_guard_clamped["left"] is False
    assert env._left_tcp_pose[1] == pytest.approx(left[1] - 0.0001)

    env._b320_carton_force_guard_targets = {
        "right": right.copy(),
        "left": left.copy(),
    }
    env._left_tcp_pose = left.copy()
    env._left_tcp_pose[1] -= 0.0003

    env._apply_carton_force_velocity_guard()

    assert env._b320_carton_force_guard_predicted_n["left"] == pytest.approx(11.0)
    assert env._b320_carton_force_guard_clamped["left"] is True
    assert env._left_tcp_pose[1] == pytest.approx(left[1])
    assert env._b320_carton_bilateral_pressure_reached["left"] is False


def test_bc004_recent_force_rise_predictively_rejects_inward_velocity() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _configure_bc004_force_guard(
        env,
        left_force=8.0,
        right_force=0.0,
    )
    env._b320_contact_force_rise_rate_n_s["left"] = 100.0
    env._right_tcp_pose = right.copy()
    env._left_tcp_pose = left.copy()
    env._left_tcp_pose[1] -= 0.00001

    env._apply_carton_force_velocity_guard()

    assert env._b320_carton_force_guard_predicted_n["left"] > 10.0
    assert env._b320_carton_force_guard_clamped["left"] is True
    assert env._left_tcp_pose[1] == pytest.approx(left[1] + 0.0005)
    assert env._b320_carton_force_relief_active["left"] is True


def test_bc004_overpressure_rebases_pose_and_hand_to_physical_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _configure_bc004_force_guard(
        env,
        left_force=10.2,
        right_force=0.0,
    )
    physical_left = left.copy()
    physical_left[1] = 0.197
    physical_matrix = np.eye(4, dtype=np.float64)
    physical_matrix[:3, 3] = physical_left[:3]
    actual_left_hand = np.linspace(
        -0.2,
        0.2,
        env._left_hand.size,
        dtype=np.float64,
    )
    monkeypatch.setattr(
        env,
        "_robot_link_pose_matrix",
        lambda side: physical_matrix if side == "left" else None,
    )
    monkeypatch.setattr(
        env,
        "_actual_hand_joint_pose",
        lambda side: (
            actual_left_hand.copy()
            if side == "left"
            else env._right_hand.copy()
        ),
    )
    env._right_tcp_pose = right.copy()
    env._left_tcp_pose = left.copy()
    env._left_tcp_pose[1] -= 0.010
    env._left_hand += 1.0

    assert env._apply_carton_force_velocity_guard() is True

    assert env._left_tcp_pose[1] == pytest.approx(physical_left[1] + 0.0005)
    np.testing.assert_allclose(env._left_hand, actual_left_hand)
    assert env._b320_carton_force_relief_active["left"] is True
    assert env._b320_carton_force_relief_count["left"] == 1


def test_bc004_force_rise_rejects_new_orientation_and_finger_requests() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _configure_bc004_force_guard(
        env,
        left_force=9.0,
        right_force=0.0,
    )
    env._right_tcp_pose = right.copy()
    env._left_tcp_pose = left.copy()
    previous_left_hand = env._left_hand.copy()
    env._left_tcp_pose[3:7] = np.asarray([0.0, 1.0, 0.0, 0.0])
    env._left_hand = np.full_like(env._left_hand, 1.5)
    env._b320_contact_force_rise_rate_n_s["left"] = 60.0

    env._apply_carton_force_velocity_guard()

    assert env._b320_carton_force_guard_predicted_n["left"] > 10.0
    assert env._b320_carton_force_guard_clamped["left"] is True
    np.testing.assert_array_equal(env._left_tcp_pose[3:7], left[3:7])
    np.testing.assert_array_equal(env._left_hand, previous_left_hand)


def test_bc004_contact_point_rotation_that_predicts_compression_is_ignored() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _configure_bc004_force_guard(
        env,
        left_force=2.0,
        right_force=0.0,
    )
    env._right_tcp_pose = right.copy()
    env._left_tcp_pose = left.copy()
    env._b320_carton_contact_outward["left"] = np.asarray(
        [0.0, 1.0, 0.0],
        dtype=np.float64,
    )
    env._b320_carton_contact_direction_available["left"] = True
    env._b320_carton_contact_lever_local["left"] = np.asarray(
        [0.10, 0.0, 0.0],
        dtype=np.float64,
    )
    env._b320_carton_contact_lever_available["left"] = True
    half_angle = -np.pi / 4.0
    env._left_tcp_pose[3:7] = np.asarray(
        [np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)],
        dtype=np.float64,
    )

    env._apply_carton_force_velocity_guard()

    assert env._b320_carton_force_guard_predicted_n["left"] > 10.0
    assert env._b320_carton_force_guard_clamped["left"] is True
    np.testing.assert_array_equal(env._left_tcp_pose[3:7], left[3:7])


def test_bc004_force_velocity_guard_allows_lift_without_bilateral_contact() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _configure_bc004_force_guard(
        env,
        left_force=5.0,
        right_force=0.0,
    )
    env._right_tcp_pose = right.copy()
    env._left_tcp_pose = left.copy()
    env._right_tcp_pose[1] += 0.010
    env._left_tcp_pose[1] -= 0.010

    env._apply_carton_force_velocity_guard()

    assert env._b320_carton_force_guard_active is True
    assert env._b320_carton_force_guard_active_by_side == {
        "left": True,
        "right": False,
    }
    assert env._left_tcp_pose[1] == pytest.approx(left[1])
    assert env._right_tcp_pose[1] == pytest.approx(right[1] + 0.010)

    env._b320_carton_force_guard_targets = {
        "right": right.copy(),
        "left": left.copy(),
    }
    env._right_tcp_pose = right.copy()
    env._left_tcp_pose = left.copy()
    env._right_tcp_pose[2] += 0.080
    env._left_tcp_pose[2] += 0.080

    env._apply_carton_force_velocity_guard()

    assert env._b320_carton_force_guard_active is True
    assert env._b320_carton_force_guard_clamped == {
        "left": False,
        "right": False,
    }
    assert env._left_tcp_pose[2] == pytest.approx(left[2] + 0.080)
    assert env._right_tcp_pose[2] == pytest.approx(right[2] + 0.080)


def test_bc004_force_velocity_guard_uses_actual_contact_patch_direction() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _configure_bc004_force_guard(
        env,
        left_force=5.0,
        right_force=0.0,
    )
    env._b320_carton_contact_outward["left"] = np.asarray(
        [1.0, 0.0, 0.0],
        dtype=np.float64,
    )
    env._b320_carton_contact_direction_available["left"] = True
    env._right_tcp_pose = right.copy()
    env._left_tcp_pose = left.copy()
    env._left_tcp_pose[0] -= 0.010
    env._left_tcp_pose[1] += 0.020

    env._apply_carton_force_velocity_guard()

    assert env._b320_carton_force_guard_clamped["left"] is True
    assert env._left_tcp_pose[0] == pytest.approx(left[0])
    assert env._left_tcp_pose[1] == pytest.approx(left[1] + 0.020)


def test_bc004_force_velocity_guard_drops_request_without_synthesizing_stop() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _configure_bc004_force_guard(
        env,
        left_force=2.0,
        right_force=0.0,
    )
    live_left = left.copy()
    live_left[1] = 0.195
    live_matrix = np.eye(4, dtype=np.float64)
    live_matrix[:3, 3] = live_left[:3]
    env._robot_link_pose_matrix = lambda side: (
        live_matrix if side == "left" else None
    )
    env._right_tcp_pose = right.copy()
    env._left_tcp_pose = left.copy()
    env._left_tcp_pose[1] = 0.194

    env._apply_carton_force_velocity_guard()

    assert env._b320_carton_force_guard_clamped["left"] is True
    assert env._left_tcp_pose[1] == pytest.approx(left[1])


def test_bc004_force_velocity_guard_can_reject_provisional_frame_after_force_rise() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _configure_bc004_force_guard(
        env,
        left_force=1.0,
        right_force=0.0,
    )
    env._right_tcp_pose = right.copy()
    env._left_tcp_pose = left.copy()
    env._left_tcp_pose[1] -= 0.0005

    env._apply_carton_force_velocity_guard(lookahead_seconds=env._sim_dt)

    assert env._b320_carton_force_guard_clamped["left"] is False
    assert env._left_tcp_pose[1] == pytest.approx(left[1] - 0.0005)

    env._b320_contact_force["left"] = 8.0
    env._apply_carton_force_velocity_guard(lookahead_seconds=env._sim_dt)

    assert env._b320_carton_force_guard_clamped["left"] is True
    assert env._left_tcp_pose[1] == pytest.approx(left[1])


def test_bc004_force_velocity_guard_keeps_other_pose_and_finger_components() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _configure_bc004_force_guard(
        env,
        left_force=5.0,
        right_force=0.0,
    )
    env._right_tcp_pose = right.copy()
    env._left_tcp_pose = left.copy()
    env._left_tcp_pose[1] -= 0.010
    env._left_tcp_pose[2] += 0.080
    env._left_tcp_pose[3:7] = np.asarray([0.0, 1.0, 0.0, 0.0])
    env._left_hand = np.full_like(env._left_hand, 1.5)

    env._apply_carton_force_velocity_guard()

    assert env._b320_carton_force_guard_clamped["left"] is True
    assert env._left_tcp_pose[1] == pytest.approx(left[1])
    assert env._left_tcp_pose[2] == pytest.approx(left[2] + 0.080)
    np.testing.assert_array_equal(
        env._left_tcp_pose[3:7],
        np.asarray([0.0, 1.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(env._left_hand, np.full_like(env._left_hand, 1.5))


def test_bc004_force_velocity_guard_learns_contact_direction_and_force_rate() -> None:
    env = _rigid_env("t10_lift_carton")
    env._poses["carton"] = np.asarray(
        [-0.30, 0.0, 0.89, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    left_view = _CartonContactView(4.0, [(-0.10, 0.0, 1.00)])
    env._b320_contact_views = {"left": left_view}

    env._read_contacts(update_latches=False)

    assert env._b320_carton_contact_direction_available["left"] is True
    assert env._b320_carton_contact_outward["left"] == pytest.approx(
        np.asarray([1.0, 0.0, 0.0])
    )

    left_view.force = 6.0
    env._read_contacts(update_latches=False)

    assert env._b320_contact_force_rise_rate_n_s["left"] > 0.0


def test_bc004_prelock_lift_guard_replans_at_physics_substep_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _rigid_env("t10_lift_carton")
    calls: list[str] = []

    def guard() -> bool:
        calls.append("prelock")
        return True

    monkeypatch.setattr(env, "_apply_carton_prelock_lift_guard", guard)
    monkeypatch.setattr(
        env,
        "_apply_carton_bilateral_common_motion_guard",
        lambda **_kwargs: calls.append("bilateral") or False,
    )
    monkeypatch.setattr(
        env,
        "_apply_joint_targets",
        lambda: calls.append("ik"),
    )
    env._launch_isaac = True

    route = env._prepare_physics_substep_joint_targets(1, 4)

    assert calls == ["prelock", "bilateral", "ik"]
    assert route.endswith("+bc004_prelock_lift_guard")


def test_bc004_bilateral_guard_runs_after_prelock_lift_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _rigid_env("t10_lift_carton")
    calls: list[str] = []
    monkeypatch.setattr(
        env,
        "_apply_carton_prelock_lift_guard",
        lambda: calls.append("prelock") or False,
    )
    monkeypatch.setattr(
        env,
        "_apply_carton_bilateral_common_motion_guard",
        lambda **_kwargs: calls.append("bilateral") or True,
    )
    monkeypatch.setattr(
        env,
        "_apply_joint_targets",
        lambda: calls.append("ik"),
    )
    env._launch_isaac = True

    route = env._prepare_physics_substep_joint_targets(1, 4)

    assert calls == ["prelock", "bilateral", "ik"]
    assert route.endswith("+bc004_bilateral_common_motion_guard")


def _enable_bc004_bilateral_lock(
    env: HandsOnB320BimanualIsaacLabEnv,
) -> tuple[np.ndarray, np.ndarray]:
    right, left = _configure_bc004_force_guard(
        env,
        left_force=10.0,
        right_force=10.0,
    )
    env._b320_carton_zone_valid = {"left": True, "right": True}
    env._right_tcp_pose = right.copy()
    env._left_tcp_pose = left.copy()
    return right, left


def test_bc004_bilateral_lock_keeps_only_common_translation_per_axis() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _enable_bc004_bilateral_lock(env)
    right_delta = np.asarray([0.002, 0.005, 0.003])
    left_delta = np.asarray([0.003, 0.002, -0.001])
    right_orientation = np.asarray([0.0, 1.0, 0.0, 0.0])
    left_orientation = np.asarray([0.0, 0.0, 1.0, 0.0])
    env._right_tcp_pose[:3] += right_delta
    env._left_tcp_pose[:3] += left_delta
    env._right_tcp_pose[3:7] = right_orientation
    env._left_tcp_pose[3:7] = left_orientation

    clamped = env._apply_carton_bilateral_common_motion_guard()

    # X remains tangential for the sampled side contacts.  The shared +Y
    # request would squeeze the right side while the carton is table-supported,
    # so the force-aware pair guard rejects Y for both hands.
    common = np.asarray([0.002, 0.0, 0.0])
    assert clamped is True
    assert env._b320_carton_bilateral_lock_active is True
    assert env._b320_carton_bilateral_lock_latch_step == env.env_step
    np.testing.assert_allclose(env._right_tcp_pose[:3], right[:3] + common)
    np.testing.assert_allclose(env._left_tcp_pose[:3], left[:3] + common)
    np.testing.assert_allclose(
        env._left_tcp_pose[:3] - env._right_tcp_pose[:3],
        left[:3] - right[:3],
    )
    np.testing.assert_array_equal(env._right_tcp_pose[3:7], right[3:7])
    np.testing.assert_array_equal(env._left_tcp_pose[3:7], left[3:7])
    np.testing.assert_array_equal(
        env._b320_carton_bilateral_lock_force_rejected_axes,
        np.asarray([False, True, False]),
    )


def test_bc004_bilateral_lock_freezes_orientation_and_hand_posture() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _enable_bc004_bilateral_lock(env)
    env._apply_carton_bilateral_common_motion_guard()
    locked_right_hand = env._right_hand.copy()
    locked_left_hand = env._left_hand.copy()

    env._b320_carton_force_guard_targets = {
        "right": env._right_tcp_pose.copy(),
        "left": env._left_tcp_pose.copy(),
    }
    env._right_tcp_pose[2] += 0.004
    env._left_tcp_pose[2] += 0.003
    env._right_tcp_pose[3:7] = np.asarray([0.0, 1.0, 0.0, 0.0])
    env._left_tcp_pose[3:7] = np.asarray([0.0, 0.0, 1.0, 0.0])
    env._right_hand = np.full_like(env._right_hand, 1.5)
    env._left_hand = np.full_like(env._left_hand, -1.5)

    assert env._apply_carton_bilateral_common_motion_guard() is True
    assert env._right_tcp_pose[2] == pytest.approx(right[2] + 0.003)
    assert env._left_tcp_pose[2] == pytest.approx(left[2] + 0.003)
    np.testing.assert_array_equal(env._right_tcp_pose[3:7], right[3:7])
    np.testing.assert_array_equal(env._left_tcp_pose[3:7], left[3:7])
    np.testing.assert_array_equal(env._right_hand, locked_right_hand)
    np.testing.assert_array_equal(env._left_hand, locked_left_hand)


def test_bc004_bilateral_force_rise_rejects_horizontal_but_keeps_lift() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _enable_bc004_bilateral_lock(env)
    env._apply_carton_bilateral_common_motion_guard()
    env._b320_carton_force_guard_targets = {
        "right": env._right_tcp_pose.copy(),
        "left": env._left_tcp_pose.copy(),
    }
    env._b320_contact_force = {"left": 10.0, "right": 10.0}
    env._b320_contact_force_rise_rate_n_s = {
        "left": 30.0,
        "right": 30.0,
    }
    env._right_tcp_pose[0] += 0.004
    env._left_tcp_pose[0] += 0.004
    env._right_tcp_pose[2] += 0.003
    env._left_tcp_pose[2] += 0.003

    assert env._apply_carton_bilateral_common_motion_guard() is True
    assert env._right_tcp_pose[0] == pytest.approx(right[0])
    assert env._left_tcp_pose[0] == pytest.approx(left[0])
    assert env._right_tcp_pose[1] == pytest.approx(right[1])
    assert env._left_tcp_pose[1] == pytest.approx(left[1])
    assert env._right_tcp_pose[2] == pytest.approx(right[2] + 0.003)
    assert env._left_tcp_pose[2] == pytest.approx(left[2] + 0.003)
    np.testing.assert_array_equal(
        env._b320_carton_bilateral_lock_force_rejected_axes,
        np.asarray([True, False, False]),
    )
    assert env._b320_carton_force_relief_active == {
        "left": False,
        "right": False,
    }
    assert env._b320_carton_bilateral_lift_bypass_count == 1


def test_bc004_bilateral_lock_keeps_10n_without_rate_based_unloading() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _enable_bc004_bilateral_lock(env)
    env._apply_carton_bilateral_common_motion_guard()
    env._b320_carton_force_guard_targets = {
        "right": env._right_tcp_pose.copy(),
        "left": env._left_tcp_pose.copy(),
    }
    env._b320_contact_force = {"left": 10.0, "right": 10.0}
    env._b320_contact_force_rise_rate_n_s = {
        "left": 200.0,
        "right": 200.0,
    }

    assert env._apply_carton_bilateral_common_motion_guard() is False

    np.testing.assert_allclose(env._right_tcp_pose, right)
    np.testing.assert_allclose(env._left_tcp_pose, left)
    assert env._b320_carton_force_relief_active == {
        "left": False,
        "right": False,
    }
    assert env._b320_carton_force_relief_count == {
        "left": 0,
        "right": 0,
    }


def test_bc004_bilateral_lock_drops_opposed_squeeze_translation() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _enable_bc004_bilateral_lock(env)
    env._right_tcp_pose[1] += 0.005
    env._left_tcp_pose[1] -= 0.004

    clamped = env._apply_carton_bilateral_common_motion_guard()

    assert clamped is True
    np.testing.assert_allclose(env._right_tcp_pose[:3], right[:3])
    np.testing.assert_allclose(env._left_tcp_pose[:3], left[:3])
    np.testing.assert_allclose(
        env._b320_carton_bilateral_lock_common_delta,
        np.zeros(3),
    )


def test_bc004_bilateral_lock_discards_the_faster_hands_excess() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _enable_bc004_bilateral_lock(env)
    env._right_tcp_pose[2] += 0.004
    env._left_tcp_pose[2] += 0.002

    env._apply_carton_bilateral_common_motion_guard()

    assert env._right_tcp_pose[2] == pytest.approx(right[2] + 0.002)
    assert env._left_tcp_pose[2] == pytest.approx(left[2] + 0.002)
    assert env._b320_carton_bilateral_lock_discarded_delta[
        "right"
    ][2] == pytest.approx(0.002)
    assert env._b320_carton_bilateral_lock_discarded_delta[
        "left"
    ][2] == pytest.approx(0.0)


def test_bc004_bilateral_lock_remains_latched_after_force_drops() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _enable_bc004_bilateral_lock(env)
    env._apply_carton_bilateral_common_motion_guard()
    assert env._b320_carton_bilateral_lock_active is True

    env._b320_contact_force = {"left": 0.0, "right": 0.0}
    env._b320_carton_zone_valid = {"left": False, "right": False}
    env._b320_carton_force_guard_targets = {
        "right": env._right_tcp_pose.copy(),
        "left": env._left_tcp_pose.copy(),
    }
    env._right_tcp_pose[0] += 0.003
    env._left_tcp_pose[0] += 0.002

    env._apply_carton_bilateral_common_motion_guard()

    assert env._b320_carton_bilateral_lock_active is True
    assert env._right_tcp_pose[0] == pytest.approx(right[0] + 0.002)
    assert env._left_tcp_pose[0] == pytest.approx(left[0] + 0.002)


def test_bc004_locked_force_hold_blocks_lift_and_recompresses_weak_side(
) -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _enable_bc004_bilateral_lock(env)
    env._apply_carton_bilateral_common_motion_guard()
    env._b320_carton_force_guard_targets = {
        "right": env._right_tcp_pose.copy(),
        "left": env._left_tcp_pose.copy(),
    }
    env._b320_contact_force = {"left": 6.0, "right": 10.0}
    env._right_tcp_pose[2] += 0.010
    env._left_tcp_pose[2] += 0.010

    assert env._apply_carton_bilateral_common_motion_guard() is True

    np.testing.assert_allclose(env._right_tcp_pose[:3], right[:3])
    expected_left = left[:3].copy()
    expected_left[1] -= 0.00025
    np.testing.assert_allclose(env._left_tcp_pose[:3], expected_left)
    assert env._b320_carton_force_hold_active == {
        "left": True,
        "right": False,
    }
    assert env._b320_carton_force_hold_inward_step_m == {
        "left": pytest.approx(0.00025),
        "right": pytest.approx(0.0),
    }
    assert env._b320_carton_bilateral_lock_force_rejected_axes.tolist() == [
        False,
        False,
        True,
    ]
    np.testing.assert_allclose(
        env._b320_carton_force_guard_targets["left"],
        env._left_tcp_pose,
    )


def test_bc004_locked_force_hold_resumes_lift_from_corrected_separation(
) -> None:
    env = _rigid_env("t10_lift_carton")
    _enable_bc004_bilateral_lock(env)
    env._apply_carton_bilateral_common_motion_guard()
    env._b320_carton_force_guard_targets = {
        "right": env._right_tcp_pose.copy(),
        "left": env._left_tcp_pose.copy(),
    }
    env._b320_contact_force = {"left": 6.0, "right": 10.0}
    env._right_tcp_pose[2] += 0.010
    env._left_tcp_pose[2] += 0.010
    env._apply_carton_bilateral_common_motion_guard()
    corrected_right = env._right_tcp_pose.copy()
    corrected_left = env._left_tcp_pose.copy()

    env._b320_contact_force = {"left": 10.0, "right": 10.0}
    env._b320_contact_force_rise_rate_n_s = {
        "left": 0.0,
        "right": 0.0,
    }
    env._right_tcp_pose[2] += 0.005
    env._left_tcp_pose[2] += 0.005

    assert env._apply_carton_bilateral_common_motion_guard() is False

    assert env._right_tcp_pose[2] == pytest.approx(
        corrected_right[2] + 0.005
    )
    assert env._left_tcp_pose[2] == pytest.approx(
        corrected_left[2] + 0.005
    )
    np.testing.assert_allclose(
        env._left_tcp_pose[:3] - env._right_tcp_pose[:3],
        corrected_left[:3] - corrected_right[:3],
    )
    assert env._b320_carton_force_hold_active == {
        "left": False,
        "right": False,
    }


@pytest.mark.parametrize(
    ("left_force", "right_force"),
    [
        (9.99, 10.0),
        (10.0, 9.99),
    ],
)
def test_bc004_bilateral_lock_waits_until_each_side_has_reached_10n(
    left_force: float,
    right_force: float,
) -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _configure_bc004_force_guard(
        env,
        left_force=left_force,
        right_force=right_force,
    )
    env._right_tcp_pose = right.copy()
    env._left_tcp_pose = left.copy()
    env._right_tcp_pose[2] += 0.003
    env._left_tcp_pose[2] += 0.002

    clamped = env._apply_carton_bilateral_common_motion_guard()

    assert clamped is False
    assert env._b320_carton_bilateral_lock_active is False
    assert env._right_tcp_pose[2] == pytest.approx(right[2] + 0.003)
    assert env._left_tcp_pose[2] == pytest.approx(left[2] + 0.002)


def test_bc004_bilateral_lock_requires_simultaneous_10n() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _configure_bc004_force_guard(
        env,
        left_force=10.0,
        right_force=5.0,
    )
    env._right_tcp_pose = right.copy()
    env._left_tcp_pose = left.copy()

    assert env._apply_carton_bilateral_common_motion_guard() is False
    assert env._b320_carton_bilateral_pressure_reached == {
        "left": True,
        "right": False,
    }

    env._b320_contact_force = {"left": 5.0, "right": 10.0}
    env._right_tcp_pose[2] += 0.003
    env._left_tcp_pose[2] += 0.002

    assert env._apply_carton_bilateral_common_motion_guard() is False
    assert env._b320_carton_bilateral_lock_active is False
    assert env._b320_carton_bilateral_pressure_reached == {
        "left": True,
        "right": True,
    }

    env._b320_contact_force = {"left": 10.0, "right": 10.0}

    assert env._apply_carton_bilateral_common_motion_guard() is True
    assert env._b320_carton_bilateral_lock_active is True
    assert env._right_tcp_pose[2] == pytest.approx(right[2] + 0.002)
    assert env._left_tcp_pose[2] == pytest.approx(left[2] + 0.002)


def test_bc004_bilateral_lock_does_not_require_lower_zone_on_latch() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _configure_bc004_force_guard(
        env,
        left_force=10.0,
        right_force=10.0,
    )
    env._b320_carton_zone_valid = {"left": False, "right": False}
    env._right_tcp_pose = right.copy()
    env._left_tcp_pose = left.copy()
    env._right_tcp_pose[2] += 0.003
    env._left_tcp_pose[2] += 0.002

    assert env._apply_carton_bilateral_common_motion_guard() is True
    assert env._b320_carton_bilateral_lock_active is True


def test_bc004_force_guard_rejects_delayed_pending_inward_target() -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _configure_bc004_force_guard(
        env,
        left_force=9.0,
        right_force=0.0,
    )
    pending = left.copy()
    pending[1] -= 0.002
    env._b320_carton_force_guard_targets["left"] = pending.copy()
    env._left_tcp_pose = pending.copy()
    env._right_tcp_pose = right.copy()
    env._b320_carton_force_guard_verified_position["left"] = left[:3].copy()
    env._b320_carton_force_guard_verified_valid["left"] = True

    env._apply_carton_force_velocity_guard()

    assert env._b320_carton_force_guard_clamped["left"] is True
    assert env._b320_carton_force_guard_pending_inward_m[
        "left"
    ] == pytest.approx(0.002)
    assert env._left_tcp_pose[1] == pytest.approx(left[1])


def test_bc004_latch_discards_backlog_and_uses_physical_relative_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _enable_bc004_bilateral_lock(env)
    env._right_tcp_pose[1] += 0.010
    env._left_tcp_pose[1] -= 0.010
    physical_right = right[:3].copy()
    physical_left = left[:3].copy()
    physical_matrices: dict[str, np.ndarray] = {}
    for side, position, yaw_deg in (
        ("right", physical_right, 25.0),
        ("left", physical_left, -20.0),
    ):
        yaw = np.radians(yaw_deg)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.asarray(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        matrix[:3, 3] = position
        physical_matrices[side] = matrix
    expected_physical = {
        side: env._pose_matrix_to_vector(matrix)
        for side, matrix in physical_matrices.items()
    }
    actual_hands = {
        "right": np.full_like(env._right_hand, 0.15),
        "left": np.full_like(env._left_hand, -0.12),
    }

    def palm(side: str) -> np.ndarray:
        return physical_matrices[side].copy()

    monkeypatch.setattr(env, "_robot_link_pose_matrix", palm)
    monkeypatch.setattr(
        env,
        "_actual_hand_joint_pose",
        lambda side: actual_hands[side].copy(),
    )

    assert env._apply_carton_bilateral_common_motion_guard() is True
    np.testing.assert_allclose(env._right_tcp_pose, expected_physical["right"])
    np.testing.assert_allclose(env._left_tcp_pose, expected_physical["left"])
    np.testing.assert_allclose(env._right_hand, actual_hands["right"])
    np.testing.assert_allclose(env._left_hand, actual_hands["left"])
    np.testing.assert_allclose(
        env._b320_carton_bilateral_lock_relative_position,
        physical_left - physical_right,
    )


def test_bc004_locked_overpressure_rebases_both_hands_and_unloads_loaded_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _rigid_env("t10_lift_carton")
    right, left = _enable_bc004_bilateral_lock(env)
    physical_matrices = {
        "right": np.eye(4, dtype=np.float64),
        "left": np.eye(4, dtype=np.float64),
    }
    physical_matrices["right"][:3, 3] = right[:3]
    physical_matrices["left"][:3, 3] = left[:3]
    actual_hands = {
        "right": np.full_like(env._right_hand, 0.10),
        "left": np.full_like(env._left_hand, -0.10),
    }
    monkeypatch.setattr(
        env,
        "_robot_link_pose_matrix",
        lambda side: physical_matrices[side].copy(),
    )
    monkeypatch.setattr(
        env,
        "_actual_hand_joint_pose",
        lambda side: actual_hands[side].copy(),
    )
    assert env._apply_carton_bilateral_common_motion_guard() is True
    env._b320_carton_force_guard_targets = {
        "right": env._right_tcp_pose.copy(),
        "left": env._left_tcp_pose.copy(),
    }
    env._b320_contact_force = {"left": 10.2, "right": 10.0}
    env._right_tcp_pose[0] += 0.020
    env._left_tcp_pose[0] += 0.020
    env._right_hand += 1.0
    env._left_hand += 1.0

    assert env._apply_carton_bilateral_common_motion_guard() is True

    np.testing.assert_allclose(env._right_tcp_pose[:3], right[:3])
    expected_left = left[:3].copy()
    expected_left[1] += 0.0005
    np.testing.assert_allclose(env._left_tcp_pose[:3], expected_left)
    np.testing.assert_allclose(env._right_hand, actual_hands["right"])
    np.testing.assert_allclose(env._left_hand, actual_hands["left"])
    assert env._b320_carton_force_relief_active == {
        "left": True,
        "right": False,
    }


def test_bc004_locked_common_motion_bypasses_individual_normal_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _rigid_env("t10_lift_carton")
    calls: list[str] = []
    env._b320_carton_bilateral_lock_active = True
    monkeypatch.setattr(
        env,
        "_apply_carton_force_velocity_guard",
        lambda **_kwargs: calls.append("individual") or True,
    )
    monkeypatch.setattr(
        env,
        "_apply_carton_bilateral_common_motion_guard",
        lambda: calls.append("bilateral") or False,
    )

    env._apply_operator_action_extras(np.zeros(54, dtype=np.float64))

    assert calls == ["bilateral"]


def test_carton_drop_is_a_physical_failure() -> None:
    env = _rigid_env("t10_lift_carton")
    _prime_contacts(env)
    env._b320_reset_poses["carton"] = env._fallback_b320_pose("carton")
    env._poses["carton"] = np.asarray(
        [-0.30, 0.0, 0.839, 1.0, 0.0, 0.0, 0.0]
    )

    assert env._compute_failure() is True
    assert env._b320_failure_reason == "carton_dropped"


def test_bc004_has_no_hidden_tilt_gate_when_all_corners_are_airborne() -> None:
    env = _rigid_env("t10_lift_carton")
    _prime_contacts(env)
    angle = np.deg2rad(35.0)
    env._poses["carton"] = np.asarray(
        [
            -0.30,
            0.0,
            1.20,
            np.cos(angle / 2.0),
            np.sin(angle / 2.0),
            0.0,
            0.0,
        ]
    )
    env._b320_carton_airborne_substeps = (
        b320_env_module._CARTON_AIRBORNE_REQUIRED_SUBSTEPS
    )

    assert all(env._carton_metrics()[1])
    assert env._compute_failure() is False


def test_b320_time_limit_truncates_but_physical_failure_terminates() -> None:
    env = _rigid_env("t20_pass_and_orient_pen")
    env.hz = 0
    env._time_limit = env._control_dt

    _, _, terminated, truncated, info = env.step(np.zeros(54, dtype=np.float64))
    assert terminated is False
    assert truncated is True
    assert info["time_limit_reached"] is True
    assert info["succeed"] is False
    assert info.get("failed", False) is False

    env = _rigid_env("t10_lift_carton")
    env.hz = 0
    env._time_limit = env._control_dt
    env._b320_contact_peak["left"] = 12.5
    _, _, terminated, truncated, info = env.step(np.zeros(54, dtype=np.float64))
    assert terminated is True
    assert truncated is False
    assert info["failed"] is True
    assert info["b320_failure_reason"] == "carton_side_force_exceeded"


def test_b320_towel_interface_and_success_geometry_are_complete() -> None:
    env = HandsOnB320TowelIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    assert env.stable_recording_origin_gate_enabled is True
    observation, info = env.reset(seed=13)
    assert env.action_space.shape == (54,)
    assert env.observation_space.contains(observation)
    assert env._compute_success_now() is False
    assert len(env._success_condition_rows()) == 6
    assert info["task_name"] == B320_TOWEL_TASK_NAME

    nodes = env._e_nodes_np.copy()
    x_min, x_max, y_min, y_max = env._target_bounds()
    nodes[[0, 40, 1640, 1680], :2] = np.asarray(
        [
            (x_min + 0.005, y_min + 0.005),
            (x_max - 0.005, y_min + 0.005),
            (x_min + 0.005, y_max - 0.005),
            (x_max - 0.005, y_max - 0.005),
        ]
    )
    env._e_nodes_np = nodes
    env._e_base_nodes_np = nodes.copy()
    env._e_velocities_np = np.zeros_like(nodes)
    env._e_readback_failed = False
    env._e_failure_reason = None
    env._b320_corner_seen = {"left": True, "right": True}
    env._b320_corner_index = {"left": 1640, "right": 1680}
    env._b320_release_frame = {"left": 10, "right": 20}

    metrics = env._metrics(update_events=False)
    assert metrics["max_stretch_ratio"] == 0.0
    assert metrics["corner_count_in_target"] == 4
    assert metrics["coverage_ratio"] > 0.80
    assert metrics["success_now"] is True
    assert env.get_physical_replay_reset_contract()[
        "b320_physical_replay_certified"
    ] is False

    image_env = HandsOnB320TowelIsaacLabEnv(
        launch_isaac=False,
        image_obs=True,
        randomize=True,
        render_resolution=(320, 240),
    )
    image_observation, _ = image_env.reset(seed=13)
    assert image_env.observation_space.contains(image_observation)
    assert image_observation["images"]["random_camera"].shape == (320, 240, 3)
    assert image_observation["images"]["wrist_left"].shape == (480, 848, 3)
    assert image_observation["images"]["wrist_right"].shape == (480, 848, 3)


def test_bc025_seeded_physical_translation_is_bounded_repeatable_and_spans() -> None:
    env = HandsOnB320TowelIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
        randomize=True,
    )
    samples: dict[int, np.ndarray] = {}
    initial_states: dict[int, np.ndarray] = {}
    for seed in (101, 102, 103):
        observation, info = env.reset(seed=seed)
        sample = np.asarray(
            env._bc025_surface_translation_sample_m,
            dtype=np.float64,
        ).copy()
        samples[seed] = sample
        initial_states[seed] = np.asarray(
            observation["state"]["garment_initial_state"],
            dtype=np.float64,
        ).copy()
        assert info["physical_randomization"] is True
        assert np.all(sample[2:] == 0.0)
        for axis, bounds in enumerate(BC025_TOWEL_TRANSLATION_BOUNDS_M):
            assert float(bounds[0]) <= sample[axis] <= float(bounds[1])
        np.testing.assert_array_equal(initial_states[seed][:2], sample[:2])
        np.testing.assert_array_equal(
            env._e_initial_state[:2],
            sample[:2],
        )

    repeated_observation, repeated_info = env.reset(seed=101)
    repeated_sample = np.asarray(
        env._bc025_surface_translation_sample_m,
        dtype=np.float64,
    )
    np.testing.assert_array_equal(repeated_sample, samples[101])
    np.testing.assert_array_equal(
        repeated_observation["state"]["garment_initial_state"][:2],
        initial_states[101][:2],
    )
    assert repeated_info["b320_surface_translation_randomization_enabled"] is True

    stack = np.stack(list(samples.values()), axis=0)
    assert float(np.ptp(stack[:, 0])) > 1.0e-6
    assert float(np.ptp(stack[:, 1])) > 1.0e-6
    expected_centroid = np.asarray(B320_TOWEL_PLACEMENT, dtype=np.float64)
    expected_centroid[:2] += samples[101][:2]
    np.testing.assert_allclose(
        np.mean(env._e_reset_nodes_np, axis=0)[:2],
        expected_centroid[:2],
        atol=1.0e-12,
    )


def test_xy4_unavailable_contact_metric_exports_null_without_changing_checker():
    env = _rigid_env("eval_comb05_scrub_bottle_then_fill_cup")
    try:
        env.reset(seed=7)
        env.step(np.zeros(env.action_space.shape))
        assert np.isinf(env._last_hs005_success_metrics["hs005_right_grasp_surface_error"])
        info = env._task_info()
        assert info["hs005_metrics"]["hs005_right_grasp_surface_error"] is None
        json.dumps(info, allow_nan=False)
        assert env._compute_success_now() is False
    finally:
        env.close()
