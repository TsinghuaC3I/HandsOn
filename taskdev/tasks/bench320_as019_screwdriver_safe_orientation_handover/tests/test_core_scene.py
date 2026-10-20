from __future__ import annotations

import ast
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from taskdev.manifest import ManifestValidator
from taskdev.tasks.bench320_as019_screwdriver_safe_orientation_handover.scene_contract import (
    AS019_TABLETOP_ARM_HOME_RAD,
    AS019_TABLETOP_TCP_HOME_WXYZ,
    ASSET_MANIFEST_RELATIVE,
    AUTHORITATIVE_EXCEL_ROW,
    AUTHORITATIVE_SOURCE_ROWS_SHA256,
    HEAD_CAMERA_RESOLUTION_HW,
    MARKER_COLLISION_ENABLED,
    MARKER_ORIENTATION_WXYZ,
    MARKER_POSITION_M,
    MARKER_SIZE_M,
    MAX_ORIENTATION_DRIFT_RAD,
    MAX_POSITION_DRIFT_M,
    OVERLAY_RELATIVE,
    PREVIEW_RELATIVE,
    REPOSITORY_ROOT,
    RUNTIME_COLLIDER_PATHS,
    RUNTIME_PHYSICS_MATERIAL_PATH,
    RUNTIME_RIGID_OWNER,
    RUNTIME_TEXTURE_RELATIVE_BY_SOURCE_MEMBER,
    RUNTIME_USD_RELATIVE,
    RUNTIME_VISUAL_MESH_PATHS,
    SCREWDRIVER_CENTER_OF_MASS_M,
    SCREWDRIVER_CONTACT_OFFSET_M,
    SCREWDRIVER_DEFAULT_FRICTION,
    SCREWDRIVER_EXPECTED_COLLIDER_COUNT,
    SCREWDRIVER_ANGULAR_DAMPING,
    SCREWDRIVER_LINEAR_DAMPING,
    SCREWDRIVER_ORIENTATION_WXYZ,
    SCREWDRIVER_POSITION_M,
    SCREWDRIVER_REST_OFFSET_M,
    SCREWDRIVER_SAFE_AXIS_WORLD,
    SCREWDRIVER_SLEEP_THRESHOLD,
    SCREWDRIVER_UNIFORM_SCALE,
    SCREWDRIVER_SOLVER_POSITION_ITERATIONS,
    SCREWDRIVER_SOLVER_VELOCITY_ITERATIONS,
    SCREWDRIVER_STABILIZATION_THRESHOLD,
    SOURCE_ARCHIVE_BYTES,
    SOURCE_ARCHIVE_RELATIVE,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_EXTENTS_M,
    SOURCE_MEMBER_SHA256,
    SUMMARY_RELATIVE,
    TABLE_TOP_Z_M,
    TASK_DIR,
    ZERO_ACTION_RELATIVE,
    ZERO_ACTION_SECONDS,
    contract_errors,
    frame_metrics,
    read_json,
    sha256_file,
    world_bounds,
)


def test_as019_uses_the_shared_mirrored_bimanual_arm_home() -> None:
    mirror_signs = np.asarray((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0))
    right = np.asarray(AS019_TABLETOP_ARM_HOME_RAD["right"], dtype=np.float64)
    left = np.asarray(AS019_TABLETOP_ARM_HOME_RAD["left"], dtype=np.float64)
    np.testing.assert_allclose(
        left, mirror_signs * right, atol=1.0e-12, rtol=0.0
    )

    for side, y_sign in (("right", -1.0), ("left", 1.0)):
        pose = np.asarray(AS019_TABLETOP_TCP_HOME_WXYZ[side], dtype=np.float64)
        assert pose.shape == (7,)
        assert y_sign * pose[1] > 0.0
        assert pose[2] - TABLE_TOP_Z_M == pytest.approx(0.15, abs=1.0e-12)
        w, x, y, z = pose[3:7]
        rotation = np.asarray(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ]
        )
        np.testing.assert_allclose(
            rotation[:, 0], (1.0, 0.0, 0.0), atol=1.0e-12
        )
        np.testing.assert_allclose(
            rotation[:, 2], (0.0, 0.0, -1.0), atol=1.0e-12
        )

    joint_pos = {
        f"joint{index}_{side}(_1)?": 99.0
        for side in ("right", "left")
        for index in range(1, 8)
    }
    cfg = SimpleNamespace(
        right_robot=SimpleNamespace(
            init_state=SimpleNamespace(joint_pos=dict(joint_pos))
        ),
        left_robot=SimpleNamespace(
            init_state=SimpleNamespace(joint_pos=dict(joint_pos))
        ),
    )
    from taskdev.tasks.bench320_as019_screwdriver_safe_orientation_handover.scene_runtime import (
        _apply_as019_tabletop_arm_home,
    )

    _apply_as019_tabletop_arm_home(cfg)
    for side in ("right", "left"):
        configured = cfg.__dict__[f"{side}_robot"].init_state.joint_pos
        assert [
            configured[f"joint{index}_{side}(_1)?"] for index in range(1, 8)
        ] == pytest.approx(AS019_TABLETOP_ARM_HOME_RAD[side], abs=1.0e-12)


def test_task_package_init_is_zero_bytes() -> None:
    assert (TASK_DIR / "__init__.py").read_bytes() == b""


def test_core_scene_contract_is_hash_bound_and_safe() -> None:
    assert contract_errors() == ()
    assert AUTHORITATIVE_EXCEL_ROW == 260
    assert AUTHORITATIVE_SOURCE_ROWS_SHA256 == "c1bf0990e8ec0ae651ae6d904c6b61e45796c8e11a670f6da782f4b23052e987"
    assert SCREWDRIVER_SAFE_AXIS_WORLD == (0.0, 1.0, 0.0)
    assert np.allclose(
        SCREWDRIVER_CENTER_OF_MASS_M,
        (-0.03348301709027515, 0.02500000005176284, 0.0),
        atol=1.0e-12,
    )
    assert MARKER_COLLISION_ENABLED is False
    assert SCREWDRIVER_LINEAR_DAMPING == 0.08
    assert SCREWDRIVER_ANGULAR_DAMPING == 2.0
    assert SCREWDRIVER_SLEEP_THRESHOLD == 0.005
    assert SCREWDRIVER_STABILIZATION_THRESHOLD == 0.0025
    assert SCREWDRIVER_SOLVER_POSITION_ITERATIONS == 8
    assert SCREWDRIVER_SOLVER_VELOCITY_ITERATIONS == 1
    assert ZERO_ACTION_SECONDS >= 3.0


def test_horizontal_reset_has_tail_screen_right_and_parallel_clear_marker() -> None:
    w, x, y, z = SCREWDRIVER_ORIENTATION_WXYZ
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )
    np.testing.assert_allclose(rotation[:, 0], [0.0, 1.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(-rotation[:, 0], [0.0, -1.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(MARKER_ORIENTATION_WXYZ, SCREWDRIVER_ORIENTATION_WXYZ)
    assert MARKER_POSITION_M[0] < -0.22
    low, high = map(np.asarray, world_bounds())
    np.testing.assert_allclose(
        high - low,
        SCREWDRIVER_UNIFORM_SCALE
        * np.asarray(
            [SOURCE_EXTENTS_M[1], SOURCE_EXTENTS_M[0], SOURCE_EXTENTS_M[2]]
        ),
        atol=1.0e-8,
    )


def test_manifest_validator_accepts_task_yaml() -> None:
    result = ManifestValidator(REPOSITORY_ROOT).validate_path(TASK_DIR / "task.yaml")
    assert result.valid, result.as_dict()


def test_runtime_script_starts_app_before_task_runtime_import() -> None:
    source = (TASK_DIR / "run_core_scene.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith(("isaacsim", "isaaclab"))
        and node.module != "isaaclab.app"
        for node in imports
    )
    assert source.index("from isaaclab.app import AppLauncher") < source.index(
        "from taskdev.tasks.bench320_as019_screwdriver_safe_orientation_handover.scene_runtime"
    )
    assert "width=640" in source and "height=480" in source
    assert "RayTracedLighting" in source
    assert 'settings.set("/rtx/post/tonemap/op", 4)' in source


def test_overlay_keeps_reference_xforms_below_task_pose() -> None:
    source = (TASK_DIR / "scene_runtime.py").read_text(encoding="utf-8")
    assert '"/World/AS019Core/Screwdriver/Asset"' in source
    assert source.index("screwdriver.AddTranslateOp()") < source.index(
        "screwdriver_asset.GetPrim().GetReferences().AddReference"
    )
    assert "marker.AddOrientOp().Set(Gf.Quatf(*MARKER_ORIENTATION_WXYZ))" in source
    assert "diffuse_color=(0.04, 0.95, 0.12)" in source
    assert "emissive_color=(0.02, 0.35, 0.04)" in source
    assert "opacity=1.0" in source
    assert "Gf.Vec3f(0.04, 0.95, 0.12)" in source
    assert "Gf.Vec3f(0.02, 0.35, 0.04)" in source
    assert 'shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)' in source


def test_task_contact_views_do_not_disable_physx_sleep() -> None:
    source = (TASK_DIR / "scene_runtime.py").read_text(encoding="utf-8")
    assert "cfg.table.spawn.activate_contact_sensors = True" in source
    assert "activate_contact_sensors=False" in source
    assert "sleepThreshold to 0.0" in source


def test_table_safety_contract_records_batched_live_stage0_verification() -> None:
    task = yaml.safe_load((TASK_DIR / "task.yaml").read_text(encoding="utf-8"))
    status = task["contact"]["table_safety_runtime"]
    assert status == {
        "contact_reporting": "implemented_pending_1_75_live_revalidation",
        "exact_filtered_views": "implemented_batched_owner_records",
        "fail_closed_support_and_penetration_gates": (
            "implemented_pending_1_75_live_revalidation"
        ),
        "invalid_query_policy": "implemented_fail_closed_and_cpu_verified",
        "terminal_arm_after_control_steps": 3,
        "terminal_debounce_control_steps": 3,
    }
    monitored = {
        record["id"]: record for record in task["contact"]["monitored_pairs"]
    }
    assert (
        monitored["screwdriver_table_support"]["runtime_status"]
        == "pending_1_75_live_revalidation"
    )
    assert monitored["screwdriver_table_support"]["max_separation_m"] == 0.003
    assert monitored["screwdriver_table_support"]["max_penetration_m"] == 0.005
    assert monitored["screwdriver_table_support"]["required_for_zone_placement"]
    assert monitored["hand_table_clearance"]["shallow_contact_allowed"]
    assert monitored["hand_table_clearance"]["max_penetration_m"] == 0.005
    assert (
        monitored["hand_table_clearance"]["runtime_status"]
        == "pending_1_75_live_revalidation"
    )
    assert monitored["arm_table_diagnostic"]["physical_contact_allowed"] is True
    assert monitored["arm_table_diagnostic"]["failure_effect"] == "none"
    contract = (TASK_DIR / "TASK_CONTRACT.md").read_text(encoding="utf-8")
    assert "arm/table contact remains diagnostic only" in contract
    assert "1.75x geometry still needs a fresh live core audit" in contract


def test_production_and_automatic_validation_horizons_are_distinct() -> None:
    task = yaml.safe_load((TASK_DIR / "task.yaml").read_text(encoding="utf-8"))
    assert task["stability"]["control_hz"] == 30
    assert task["success"]["timeout_steps"] == 3600
    automatic = task["validation"]["automatic_policy"]
    assert automatic == {
        "stage0_canary_max_steps": 320,
        "positive_max_steps": 1200,
        "production_episode_timeout_steps": 3600,
        "scope": "validation_budget_with_separate_production_hard_limit",
    }
    contract = (TASK_DIR / "TASK_CONTRACT.md").read_text(encoding="utf-8")
    assert "120 s / 3600-control-step production hard limit" in contract
    assert "separate 1200-step budget" in contract


def test_delivered_python_has_no_deprecated_imports_or_hardcoded_user_paths() -> None:
    for path in (TASK_DIR / "scene_contract.py", TASK_DIR / "scene_runtime.py", TASK_DIR / "run_core_scene.py"):
        source = path.read_text(encoding="utf-8")
        assert "omni.isaac.core" not in source
        assert not re.search(r"/(?:home|Users|mnt/pfs)/[^\"']+", source)


def test_frame_metrics_reject_black_and_accept_structured_rgb() -> None:
    height, width = HEAD_CAMERA_RESOLUTION_HW
    assert frame_metrics(np.zeros((height, width, 3), dtype=np.uint8))["pass"] is False
    y = np.arange(height, dtype=np.uint16)[:, None]
    x = np.arange(width, dtype=np.uint16)[None, :]
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[..., 0] = ((x + y) % 256).astype(np.uint8)
    frame[..., 1] = ((2 * x + y) % 256).astype(np.uint8)
    frame[..., 2] = ((x + 3 * y) % 256).astype(np.uint8)
    assert frame_metrics(frame)["pass"] is True


def test_collection_smoke_counts_hidden_published_recorder_episode(
    tmp_path: Path,
) -> None:
    from taskdev.tasks.bench320_as019_screwdriver_safe_orientation_handover.run_collection_ui_smoke import (
        TASK_NAME,
        _completed_demo_dirs,
    )

    incomplete = tmp_path / "dataset" / ".recording_staging" / (
        f"{TASK_NAME}_demo_incomplete"
    )
    incomplete.mkdir(parents=True)
    published = tmp_path / "dataset" / ".physical_replay_pending" / (
        f"{TASK_NAME}_demo_complete"
    )
    (published / "replay.zarr").mkdir(parents=True)

    assert _completed_demo_dirs(tmp_path) == [published]


def test_live_artifacts_are_consistent_when_present() -> None:
    report = yaml.safe_load(
        (TASK_DIR / "validation_report.yaml").read_text(encoding="utf-8")
    )
    current_adjustment = report.get("current_operator_adjustment", {})
    if str(current_adjustment.get("live_core_revalidation", "")).startswith(
        "pending_"
    ):
        pytest.skip(
            "committed live artifacts are the explicitly historical 1.5x "
            "baseline; the 1.75x core audit requires GPU access"
        )
    paths = [
        REPOSITORY_ROOT / RUNTIME_USD_RELATIVE,
        REPOSITORY_ROOT / ASSET_MANIFEST_RELATIVE,
        REPOSITORY_ROOT / OVERLAY_RELATIVE,
        REPOSITORY_ROOT / PREVIEW_RELATIVE,
        REPOSITORY_ROOT / SUMMARY_RELATIVE,
        REPOSITORY_ROOT / ZERO_ACTION_RELATIVE,
        TASK_DIR / "scene_manifest.json",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    assert not missing, f"missing required AS019 artifacts: {missing}"
    asset_manifest = read_json(REPOSITORY_ROOT / ASSET_MANIFEST_RELATIVE)
    runtime = asset_manifest["runtime"]
    assert runtime["sha256"] == sha256_file(REPOSITORY_ROOT / RUNTIME_USD_RELATIVE)
    assert runtime["size_bytes"] == (REPOSITORY_ROOT / RUNTIME_USD_RELATIVE).stat().st_size
    assert runtime["rigid_owner"] == RUNTIME_RIGID_OWNER
    assert runtime["visual_mesh_paths"] == list(RUNTIME_VISUAL_MESH_PATHS)
    assert runtime["collider_paths"] == list(RUNTIME_COLLIDER_PATHS)
    assert runtime["collider_count"] == SCREWDRIVER_EXPECTED_COLLIDER_COUNT
    assert runtime["collision_approximation"] == "convexHull"
    assert runtime["starts_asleep"] is True
    assert runtime["physics_material_path"] == RUNTIME_PHYSICS_MATERIAL_PATH
    assert runtime["default_static_friction"] == SCREWDRIVER_DEFAULT_FRICTION
    assert runtime["default_dynamic_friction"] == SCREWDRIVER_DEFAULT_FRICTION
    assert runtime["isaac_sim_version"] == "5.1.0.0"
    assert runtime["stage_meters_per_unit"] == 1.0
    assert runtime["stage_up_axis"] == "Z"
    assert runtime["default_prim"] == RUNTIME_RIGID_OWNER
    assert asset_manifest["provenance"]["archive_sha256"] == SOURCE_ARCHIVE_SHA256
    source_members = asset_manifest["source"]["members"]
    if isinstance(source_members, list):
        source_members = {row["path"]: row for row in source_members}
    for member, expected_hash in SOURCE_MEMBER_SHA256.items():
        record = source_members[member]
        actual_hash = record["sha256"] if isinstance(record, dict) else record
        assert actual_hash == expected_hash
    for member, relative_texture in RUNTIME_TEXTURE_RELATIVE_BY_SOURCE_MEMBER.items():
        texture = REPOSITORY_ROOT / relative_texture
        assert texture.is_file()
        assert sha256_file(texture) == SOURCE_MEMBER_SHA256[member]
    assert SCREWDRIVER_CONTACT_OFFSET_M == 0.001
    assert SCREWDRIVER_REST_OFFSET_M == 0.0
    overlay = (REPOSITORY_ROOT / OVERLAY_RELATIVE).read_text(encoding="utf-8")
    forbidden_shared_prefix = "/" + "mnt" + "/" + "pfs" + "/"
    forbidden_home_prefix = "/" + "home" + "/"
    assert forbidden_shared_prefix not in overlay
    assert forbidden_home_prefix not in overlay

    zero = read_json(REPOSITORY_ROOT / ZERO_ACTION_RELATIVE)
    assert zero["sim_seconds"] >= 3.0
    assert zero["object_state_writes"] == 0
    assert zero["position_drift_m"] <= MAX_POSITION_DRIFT_M
    assert zero["orientation_drift_rad"] <= MAX_ORIENTATION_DRIFT_RAD
    assert zero["pass"] is True

    summary = read_json(REPOSITORY_ROOT / SUMMARY_RELATIVE)
    assert summary["core_scene_pass"] is True
    assert summary["handover_path_executed"] is False
    assert summary["grasp_executed"] is False
    assert summary["task_complete"] is False
    assert summary["head_frame_metrics"]["pass"] is True
    marker_inventory = summary["scene_inventory"]["marker"]
    np.testing.assert_allclose(
        marker_inventory["position_m"], MARKER_POSITION_M, atol=1.0e-6
    )
    actual_marker_quaternion = np.asarray(
        marker_inventory["orientation_wxyz"], dtype=np.float64
    )
    expected_marker_quaternion = np.asarray(
        MARKER_ORIENTATION_WXYZ, dtype=np.float64
    )
    assert abs(
        float(
            np.dot(
                actual_marker_quaternion
                / np.linalg.norm(actual_marker_quaternion),
                expected_marker_quaternion
                / np.linalg.norm(expected_marker_quaternion),
            )
        )
    ) == pytest.approx(1.0, abs=1.0e-6)
    assert marker_inventory["pose_matches_contract"] is True
    assert marker_inventory["size_matches_contract"] is True
    assert marker_inventory["material_matches_contract"] is True
    np.testing.assert_allclose(
        sorted(marker_inventory["world_aabb_size_m"]),
        sorted(MARKER_SIZE_M),
        atol=1.0e-6,
    )

    screwdriver_inventory = summary["scene_inventory"]["screwdriver"]
    assert screwdriver_inventory["rigid_body_count"] == 1
    assert screwdriver_inventory["visual_mesh_count"] == 1
    assert screwdriver_inventory["visual_collision_free"] is True
    assert screwdriver_inventory["colliders_hidden"] is True
    assert screwdriver_inventory["paths_match_asset_manifest"] is True

    manifest = read_json(TASK_DIR / "scene_manifest.json")
    assert manifest["claims"] == {
        "core_scene_ready": True,
        "handover_path_validated": False,
        "motion_or_grasp_validated": False,
        "task_complete": False,
    }
    assert manifest["components"]["safe_orientation_marker"]["collider_count"] == 0
    np.testing.assert_allclose(
        manifest["components"]["safe_orientation_marker"]["position_m"],
        MARKER_POSITION_M,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        manifest["components"]["safe_orientation_marker"]["orientation_wxyz"],
        MARKER_ORIENTATION_WXYZ,
        atol=1.0e-12,
    )
    assert (
        manifest["components"]["screwdriver"]["collider_count"]
        == SCREWDRIVER_EXPECTED_COLLIDER_COUNT
    )
    assert manifest["components"]["screwdriver"]["rigid_owner"] == RUNTIME_RIGID_OWNER
    assert manifest["components"]["screwdriver"]["visual_mesh_paths"] == list(
        RUNTIME_VISUAL_MESH_PATHS
    )
    assert manifest["components"]["screwdriver"]["collider_paths"] == list(
        RUNTIME_COLLIDER_PATHS
    )
    assert manifest["live_core_scene_evidence"]["object_state_writes_during_action"] == 0
    evidence = manifest["live_core_scene_evidence"]
    assert evidence["overlay_sha256"] == sha256_file(
        REPOSITORY_ROOT / OVERLAY_RELATIVE
    )
    assert evidence["preview_sha256"] == sha256_file(
        REPOSITORY_ROOT / PREVIEW_RELATIVE
    )
    assert evidence["summary_sha256"] == sha256_file(
        REPOSITORY_ROOT / SUMMARY_RELATIVE
    )
    assert evidence["zero_action_sha256"] == sha256_file(
        REPOSITORY_ROOT / ZERO_ACTION_RELATIVE
    )
    assert summary["preview_sha256"] == sha256_file(
        REPOSITORY_ROOT / PREVIEW_RELATIVE
    )
    assert manifest["components"]["screwdriver"]["runtime_sha256"] == sha256_file(
        REPOSITORY_ROOT / RUNTIME_USD_RELATIVE
    )
    assert manifest["components"]["screwdriver"]["asset_manifest_sha256"] == sha256_file(
        REPOSITORY_ROOT / ASSET_MANIFEST_RELATIVE
    )


def test_task_yaml_does_not_claim_task_completion() -> None:
    task = yaml.safe_load((TASK_DIR / "task.yaml").read_text(encoding="utf-8"))
    assert task["task"]["id"] == "bench320_as019_screwdriver_safe_orientation_handover"
    assert task["task"]["task_complete"] is False
    assert task["task"]["development_state"] == "bug_iteration"
    assert task["validation"]["workflow_state"] == "bug_iteration"
    assert (
        task["validation"]["validation_status"]
        == "partial_pass_resize_and_runtime_rate_stage0_acquisition_pending"
    )
    assert task["source"]["author"] == "Dylan Detwiller"
    assert task["source"]["license"] == "CC0-1.0"
    assert task["source"]["reference"] == "https://polyhaven.com/a/flathead_screwdriver"
    assert task["source"]["source_archive"] == str(SOURCE_ARCHIVE_RELATIVE)
    assert task["source"]["source_archive_sha256"] == SOURCE_ARCHIVE_SHA256
    assert task["assets"][0]["path"] == str(RUNTIME_USD_RELATIVE)
    assert task["assets"][0]["collision"]["approximation"] == "convexHull"
    assert task["assets"][0]["collision"]["derivation"] == "source_derived"
    assert (
        task["assets"][0]["collision"]["expected_shape_count"]
        == SCREWDRIVER_EXPECTED_COLLIDER_COUNT
    )
    assert task["assets"][1]["collision"]["enabled"] is False
    np.testing.assert_allclose(
        task["assets"][0]["reset_pose"]["position_m"],
        SCREWDRIVER_POSITION_M,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        task["assets"][0]["reset_pose"]["orientation_wxyz"],
        SCREWDRIVER_ORIENTATION_WXYZ,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        task["assets"][0]["reset_pose"]["safe_tip_axis_world"],
        SCREWDRIVER_SAFE_AXIS_WORLD,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        task["assets"][1]["reset_pose"]["position_m"],
        MARKER_POSITION_M,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        task["assets"][1]["reset_pose"]["orientation_wxyz"],
        MARKER_ORIENTATION_WXYZ,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        task["assets"][1]["size_m"], MARKER_SIZE_M, atol=1.0e-12
    )
    assert task["workspace"]["targets"][0]["arm"] == "right"
    assert task["workspace"]["targets"][1]["arm"] == "left"
    assert task["source"]["authoritative_record"]["excel_row"] == 260
    assert "handover_path" not in task


def test_source_asset_is_committed_not_an_lfs_pointer() -> None:
    source = REPOSITORY_ROOT / SOURCE_ARCHIVE_RELATIVE
    assert source.stat().st_size == SOURCE_ARCHIVE_BYTES
    assert sha256_file(source) == SOURCE_ARCHIVE_SHA256
    assert not source.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")
