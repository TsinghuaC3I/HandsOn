from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import yaml

from taskdev.tasks.bench320_bc012_dual_can_synchronized_bin_load import core_scene_spec as spec


TASK_DIR = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_exact_asset_hashes_and_static_contract() -> None:
    assert spec.validate_static_contract() == []
    for path, expected in spec.EXPECTED_SHA256.items():
        assert _sha256(path) == expected


def test_authoritative_source_row_snapshot() -> None:
    row = json.loads((TASK_DIR / "source_row.json").read_text(encoding="utf-8"))
    inventory = json.loads((TASK_DIR / "asset_inventory.json").read_text(encoding="utf-8"))
    assert row["task_id"] == spec.TASK_ID
    assert row["excel_row"] == spec.AUTHORITATIVE_EXCEL_ROW
    assert row["objective"] == "同时拾取两个罐头并放入箱内左右分区。"
    assert inventory["source"]["catalog"] == spec.AUTHORITATIVE_SOURCE_CATALOG
    assert inventory["source"]["catalog_sha256"] == spec.AUTHORITATIVE_SOURCE_CATALOG_SHA256


def test_scale_pose_orientation_and_partition_contract() -> None:
    assert spec.CAN_SOURCE_SCALE_XYZ == (1.0, 1.0, 1.0)
    assert spec.OPEN_BIN_SOURCE_SCALE_XYZ == (1.0, 1.0, 1.0)
    assert math.isclose(spec.quaternion_norm(spec.CAN_SOURCE_ORIENTATION_XYZW), 1.0)
    assert math.isclose(spec.quaternion_norm(spec.OPEN_BIN_SOURCE_ORIENTATION_XYZW), 1.0)
    assert spec.CAN_ROOT_POSITIONS_M["left"][1] > 0.0
    assert spec.CAN_ROOT_POSITIONS_M["right"][1] < 0.0
    assert math.isclose(
        spec.CAN_ROOT_POSITIONS_M["left"][0]
        - spec.OPEN_BIN_ROOT_POSITION_M[0],
        -0.03,
    )
    assert math.isclose(
        spec.CAN_ROOT_POSITIONS_M["right"][0]
        - spec.OPEN_BIN_ROOT_POSITION_M[0],
        -0.03,
    )
    assert len(spec.OPEN_BIN_COLLISION_PANELS) == 5
    assert len(spec.GOAL_ZONES) == 2
    assert spec.PARTITION_DIVIDER[1][1] > 0.0


def test_manifest_is_truthful_and_placeholder_free() -> None:
    manifest = yaml.safe_load((TASK_DIR / "task.yaml").read_text(encoding="utf-8"))
    serialized = json.dumps(manifest, sort_keys=True)
    assert "REPLACE_WITH" not in serialized
    assert manifest["task"]["id"] == spec.RUNTIME_TASK_NAME
    assert manifest["robot"]["arms"]["count"] == 2
    assert manifest["robot"]["hands"]["count"] == 2
    assert manifest["camera"]["prim_path"] == spec.HEAD_CAMERA_PRIM
    assert manifest["camera"]["resolution"] == [640, 480]
    assert set(manifest["camera"]["streams"]) == {
        "ego",
        "wrist_left",
        "wrist_right",
    }
    assert manifest["success"]["hold_steps"] == 15
    assert manifest["collection"]["min_successful_episodes"] == 1
    assert len(manifest["assets"]) == 3
    assert "crossed_transport" in manifest["task"]["taxonomy"]["interactions"]
    predicates = {
        predicate["id"]: predicate
        for predicate in manifest["success"]["predicates"]
    }
    assert predicates["left_can_in_right_partition"]["parameters"]["target"] == (
        "goal/right_bin_partition"
    )
    assert predicates["right_can_in_left_partition"]["parameters"]["target"] == (
        "goal/left_bin_partition"
    )


def test_runner_is_zero_action_core_scene_only() -> None:
    source = (TASK_DIR / "run_core_scene.py").read_text(encoding="utf-8")
    assert "self.env.step(" not in source
    assert "success_action_executed\": False" in source
    assert "grasp_tuned\": False" in source
    assert "task_complete\": False" in source
    assert '"action_commands_issued": 0' in source
    assert "RayTracedLighting" in source
    assert 'settings.set("/rtx/post/tonemap/op", 4)' in source
    assert "DomeLight" in source and "DistantLight" in source
    assert "args.width = 640" in source and "args.height = 480" in source
    assert "shutdown_guard = threading.Timer" in source
    assert "simulation_app.close(wait_for_replicator=False)" in source
    assert "/home/" not in source
    assert "10.0.0." not in source and "192.168.1." not in source


def test_scene_manifest_never_claims_task_completion() -> None:
    manifest = json.loads((TASK_DIR / "scene_manifest.json").read_text(encoding="utf-8"))
    assert manifest["task_id"] == "B320-BC012"
    assert manifest["scope"] == "full_collection_vertical_slice"
    attempts = manifest["live_attempts"]
    assert attempts[0]["status"] == "failed_before_physics"
    assert attempts[0]["static_fix_present"] is True
    assert attempts[-1]["semantic_success"] is False
    assert attempts[-1]["zarr_created"] is True
    assert attempts[-1]["preview_created"] is True
    assert manifest["claims"]["live_collection_validated"] is False
    assert manifest["claims"]["fresh_process_replay_validated"] is False
    assert manifest["claims"]["task_complete"] is False


def test_latest_live_reach_diagnosis_uses_measured_physx_palms() -> None:
    failure = json.loads(
        (
            TASK_DIR
            / "evidence/live_collection_attempt_20260817T174242Z/failure_summary.json"
        ).read_text(encoding="utf-8")
    )
    reach = failure["physx_palm_reach_diagnosis"]
    assert reach["command_state_is_not_measured_state"] is True
    assert reach["right_minimum_physx_palm_to_can_center_m"] > 0.115
    assert reach["left_minimum_physx_palm_to_can_center_m"] > 0.115
    assert failure["atomic_joint_validation"][
        "legacy_missing_body1_warning_present"
    ] is False
    assert failure["cpu_fix_after_attempt"][
        "shared_robot_controller_modified"
    ] is False


def test_translated_layout_enters_owner_radius_before_binding_failure() -> None:
    failure = json.loads(
        (
            TASK_DIR
            / "evidence/live_collection_attempt_20260817T180039Z/failure_summary.json"
        ).read_text(encoding="utf-8")
    )
    owner = failure["reach_and_owner_diagnosis"]
    assert owner["right_minimum_physx_palm_to_can_center_m"] < owner[
        "owner_radius_m"
    ]
    assert owner["left_minimum_physx_palm_to_can_center_m"] < owner[
        "owner_radius_m"
    ]
    assert owner["right_grasp_attempt_count"] > 0
    assert owner["left_grasp_attempt_count"] > 0
    assert "numpy.float32" in owner["blocking_error"]
    assert failure["cpu_fix_after_attempt"][
        "relative_joint_position_converted_to_python_float_tuple"
    ] is True


def test_handoff_and_validation_report_are_fail_closed_without_placeholders() -> None:
    handoff = (TASK_DIR / "SCENE_HANDOFF.md").read_text(encoding="utf-8")
    report = (TASK_DIR / "validation_report.yaml").read_text(encoding="utf-8")
    assert "physical-only runtime" in handoff
    assert "`task_complete=false`" in handoff
    assert "task_complete: false" in report
    assert "fresh-process replay" in handoff
    assert "REPLACE_WITH" not in report


def test_vertical_slice_runner_is_action_only_and_separates_fresh_replay() -> None:
    source = (TASK_DIR / "run_vertical_slice.py").read_text(encoding="utf-8")
    assert '--phase", required=True, choices=("collect", "replay")' in source
    assert "automatic_success_actions" in source
    assert "write_episode" in source
    assert "validate_episode" in source
    assert "source_pid == os.getpid()" in source
    assert "restore_recorded_initial_state(" not in source
    assert "write_root_pose_to_sim(" not in source
    assert "write_root_velocity_to_sim(" not in source


def test_zero_action_duration_meets_validator_floor() -> None:
    assert spec.ZERO_ACTION_STEPS / spec.PHYSICS_HZ >= 3.0


def test_glb_node_names_are_sanitized_for_usd_prim_paths() -> None:
    assert spec.usd_prim_identifier("bin_back") == "bin_back"
    assert spec.usd_prim_identifier("00_bin_back") == "_00_bin_back"
    assert spec.usd_prim_identifier("bin:back.panel") == "bin_back_panel"
