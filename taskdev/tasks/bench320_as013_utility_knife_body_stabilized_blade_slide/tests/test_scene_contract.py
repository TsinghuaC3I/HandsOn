from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml

from taskdev.tasks.bench320_as013_utility_knife_body_stabilized_blade_slide import (
    scene_contract,
)


TASK_DIR = Path(scene_contract.__file__).resolve().parent


def test_static_scene_contract() -> None:
    result = scene_contract.validate(require_live=False)
    assert result["pass"], result["errors"]


def test_exact_asset_bundle_is_hash_bound() -> None:
    assert scene_contract.asset_hashes() == scene_contract.EXPECTED_ASSET_SHA256
    assert (
        scene_contract.sha256_file(scene_contract.ASSET_PATH)
        == scene_contract.INTEGRATED_ASSET_SHA256
    )


def test_slider_collision_hulls_are_full_rank_and_fail_closed() -> None:
    metadata = json.loads(scene_contract.METADATA_PATH.read_text(encoding="utf-8"))
    decomposition = metadata["slider_collision_decomposition"]
    assert decomposition["method"] == scene_contract.SLIDER_COLLISION_METHOD
    assert decomposition["hull_count"] == scene_contract.SLIDER_COLLIDER_COUNT
    assert decomposition["certification_samples_per_hull"] == 30_000
    assert decomposition["geometry_quality_gates"] == (
        scene_contract.SLIDER_GEOMETRY_QUALITY_GATES
    )
    assert decomposition["geometry_quality_all_pass"] is True
    assert decomposition["sampled_surface_gap_max_m"] <= 0.003

    slider_colliders = [
        collider
        for collider in metadata["colliders"]
        if collider["role"] == "slider"
    ]
    assert len(slider_colliders) == 6
    for collider in slider_colliders:
        quality = collider["geometry_quality"]
        assert quality["pass"] is True
        assert quality["affine_rank"] == 3
        assert all(quality["checks"].values())
        assert quality["minimum_axis_extent_m"] >= 0.00030
        assert quality["minimum_singular_value_m"] >= 0.00009
        assert quality["singular_value_ratio"] >= 0.002
        assert quality["volume_m3"] >= 5.0e-10
        assert collider["sampled_surface_gap_m"] <= 0.003
    assert (
        scene_contract.sha256_file(scene_contract.METADATA_PATH)
        == scene_contract.INTEGRATED_METADATA_SHA256
    )


def test_blade_q0_envelope_is_inside_housing_with_clearance() -> None:
    metadata = json.loads(scene_contract.METADATA_PATH.read_text(encoding="utf-8"))
    correction = metadata["blade_envelope_correction"]
    assert correction["method"] == scene_contract.BLADE_ENVELOPE_METHOD
    assert correction["axis"] == "Y"
    assert correction["clearance_m"] == scene_contract.BLADE_RETRACTED_CLEARANCE_M
    assert correction["q0_fully_retracted"] is True
    housing = metadata["housing_bounds_m"]
    blade = metadata["blade_bounds_m"]
    clearance = float(correction["clearance_m"])
    assert blade["min_m"][1] >= housing["min_m"][1] + clearance - 1.0e-9
    assert blade["max_m"][1] <= housing["max_m"][1] - clearance + 1.0e-9
    assert blade["extent_m"][1] <= housing["extent_m"][1] - 2.0 * clearance + 1.0e-9
    assert correction["corrected_bounds_m"] == blade


def test_selected_download_and_task_yaml_are_fail_closed() -> None:
    workflow = yaml.safe_load(
        (TASK_DIR / "workflow_state.yaml").read_text(encoding="utf-8")
    )
    assert workflow["selected_asset"] == "as013-sketchfab-ahtusik"
    assert workflow["download_checkpoint"]["license_verified"] is True
    assert workflow["download_checkpoint"]["source_hash_bound"] is True
    task = yaml.safe_load((TASK_DIR / "task.yaml").read_text(encoding="utf-8"))
    assert task["task"]["task_complete"] is False
    assert task["source"]["license"] == "CC-BY-4.0"
    assert task["assets"][0]["scale_m"] == [1.0, 1.0, 1.0]
    articulation = task["assets"][0]["articulation"]
    assert articulation["drive_stiffness"] == 80.0
    assert articulation["drive_damping"] == 25.0
    assert articulation["joint_static_friction"] == 3.0
    assert articulation["joint_dynamic_friction"] == 2.4
    assert articulation["blade_contact_static_friction_default"] == 2.0
    assert articulation["blade_contact_friction_combine_mode"] == "max"
    assert articulation["housing_contact_static_friction_default"] == 2.5
    assert articulation["housing_contact_dynamic_friction_default"] == 2.3
    assert articulation["housing_contact_static_friction_range"] == [2.3, 2.7]
    assert articulation["housing_contact_friction_combine_mode"] == "max"
    assert articulation["housing_grip_material_scope"] == (
        "single_housing_collider_including_side_faces"
    )
    assert articulation["source_asset_drive_damping"] == 1.2
    assert articulation["runtime_target_policy"] == {
        "passive_target": (
            "measured_joint_position_during_action_with_0_1mm_write_deadband"
        ),
        "passive_target_requires_contact": False,
        "stable_origin_target_writes": (
            "forbidden_reset_target_already_synchronized"
        ),
        "maximum_deadband_restoring_force_n": 0.008,
        "cumulative_return_force_to_reset_pose": False,
        "causal_forward_assist_default": False,
        "causal_forward_assist_gate": (
            "ordered_right_grip_seen_then_pickup_seen_then_"
            "left_fingertip_slider_contact_plus_joint_axis_operator_motion"
        ),
    }


def test_runtime_slider_contract_rejects_a_stale_reset_target() -> None:
    contract = (TASK_DIR / "COLLECTION_TASK_CONTRACT.md").read_text(
        encoding="utf-8"
    )
    worklog = (TASK_DIR / "WORKLOG.md").read_text(encoding="utf-8")
    assert "follows the measured blade DOF during action" in contract
    assert "Stable-origin settling performs no task-target" in contract
    assert "`0.1 mm` write deadband" in contract
    assert "`0.008 N`" in contract
    assert "without accumulating meaningful displacement error" in contract
    assert "`25.0 N*s/m` damping" in contract
    assert "`3.0` static" in contract
    assert "`2.4`\n  dynamic" in contract
    assert "PhysX `max` friction combine mode" in contract
    assert "returning the\n  slider toward its reset position" in contract
    assert "Diagnostic action path (not completion evidence)" in contract
    assert "right-hand housing grasp with a target-direction-aware" in contract
    assert "cannot serve as\n  completion evidence in this round" in contract
    assert "optional causal prismatic forward\n  assist is disabled by" in contract
    assert "the 16-value public\n  `knife_interaction_state`" in contract
    assert "complete current/next replay\n  contract" in contract
    assert "no fresh-process full replay is claimed" in contract
    assert "blade position `0.0632099 m`" in worklog
    assert "drive target remained at" in worklog
    assert "Fresh manual" in worklog
    assert "remain pending" in worklog


def test_current_operator_roles_are_consistent_across_contracts() -> None:
    expected_prompt = (
        "Pick up the utility knife with the right hand. After pickup, touch the "
        "slider with the left fingertips and move it to the episode target: "
        "bottom to retract the blade or top to extend it."
    )
    workflow = yaml.safe_load(
        (TASK_DIR / "workflow_state.yaml").read_text(encoding="utf-8")
    )
    task = yaml.safe_load((TASK_DIR / "task.yaml").read_text(encoding="utf-8"))
    multi_task = yaml.safe_load(
        (
            scene_contract.REPOSITORY_ROOT
            / "configs"
            / "multi_task"
            / "t40_slide_knife_blade.yaml"
        ).read_text(encoding="utf-8")
    )
    contract_text = (TASK_DIR / "COLLECTION_TASK_CONTRACT.md").read_text(
        encoding="utf-8"
    )
    handoff_text = (TASK_DIR / "SCENE_HANDOFF.md").read_text(encoding="utf-8")
    worklog_text = (TASK_DIR / "WORKLOG.md").read_text(encoding="utf-8")
    validation = yaml.safe_load(
        (TASK_DIR / "validation_report.yaml").read_text(encoding="utf-8")
    )

    assert workflow["contract"]["operator_prompt"] == expected_prompt
    assert workflow["contract"]["operation_mode"] == (
        "bimanual_handheld_right_pickup_left_touch_bidirectional_endpoint"
    )
    assert workflow["contract"]["operator_roles"] == {
        "left": "distal_fingertips_touch_and_operate_slider_after_pickup",
        "right": "full_hand_multi_point_housing_grasp_and_pickup",
    }
    assert workflow["contract"]["completion_requires_release"] is False
    assert workflow["contract"]["completion_requires_table_support"] is False
    assert workflow["contract"]["both_arms_active_in_hardware_interface"] is True
    assert multi_task["prompt"] == expected_prompt
    assert multi_task["robot_type"] == "dual_arm"
    assert "right-palm\n  contact is diagnostic and is not individually required" in (
        contract_text
    )
    assert "at least two distinct rigid bodies from the full\n  right hand" in (
        contract_text
    )
    assert "left distal fingertip" in contract_text
    assert "housing/table and moving-component/table contact signals are false" in (
        contract_text
    )
    assert "Current right grip" in contract_text
    assert "multi-frame terminal hold are not success predicates" in contract_text
    assert "Incidental hand contact\n  with the knife" in contract_text
    assert "never terminates the\n  episode" in contract_text
    assert "Pre-pickup slider travel is recoverable" in contract_text
    assert "not credited as\n  ordered operation progress" in contract_text
    assert "relative XY displacement-from-reset threshold" in contract_text
    assert "normalized progress\n  `<=0.05`" in contract_text
    for text in (handoff_text, worklog_text):
        assert "pickup_seen" in text
        assert "endpoint" in text
        assert "workspace" in text
        assert "recoverable" in text

    validation_contract = validation["current_operator_contract"]
    assert validation["task_complete"] is False
    assert validation_contract["mode"] == (
        "bimanual_handheld_right_pickup_left_touch_bidirectional_endpoint"
    )
    assert validation_contract["prepickup_left_fingertip_button_contact"] == (
        "allowed_but_not_counted"
    )
    assert validation_contract["prepickup_slider_progress"] == (
        "recoverable_but_not_counted"
    )
    assert validation_contract["incidental_hand_contact"] == (
        "allowed_diagnostic_only_never_terminal"
    )
    assert validation_contract["runtime_slider_resistance"] == {
        "drive_damping": 25.0,
        "joint_static_friction": 3.0,
        "joint_dynamic_friction": 2.4,
        "blade_contact_static_friction_default": 2.0,
        "friction_combine_mode": "max",
        "manual_hand_feel_status": "pending",
    }
    assert validation_contract["grip_contact_now_allows_motion"] is True
    assert validation_contract["current_terminal_contact_required"] is False
    assert validation_contract["held_stable_now_required_for_completion"] is False
    assert validation_contract[
        "relative_xy_displacement_over_0_060_m_terminal"
    ] is False

    config_source = (
        scene_contract.REPOSITORY_ROOT
        / "handson"
        / "handson"
        / "tasks"
        / "t40_slide_knife_blade"
        / "config.py"
    ).read_text(encoding="utf-8")
    config_tree = ast.parse(config_source)
    task_prompt_values = [
        ast.literal_eval(node.value)
        for node in ast.walk(config_tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "task_prompt"
            for target in node.targets
        )
    ]
    assert task_prompt_values == [expected_prompt]

    assert task["robot"]["arms"]["count"] == 2
    assert task["robot"]["hands"]["count"] == 2
    assert task["task"]["taxonomy"]["handedness"] == (
        "bimanual_right_hold_left_slide"
    )
    assert "inactive_arm" not in task["workspace"]

    targets = {target["id"]: target for target in task["workspace"]["targets"]}
    assert targets["left_fingertip_blade_slider"]["arm"] == "left"
    assert targets["left_fingertip_blade_slider"]["digits"] == [
        "thumb", "index", "middle", "ring", "pinky"
    ]
    assert targets["right_hand_housing_grasp"] == {
        "id": "right_hand_housing_grasp",
        "arm": "right",
        "digits": ["thumb", "index", "middle", "ring", "pinky", "palm"],
        "frame": "world",
        "position_m": [-0.12, -0.045, 0.96],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "clearance_m": 0.02,
    }

    monitored_pairs = {
        pair["id"]: pair["bodies"] for pair in task["contact"]["monitored_pairs"]
    }
    assert monitored_pairs == {
        "right_hand_knife_housing": [
            "robot/right_hand/*",
            "assets/utility_knife/body",
        ],
        "left_fingertip_blade_slider": [
            "robot/left_hand/fingertips",
            "assets/utility_knife/blade_slider",
        ],
        "housing_table_support": [
            "assets/utility_knife/body",
            "environment/table",
        ],
        "moving_component_table_support": [
            "assets/utility_knife/blade_slider",
            "environment/table",
        ],
    }
    allowed_pairs = {
        pair["id"]: pair for pair in task["contact"]["allowed_pairs"]
    }
    assert set(allowed_pairs) == {"left_fingertips_slider_button"}
    allowed_moving_contact = allowed_pairs[
        "left_fingertips_slider_button"
    ]
    assert allowed_moving_contact["bodies"] == [
        "robot/left_hand/left_finger1_to_5_link4",
        "assets/utility_knife/blade_slider",
    ]
    assert allowed_moving_contact["hand_region"] == "any_left_distal_fingertip"
    assert allowed_moving_contact["object_region"] == (
        "slider_button_local_aabb"
    )
    assert allowed_moving_contact["local_aabb_source"].endswith(
        "#button_region.local_bounds_m"
    )
    assert allowed_moving_contact["completion_evidence_contact"] is True
    metadata = json.loads(
        scene_contract.METADATA_PATH.read_text(encoding="utf-8")
    )
    button_bounds = metadata["button_region"]["local_bounds_m"]
    assert allowed_moving_contact["local_aabb_m"] == {
        "min": button_bounds["min_m"],
        "max": button_bounds["max_m"],
    }
    forbidden_pairs = {
        pair["id"]: pair for pair in task["contact"]["forbidden_pairs"]
    }
    assert "right_hand_blade_edge" not in forbidden_pairs
    assert set(forbidden_pairs) == {"robot_table"}
    incidental_pairs = {
        pair["id"]: pair for pair in task["contact"]["incidental_pairs"]
    }
    assert incidental_pairs["either_hand_nonbutton_contact"]["bodies"] == [
        "robot/left_or_right_hand/*",
        "assets/utility_knife/*",
    ]
    assert incidental_pairs["either_hand_nonbutton_contact"]["terminal"] is False
    assert incidental_pairs["either_hand_nonbutton_contact"][
        "completion_evidence"
    ] is False

    predicates = {
        predicate["id"]: predicate for predicate in task["success"]["predicates"]
    }
    assert predicates["pickup_seen"]["parameters"] == {
        "entity": "utility_knife_body",
        "reference": "authored_or_reset_housing_root_height",
        "min_delta_m": 0.02,
        "requires_current_during_acquisition": {
            "right_hand_housing_contact": True,
            "minimum_distinct_contact_owners": 2,
            "palm_contact_required": False,
        },
        "housing_table_contact_current": False,
        "moving_component_table_contact_current": False,
        "debounce_physics_substeps": 4,
    }
    assert predicates["left_slider_contact_seen"]["parameters"] == {
        "arm": "left",
        "hand_bodies": [
            "left_finger1_link4", "left_finger2_link4", "left_finger3_link4",
            "left_finger4_link4", "left_finger5_link4",
        ],
        "hand_region": "any_distal_fingertip",
        "object_region": "slider_button_local_aabb",
        "object_region_frame": "assets/utility_knife/blade_slider",
        "requires": ["pickup_seen"],
        "debounce_physics_substeps": 4,
        "prepickup_contact": "allowed_but_not_counted",
    }
    assert predicates["sampled_target_endpoint_reached"] == {
        "id": "sampled_target_endpoint_reached",
        "type": "randomized_bidirectional_joint_endpoint",
        "parameters": {
            "joint": "PrismaticJoint_utility_knife_3_middle",
            "target_choice": ["retracted_bottom", "extended_top"],
            "retracted_bottom_normalized_progress_max": 0.05,
            "extended_top_normalized_progress_min": 0.95,
            "requires_latched": ["pickup_seen", "left_slider_contact_seen"],
            "requires_current_contact": False,
            "requires_terminal_stability_hold": False,
        },
    }
    assert set(predicates) == {
        "pickup_seen",
        "left_slider_contact_seen",
        "sampled_target_endpoint_reached",
    }
    assert task["success"]["hold_steps"] == 1
    assert "safe_release" not in predicates
    assert task["failure"]["recoverable_guards"][0] == {
        "id": "ordinary_root_xy_displacement_from_reset",
        "former_relative_threshold_m": 0.06,
        "terminal": False,
        "effect": "clear_or_pause_progress_only",
    }
    hard_failures = {
        failure["id"]: failure
        for failure in task["failure"]["hard_terminal_failures"]
    }
    assert set(hard_failures) == {
        "knife_dropped_after_pickup",
        "nonfinite_physics_state",
        "contact_views_unavailable",
        "contact_report_invalid",
        "knife_escaped_handheld_workspace",
        "blade_joint_out_of_limits",
    }
    recoverable = {
        guard["id"]: guard for guard in task["failure"]["recoverable_guards"]
    }
    assert recoverable["blade_progress_before_pickup"] == {
        "id": "blade_progress_before_pickup",
        "normalized_progress_delta_from_reset_gt": 0.03,
        "terminal": False,
        "effect": "do_not_credit_as_ordered_left_fingertip_slide_progress",
    }
    assert recoverable["either_hand_incidental_nonbutton_contact"] == {
        "id": "either_hand_incidental_nonbutton_contact",
        "contact_force_threshold_n": 0.01,
        "terminal": False,
        "effect": (
            "diagnostic_only_left_button_region_still_required_for_completion"
        ),
    }
    assert hard_failures["knife_dropped_after_pickup"]["parameters"] == {
        "requires": {
            "pickup_seen": True,
            "right_grip_contact_now": False,
            "contact_report_valid": True,
        },
        "any_of": [
            {"pickup_lift_m_lte": 0.005},
            {
                "all_of": [
                    {"pickup_peak_minus_current_lift_m_gte": 0.050},
                    {"body_velocity_report_valid": True},
                    {
                        "any_of": [
                            {"body_vertical_velocity_mps_lte": -0.02},
                            {
                                "housing_or_moving_table_impact_force_n_gt": 0.1
                            },
                        ]
                    },
                ]
            },
        ],
        "debounce_physics_substeps": 4,
    }
    assert hard_failures["knife_escaped_handheld_workspace"]["bounds_m"] == {
        "min": [-0.65, -0.60, 0.84],
        "max": [0.30, 0.60, 1.44],
    }
    assert hard_failures["blade_joint_out_of_limits"]["parameters"] == {
        "joint": "PrismaticJoint_utility_knife_3_middle",
        "authored_limits_m": [0.0, 0.08],
        "tolerance_m": 0.0001,
    }
    for failure_id in (
        "nonfinite_physics_state",
        "contact_views_unavailable",
        "contact_report_invalid",
    ):
        assert hard_failures[failure_id]["immediate"] is True
    assert all(failure["terminal"] is True for failure in hard_failures.values())

    collection = task["collection"]
    assert collection["diagnostic_action_path"] == {
        "flag": "--scripted_as013_success=True",
        "implementation_status": "implemented",
            "completion_evidence": False,
            "right_hand_grasp_pose_live_calibration": "pending",
            "left_fingertip_slider_trajectory_live_calibration": "pending",
        }
    assert collection["public_interaction_state"]["shape"] == [16]
    assert collection["public_interaction_state"]["layout"][-3:] == [
        "target_direction",
        "target_progress",
        "target_endpoint_reached",
    ]
    assert collection["public_interaction_state"]["recorded"] is True
    assert collection["public_metrics_recorded"] is True
    assert collection[
        "full_debounce_latch_hidden_current_next_replay_contract"
    ] == "pending"
    assert collection["fresh_process_full_replay_status"] == "pending"
    assert collection["task_complete"] is False

    ui_source = (
        scene_contract.REPOSITORY_ROOT / "collection_ui" / "server.py"
    ).read_text(encoding="utf-8")
    assert "B320-AS013 Bimanual Handheld Utility Knife Slide" in ui_source
    assert "Pre-pickup contact is recoverable but is not credited" in ui_source
    assert "RETRACT means move the slider to the bottom/inside endpoint" in ui_source
    assert "EXTEND means move it to the top/outside endpoint" in ui_source
    assert "blade progress more than 3% beyond reset before pickup" not in ui_source
    assert "no final contact or stability hold is required" in ui_source

    preview_source = (
        scene_contract.REPOSITORY_ROOT
        / "handson"
        / "handson"
        / "sim"
        / "isaaclab"
        / "task_preview.py"
    ).read_text(encoding="utf-8")
    assert "Right hand picks up the utility knife by at least 2 cm" in preview_source
    assert "Left fingertips touch the slider after pickup" in preview_source
    assert "Per-episode target is randomized: retract to bottom or extend to top" in preview_source
    assert "Success is immediate at the requested endpoint; no final hold" in preview_source


def test_collection_checkpoint_claims_certified_live_completion() -> None:
    manifest = json.loads(
        (TASK_DIR / "collection_checkpoint_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["task_code"] == "B320-AS013"
    # A fresh download invalidates the previous live proof.  Keep the old
    # episode under historical_live_evidence, but expose only fail-closed
    # pending claims for the selected asset.
    assert manifest["claims"] == {
        "cpu_checkpoint_ready": True,
        "live_episode_pass": False,
        "fresh_process_replay_pass": False,
        "task_complete": False,
    }
    assert manifest["live_evidence"]["status"] == (
        "pending_fresh_bimanual_handheld_recertification"
    )
    assert manifest["current_operator_contract"]["mode"] == (
        "bimanual_handheld_right_grip_left_slide"
    )
    assert manifest["current_operator_contract"]["release_required"] is False
    assert manifest["current_operator_contract"][
        "table_support_required_at_completion"
    ] is False
    assert manifest["current_operator_contract"][
        "grip_contact_now_allows_motion"
    ] is True
    assert manifest["current_operator_contract"][
        "held_stable_now_only_after_completion"
    ] is True
    assert manifest["current_operator_contract"][
        "relative_xy_displacement_over_0.060_m_terminal"
    ] is False
    assert "historical_live_evidence" in manifest
    assert manifest["historical_live_evidence"]["successful_episode"]["episode_published"] is True
    assert manifest["historical_live_evidence"]["fresh_process_replay"]["status"] == "passed"
    assert manifest["collection_capability"]["registered"] is True
    assert manifest["runtime"]["physics_backend"] == "physx"
    assert manifest["runtime"]["newton_allowed"] is False
    assert manifest["runtime"]["sim6_allowed"] is False


def test_live_evidence_is_certified_after_scheduler_runs() -> None:
    report = (TASK_DIR / "evidence" / "live_scene_report.json")
    preview = TASK_DIR / "previews" / "reset_head.png"
    if report.is_file() or preview.is_file():
        assert report.is_file() and preview.is_file()
        assert not scene_contract.live_evidence_errors()
    else:
        validation = (TASK_DIR / "validation_report.yaml").read_text(
            encoding="utf-8"
        )
        assert "pending_fresh_asset_live_evidence" in validation
        assert "status: pending_fresh_asset_recertification" in validation


def test_handoff_binds_discovered_articulation_and_reset_only_state_writes() -> None:
    handoff = (TASK_DIR / "SCENE_HANDOFF.md").read_text(encoding="utf-8")
    assert "ArticulationRootAPI" in handoff
    assert "never calls" in handoff
    assert "reset boundary" in handoff
    assert "action frame zero" in handoff


def test_static_asset_audit_is_bound_and_live_fail_closed() -> None:
    audit = json.loads(
        (TASK_DIR / "evidence" / "static_asset_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["schema"] == "as013_static_asset_audit_v4"
    assert audit["status"] == "static_pass_live_gates_pending"
    assert audit["asset"]["task_local_usda_sha256"] == scene_contract.INTEGRATED_ASSET_SHA256
    assert audit["stage"]["orientation_preserving"] is True
    assert audit["support_audit"]["com_projection_inside"] is True
    assert len(audit["collision_geometry"]) == scene_contract.TOTAL_COLLIDER_COUNT
    assert all(
        collider["computed_visibility"] == "invisible"
        for collider in audit["collision_geometry"]
    )
    quality_audit = audit["slider_collider_quality_audit"]
    assert quality_audit["status"] == "pass"
    assert quality_audit["welded_component_count"] == 6
    assert all(quality_audit["assertions"].values())
    proxy_audit = audit["same_body_proxy_audit"]
    assert (
        proxy_audit[
            "slider_conservative_independent_all_6_hulls_30000_each_gap_max_m"
        ]
        <= 0.003
    )
    coverage = audit["visual_to_collider_coverage_audit"]
    assert coverage["sample_count"] == 30_000
    assert coverage["maximum_distance_m"] <= 0.003
    assert coverage["count_over_0_003_m"] == 0
    assert coverage["status"] == "static_pass_live_hand_sweep_pending"
    workflow = yaml.safe_load(
        (TASK_DIR / "workflow_state.yaml").read_text(encoding="utf-8")
    )
    assert workflow["state"] == "physical_integration_audit"
    assert "first_physx_overlap_and_depenetration_audit" in workflow[
        "physical_integration_audit"
    ]["pending_live_checks"]
