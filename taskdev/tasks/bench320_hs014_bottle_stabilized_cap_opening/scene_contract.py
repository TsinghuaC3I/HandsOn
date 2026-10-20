"""Pure-Python contract helpers for the HS014 core-scene artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TASK_ID = "B320-HS014"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_scene_manifest(manifest: dict[str, Any]) -> list[str]:
    errors = []
    if manifest.get("schema") != "hs014_core_scene_manifest_v1":
        errors.append("wrong scene-manifest schema")
    if manifest.get("task_id") != TASK_ID:
        errors.append("wrong task id")
    if manifest.get("task_complete") is not False:
        errors.append("task_complete must remain false")
    if manifest.get("scope") != "core_scene_only":
        errors.append("scope must remain core_scene_only")
    if manifest.get("exact_production_child") is not False:
        errors.append("core scene must not claim to be the exact production child")
    if manifest.get("priority_zero_qualified") is not False:
        errors.append("core scene must not claim priority-zero qualification")
    forbidden = manifest.get("forbidden_scope", {})
    for key in ("ik_path", "grasp", "insertion_motion", "success_action"):
        if forbidden.get(key) is not True:
            errors.append(f"forbidden scope is not explicit: {key}")
    assets = manifest.get("assets", {})
    for name in ("bottle", "cap"):
        item = assets.get(name, {})
        if item.get("scale_xyz") != [1.0, 1.0, 1.0]:
            errors.append(f"{name} must use identity scale")
        if not item.get("sha256"):
            errors.append(f"{name} source hash missing")
    cap = assets.get("cap", {})
    bottle = assets.get("bottle", {})
    if bottle.get("source_mesh_axis") != "+Y":
        errors.append("bottle source mesh axis must be +Y")
    if bottle.get("source_axis_semantics") != "base_to_open_mouth":
        errors.append("bottle source +Y semantics must be base-to-open-mouth")
    if bottle.get("composed_axis") != "+Z_from_base_to_open_mouth":
        errors.append("bottle must map source base-to-mouth onto local +Z")
    if bottle.get("visual_correction_wxyz") != [
        0.7071067811865476,
        0.7071067811865476,
        0.0,
        0.0,
    ]:
        errors.append("bottle visual correction changed")
    if bottle.get("visual_translation_m") != [0.0, 0.0, -0.2]:
        errors.append("bottle mouth-root translation changed")
    expected = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]
    if cap.get("orientation_xyzw") != expected:
        errors.append("cap orientation must map source +Y to world +Z")
    runtime = manifest.get("runtime", {})
    if runtime.get("head_camera_prim_path") != "/World/RecordCameras/head_camera":
        errors.append("canonical head camera path missing")
    if runtime.get("arm_count") != 2 or runtime.get("hand_count") != 2:
        errors.append("dual-arm/dual-hand runtime is required")
    rigid_solver = manifest.get("physics", {}).get("rigid_owner_solver", {})
    if rigid_solver != {
        "position_iteration_count": 16,
        "velocity_iteration_count": 8,
        "max_depenetration_velocity_m_s": 1.0,
    }:
        errors.append("bottle/cap rigid-owner solver contract changed")
    collision = manifest.get("physics", {}).get("bottle_collision", {})
    if collision.get("type") != "source_convex_decomposition_dynamic":
        errors.append("bottle must use the authored source convex decomposition")
    if collision.get("authoritative_source_collision_enabled") is not True:
        errors.append("bottle source collision must remain enabled")
    if collision.get("extra_stability_proxy_present") is not False:
        errors.append("bottle must not contain a hidden stability proxy")
    expected_supports = {
        "type": "three_invisible_spheres",
        "count": 3,
        "sphere_radius_m": 0.0015,
        "ring_radius_m": 0.0024,
        "center_z_m": -0.0055,
        "angles_deg": [0.0, 120.0, 240.0],
        "centers_xyz_m": [
            [0.0024, 0.0, -0.0055],
            [-0.0012, 0.002078460969083, -0.0055],
            [-0.0012, -0.002078460969083, -0.0055],
        ],
        "radial_envelope_m": 0.0039,
        "top_z_m": -0.004,
        "inside_bottle_mouth": True,
        "nested_rigid_owner": False,
    }
    if collision.get("cradle_supports") != expected_supports:
        errors.append("bottle cradle support geometry changed")
    cap_collision = manifest.get("physics", {}).get("cap_collision", {})
    expected_cap_collision = {
        "type": "outer_cylinder_and_com_centered_cradle_sphere",
        "axis": "Y",
        "radius_m": 0.0142,
        "height_m": 0.014,
        "center_y_m": 0.009,
        "bottom_inset_m": 0.002,
        "top_inset_m": 0.002,
        "inside_visible_geometry": True,
        "outer_physics_material_binding": "default_grasp_friction",
        "cradle_sphere": {
            "type": "invisible_sphere",
            "radius_m": 0.013,
            "center_y_m": 0.009,
            "local_aabb_m": [[-0.013, 0.0, -0.013], [0.013, 0.017, 0.013]],
            "inside_visible_bounds": True,
            "center_matches_cap_com": True,
            "nested_rigid_owner": False,
        },
        "helper_physics_material": {
            "binding_scope": "cradle_sphere_and_three_support_spheres",
            "binding_purpose": "physics",
            "contact_offset_m": 0.00005,
            "rest_offset_m": 0.0,
            "static_friction": 0.0,
            "dynamic_friction": 0.0,
            "restitution": 0.0,
            "friction_combine_mode": "min",
            "applies_to_cap_outer_or_cap_hand": False,
        },
    }
    if cap_collision != expected_cap_collision:
        errors.append("cap collision must remain inside the visible straight-wall region")
    cap_mass_properties = manifest.get("physics", {}).get(
        "cap_mass_properties", {}
    )
    expected_cap_mass_properties = {
        "mass_kg": 0.009,
        "center_of_mass_local_m": [0.0, 0.009, 0.0],
        "diagonal_inertia_kg_m2": [
            2.1e-6,
            2.8e-6,
            2.1e-6,
        ],
        "principal_axes_wxyz": [1.0, 0.0, 0.0, 0.0],
        "source": "uniform_surface_integral_of_hash_bound_visible_mesh",
    }
    if cap_mass_properties != expected_cap_mass_properties:
        errors.append("cap thin-wall mass properties changed")
    expected_topology = {
        "contact_truth_topology": "three_sphere_cradle_to_com_sphere",
        "filtered_shape_role_pairs": [
            ["bottle_source", "cap_outer"],
            ["bottle_source", "cap_cradle"],
            ["bottle_support_0", "cap_outer"],
            ["bottle_support_1", "cap_outer"],
            ["bottle_support_2", "cap_outer"],
        ],
        "unfiltered_contact_role_pairs": [
            ["bottle_support_0", "cap_cradle"],
            ["bottle_support_1", "cap_cradle"],
            ["bottle_support_2", "cap_cradle"],
        ],
        "cap_seated_height_m": -0.0002,
        "center_delta_z_m": 0.0143,
        "center_horizontal_offset_m": 0.0024,
        "center_distance_m": 0.0145,
        "radius_sum_m": 0.0145,
        "exact_initial_tangency": True,
        "contact_normals_intersect_cap_com": True,
        "axial_lift_separates_immediately": True,
    }
    if (
        manifest.get("physics", {}).get("cap_bottle_contact_topology")
        != expected_topology
    ):
        errors.append("cap-bottle torque-neutral cradle topology changed")
    evidence = manifest.get("evidence", {})
    if (
        evidence.get("status") != "historical_stale_pending_regeneration"
        or evidence.get("qualifies_current_source") is not False
        or evidence.get("invalidated_by")
        != "torque_neutral_three_sphere_cradle_20260901"
    ):
        errors.append("historical core artifact must remain explicitly stale")
    return errors


def validate_core_report(report: dict[str, Any]) -> list[str]:
    errors = []
    if report.get("task_complete") is not False:
        errors.append("runtime report must keep task_complete false")
    for field in ("ik_executed", "grasp_executed", "success_motion_executed"):
        if report.get(field) is not False:
            errors.append(f"{field} must be false")
    if report.get("action_command_count") != 0:
        errors.append("zero-action audit issued an action")
    if report.get("object_state_writes_during_no_action") != 0:
        errors.append("object state was written during zero-action audit")
    if float(report.get("zero_action_sim_seconds", 0.0)) < 10.0:
        errors.append("zero-action simulation is shorter than 10 seconds")
    if report.get("exact_production_child") is not False:
        errors.append("core scene must not claim to be the exact production child")
    if report.get("priority_zero_qualified") is not False:
        errors.append("core scene must not claim priority-zero qualification")
    checks = report.get("checks", {})
    if not checks or not all(value is True for value in checks.values()):
        errors.append("one or more core-scene runtime checks failed")
    if report.get("core_scene_pass") is not True:
        errors.append("core_scene_pass is false")
    return errors
