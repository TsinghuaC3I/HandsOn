#!/usr/bin/env python3
"""Static and evidence-backed contract helpers for the AS013 core scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


TASK_ID = "bench320_as013_utility_knife_body_stabilized_blade_slide"
TASK_CODE = "B320-AS013"
TASK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_DIR.parents[2]
ASSET_ROOT = TASK_DIR / "assets" / "utility_knife_ahtusik"
SOURCE_ROOT = ASSET_ROOT / "source" / "sketchfab_utility_knife"
RUNTIME_ROOT = ASSET_ROOT / "runtime"
ASSET_RELATIVE = Path(
    "assets/utility_knife_ahtusik/runtime/utility_knife_ahtusik.usda"
)
ASSET_PATH = TASK_DIR / ASSET_RELATIVE
RUNTIME_ASSET_RELATIVE = Path(
    "handson/handson/sim/isaaclab/usd/"
    "t40_slide_knife_blade/"
    "utility_knife_ahtusik/utility_knife_ahtusik.usda"
)
RUNTIME_ASSET_PATH = REPOSITORY_ROOT / RUNTIME_ASSET_RELATIVE
METADATA_PATH = RUNTIME_ROOT / "utility_knife_ahtusik_metadata.json"
RUNTIME_METADATA_PATH = RUNTIME_ASSET_PATH.with_name(
    "utility_knife_ahtusik_metadata.json"
)
ARCHIVE_PATH = ASSET_ROOT / "source" / "utility_knife.zip"
STATIC_AUDIT_PATH = TASK_DIR / "evidence" / "static_asset_audit.json"
INTEGRATED_ASSET_SHA256 = "2722183ee64303f8c85dac65f17acbea0977ed4d9604fd589ee5bf627b7a18d7"
INTEGRATED_METADATA_SHA256 = "f67a448283852e90a4349947a0e342245f04a65f3533eaa78621947218f725f6"
JOINT_NAME = "PrismaticJoint_utility_knife_3_middle"
# The generated USD is authored in metres.  The one-time source conversion is
# recorded separately in metadata and is deliberately not hidden as a runtime
# non-uniform scale.
UNIFORM_SCALE = 1.0
SOURCE_CONVERSION_SCALE = 0.008
ERGONOMIC_SCALE_XYZ = [1.30, 1.30, 2.35]
BLADE_ENVELOPE_METHOD = "baked_blade_y_fit_to_retracted_housing_envelope"
BLADE_RETRACTED_CLEARANCE_M = 0.004
SLIDER_COLLIDER_COUNT = 6
MOVING_COLLIDER_COUNT = 8
TOTAL_COLLIDER_COUNT = 9
SLIDER_COLLISION_METHOD = (
    "coordinate_welded_shared_edge_component_convex_hulls"
)
SLIDER_COLLIDER_AUDIT_SAMPLES = 30_000
SLIDER_COLLIDER_SURFACE_GAP_TARGET_M = 0.003
SLIDER_GEOMETRY_QUALITY_GATES = {
    "minimum_vertex_count": 4,
    "minimum_face_count": 4,
    "finite": True,
    "affine_rank": 3,
    "minimum_axis_extent_m": 0.00030,
    "minimum_singular_value_m": 0.00009,
    "minimum_singular_value_ratio": 0.002,
    "minimum_volume_m3": 5.0e-10,
    "watertight": True,
    "winding_consistent": True,
}
PHYSICS_HZ = 120
ZERO_ACTION_STEPS = 1200
HEAD_RESOLUTION_HW = (480, 640)
# These hashes bind the downloaded archive and every extracted source file.  A
# generated USD is checked independently against its metadata and runtime
# mirror; it is intentionally not substituted for source identity.
EXPECTED_ASSET_SHA256 = {
    "source/utility_knife.zip": "85fe8c49f43383aa15e5ea9803dfca52fec196d3dc00424ea8eaaca326aa648b",
    "source/sketchfab_utility_knife/license.txt": "5f89a19cb535af5dd35016f8315ff3c4ea1cc07bd58da421f5eb7a05642eddd1",
    "source/sketchfab_utility_knife/scene.bin": "5b41b20a8a2a0c523fcb06dcca2da09301dbebbb7ea09cb37a14becef81755ba",
    "source/sketchfab_utility_knife/scene.gltf": "7e155ea959934f7d37380e1d0eb5f35d0d36076dd71d063f68b66acc5b8100ee",
    "source/sketchfab_utility_knife/textures/Material.001_baseColor.png": "fea7c19eb3a7898bdf6edf231eefe110ebd2a0c09dc519a40f4791d2b0ae507b",
    "source/sketchfab_utility_knife/textures/Material.001_metallicRoughness.png": "a2e686f398053f2ae43aa98eaab9e2c33f4f695a8b94cbb43a8a237abf8066ca",
    "source/sketchfab_utility_knife/textures/Material.001_normal.png": "40f7b47ae84a175ec7bb83256e2f6bee03d9e663ec2a598c645ee59d468b04d7",
}
EXPECTED_SOURCE_SHA256 = EXPECTED_ASSET_SHA256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def image_metrics(path: Path) -> dict[str, Any]:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float64)
    return {
        "shape_hwc": [int(array.shape[0]), int(array.shape[1]), 3],
        "mean_rgb": float(array.mean()),
        "max_rgb": int(array.max()),
        "pixel_std": float(array.std()),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def asset_hashes() -> dict[str, str]:
    return {
        relative: sha256_file(ASSET_ROOT / relative)
        for relative in EXPECTED_ASSET_SHA256
        if (ASSET_ROOT / relative).is_file()
    }


def _metadata_text() -> str:
    if not METADATA_PATH.is_file():
        return ""
    return METADATA_PATH.read_text(encoding="utf-8")


def _metadata_value(payload: Any, *keys: str) -> Any:
    """Find a metadata value through shallow/nested provenance blocks.

    Asset builders have historically used both a flat provenance object and
    nested ``source``/``conversion`` objects.  Keeping this lookup tolerant
    lets the contract validate identity without prescribing a private builder
    schema.
    """

    # Treat camelCase and snake_case metadata keys equivalently, but do not
    # use a broad alias such as ``source`` for a URL: builders commonly encode
    # ``source`` as a full dictionary and that must not be mistaken for the
    # scalar ``source_url`` value.
    wanted = {re.sub(r"[^a-z0-9]", "", key.lower()) for key in keys}
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized in wanted:
                return value
        for value in payload.values():
            found = _metadata_value(value, *keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _metadata_value(value, *keys)
            if found is not None:
                return found
    return None


def _metadata_errors() -> list[str]:
    errors: list[str] = []
    if not METADATA_PATH.is_file():
        return [f"integrated asset metadata is missing: {METADATA_PATH}"]
    raw = _metadata_text()
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [f"integrated asset metadata is not valid JSON: {exc}"]
    if not isinstance(metadata, dict):
        return ["integrated asset metadata must be a JSON object"]

    # Identity and license are checked both structurally (when present) and
    # textually.  The text checks prevent an apparently valid metadata schema
    # from silently dropping the exact downloaded provenance.
    archive_sha = _metadata_value(
        metadata, "archive_sha256", "source_archive_sha256", "download_sha256"
    )
    if archive_sha != EXPECTED_ASSET_SHA256["source/utility_knife.zip"]:
        errors.append("metadata archive SHA-256 does not match the downloaded ZIP")
    source_url = _metadata_value(metadata, "source_url", "sourceUrl")
    if source_url is not None and "083c2d52b3d04eedb5350cce6e2d1b92" not in str(source_url):
        errors.append("metadata source URL is not the selected Sketchfab model")
    author = _metadata_value(metadata, "author", "creator")
    if author is not None and "ahtusik" not in str(author).lower():
        errors.append("metadata author is not ahtusik")
    conversion = _metadata_value(
        metadata,
        "conversion_scale",
        "conversion_scale_to_m",
        "source_scale_to_m",
        "scale_to_meters",
    )
    if conversion is not None:
        try:
            if not math.isclose(float(conversion), SOURCE_CONVERSION_SCALE, abs_tol=1.0e-12):
                errors.append("metadata source conversion scale is not 0.008")
        except (TypeError, ValueError):
            errors.append("metadata source conversion scale is not numeric")
    for token in (
        EXPECTED_ASSET_SHA256["source/utility_knife.zip"],
        "083c2d52b3d04eedb5350cce6e2d1b92",
        "ahtusik",
        "CC-BY-4.0",
        "0.008",
        JOINT_NAME,
        "E_body_1",
        "E_link_blade_2",
    ):
        if token.lower() not in raw.lower():
            errors.append(f"metadata is missing provenance/structure token: {token}")

    ergonomic = metadata.get("ergonomic_resize", {})
    if ergonomic.get("scale_xyz") != ERGONOMIC_SCALE_XYZ:
        errors.append("metadata ergonomic XYZ resize does not match the audited build")
    try:
        button_width = float(ergonomic["button_actual_width_m"])
        button_protrusion = float(ergonomic["button_actual_protrusion_m"])
        if not math.isclose(button_width, 0.023, abs_tol=1.0e-9):
            errors.append("metadata button width is not the audited 23 mm")
        if not math.isclose(button_protrusion, 0.0025, abs_tol=1.0e-9):
            errors.append("metadata button protrusion is not the audited 2.5 mm")
    except (KeyError, TypeError, ValueError):
        errors.append("metadata ergonomic button record is malformed")

    # q=0 is the fully retracted joint pose.  The source blade originally
    # extended ~29 mm past the housing's positive Y end; require the baked
    # envelope correction to cover both visual vertices and the bounds used to
    # regenerate the blade convex hull.  Keep this check numeric (rather than
    # trusting a descriptive flag) so a stale sidecar cannot pass silently.
    envelope = metadata.get("blade_envelope_correction", {})
    if envelope.get("method") != BLADE_ENVELOPE_METHOD:
        errors.append("metadata blade envelope correction method is missing or stale")
    try:
        clearance = float(envelope["clearance_m"])
        housing_bounds = metadata["housing_bounds_m"]
        blade_bounds = metadata["blade_bounds_m"]
        housing_min_y = float(housing_bounds["min_m"][1])
        housing_max_y = float(housing_bounds["max_m"][1])
        blade_min_y = float(blade_bounds["min_m"][1])
        blade_max_y = float(blade_bounds["max_m"][1])
        housing_extent_y = float(housing_bounds["extent_m"][1])
        blade_extent_y = float(blade_bounds["extent_m"][1])
        if not math.isclose(clearance, BLADE_RETRACTED_CLEARANCE_M, abs_tol=1.0e-12):
            errors.append("metadata blade retracted clearance is not 4 mm")
        tolerance = 1.0e-8
        if blade_min_y < housing_min_y + clearance - tolerance:
            errors.append("q=0 blade negative end exits the housing envelope")
        if blade_max_y > housing_max_y - clearance + tolerance:
            errors.append("q=0 blade positive tip exits the housing envelope")
        if blade_extent_y > housing_extent_y - 2.0 * clearance + tolerance:
            errors.append("q=0 blade length is not shorter than the housing with clearance")
        if envelope.get("q0_fully_retracted") is not True:
            errors.append("metadata q0_fully_retracted contract is not true")
        corrected = envelope.get("corrected_bounds_m", {})
        if (
            corrected.get("min_m") != blade_bounds.get("min_m")
            or corrected.get("max_m") != blade_bounds.get("max_m")
        ):
            errors.append("metadata corrected blade bounds disagree with blade_bounds_m")
    except (KeyError, IndexError, TypeError, ValueError):
        errors.append("metadata blade envelope correction record is malformed")
    collision_summary = metadata.get("collision_summary", {})
    if collision_summary.get("slider_colliders") != SLIDER_COLLIDER_COUNT:
        errors.append("metadata slider collider count differs from the audited decomposition")
    if collision_summary.get("moving_colliders") != MOVING_COLLIDER_COUNT:
        errors.append("metadata moving collider count differs from the audited decomposition")
    decomposition = metadata.get("slider_collision_decomposition", {})
    if metadata.get("schema_version") != "as013.asset.v4":
        errors.append("metadata schema must be as013.asset.v4")
    if decomposition.get("method") != SLIDER_COLLISION_METHOD:
        errors.append("metadata slider decomposition method is not the welded component build")
    if decomposition.get("hull_count") != SLIDER_COLLIDER_COUNT:
        errors.append("metadata slider decomposition hull count is not six")
    if decomposition.get("certification_samples_per_hull") != SLIDER_COLLIDER_AUDIT_SAMPLES:
        errors.append("metadata slider surface audit did not use 30,000 samples per hull")
    if decomposition.get("geometry_quality_gates") != SLIDER_GEOMETRY_QUALITY_GATES:
        errors.append("metadata slider geometry quality gates differ from the contract")
    if decomposition.get("geometry_quality_all_pass") is not True:
        errors.append("metadata slider geometry quality gate did not pass")
    try:
        if not 0.0 <= float(decomposition["sampled_surface_gap_max_m"]) <= (
            SLIDER_COLLIDER_SURFACE_GAP_TARGET_M
        ):
            errors.append("metadata slider surface gap exceeds 3 mm")
    except (KeyError, TypeError, ValueError):
        errors.append("metadata slider surface gap result is malformed")

    slider_colliders = [
        collider
        for collider in metadata.get("colliders", [])
        if isinstance(collider, dict) and collider.get("role") == "slider"
    ]
    if len(slider_colliders) != SLIDER_COLLIDER_COUNT:
        errors.append("metadata does not enumerate exactly six slider hulls")
    for collider in slider_colliders:
        quality = collider.get("geometry_quality", {})
        prim = collider.get("prim", "slider collider")
        try:
            checks = quality["checks"]
            if quality.get("pass") is not True or not isinstance(checks, dict) or not all(
                value is True for value in checks.values()
            ):
                errors.append(f"{prim} did not pass every geometry-quality check")
            if int(quality["affine_rank"]) != 3:
                errors.append(f"{prim} is not full-rank 3-D geometry")
            if float(quality["minimum_axis_extent_m"]) < 0.00030:
                errors.append(f"{prim} is thinner than the 0.30 mm axis gate")
            if float(quality["minimum_singular_value_m"]) < 0.00009:
                errors.append(f"{prim} fails the minimum singular-value gate")
            if float(quality["singular_value_ratio"]) < 0.002:
                errors.append(f"{prim} fails the singular-value ratio gate")
            if float(quality["volume_m3"]) < 5.0e-10:
                errors.append(f"{prim} fails the minimum physical-volume gate")
            if float(collider["sampled_surface_gap_m"]) > (
                SLIDER_COLLIDER_SURFACE_GAP_TARGET_M
            ):
                errors.append(f"{prim} exceeds the 3 mm surface-gap gate")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{prim} geometry-quality evidence is malformed")

    # The generated source has no DCC physical properties.  Require the
    # deterministic, dimension-derived inertia records emitted by the asset
    # builder rather than accepting an isotropic placeholder.  This is a
    # static contract; a future live-measured asset revision may replace this
    # approximation, but PhysX will use the authored values as-is.
    rigid_bodies = metadata.get("rigid_bodies")
    if not isinstance(rigid_bodies, list) or len(rigid_bodies) != 2:
        errors.append("metadata must describe exactly two rigid-body mass properties")
    else:
        for body in rigid_bodies:
            if not isinstance(body, dict):
                errors.append("metadata rigid-body entry is not an object")
                continue
            if body.get("inertia_model") != "uniform_aabb_box":
                errors.append("rigid-body inertia model must be uniform_aabb_box")
                continue
            try:
                mass = float(body["mass_kg"])
                dimensions = np.asarray(body["inertia_dimensions_m"], dtype=float)
                authored = np.asarray(body["diagonal_inertia_kg_m2"], dtype=float)
                expected = _uniform_box_diagonal_inertia(mass, dimensions)
                if (
                    dimensions.shape != (3,)
                    or not np.isfinite(dimensions).all()
                    or np.any(dimensions <= 0.0)
                ):
                    raise ValueError("invalid dimensions")
                if authored.shape != (3,) or not np.allclose(
                    authored, expected, rtol=1.0e-9, atol=1.0e-12
                ):
                    errors.append(
                        f"{body.get('prim', 'rigid body')} inertia does not match its dimensions"
                    )
            except (KeyError, TypeError, ValueError):
                errors.append("rigid-body inertia record is malformed")
    return errors


def _uniform_box_diagonal_inertia(mass_kg: float, dimensions_m: np.ndarray) -> np.ndarray:
    dimensions = np.asarray(dimensions_m, dtype=float)
    mass = float(mass_kg)
    return mass / 12.0 * np.asarray(
        [
            dimensions[1] ** 2 + dimensions[2] ** 2,
            dimensions[0] ** 2 + dimensions[2] ** 2,
            dimensions[0] ** 2 + dimensions[1] ** 2,
        ],
        dtype=float,
    )


def static_source_errors() -> list[str]:
    errors: list[str] = []
    if asset_hashes() != EXPECTED_ASSET_SHA256:
        errors.append("downloaded Sketchfab source hash set differs from the bound archive")
    if not ASSET_PATH.is_file():
        errors.append(f"integrated task-local USD is missing: {ASSET_PATH}")
        root_bytes = b""
    else:
        root_bytes = ASSET_PATH.read_bytes()
    if not RUNTIME_ASSET_PATH.is_file():
        errors.append(f"runtime asset mirror is missing: {RUNTIME_ASSET_PATH}")
    elif ASSET_PATH.is_file() and sha256_file(ASSET_PATH) != sha256_file(RUNTIME_ASSET_PATH):
        errors.append("task-local USD and runtime USD mirror differ")
    if ASSET_PATH.is_file() and sha256_file(ASSET_PATH) != INTEGRATED_ASSET_SHA256:
        errors.append("integrated task-local USD hash differs from the audited build")
    if METADATA_PATH.is_file() and sha256_file(METADATA_PATH) != INTEGRATED_METADATA_SHA256:
        errors.append("integrated asset metadata hash differs from the audited build")
    if not RUNTIME_METADATA_PATH.is_file():
        errors.append(f"runtime asset metadata mirror is missing: {RUNTIME_METADATA_PATH}")
    elif METADATA_PATH.is_file() and sha256_file(RUNTIME_METADATA_PATH) != sha256_file(METADATA_PATH):
        errors.append("task-local and runtime metadata mirrors differ")
    binary_paths = (
        ASSET_PATH.with_suffix(".usd"),
        RUNTIME_ASSET_PATH.with_suffix(".usd"),
    )
    binary_hashes: list[str] = []
    for binary_path in binary_paths:
        if not binary_path.is_file():
            errors.append(f"binary USD mirror is missing: {binary_path}")
        else:
            binary_hashes.append(sha256_file(binary_path))
    if len(binary_hashes) == 2 and binary_hashes[0] != binary_hashes[1]:
        errors.append("task-local and runtime binary USD mirrors differ")
    if not STATIC_AUDIT_PATH.is_file():
        errors.append(f"static asset audit evidence is missing: {STATIC_AUDIT_PATH}")
    else:
        try:
            static_audit = read_json(STATIC_AUDIT_PATH)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            errors.append(f"static asset audit evidence is invalid: {exc}")
        else:
            if static_audit.get("status") != "static_pass_live_gates_pending":
                errors.append("static asset audit must remain live-gate fail-closed")
            if static_audit.get("schema") != "as013_static_asset_audit_v4":
                errors.append("static asset audit schema must be v4")
            if static_audit.get("asset", {}).get("task_local_usda_sha256") != INTEGRATED_ASSET_SHA256:
                errors.append("static asset audit is not bound to the task-local USD")
            observed_binary = (
                sha256_file(ASSET_PATH.with_suffix(".usd"))
                if ASSET_PATH.with_suffix(".usd").is_file()
                else None
            )
            if static_audit.get("asset", {}).get("binary_usd_sha256") != observed_binary:
                errors.append("static asset audit is not bound to the current binary USD mirror")
            observed_runtime_metadata = (
                sha256_file(RUNTIME_METADATA_PATH)
                if RUNTIME_METADATA_PATH.is_file()
                else None
            )
            if static_audit.get("asset", {}).get("runtime_metadata_sha256") != observed_runtime_metadata:
                errors.append("static asset audit is not bound to the runtime metadata mirror")
            if static_audit.get("stage", {}).get("visual_face_count_checked", 0) < 1:
                errors.append("static asset audit did not inspect visual face winding")
            collisions = static_audit.get("collision_geometry", [])
            if len(collisions) != TOTAL_COLLIDER_COUNT or not all(
                item.get("computed_visibility") == "invisible" for item in collisions
            ):
                errors.append(
                    "static asset audit must enumerate all ergonomic-build "
                    "invisible collision meshes"
                )
            quality_audit = static_audit.get("slider_collider_quality_audit", {})
            if (
                quality_audit.get("method") != SLIDER_COLLISION_METHOD
                or quality_audit.get("status") != "pass"
                or quality_audit.get("welded_component_count")
                != SLIDER_COLLIDER_COUNT
                or not quality_audit.get("assertions")
                or not all(
                    value is True
                    for value in quality_audit.get("assertions", {}).values()
                )
            ):
                errors.append("static asset audit does not prove all slider geometry gates")
            proxy_audit = static_audit.get("same_body_proxy_audit", {})
            try:
                if float(
                    proxy_audit[
                        "slider_conservative_independent_all_6_hulls_30000_each_gap_max_m"
                    ]
                ) > SLIDER_COLLIDER_SURFACE_GAP_TARGET_M:
                    errors.append("independent static slider surface gap exceeds 3 mm")
            except (KeyError, TypeError, ValueError):
                errors.append("independent static slider surface-gap evidence is missing")
            coverage = static_audit.get("visual_to_collider_coverage_audit", {})
            try:
                if (
                    coverage.get("status")
                    != "static_pass_live_hand_sweep_pending"
                    or int(coverage["sample_count"]) != 30_000
                    or int(coverage["count_over_0_003_m"]) != 0
                    or float(coverage["maximum_distance_m"])
                    > SLIDER_COLLIDER_SURFACE_GAP_TARGET_M
                ):
                    errors.append("visible slider-to-collider sampled coverage did not pass")
            except (KeyError, TypeError, ValueError):
                errors.append("visible slider-to-collider coverage evidence is missing")
            support = static_audit.get("support_audit", {})
            if support.get("com_projection_inside") is not True or float(
                support.get("minimum_edge_margin_m", 0.0)
            ) <= 0.0:
                errors.append("static asset audit support COM projection is not inside the hull")
            solver = static_audit.get("solver_audit", {})
            if solver.get("runtime_physx_position_iteration_cap") != 64 or solver.get(
                "runtime_physx_velocity_iteration_cap"
            ) != 4:
                errors.append("static asset audit solver caps are not bound to the shared PhysX defaults")
    for required in (
        b"PhysicsRigidBodyAPI",
        b"PhysicsArticulationRootAPI",
        b"PrismaticJoint_utility_knife_3_middle",
        b"E_body_1",
        b"E_link_blade_2",
        b"physics:axis = \"Y\"",
        b"inertiaModel = \"uniformAabbBox\"",
        BLADE_ENVELOPE_METHOD.encode(),
        b"bladeRetractedClearanceM",
    ):
        if required not in root_bytes:
            errors.append(f"integrated USD is missing token {required.decode(errors='replace')}")
    errors.extend(_metadata_errors())

    task_text = (TASK_DIR / "task.yaml").read_text(encoding="utf-8")
    task = yaml.safe_load(task_text)
    if task["task"]["id"] != TASK_ID:
        errors.append("task.yaml task id mismatch")
    if task["task"].get("task_complete") is not False:
        errors.append("task.yaml must keep task_complete=false until fresh live recertification")
    if "utility_knife002" in task_text:
        errors.append("task.yaml still references the historical utility_knife002 asset")
    if task["assets"][0]["path"] != str(
        Path("taskdev/tasks") / TASK_ID / ASSET_RELATIVE
    ):
        errors.append("task.yaml asset path is not task-local")
    if task["assets"][0].get("runtime_path") != str(RUNTIME_ASSET_RELATIVE):
        errors.append("task.yaml runtime asset path is not the selected mirror")
    if task["assets"][0]["scale_m"] != [UNIFORM_SCALE] * 3:
        errors.append("task.yaml must use the generated meter-scale USD without extra deformation")
    source_conversion = task["assets"][0].get("source_conversion", {})
    blade_contract = source_conversion.get("blade_envelope_correction", {})
    if blade_contract.get("method") != BLADE_ENVELOPE_METHOD:
        errors.append("task.yaml blade envelope correction method is missing or stale")
    try:
        if not math.isclose(
            float(blade_contract["clearance_m"]),
            BLADE_RETRACTED_CLEARANCE_M,
            abs_tol=1.0e-12,
        ):
            errors.append("task.yaml blade retracted clearance is not 4 mm")
        if blade_contract.get("q0_contract") != (
            "fully_retracted_blade_within_housing_envelope"
        ):
            errors.append("task.yaml does not declare the q=0 blade envelope contract")
    except (KeyError, TypeError, ValueError):
        errors.append("task.yaml blade envelope correction record is malformed")
    if task["assets"][0]["articulation"]["joint_name"] != JOINT_NAME:
        errors.append("task.yaml blade joint mismatch")
    collision_contract = task["assets"][0].get("collision", {})
    if collision_contract.get("approximation") != "welded_connected_component_convexHulls":
        errors.append("task.yaml slider collision approximation is not the welded build")
    if collision_contract.get("slider_segment_count") != SLIDER_COLLIDER_COUNT:
        errors.append("task.yaml must count every welded slider collider")
    if collision_contract.get("blade_slider_collider_count") != MOVING_COLLIDER_COUNT:
        errors.append("task.yaml must count every moving-link collider")
    if collision_contract.get("total_collider_count") != TOTAL_COLLIDER_COUNT:
        errors.append("task.yaml must count the complete ergonomic collider set")
    task_quality_gates = collision_contract.get("slider_geometry_quality_gates", {})
    for key in (
        "affine_rank",
        "minimum_axis_extent_m",
        "minimum_singular_value_m",
        "minimum_singular_value_ratio",
        "minimum_volume_m3",
        "watertight",
        "winding_consistent",
    ):
        if task_quality_gates.get(key) != SLIDER_GEOMETRY_QUALITY_GATES[key]:
            errors.append(f"task.yaml slider geometry gate {key} differs from metadata")
    try:
        if float(collision_contract["sampled_slider_surface_gap_max_m"]) > (
            SLIDER_COLLIDER_SURFACE_GAP_TARGET_M
        ):
            errors.append("task.yaml independent slider surface gap exceeds 3 mm")
    except (KeyError, TypeError, ValueError):
        errors.append("task.yaml independent slider surface gap is missing")
    mass_properties = task["assets"][0].get("mass_properties", {})
    if mass_properties.get("model") != "uniform_aabb_box":
        errors.append("task.yaml must declare the dimension-derived inertia model")
    if mass_properties.get("runtime_mass_randomization_scales_inertia") is not True:
        errors.append("task.yaml must bind mass/inertia randomization together")
    if not isinstance(mass_properties.get("bodies"), list) or len(
        mass_properties["bodies"]
    ) != 2:
        errors.append("task.yaml must list both rigid-body mass properties")
    solver_properties = task["assets"][0].get("solver_properties", {})
    if solver_properties != {
        "rigid_body_position_iterations": 16,
        "rigid_body_velocity_iterations": 4,
        "scene_position_iteration_cap": 64,
        "scene_velocity_iteration_cap": 4,
    }:
        errors.append("task.yaml solver properties must match the AS013 PhysX audit")
    if str(task.get("source", {}).get("license", "")).lower() not in {
        "cc-by-4.0",
        "cc by 4.0",
        "cc-by",
    }:
        errors.append("task.yaml must record the selected CC-BY-4.0 license")
    if EXPECTED_ASSET_SHA256["source/utility_knife.zip"] not in task_text:
        errors.append("task.yaml does not bind the downloaded archive SHA-256")

    runner = (TASK_DIR / "run_core_scene.py").read_text(encoding="utf-8")
    required_source = (
        "from isaaclab.app import AppLauncher",
        "args.width = 640",
        "args.height = 480",
        "RayTracedLighting",
        'settings.set("/rtx/post/tonemap/op", 4)',
        "DomeLight",
        "DistantLight",
    )
    for token in required_source:
        if token not in runner:
            errors.append(f"run_core_scene.py missing validator token: {token}")
    if "omni.isaac." in runner:
        errors.append("deprecated omni.isaac namespace is forbidden")
    if re.search(r"/(?:home|Users)/[^/\s]+/", runner):
        errors.append("runner contains a hardcoded user path")
    if re.search(r"(?<![\d.])(?:10|192\.168)\.\d+\.\d+\.\d+", runner):
        errors.append("runner contains a bare private network endpoint")
    return errors


def live_evidence_errors() -> list[str]:
    errors: list[str] = []
    manifest_path = TASK_DIR / "scene_manifest.json"
    report_path = TASK_DIR / "evidence" / "live_scene_report.json"
    preview_path = TASK_DIR / "previews" / "reset_head.png"
    # A fresh download invalidates the prior live episode.  Until a new
    # Isaac/PhysX run produces all three artifacts, report a single explicit
    # pending gate rather than treating historical evidence as current.
    if not (manifest_path.is_file() and report_path.is_file() and preview_path.is_file()):
        return ["pending_fresh_asset_live_evidence"]

    manifest = read_json(manifest_path)
    report = read_json(report_path)
    if manifest.get("claims", {}).get("cpu_checkpoint_ready") is not True:
        errors.append("manifest does not claim a passing fresh-asset CPU checkpoint")
    if manifest.get("claims", {}).get("live_episode_pass") is not True:
        errors.append("manifest does not claim a fresh-asset live episode")
    if manifest.get("claims", {}).get("task_complete") is not False:
        errors.append("manifest must keep task_complete=false")
    if report.get("core_scene_pass") is not True:
        errors.append("live report did not pass")
    if report.get("task_complete") is not False:
        errors.append("live report must keep task_complete=false")
    if report.get("motion_path_executed") is not False:
        errors.append("core-scene runner executed a forbidden motion path")
    if report.get("object_state_writes_during_settle") != 0:
        errors.append("zero-action settle wrote object state")
    if report.get("runtime", {}).get("physics_steps", 0) < ZERO_ACTION_STEPS:
        errors.append("zero-action settle is shorter than 10 seconds")
    if not math.isclose(
        float(report.get("asset", {}).get("uniform_scale", math.nan)),
        UNIFORM_SCALE,
        abs_tol=1.0e-12,
    ):
        errors.append("live composed scale mismatch")
    if report.get("asset", {}).get("sha256") != sha256_file(ASSET_PATH):
        errors.append("live report is not bound to the selected task-local USD")
    if report.get("asset", {}).get("blade_joint", {}).get("name") != JOINT_NAME:
        errors.append("live blade joint mismatch")
    if report.get("asset", {}).get("collision_shape_count", 0) < 2:
        errors.append("live asset exposes fewer than two collision shapes")
    if report.get("stability", {}).get("finite") is not True:
        errors.append("zero-action state contains non-finite values")
    if report.get("stability", {}).get("max_body_translation_m", math.inf) > 0.01:
        errors.append("knife moved more than 1 cm during zero action")
    if report.get("stability", {}).get("blade_relative_drift_m", math.inf) > 0.002:
        errors.append("blade relative pose drifted more than 2 mm")

    metrics = image_metrics(preview_path)
    if metrics["shape_hwc"] != [*HEAD_RESOLUTION_HW, 3]:
        errors.append("preview is not native 640x480 RGB")
    if metrics["bytes"] < 150_000:
        errors.append("preview is below the 150 KB render gate")
    if metrics["mean_rgb"] <= 30.0:
        errors.append("preview mean RGB is too dark")
    if metrics["pixel_std"] <= 15.0:
        errors.append("preview is too flat")
    if metrics["sha256"] != report.get("head_camera", {}).get("sha256"):
        errors.append("preview hash differs from live report")
    return errors


def validate(*, require_live: bool) -> dict[str, Any]:
    errors = static_source_errors()
    if require_live:
        errors.extend(live_evidence_errors())
    return {
        "schema": "as013_static_contract_v1",
        "task_id": TASK_CODE,
        "require_live": require_live,
        "errors": errors,
        "pass": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-live", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(require_live=args.require_live)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
