from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


TASK_DIR = Path(__file__).resolve().parents[1]
BUILDER_PATH = TASK_DIR / "build_asset.py"
ARCHIVE_PATH = (
    TASK_DIR
    / "assets/polyhaven_flathead_screwdriver/source/flathead_screwdriver_4k.gltf.zip"
)


def _load_builder():
    spec = importlib.util.spec_from_file_location("as019_asset_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


@pytest.fixture(scope="module")
def plan():
    return builder.create_build_plan(ARCHIVE_PATH)


def test_archive_is_byte_exact_and_has_only_the_reviewed_members(plan):
    assert plan.archive.archive_path == ARCHIVE_PATH.resolve()
    assert ARCHIVE_PATH.stat().st_size == 27_238_057
    assert builder.sha256_file(ARCHIVE_PATH) == builder.ARCHIVE_SHA256
    assert tuple(plan.archive.members) == builder.MEMBER_PATHS
    assert tuple(record["path"] for record in plan.archive.member_records) == (
        "flathead_screwdriver_4k.gltf",
        "flathead_screwdriver.bin",
        "textures/flathead_screwdriver_arm_4k.jpg",
        "textures/flathead_screwdriver_diff_4k.jpg",
        "textures/flathead_screwdriver_nor_gl_4k.jpg",
    )
    for record, expected in zip(
        plan.archive.member_records, builder.MEMBER_SPECS, strict=True
    ):
        assert record["size_bytes"] == expected.size_bytes
        assert record["sha256"] == expected.sha256


def test_single_source_pbr_mesh_and_uniform_normalization_are_locked(plan):
    assert plan.mesh.points.shape == (1558, 3)
    assert plan.mesh.normals.shape == (1558, 3)
    assert plan.mesh.uv.shape == (1558, 2)
    assert plan.mesh.faces.shape == (2300, 3)
    assert np.array_equal(builder._bounds(plan.mesh.points), builder.EXPECTED_SOURCE_BOUNDS_M)
    assert np.allclose(
        builder._bounds(plan.task_points),
        builder.EXPECTED_TASK_BOUNDS_M,
        atol=2.0e-15,
        rtol=0.0,
    )
    assert builder.UNIFORM_SCALE == 0.9950483266936156
    assert np.allclose(
        builder.ROTATION_SOURCE_TO_TASK,
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=0.0,
        rtol=0.0,
    )
    assert np.linalg.det(builder.ROTATION_SOURCE_TO_TASK) == pytest.approx(1.0)
    assert builder.ROTATION_QUATERNION_WXYZ == pytest.approx(
        [math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)]
    )
    assert np.allclose(
        builder.inverse_transform_points(plan.task_points),
        plan.mesh.points,
        atol=3.0e-17,
        rtol=0.0,
    )
    assert np.allclose(np.linalg.norm(plan.task_normals, axis=1), 1.0, atol=1.0e-14)
    # Task +X is the sharp end; task -X is the blunt wooden tail.
    source_min_y = int(np.argmin(plan.mesh.points[:, 1]))
    source_max_y = int(np.argmax(plan.mesh.points[:, 1]))
    assert plan.task_points[source_min_y, 0] == pytest.approx(-0.125, abs=1.0e-15)
    assert plan.task_points[source_max_y, 0] == pytest.approx(
        0.13500000163912776, abs=1.0e-15
    )


def test_source_shells_and_five_collider_path_set_are_exact(plan):
    assert tuple(component.name for component in plan.components) == (
        "wood_handle",
        "steel_ferrule",
        "steel_shaft_blade",
    )
    assert tuple(len(component.points) for component in plan.components) == (942, 160, 77)
    assert tuple(collider.path for collider in plan.colliders) == (
        "/FlatheadScrewdriver/Colliders/TailMain",
        "/FlatheadScrewdriver/Colliders/HandleNeck",
        "/FlatheadScrewdriver/Colliders/Ferrule",
        "/FlatheadScrewdriver/Colliders/Stem",
        "/FlatheadScrewdriver/Colliders/FlatBlade",
    )
    assert builder.ROOT_PATH == "/FlatheadScrewdriver"
    assert builder.VISUAL_PATH == "/FlatheadScrewdriver/Visual/SourceMesh"
    assert builder.PHYSICS_MATERIAL_PATH == "/FlatheadScrewdriver/Looks/PhysicsMaterial"
    assert tuple(collider.material for collider in plan.colliders) == (
        "wood",
        "wood",
        "steel",
        "steel",
        "steel",
    )
    expected_x_ranges = (
        (-0.125, -0.0508144237799039),
        (-0.0508144237799039, -0.016476666758274833),
        (-0.017030295011975685, 0.006049148842335109),
        (-0.01414767899083312, 0.11085820227730864),
        (0.11085820227730864, 0.13500000163912776),
    )
    for collider, expected_x in zip(plan.colliders, expected_x_ranges, strict=True):
        bounds = builder._bounds(collider.task_points)
        assert tuple(bounds[:, 0]) == pytest.approx(expected_x, abs=2.0e-15)
        assert len(collider.task_points) >= 4
        assert len(collider.faces) >= 4
        assert collider.volume_m3 > 0.0


def test_every_hull_vertex_is_source_derived_and_tight_protrusion_audit_passes(plan):
    component_by_name = {
        component.name: component for component in plan.components
    }
    observed_gaps = []
    for collider in plan.colliders:
        component_points = component_by_name[collider.source_component].points
        for point in collider.source_points:
            assert np.any(np.all(component_points == point, axis=1))
        audit = collider.protrusion_audit
        assert audit["source_derived_hull_vertices"] is True
        assert audit["maximum_hull_vertex_to_source_vertex_distance_m"] <= 2.0e-8
        assert audit["maximum_sampled_surface_gap_m"] <= 0.0021
        assert audit["acceptance_limit_m"] == 0.0021
        assert audit["pass"] is True
        observed_gaps.append(audit["maximum_sampled_surface_gap_m"])
    assert observed_gaps == pytest.approx(
        [
            0.001071206080625018,
            0.0020438335134678473,
            0.0009909420477220061,
            0.0,
            0.00022099396455010312,
        ],
        abs=1.0e-12,
    )


def test_explicit_wood_steel_mass_model_is_finite_normalized_and_supported(plan):
    mass_model = plan.mass_model
    assert mass_model["wood_density_kg_m3"] == 700.0
    assert mass_model["steel_density_kg_m3"] == 7850.0
    assert mass_model["nominal_mass_kg"] == 0.18
    assert sum(piece["normalized_mass_kg"] for piece in mass_model["pieces"]) == pytest.approx(
        0.18, abs=1.0e-15
    )
    center_of_mass = np.asarray(mass_model["center_of_mass_m"])
    assert np.all(np.isfinite(center_of_mass))
    assert center_of_mass == pytest.approx(
        [-0.03348301709027515, 0.02500000005176284, 0.0], abs=2.0e-15
    )
    assert mass_model["support_region"]["contains_center_of_mass"] is True
    support_min = np.asarray(mass_model["support_region"]["min_m"])
    support_max = np.asarray(mass_model["support_region"]["max_m"])
    assert np.all(center_of_mass >= support_min)
    assert np.all(center_of_mass <= support_max)
    diagonal_inertia = np.asarray(mass_model["diagonal_inertia_kg_m2"])
    assert np.all(np.isfinite(diagonal_inertia))
    assert np.all(diagonal_inertia > 0.0)


def test_runtime_contract_constants_cover_sleep_and_randomizable_physics_material():
    assert builder.NOMINAL_MASS_KG == 0.18
    assert builder.DEFAULT_STATIC_FRICTION == 0.68
    assert builder.DEFAULT_DYNAMIC_FRICTION == 0.68
    assert builder.DEFAULT_RESTITUTION == 0.0
    assert builder.CONTACT_REPORT_THRESHOLD_N == 0.0
    assert builder.ASSET_REPO_RELATIVE_DIR == (
        "taskdev/tasks/bench320_as019_screwdriver_safe_orientation_handover/"
        "assets/polyhaven_flathead_screwdriver"
    )


def test_generated_manifest_binds_the_exact_runtime_and_inventory():
    asset_dir = TASK_DIR / "assets/polyhaven_flathead_screwdriver"
    manifest_path = asset_dir / "asset_manifest.json"
    runtime_path = asset_dir / "runtime/flathead_screwdriver.usd"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = manifest["runtime"]
    assert runtime["path"] == (
        "taskdev/tasks/bench320_as019_screwdriver_safe_orientation_handover/"
        "assets/polyhaven_flathead_screwdriver/runtime/flathead_screwdriver.usd"
    )
    assert runtime["sha256"] == builder.sha256_file(runtime_path)
    assert runtime["size_bytes"] == runtime_path.stat().st_size
    assert runtime["isaac_sim_version"] == "5.1.0.0"
    assert runtime["rigid_owner"] == "/FlatheadScrewdriver"
    assert runtime["visual_mesh_paths"] == [
        "/FlatheadScrewdriver/Visual/SourceMesh"
    ]
    assert tuple(runtime["collider_paths"]) == builder.COLLIDER_PATHS
    assert runtime["collider_count"] == 5
    assert runtime["collision_approximation"] == "convexHull"
    assert runtime["mass_kg"] == 0.18
    assert runtime["center_of_mass_m"] == pytest.approx(
        [-0.03348301709027515, 0.02500000005176284, 0.0], abs=2.0e-15
    )
    assert runtime["starts_asleep"] is True
    assert runtime["physics_material_path"] == builder.PHYSICS_MATERIAL_PATH
    assert manifest["provenance"]["archive_sha256"] == builder.ARCHIVE_SHA256
    assert manifest["source"]["member_allowlist"] == list(builder.MEMBER_PATHS)
    assert np.array_equal(
        np.asarray(manifest["normalization"]["matrix_row_major"]),
        builder.NORMALIZATION_MATRIX_ROW_MAJOR,
    )
    assert manifest["visual"]["mesh_count"] == 1
    assert manifest["visual"]["visible_mesh_count"] == 1
    assert manifest["visual"]["collision_api_authored"] is False
    assert manifest["physics_material"] == {
        "binding_purpose": "physics",
        "binding_strength": "strongerThanDescendants",
        "bound_collider_paths": list(builder.COLLIDER_PATHS),
        "dynamic_friction": 0.68,
        "episode_randomization_owner": "AS019 collection environment",
        "material_api_count": 1,
        "path": "/FlatheadScrewdriver/Looks/PhysicsMaterial",
        "restitution": 0.0,
        "static_friction": 0.68,
    }
    assert manifest["protrusion_audit"]["all_pass"] is True
    assert manifest["protrusion_audit"]["acceptance_limit_m"] == 0.0021
