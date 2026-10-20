from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TASK_DIR.parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from taskdev.tasks.bench320_hs014_bottle_stabilized_cap_opening import core_scene
from taskdev.tasks.bench320_hs014_bottle_stabilized_cap_opening.scene_contract import (
    load_json,
    validate_core_report,
    validate_scene_manifest,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CoreSceneContractTests(unittest.TestCase):
    def test_authoritative_asset_hashes_and_cap_topology(self) -> None:
        contract = core_scene.source_asset_contract()
        self.assertTrue(contract["pass"])
        self.assertEqual(contract["cap_vertex_count"], 1200)
        self.assertEqual(contract["cap_triangle_count"], 636)
        self.assertEqual(
            contract["cap_source_local_aabb_m"],
            [[-0.023000000044703484, 0.0, -0.023000000044703484], [0.023000000044703484, 0.026000000536441803, 0.023000000044703484]],
        )
        core_scene.np.testing.assert_allclose(
            contract["cap_collision_local_aabb_m"],
            [[-0.023, 0.0, -0.023], [0.023, 0.026, 0.023]],
        )
        self.assertTrue(contract["cap_collision_inside_visible_bounds"])
        self.assertTrue(contract["cap_collision_inside_source_convex_profile"])
        straight_wall = contract[
            "cap_collision_visible_straight_wall_interval_y_m"
        ]
        self.assertLessEqual(straight_wall[0], 0.003)
        self.assertGreaterEqual(straight_wall[1], 0.023)

    def test_cap_bevel_collider_is_closed_convex_and_source_derived(self) -> None:
        """No invisible enlargement; both formerly missing 3 mm rims can contact."""
        from collections import Counter

        np = core_scene.np
        cap = core_scene.decode_authoritative_cap()
        mesh = core_scene.cap_collision_mesh(cap)
        points = mesh["points"]
        source = np.asarray(cap["points"]) + np.asarray(cap["node_translation"])
        self.assertEqual(points.shape, (160, 3))
        self.assertEqual(len(np.unique(points[:, 1])), 8)
        self.assertTrue(set(map(tuple, points)) <= set(map(tuple, source)))
        np.testing.assert_allclose(points.min(0), cap["bounds"][0], atol=1e-9)
        np.testing.assert_allclose(points.max(0), cap["bounds"][1], atol=1e-9)
        edges = Counter()
        offset = 0
        for count in mesh["counts"]:
            indices = mesh["indices"][offset:offset + count]
            offset += count
            face = points[indices]
            normal = np.cross(face[1] - face[0], face[2] - face[0])
            self.assertGreater(np.linalg.norm(normal), 1e-10)
            normal /= np.linalg.norm(normal)
            self.assertLessEqual(float(np.max((points - face[0]) @ normal)), 1e-8)
            self.assertGreater(float((face[0] - points.mean(0)) @ normal), 0.0)
            edges.update(zip(indices, indices[1:] + indices[:1]))
        self.assertEqual(offset, len(mesh["indices"]))
        self.assertTrue(all(count == 1 and edges[b, a] == 1
                            for (a, b), count in edges.items()))
        self.assertEqual(len(points) - len(edges) // 2 + len(mesh["counts"]), 2)

    def test_scene_manifest_is_fail_closed(self) -> None:
        manifest = load_json(TASK_DIR / "scene_manifest.json")
        self.assertEqual(validate_scene_manifest(manifest), [])
        self.assertEqual(manifest["task_id"], "B320-HS014")
        self.assertEqual(core_scene.TASK_ID, "B320-HS014")
        self.assertEqual(
            core_scene.TASK_NAME,
            "bench320_hs014_bottle_stabilized_cap_opening",
        )
        self.assertFalse(manifest["exact_production_child"])
        self.assertFalse(manifest["priority_zero_qualified"])
        self.assertEqual(
            manifest["evidence"]["status"],
            "historical_stale_pending_regeneration",
        )
        self.assertFalse(manifest["evidence"]["qualifies_current_source"])

    def test_cap_uses_hash_bound_thin_wall_mass_properties(self) -> None:
        cap = core_scene.decode_authoritative_cap()
        points = core_scene.np.asarray(cap["points"], dtype=core_scene.np.float64)
        points += core_scene.np.asarray(
            cap["node_translation"], dtype=core_scene.np.float64
        )
        triangles = points[
            core_scene.np.asarray(cap["indices"], dtype=core_scene.np.int64).reshape(
                -1, 3
            )
        ]
        areas = 0.5 * core_scene.np.linalg.norm(
            core_scene.np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        total_area = float(core_scene.np.sum(areas))
        center = core_scene.np.sum(
            areas[:, None] * core_scene.np.mean(triangles, axis=1), axis=0
        ) / total_area
        second_moment = core_scene.np.zeros((3, 3), dtype=core_scene.np.float64)
        for triangle, area in zip(triangles, areas):
            summed = core_scene.np.sum(triangle, axis=0)
            vertex_outer = sum(
                (core_scene.np.outer(vertex, vertex) for vertex in triangle),
                start=core_scene.np.zeros((3, 3), dtype=core_scene.np.float64),
            )
            second_moment += area * (
                vertex_outer + core_scene.np.outer(summed, summed)
            ) / 12.0
        second_moment /= total_area
        covariance = second_moment - core_scene.np.outer(center, center)
        inertia = core_scene.CAP_MASS_KG * (
            core_scene.np.trace(covariance) * core_scene.np.eye(3) - covariance
        )

        self.assertEqual(core_scene.CAP_MASS_KG, 0.012)
        core_scene.np.testing.assert_allclose(
            center,
            core_scene.CAP_CENTER_OF_MASS_LOCAL_M,
            rtol=0.0,
            atol=5.0e-10,
        )
        core_scene.np.testing.assert_allclose(
            core_scene.np.diag(inertia),
            core_scene.CAP_DIAGONAL_INERTIA_KG_M2,
            rtol=0.0,
            atol=5.0e-13,
        )
        self.assertLess(
            float(
                core_scene.np.max(
                    core_scene.np.abs(
                        inertia - core_scene.np.diag(core_scene.np.diag(inertia))
                    )
                )
            ),
            1.0e-18,
        )
        self.assertEqual(core_scene.CAP_PRINCIPAL_AXES_WXYZ, (1.0, 0.0, 0.0, 0.0))
        largest = max(core_scene.CAP_DIAGONAL_INERTIA_KG_M2)
        self.assertLess(largest, sum(core_scene.CAP_DIAGONAL_INERTIA_KG_M2) - largest)
        manifest = load_json(TASK_DIR / "scene_manifest.json")
        self.assertEqual(
            manifest["physics"]["cap_mass_properties"],
            {
                "mass_kg": 0.012,
                "center_of_mass_local_m": [0.0, 0.013, 0.0],
                "diagonal_inertia_kg_m2": [
                    3.635590e-6,
                    4.745227e-6,
                    3.635590e-6,
                ],
                "principal_axes_wxyz": [1.0, 0.0, 0.0, 0.0],
                "source": "uniform_surface_integral_of_hash_bound_visible_mesh",
            },
        )
        source = (TASK_DIR / "core_scene.py").read_text(encoding="utf-8")
        self.assertIn("mass_kg=CAP_MASS_KG", source)
        self.assertIn("center_of_mass=CAP_CENTER_OF_MASS_LOCAL_M", source)
        self.assertIn("diagonal_inertia=CAP_DIAGONAL_INERTIA_KG_M2", source)

    def test_identity_scale_and_explicit_axis_mapping(self) -> None:
        manifest = load_json(TASK_DIR / "scene_manifest.json")
        self.assertEqual(manifest["assets"]["bottle"]["scale_xyz"], [1.0, 1.0, 1.0])
        self.assertEqual(manifest["assets"]["cap"]["scale_xyz"], [1.0, 1.0, 1.0])
        bottle_quaternion = core_scene.BOTTLE_VISUAL_CORRECTION_WXYZ
        bottle_rotation = core_scene._quat_matrix_xyzw(
            core_scene.np.asarray(
                [
                    bottle_quaternion[1],
                    bottle_quaternion[2],
                    bottle_quaternion[3],
                    bottle_quaternion[0],
                ]
            )
        )
        bottle_axis = bottle_rotation @ core_scene.np.asarray([0.0, 1.0, 0.0])
        self.assertTrue(
            core_scene.np.allclose(bottle_axis, [0.0, 0.0, 1.0], atol=1.0e-12)
        )
        self.assertEqual(
            core_scene.BOTTLE_VISUAL_TRANSLATION_M.tolist(),
            [0.0, 0.0, -0.2],
        )
        self.assertTrue(
            core_scene.np.allclose(
                core_scene.BOTTLE_VISUAL_TRANSLATION_M,
                [0.0, 0.0, -0.2],
            )
        )
        self.assertTrue(
            core_scene.np.allclose(
                0.2 * bottle_axis + core_scene.BOTTLE_VISUAL_TRANSLATION_M,
                [0.0, 0.0, 0.0],
                atol=1.0e-12,
            )
        )
        rotation = core_scene._quat_matrix_xyzw(core_scene.CAP_UPRIGHT_QUAT_XYZW)
        mapped_axis = rotation @ core_scene.np.asarray([0.0, 1.0, 0.0])
        self.assertTrue(core_scene.np.allclose(mapped_axis, [0.0, 0.0, 1.0], atol=1.0e-12))

    def test_dynamic_bottle_collision_contract(self) -> None:
        manifest = load_json(TASK_DIR / "scene_manifest.json")
        bottle = manifest["physics"]["bottle_collision"]
        self.assertEqual(bottle["type"], "source_convex_decomposition_dynamic")
        self.assertTrue(bottle["authoritative_source_collision_enabled"])
        self.assertFalse(bottle["extra_stability_proxy_present"])
        self.assertEqual(bottle["center_of_mass_local_m"], [0.0, 0.0, -0.1])
        source = (TASK_DIR / "core_scene.py").read_text(encoding="utf-8")
        self.assertIn("source_geometry.SetInstanceable(False)", source)
        self.assertNotIn("StabilityBaseProxy", source)
        self.assertNotIn("BodyProxy", source)
        self.assertNotIn("NeckProxy", source)
        self.assertEqual(core_scene.COLLISION_CONTACT_OFFSET_M, 0.0005)
        self.assertEqual(core_scene.COLLISION_REST_OFFSET_M, 0.0)
        self.assertIn(
            "contact_offset_m: float = COLLISION_CONTACT_OFFSET_M", source
        )
        self.assertIn(
            "rest_offset_m: float = COLLISION_REST_OFFSET_M", source
        )
        self.assertIn("CreateContactOffsetAttr(float(contact_offset_m))", source)
        self.assertIn("CreateRestOffsetAttr(float(rest_offset_m))", source)

    def test_bottle_and_cap_use_qualified_local_solver_iterations(self) -> None:
        manifest = load_json(TASK_DIR / "scene_manifest.json")
        solver = manifest["physics"]["rigid_owner_solver"]
        self.assertEqual(core_scene.RIGID_SOLVER_POSITION_ITERATIONS, 64)
        self.assertEqual(core_scene.RIGID_SOLVER_VELOCITY_ITERATIONS, 1)
        self.assertEqual(
            core_scene.RIGID_MAX_DEPENETRATION_VELOCITY_M_S,
            1.0,
        )
        self.assertEqual(
            solver,
            {
                "position_iteration_count": 64,
                "velocity_iteration_count": 1,
                "max_depenetration_velocity_m_s": 1.0,
            },
        )
        source = (TASK_DIR / "core_scene.py").read_text(encoding="utf-8")
        self.assertIn(
            "CreateSolverPositionIterationCountAttr(\n"
            "        RIGID_SOLVER_POSITION_ITERATIONS",
            source,
        )
        self.assertIn(
            "CreateSolverVelocityIterationCountAttr(\n"
            "        RIGID_SOLVER_VELOCITY_ITERATIONS",
            source,
        )
        self.assertNotIn("CreateSolverVelocityIterationCountAttr(4)", source)

    def test_cap_collision_is_inset_inside_source_convex_profile(self) -> None:
        manifest = load_json(TASK_DIR / "scene_manifest.json")
        collision = manifest["physics"]["cap_collision"]
        self.assertEqual(
            collision["type"],
            "source_bevel_hull_and_com_centered_cradle_sphere",
        )
        self.assertEqual(core_scene.CAP_COLLISION_RADIUS_M, 0.023)
        self.assertEqual(core_scene.CAP_COLLISION_BOTTOM_INSET_M, 0.0)
        self.assertEqual(core_scene.CAP_COLLISION_TOP_INSET_M, 0.0)
        self.assertAlmostEqual(core_scene.CAP_COLLISION_HEIGHT_M, 0.026)
        self.assertAlmostEqual(core_scene.CAP_COLLISION_CENTER_Y_M, 0.013)
        self.assertTrue(collision["inside_visible_geometry"])
        self.assertTrue(
            core_scene.np.all(
                core_scene.CAP_COLLISION_LOCAL_AABB_M[0]
                >= core_scene.CAP_LOCAL_AABB_M[0]
            )
        )
        self.assertTrue(
            core_scene.np.all(
                core_scene.CAP_COLLISION_LOCAL_AABB_M[1]
                <= core_scene.CAP_LOCAL_AABB_M[1]
            )
        )
        source = (TASK_DIR / "core_scene.py").read_text(encoding="utf-8")
        self.assertIn("cap_collision_mesh(cap_data)", source)
        self.assertIn("hull.CreateHullVertexLimitAttr(256)", source)

    def test_torque_neutral_three_sphere_cradle_contract(self) -> None:
        manifest = load_json(TASK_DIR / "scene_manifest.json")
        bottle = manifest["physics"]["bottle_collision"]
        cap = manifest["physics"]["cap_collision"]
        topology = manifest["physics"]["cap_bottle_contact_topology"]
        self.assertEqual(bottle["cradle_supports"]["count"], 3)
        self.assertEqual(bottle["cradle_supports"]["sphere_radius_m"], 0.0015)
        self.assertEqual(bottle["cradle_supports"]["ring_radius_m"], 0.0024)
        self.assertEqual(bottle["cradle_supports"]["radial_envelope_m"], 0.0039)
        self.assertTrue(bottle["cradle_supports"]["inside_bottle_mouth"])
        self.assertEqual(cap["cradle_sphere"]["radius_m"], 0.013)
        self.assertEqual(cap["cradle_sphere"]["center_y_m"], 0.013)
        self.assertTrue(cap["cradle_sphere"]["inside_visible_bounds"])
        self.assertTrue(cap["cradle_sphere"]["center_matches_cap_com"])
        self.assertEqual(cap["outer_physics_material_binding"], "default_grasp_friction")
        self.assertEqual(
            cap["helper_physics_material"],
            {
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
        )
        self.assertEqual(topology["contact_truth_topology"], "three_sphere_cradle_to_com_sphere")
        self.assertEqual(len(topology["filtered_shape_role_pairs"]), 5)
        self.assertEqual(len(topology["unfiltered_contact_role_pairs"]), 3)
        self.assertEqual(topology["cap_seated_height_m"], -0.0002)
        self.assertEqual(topology["center_delta_z_m"], 0.0143)
        self.assertEqual(topology["center_horizontal_offset_m"], 0.0024)
        self.assertEqual(topology["center_distance_m"], 0.0145)
        self.assertEqual(topology["radius_sum_m"], 0.0145)
        self.assertTrue(topology["exact_initial_tangency"])
        self.assertTrue(topology["contact_normals_intersect_cap_com"])
        self.assertTrue(topology["axial_lift_separates_immediately"])
        self.assertAlmostEqual(core_scene.CRADLE_CENTER_DZ_M, 0.0143)
        self.assertAlmostEqual(core_scene.CRADLE_CENTER_DISTANCE_M, 0.0145)
        self.assertAlmostEqual(core_scene.CRADLE_RADIUS_SUM_M, 0.0145)
        self.assertLessEqual(
            core_scene.BOTTLE_CRADLE_SUPPORT_RADIAL_ENVELOPE_M,
            core_scene.BOTTLE_MOUTH_CONTAINMENT_RADIUS_M,
        )
        core_scene.np.testing.assert_allclose(
            [0.0, core_scene.CAP_CRADLE_CENTER_Y_M, 0.0],
            core_scene.CAP_CENTER_OF_MASS_LOCAL_M,
            rtol=0.0,
            atol=1.0e-12,
        )
        source = (TASK_DIR / "core_scene.py").read_text(encoding="utf-8")
        self.assertIn("def _author_cradle_helpers", source)
        self.assertIn("def _author_cradle_filters", source)
        self.assertIn("CradleSupport", source)
        self.assertIn("CradleSphere", source)
        self.assertIn("CRADLE_HELPER_CONTACT_OFFSET_M = 0.00005", source)
        self.assertIn('CRADLE_HELPER_FRICTION_COMBINE_MODE = "min"', source)
        self.assertNotIn("D6Joint", source)
        self.assertNotIn("/Seat", source)


    def test_inventory_matches_copied_files(self) -> None:
        inventory = load_json(TASK_DIR / "asset_inventory.json")
        for item in inventory["assets"]:
            copied = TASK_DIR / item["copied_path"]
            self.assertTrue(copied.is_file())
            self.assertEqual(copied.stat().st_size, item["bytes"])
            self.assertEqual(sha256(copied), item["sha256"])
            for dependency in item.get("runtime_dependencies", []):
                path = TASK_DIR / dependency["path"]
                self.assertTrue(path.is_file())
                self.assertEqual(path.stat().st_size, dependency["bytes"])
                if dependency.get("sha256"):
                    self.assertEqual(sha256(path), dependency["sha256"])

    def test_runtime_report_and_preview(self) -> None:
        report = load_json(TASK_DIR / "evidence" / "core_scene_report.json")
        self.assertEqual(
            validate_core_report(report),
            [
                "zero-action simulation is shorter than 10 seconds",
                "core scene must not claim to be the exact production child",
                "core scene must not claim priority-zero qualification",
            ],
        )
        raw = load_json(TASK_DIR / "evidence" / "core_scene_report.runner_raw.json")
        self.assertEqual(report, raw)
        self.assertTrue(raw["checks"]["head_camera_preview"])
        self.assertTrue(report["checks"]["head_camera_preview"])
        self.assertEqual(
            report["preview"]["metrics"]["pixel_variance"],
            report["preview"]["metrics"]["pixel_std"] ** 2,
        )
        self.assertGreater(report["preview"]["metrics"]["pixel_variance"], 15.0)
        self.assertEqual(report["final_owner_states"], raw["final_owner_states"])
        self.assertEqual(report["zero_action_drift"], raw["zero_action_drift"])
        preview = TASK_DIR / "previews" / "reset_head.png"
        self.assertTrue(preview.is_file())
        self.assertGreaterEqual(preview.stat().st_size, 150_000)
        self.assertEqual(sha256(preview), report["preview"]["sha256"])
        usd = TASK_DIR / "hs014_core_scene.usda"
        self.assertTrue(usd.is_file())
        self.assertGreater(usd.stat().st_size, 1024)
        # Historical evidence is intentionally not qualification of today's scene.
        manifest = load_json(TASK_DIR / "scene_manifest.json")
        self.assertFalse(manifest["evidence"]["qualifies_current_source"])
        self.assertNotEqual(sha256(usd), report["portable_usd"]["sha256"])
        source = (TASK_DIR / "core_scene.py").read_text(encoding="utf-8")
        self.assertIn("10 * PHYSICS_HZ", source)
        self.assertIn('"exact_production_child": False', source)
        self.assertIn('"priority_zero_qualified": False', source)

    def test_no_manipulation_implementation_in_runner(self) -> None:
        source = (TASK_DIR / "run_core_scene.py").read_text(encoding="utf-8")
        self.assertNotIn("env.step(", source)
        self.assertNotIn("solve_ik", source)
        self.assertNotIn("grasp_action", source)


if __name__ == "__main__":
    unittest.main()
