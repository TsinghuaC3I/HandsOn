from __future__ import annotations

import json
import unittest

from taskdev.tasks.bench320_fu001_synchronized_double_cabinet_open.scene_spec import (
    TASK_DIR,
)


class FU001EvidenceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(
            (TASK_DIR / "evidence/live_run/core_scene_summary.json").read_text(
                encoding="utf-8"
            )
        )
        cls.manifest = json.loads(
            (TASK_DIR / "scene_manifest.json").read_text(encoding="utf-8")
        )

    def test_historical_core_scene_is_green_but_not_task_complete(self) -> None:
        """Keep the retired 44781 run auditable without relabeling it selected."""
        self.assertIs(self.summary["core_scene_pass"], True)
        self.assertTrue(all(self.summary["checks"].values()))
        self.assertIs(self.summary["task_complete"], False)
        self.assertIs(self.summary["success_motion_executed"], False)

    def test_scale_pose_and_clearance(self) -> None:
        placement = self.summary["placement"]
        self.assertEqual(placement["composed_world_scale_xyz"], [1.0, 1.0, 1.0])
        self.assertEqual(placement["yaw_deg"], 0.0)
        self.assertLessEqual(abs(placement["table_clearance_m"] - 0.002), 1e-9)
        self.assertEqual(
            self.summary["initial_clearance"]["maximum_aabb_overlap_m"], 0.0
        )

    def test_joint_and_collider_owners_are_explicit(self) -> None:
        joints = self.summary["joints"]["joints"]
        self.assertEqual(set(joints), {"target_joint_revolute", "joint_11"})
        for record in joints.values():
            self.assertTrue(record["body0_owner_paths"])
            self.assertTrue(record["body1_owner_paths"])
        topology = self.summary["topology"]
        self.assertEqual(topology["collision_count"], 54)
        self.assertEqual(len(topology["collision_owner_map"]), 54)
        self.assertTrue(all(topology["collision_owner_map"].values()))

    def test_dual_robot_and_zero_action_contract(self) -> None:
        sides = self.summary["robot"]["sides"]
        self.assertEqual(set(sides), {"left", "right"})
        self.assertTrue(all(side["pass"] for side in sides.values()))
        stability = self.summary["stability"]
        self.assertEqual(stability["zero_action_seconds"], 3.0)
        self.assertIs(stability["finite"], True)
        self.assertIs(stability["pass"], True)

    def test_selected_manifest_is_honest_and_historical_preview_is_preserved(self) -> None:
        # The checked-in historical PNG remains provenance for 44781.  The
        # selected asset now has a separate live CPU smoke frame and summary;
        # neither evidence bundle is allowed to masquerade as the other.
        preview = TASK_DIR / self.manifest["historical_head_camera"]["preview"]
        self.assertTrue(preview.is_file())
        self.assertEqual(preview.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(preview.stat().st_size, 150_000)
        selected_preview = TASK_DIR / self.manifest["head_camera"]["preview"]
        self.assertTrue(selected_preview.is_file())
        self.assertEqual(selected_preview.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(selected_preview.stat().st_size, 150_000)
        self.assertIs(self.manifest["head_camera"]["live_verified"], True)
        self.assertEqual(
            self.manifest["head_camera"]["prim_path"],
            "/World/RecordCameras/head_camera",
        )
        selected_summary = TASK_DIR / self.manifest["evidence"]["selected_asset_live_summary"]
        self.assertTrue(selected_summary.is_file())
        selected_payload = json.loads(selected_summary.read_text(encoding="utf-8"))
        self.assertIs(selected_payload["core_scene_pass"], True)
        self.assertTrue(all(selected_payload["checks"].values()))
        self.assertEqual(
            selected_payload["head_camera"]["prim_path"],
            "/World/RecordCameras/head_camera",
        )
        self.assertIs(selected_payload["head_camera"]["stage_prim_is_camera"], True)
        self.assertIs(selected_payload["head_camera"]["deferred_camera_render"], False)
        self.assertEqual(selected_payload["stability"]["maximum_dof_position_drift_rad"], 0.0)
        self.assertIs(selected_payload["reset_staging"]["hand_door_collision_filtering"], False)
        self.assertIs(self.manifest["zero_action"]["live_verified"], True)
        self.assertIs(self.manifest["checks"]["selected_scene_live_audit"], True)
        self.assertEqual(self.manifest["runtime_task_name"], "b320_synchronized_double_cabinet_open")
        self.assertIs(self.manifest["checks"]["selected_asset_provenance"], True)
        self.assertIs(self.manifest["checks"]["selected_runtime_closure"], True)
        self.assertIs(self.manifest["checks"]["core_scene_pass"], True)
        self.assertEqual(
            self.manifest["reset_staging"]["hand_door_collision_filtering"],
            False,
        )
        self.assertIs(self.manifest["task_complete"], False)

    def test_selected_physical_audits_are_bound_without_claiming_operator_success(self) -> None:
        exact_path = TASK_DIR / "evidence/live_run/fu001_pre_post_shape_audit.json"
        exact = json.loads(exact_path.read_text(encoding="utf-8"))
        self.assertEqual(exact["verdict"]["query_errors"], 0)
        self.assertIs(exact["verdict"]["all_three_exact_query_phases_clear"], True)
        self.assertIs(exact["verdict"]["contact_report_clear"], True)
        for phase in ("authored", "post_constructor", "post_settle"):
            self.assertEqual(exact[phase]["pair_count"], 0)
            self.assertEqual(exact[phase]["counts"], {
                "cabinet_collision_shapes": 5,
                "robot_collision_shapes": 60,
                "table_collision_shapes": 1,
            })
        long_path = TASK_DIR / "evidence/live_run/fu001_long_stability_camera.json"
        long_run = json.loads(long_path.read_text(encoding="utf-8"))
        self.assertIs(long_run["aggregate"]["pass"], True)
        self.assertEqual(long_run["aggregate"]["steps_completed"], 1200)
        self.assertEqual(long_run["camera"]["frame_count"], 30)
        self.assertEqual(long_run["camera"]["successful_frame_count"], 30)
        self.assertEqual(self.manifest["selected_asset_checks"]["operator_contact_and_success"], "pending")
        self.assertIs(self.manifest["task_complete"], False)


if __name__ == "__main__":
    unittest.main()
