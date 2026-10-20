from __future__ import annotations

import json
import unittest

from taskdev.tasks.bench320_fu001_synchronized_double_cabinet_open.scene_spec import (
    CABINET_RELATIVE,
    DOOR_INITIAL_FRACTION,
    DOOR_JOINTS,
    EXPECTED_SHA256,
    HEAD_CAMERA_PATH,
    HEAD_SHAPE,
    REPOSITORY_ROOT,
    SELECTED_FIXTURE_TRANSLATION_RANDOMIZATION_MODE,
    SELECTED_FIXTURE_TRANSLATION_RANDOMIZATION_RANGE_M,
    SELECTED_SOURCE_FIXTURE_TRANSLATION_RANDOMIZATION_RANGE_M,
    TASK_DIR,
    ZERO_ACTION_SECONDS,
)
from taskdev.tasks.bench320_fu001_synchronized_double_cabinet_open.validate_cpu import (
    validate,
)


class FU001SceneContractTest(unittest.TestCase):
    def test_cpu_contract_passes(self) -> None:
        payload = validate()
        self.assertTrue(payload["pass"], payload)

    def test_asset_bundle_is_repository_local_and_complete(self) -> None:
        self.assertFalse(CABINET_RELATIVE.is_absolute())
        self.assertTrue((REPOSITORY_ROOT / CABINET_RELATIVE).is_file())
        self.assertEqual(len(EXPECTED_SHA256), 17)
        self.assertTrue(all(len(value) == 64 for value in EXPECTED_SHA256.values()))

    def test_two_authored_door_joints_are_bound(self) -> None:
        meta = json.loads(
            (TASK_DIR / "assets/dv_double_cabinet_44781/meta.json").read_text(
                encoding="utf-8"
            )
        )
        real = {
            slot["slot_joint"]
            for slot in meta["slots"]
            if slot["type"] == "revolute" and slot["is_dummy"] is False
        }
        self.assertEqual(real, set(DOOR_JOINTS))

    def test_initial_and_camera_contract(self) -> None:
        self.assertGreaterEqual(DOOR_INITIAL_FRACTION, 0.10)
        self.assertLessEqual(DOOR_INITIAL_FRACTION, 0.30)
        self.assertEqual(HEAD_CAMERA_PATH, "/World/RecordCameras/head_camera")
        self.assertEqual(HEAD_SHAPE, (480, 640, 3))
        self.assertGreaterEqual(ZERO_ACTION_SECONDS, 3.0)

    def test_fixed_fixture_randomization_exemption_is_explicit(self) -> None:
        """A fixed articulation must not claim an unapplied root jitter."""
        self.assertEqual(
            SELECTED_FIXTURE_TRANSLATION_RANDOMIZATION_MODE,
            "fixed_articulated_root_exempt",
        )
        self.assertEqual(
            SELECTED_SOURCE_FIXTURE_TRANSLATION_RANDOMIZATION_RANGE_M,
            (0.030, 0.030, 0.0),
        )
        self.assertEqual(
            SELECTED_FIXTURE_TRANSLATION_RANDOMIZATION_RANGE_M,
            (0.0, 0.0, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
