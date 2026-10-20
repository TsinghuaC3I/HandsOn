from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from teleoperation.manus import calibration_store as store


ROOT = Path(__file__).resolve().parents[2]
BRIDGE_SCRIPT = ROOT / "teleoperation" / "manus" / "manus_wuji_ros2_bridge.py"
SIDES = ("right", "left")
SOLVER = "test-solver-v1"


def neutral_anchors(*, right_rotation: np.ndarray | None = None) -> dict[str, np.ndarray]:
    right = np.eye(4, dtype=np.float64)
    left = np.eye(4, dtype=np.float64)
    right[:3, 3] = [0.0, -0.3, 1.0]
    left[:3, 3] = [0.0, 0.3, 1.0]
    if right_rotation is not None:
        right[:3, :3] = right_rotation
    return {"right": right, "left": left}


def fingerprint(**changes: object) -> str:
    arguments: dict[str, object] = {
        "mapping_content": "startup_calibration: {enabled: true}\n",
        "openvr_content": "tracker_serials: {right_wrist: R, left_wrist: L}\n",
        "requested_sides": SIDES,
        "solver_version": SOLVER,
        "runtime_options": {"swap_wrist_trackers": False},
    }
    arguments.update(changes)
    return store.build_calibration_fingerprint(**arguments)  # type: ignore[arg-type]


def signatures(anchors: dict[str, np.ndarray] | None = None):
    return store.build_anchor_signatures(
        anchors or neutral_anchors(),
        SIDES,
        world_up=np.array([0.0, 0.0, 1.0]),
    )


def load_bridge_module():
    rclpy = types.ModuleType("rclpy")
    rclpy.duration = types.ModuleType("rclpy.duration")
    rclpy.executors = types.ModuleType("rclpy.executors")
    rclpy.node = types.ModuleType("rclpy.node")
    rclpy.qos = types.ModuleType("rclpy.qos")
    rclpy.time = types.ModuleType("rclpy.time")
    rclpy.duration.Duration = type("Duration", (), {})
    rclpy.executors.ExternalShutdownException = type(
        "ExternalShutdownException", (Exception,), {}
    )
    rclpy.node.Node = type("Node", (), {})
    rclpy.qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST=1)
    rclpy.qos.QoSProfile = type("QoSProfile", (), {})
    rclpy.qos.ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT=1)
    rclpy.time.Time = type("Time", (), {})

    std_msgs = types.ModuleType("std_msgs")
    std_msgs.msg = types.ModuleType("std_msgs.msg")
    std_msgs.msg.Float32MultiArray = type("Float32MultiArray", (), {})
    manus_ros2_msgs = types.ModuleType("manus_ros2_msgs")
    manus_ros2_msgs.msg = types.ModuleType("manus_ros2_msgs.msg")
    manus_ros2_msgs.msg.ManusGlove = type("ManusGlove", (), {})
    retargeting = types.ModuleType("wuji_retargeting")
    retargeting.Retargeter = type("Retargeter", (), {})
    stubs = {
        "rclpy": rclpy,
        "rclpy.duration": rclpy.duration,
        "rclpy.executors": rclpy.executors,
        "rclpy.node": rclpy.node,
        "rclpy.qos": rclpy.qos,
        "rclpy.time": rclpy.time,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs.msg,
        "manus_ros2_msgs": manus_ros2_msgs,
        "manus_ros2_msgs.msg": manus_ros2_msgs.msg,
        "wuji_retargeting": retargeting,
    }
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in stubs}
    try:
        sys.modules.update(stubs)
        spec = importlib.util.spec_from_file_location(
            "manus_calibration_cache_bridge",
            BRIDGE_SCRIPT,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in previous.items():
            if old is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


bridge = load_bridge_module()


def translated_pose(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [x, y, z]
    return pose


class GuidedDirectionCalibrationTest(unittest.TestCase):
    def test_phase_summary_uses_endpoints_instead_of_midphase_overshoot(self) -> None:
        samples = (
            [translated_pose(0.0) for _ in range(5)]
            + [translated_pose(-0.50) for _ in range(3)]
            + [translated_pose(0.10) for _ in range(5)]
        )

        delta, distance = bridge.phase_motion_summary(samples)

        self.assertGreater(delta[0], 0.0)
        self.assertAlmostEqual(distance, 0.10)

    def test_pair_direction_alignment_detects_opposite_motion(self) -> None:
        right = [translated_pose(0.0) for _ in range(5)] + [
            translated_pose(0.10) for _ in range(5)
        ]
        left = [translated_pose(0.0) for _ in range(5)] + [
            translated_pose(-0.10) for _ in range(5)
        ]

        alignment = bridge.guided_pair_direction_alignment(right, left)

        self.assertIsNotNone(alignment)
        self.assertAlmostEqual(float(alignment), -1.0)


class CalibrationStoreTest(unittest.TestCase):
    def test_default_path_uses_xdg_state_home(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with mock.patch.dict(
                os.environ,
                {"XDG_STATE_HOME": temporary_directory},
                clear=False,
            ):
                self.assertEqual(
                    store.default_calibration_cache_path(),
                    Path(temporary_directory)
                    / "handson"
                    / "calibration"
                    / "v1"
                    / "mode_a.json",
                )

    def test_fingerprint_covers_configs_sides_solver_and_runtime(self) -> None:
        baseline = fingerprint()
        variants = [
            fingerprint(mapping_content="changed mapping"),
            fingerprint(openvr_content="changed openvr"),
            fingerprint(requested_sides=("right",)),
            fingerprint(solver_version="test-solver-v2"),
            fingerprint(runtime_options={"swap_wrist_trackers": True}),
        ]
        self.assertEqual(len(set([baseline, *variants])), 6)

    def test_roundtrip_is_private_atomic_and_contains_no_anchor_position(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "private" / "mode_a.json"
            expected_fingerprint = fingerprint()
            matrices = {side: np.eye(3) for side in SIDES}
            written = store.write_calibration_cache(
                path,
                fingerprint=expected_fingerprint,
                solver_version=SOLVER,
                requested_sides=SIDES,
                tracker_raw_to_sim=matrices,
                anchor_signatures=signatures(),
            )
            loaded = store.load_calibration_cache(
                path,
                expected_fingerprint=expected_fingerprint,
                expected_solver_version=SOLVER,
                expected_sides=SIDES,
            )

            self.assertEqual(written.sides, SIDES)
            self.assertEqual(loaded.sides, SIDES)
            for side in SIDES:
                np.testing.assert_allclose(loaded.tracker_raw_to_sim[side], np.eye(3))
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("tracker_anchor_raw", payload)
            self.assertNotIn("position", path.read_text(encoding="utf-8"))
            self.assertFalse(list(path.parent.glob(f".{path.name}.*")))

    def test_existing_parent_permissions_are_not_changed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory) / "shared"
            parent.mkdir(mode=0o755)
            parent.chmod(0o755)
            path = parent / "mode_a.json"

            store.write_calibration_cache(
                path,
                fingerprint=fingerprint(),
                solver_version=SOLVER,
                requested_sides=SIDES,
                tracker_raw_to_sim={side: np.eye(3) for side in SIDES},
                anchor_signatures=signatures(),
            )

            self.assertEqual(stat.S_IMODE(parent.stat().st_mode), 0o755)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_corrupt_or_non_rotation_matrix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "mode_a.json"
            expected_fingerprint = fingerprint()
            store.write_calibration_cache(
                path,
                fingerprint=expected_fingerprint,
                solver_version=SOLVER,
                requested_sides=SIDES,
                tracker_raw_to_sim={side: np.eye(3) for side in SIDES},
                anchor_signatures=signatures(),
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["tracker_raw_to_sim"]["right"][0][0] = 2.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(store.CalibrationCacheError, "orthonormal"):
                store.load_calibration_cache(
                    path,
                    expected_fingerprint=expected_fingerprint,
                    expected_solver_version=SOLVER,
                    expected_sides=SIDES,
                )

            path.write_text("{truncated", encoding="utf-8")
            with self.assertRaisesRegex(store.CalibrationCacheError, "cannot read"):
                store.load_calibration_cache(
                    path,
                    expected_fingerprint=expected_fingerprint,
                    expected_solver_version=SOLVER,
                    expected_sides=SIDES,
                )

    def test_world_readable_cache_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "mode_a.json"
            expected_fingerprint = fingerprint()
            store.write_calibration_cache(
                path,
                fingerprint=expected_fingerprint,
                solver_version=SOLVER,
                requested_sides=SIDES,
                tracker_raw_to_sim={side: np.eye(3) for side in SIDES},
                anchor_signatures=signatures(),
            )
            path.chmod(0o644)
            with self.assertRaisesRegex(store.CalibrationCacheError, "0600"):
                store.load_calibration_cache(
                    path,
                    expected_fingerprint=expected_fingerprint,
                    expected_solver_version=SOLVER,
                    expected_sides=SIDES,
                )

    def test_fingerprint_mismatch_rejects_without_modifying_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "mode_a.json"
            store.write_calibration_cache(
                path,
                fingerprint=fingerprint(),
                solver_version=SOLVER,
                requested_sides=SIDES,
                tracker_raw_to_sim={side: np.eye(3) for side in SIDES},
                anchor_signatures=signatures(),
            )
            before = path.read_bytes()
            with self.assertRaisesRegex(store.CalibrationCacheError, "fingerprint"):
                store.load_calibration_cache(
                    path,
                    expected_fingerprint=fingerprint(mapping_content="new mapping"),
                    expected_solver_version=SOLVER,
                    expected_sides=SIDES,
                )
            self.assertEqual(path.read_bytes(), before)

    def test_failed_atomic_replace_preserves_previous_good_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "mode_a.json"
            expected_fingerprint = fingerprint()
            store.write_calibration_cache(
                path,
                fingerprint=expected_fingerprint,
                solver_version=SOLVER,
                requested_sides=SIDES,
                tracker_raw_to_sim={side: np.eye(3) for side in SIDES},
                anchor_signatures=signatures(),
            )
            before = path.read_bytes()
            angle = np.deg2rad(10.0)
            rotated = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            with mock.patch.object(
                store.os,
                "replace",
                side_effect=PermissionError("replace denied"),
            ):
                with self.assertRaisesRegex(store.CalibrationCacheError, "replace denied"):
                    store.write_calibration_cache(
                        path,
                        fingerprint=expected_fingerprint,
                        solver_version=SOLVER,
                        requested_sides=SIDES,
                        tracker_raw_to_sim={"right": rotated, "left": np.eye(3)},
                        anchor_signatures=signatures(),
                    )
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(list(path.parent.glob(f".{path.name}.*")))

    def test_post_replace_directory_fsync_failure_keeps_new_cache_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "mode_a.json"
            expected_fingerprint = fingerprint()
            fsync_calls = 0
            real_fsync = store.os.fsync

            def fail_directory_fsync(descriptor: int) -> None:
                nonlocal fsync_calls
                fsync_calls += 1
                if fsync_calls == 2:
                    raise OSError("directory fsync unavailable")
                real_fsync(descriptor)

            with mock.patch.object(store.os, "fsync", side_effect=fail_directory_fsync):
                store.write_calibration_cache(
                    path,
                    fingerprint=expected_fingerprint,
                    solver_version=SOLVER,
                    requested_sides=SIDES,
                    tracker_raw_to_sim={side: np.eye(3) for side in SIDES},
                    anchor_signatures=signatures(),
                )

            loaded = store.load_calibration_cache(
                path,
                expected_fingerprint=expected_fingerprint,
                expected_solver_version=SOLVER,
                expected_sides=SIDES,
            )
            self.assertEqual(loaded.sides, SIDES)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_mount_signature_change_is_rejected(self) -> None:
        angle = np.deg2rad(30.0)
        rotated = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        current = signatures(neutral_anchors(right_rotation=rotated))
        with self.assertRaisesRegex(store.CalibrationCacheError, "changed by"):
            store.validate_anchor_signatures(
                current,
                signatures(),
                SIDES,
                tolerance_deg=15.0,
            )


class BridgeCacheValidationTest(unittest.TestCase):
    def test_cache_path_cannot_replace_mapping_or_openvr_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            mapping_path = root / "mapping.yaml"
            openvr_path = root / "openvr.yaml"
            safe_cache = root / "state" / "mode_a.json"
            mapping_path.write_text("mapping", encoding="utf-8")
            openvr_path.write_text("openvr", encoding="utf-8")

            self.assertTrue(
                bridge.calibration_cache_path_conflicts(
                    mapping_path,
                    (mapping_path, openvr_path),
                )
            )
            self.assertTrue(
                bridge.calibration_cache_path_conflicts(
                    openvr_path,
                    (mapping_path, openvr_path),
                )
            )
            self.assertFalse(
                bridge.calibration_cache_path_conflicts(
                    safe_cache,
                    (mapping_path, openvr_path),
                )
            )

    def test_missing_fingerprint_config_fails_closed(self) -> None:
        with self.assertRaises(FileNotFoundError):
            bridge.read_config_content(None)
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing-openvr.yaml"
            with self.assertRaises(FileNotFoundError):
                bridge.read_config_content(missing)

    def test_single_hand_mode_does_not_support_calibration_cache(self) -> None:
        self.assertTrue(
            bridge.startup_calibration_cache_supported(("right", "left"))
        )
        self.assertFalse(bridge.startup_calibration_cache_supported(("right",)))
        self.assertFalse(bridge.startup_calibration_cache_supported(("left",)))

    def test_force_environment_bypasses_cache_by_defaulting_cli_flag(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"HANDSON_FORCE_RECALIBRATE": "1"},
            clear=False,
        ):
            args = bridge.build_argparser().parse_args([])
        self.assertTrue(args.force_recalibrate)

    def test_guided_cache_write_targets_candidate_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            permanent = root / "mode_a.json"
            candidate = root / "mode_a.candidate"
            permanent.write_text("old-cache", encoding="utf-8")
            controller = object.__new__(bridge.ManusWujiHandsOnBridge)
            controller.hands = SIDES
            controller._tracker_anchor_raw = neutral_anchors()
            controller.tracker_raw_to_sim = {
                side: np.eye(3, dtype=np.float64) for side in SIDES
            }
            controller._calibration_fingerprint = fingerprint()
            controller._calibration_cache_path = permanent
            controller._calibration_cache_output_path = candidate
            messages: list[str] = []
            controller.get_logger = lambda: types.SimpleNamespace(
                info=messages.append,
            )

            controller._persist_startup_calibration_cache()

            self.assertEqual(permanent.read_text(encoding="utf-8"), "old-cache")
            self.assertTrue(candidate.exists())
            self.assertEqual(candidate.stat().st_mode & 0o777, 0o600)
            self.assertTrue(
                any("cache candidate saved" in message for message in messages)
            )

    def make_controller(
        self,
        current_anchors: dict[str, np.ndarray],
        cached_signatures=None,
    ):
        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        controller.hands = SIDES
        controller._calibration_cache_candidate = store.CalibrationCache(
            fingerprint="a" * 64,
            solver_version=SOLVER,
            sides=SIDES,
            tracker_raw_to_sim={side: np.eye(3) for side in SIDES},
            anchor_signatures=cached_signatures or signatures(),
            created_at="2026-01-01T00:00:00+00:00",
        )
        controller._calib_phase = "cache_anchor"
        controller._calib_phase_t0 = 0.0
        controller._cache_anchor_sec = 0.1
        controller._cache_anchor_samples = {side: [] for side in SIDES}
        controller._calib_settle_samples = {side: [] for side in SIDES}
        controller._calib_rotate_samples = {side: [] for side in SIDES}
        controller._guided_samples = {
            phase: {side: [] for side in SIDES}
            for phase in bridge.GUIDED_PHASES
        }
        controller._calib_min_samples = 1
        controller._guided_anchor_max_stddev_m = 0.01
        controller._cache_max_rotation_spread_deg = 5.0
        controller._cache_signature_tolerance_deg = 15.0
        controller._last_tracker_missing_log = 0.0
        controller._tracker_anchor_raw = {side: None for side in SIDES}
        controller._calib_solved = {side: False for side in SIDES}
        controller.tracker_raw_to_sim = {side: np.zeros((3, 3)) for side in SIDES}
        controller._pair_binding_done = False
        controller._calib_mode = "guided"
        controller._raw_tracker_set = lambda: {
            side: pose.copy() for side, pose in current_anchors.items()
        }
        controller._check_pair_binding = lambda: True
        messages: list[tuple[str, str]] = []
        controller.get_logger = lambda: types.SimpleNamespace(
            info=lambda message: messages.append(("info", message)),
            warn=lambda message: messages.append(("warn", message)),
        )
        return controller, messages

    def test_cache_hit_captures_fresh_anchor_before_ready(self) -> None:
        if bridge.Rotation is None:
            self.skipTest("scipy is unavailable")
        current = neutral_anchors()
        current["right"][:3, 3] += [1.0, 2.0, 0.0]
        current["left"][:3, 3] += [1.0, 2.0, 0.0]
        controller, messages = self.make_controller(current)
        with mock.patch.object(bridge.time, "monotonic", return_value=1.0):
            self.assertTrue(controller._collect_cached_startup_calibration())

        self.assertEqual(controller._calib_phase, "done")
        self.assertTrue(all(controller._calib_solved.values()))
        for side in SIDES:
            np.testing.assert_allclose(
                controller._tracker_anchor_raw[side][:3, 3],
                current[side][:3, 3],
            )
            np.testing.assert_allclose(controller.tracker_raw_to_sim[side], np.eye(3))
        self.assertIn(
            ("info", "Startup calibration ready: source=cache"),
            messages,
        )

    def test_cache_signature_failure_falls_back_without_writing(self) -> None:
        if bridge.Rotation is None:
            self.skipTest("scipy is unavailable")
        angle = np.deg2rad(30.0)
        rotated = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        current = neutral_anchors(right_rotation=rotated)
        controller, messages = self.make_controller(current)
        with mock.patch.object(bridge.time, "monotonic", return_value=1.0):
            self.assertFalse(controller._collect_cached_startup_calibration())

        self.assertEqual(controller._calib_phase, "anchor")
        self.assertIsNone(controller._calibration_cache_candidate)
        self.assertFalse(any(controller._calib_solved.values()))
        self.assertTrue(
            any(
                level == "warn" and "cache rejected" in message
                for level, message in messages
            )
        )

    def test_transient_short_anchor_retries_without_invalidating_cache(self) -> None:
        if bridge.Rotation is None:
            self.skipTest("scipy is unavailable")
        controller, messages = self.make_controller(neutral_anchors())
        controller._calib_min_samples = 2
        with mock.patch.object(bridge.time, "monotonic", return_value=1.0):
            self.assertFalse(controller._collect_cached_startup_calibration())

        self.assertEqual(controller._calib_phase, "cache_anchor")
        self.assertIsNotNone(controller._calibration_cache_candidate)
        self.assertTrue(
            any(
                level == "warn" and "repeat the neutral hold" in message
                for level, message in messages
            )
        )

    def test_pair_binding_distance_tolerance_override_allows_wide_anchor(self) -> None:
        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        right = np.eye(4, dtype=np.float64)
        left = np.eye(4, dtype=np.float64)
        left[:3, 3] = [-0.122, -0.802, -0.006]
        controller._tracker_anchor_raw = {"right": right, "left": left}
        controller._pair_binding_cfg = {
            "expected_distance_m": 0.60,
            "distance_tolerance_m": 0.15,
            "same_axis_tolerance_m": 0.10,
            "enforce": True,
        }
        controller._pair_binding_enabled = True
        controller._pair_binding_done = False
        controller._pair_binding_distance_tolerance_m = None
        messages: list[tuple[str, str]] = []
        controller.get_logger = lambda: types.SimpleNamespace(
            info=lambda message: messages.append(("info", message)),
            warn=lambda message: messages.append(("warn", message)),
        )

        self.assertFalse(controller._check_pair_binding())
        self.assertFalse(controller._pair_binding_done)

        controller._pair_binding_distance_tolerance_m = 0.25
        self.assertTrue(controller._check_pair_binding())
        self.assertTrue(controller._pair_binding_done)
        self.assertTrue(
            any(level == "info" and "range=0.350-0.850" in message for level, message in messages)
        )

    def test_environment_float_rejects_invalid_values(self) -> None:
        with mock.patch.dict(os.environ, {"HANDSON_TEST_FLOAT": "0.25"}, clear=False):
            self.assertEqual(bridge.environment_float("HANDSON_TEST_FLOAT"), 0.25)
        with mock.patch.dict(os.environ, {"HANDSON_TEST_FLOAT": "wide"}, clear=False):
            with self.assertRaises(ValueError):
                bridge.environment_float("HANDSON_TEST_FLOAT")


if __name__ == "__main__":
    unittest.main()
