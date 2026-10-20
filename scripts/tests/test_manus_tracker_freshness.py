from __future__ import annotations

import importlib.util
import sys
import threading
import time
import types
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BRIDGE_SCRIPT = ROOT / "teleoperation" / "manus" / "manus_wuji_ros2_bridge.py"


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
        spec = importlib.util.spec_from_file_location("tracker_freshness_bridge", BRIDGE_SCRIPT)
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


class TrackerFreshnessTest(unittest.TestCase):
    def test_executor_propagates_receiver_failure_and_joins_before_shutdown(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        shutdown_done = threading.Event()
        caught = []

        class SerialExecutor:
            def __init__(self, **_kwargs):
                self.woken = threading.Event()

            def add_node(self, node):
                self.node = node

            def spin_once(self, **_kwargs):
                if self.node == "receiver":
                    entered.set()
                    release.wait(3.0)
                    raise RuntimeError("receiver failed")
                self.woken.wait(0.1)
                self.woken.clear()

            def wake(self):
                self.woken.set()

            def shutdown(self):
                pass

        node = types.SimpleNamespace(
            context=types.SimpleNamespace(ok=lambda: True),
            receiver_nodes=["receiver"],
            _raw_recording_enabled=True,
            get_logger=lambda: types.SimpleNamespace(info=lambda _message: None),
        )
        with mock.patch.dict(sys.modules, {
            "rclpy.executors": types.SimpleNamespace(SingleThreadedExecutor=SerialExecutor),
        }):
            executor = bridge.BridgeExecutor(node)

        def spin():
            try:
                executor.spin()
            except RuntimeError as exc:
                caught.append(str(exc))

        def shutdown():
            executor.shutdown()
            shutdown_done.set()

        thread = threading.Thread(target=spin)
        closer = threading.Thread(target=shutdown)
        thread.start()
        try:
            self.assertTrue(entered.wait(2.0))
            closer.start()
            self.assertFalse(shutdown_done.wait(0.05))
        finally:
            release.set()
            thread.join(timeout=3.0)
            if closer.ident is not None:
                closer.join(timeout=3.0)
            executor.shutdown()
        self.assertFalse(thread.is_alive())
        self.assertTrue(shutdown_done.is_set())
        self.assertEqual(caught, ["receiver failed"])

    def test_collection_mapping_enables_continuous_tracker_reconnect(self) -> None:
        mapping_path = (
            ROOT
            / "teleoperation"
            / "manus"
            / "config"
            / "handson_manus_mapping.yaml"
        )
        mapping = bridge.yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
        safety = mapping["tracker_safety"]
        motion_correction = mapping["tracker_motion_correction"]

        self.assertTrue(safety["preserve_baseline_on_reconnect"])
        self.assertFalse(safety["reacquire_reset_on_return"])
        self.assertFalse(safety["reset_after_persistent_jump"])
        self.assertLessEqual(safety["reconnect_stable_sec"], 0.05)
        self.assertGreaterEqual(safety["max_tf_age_sec"], 0.10)
        self.assertFalse(motion_correction["invert_world_z"]["right"])
        self.assertFalse(motion_correction["invert_world_z"]["left"])

    def test_transform_age_uses_ros_timestamp(self) -> None:
        now = 13_250_000_000
        self.assertAlmostEqual(bridge.transform_age_seconds(now, 12, 500_000_000), 0.75)

    def test_future_transform_is_not_considered_stale(self) -> None:
        self.assertEqual(bridge.transform_age_seconds(1_000_000_000, 2, 0), 0.0)

    def test_calibration_boundary_resets_guard_before_live_pose(self) -> None:
        np = __import__("numpy")
        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        controller.hands = ("right", "left")
        controller._calib_phase = "up"
        controller._calib_phase_t0 = 1.0
        controller._calibration_cache_supported = False
        controller._latest_tracker = {
            "right": np.eye(4),
            "left": np.eye(4),
        }
        controller._tracker_rebase_correction = {
            side: bridge.identity_tracker_pose_correction()
            for side in controller.hands
        }
        controller._tracker_jump_reject_count = {"right": 4, "left": 3}
        controller._tracker_missing_since = {"right": 1.0, "left": 1.0}
        controller._tracker_recovery_active = {"right": True, "left": True}
        controller._tracker_recovery_samples = {
            "right": [(1.0, np.eye(4))],
            "left": [(1.0, np.eye(4))],
        }
        controller._tracker_debug_start_raw = {
            "right": np.eye(4),
            "left": np.eye(4),
        }
        controller._tracker_debug_start_out = {
            "right": np.eye(4),
            "left": np.eye(4),
        }
        controller.get_logger = lambda: types.SimpleNamespace(
            info=lambda _message: None,
            warn=lambda _message: None,
        )

        controller._mark_startup_calibration_ready("cache")

        self.assertEqual(controller._calib_phase, "done")
        self.assertTrue(
            all(controller._latest_tracker[side] is None for side in controller.hands)
        )
        self.assertFalse(any(controller._tracker_recovery_active.values()))
        self.assertTrue(
            all(not controller._tracker_recovery_samples[side] for side in controller.hands)
        )

        # The first pose in the calibrated frame may be far from the fixed
        # pre-calibration anchor.  It establishes the guard baseline instead
        # of being rejected as a one-frame SteamVR jump.
        controller.tracker_hold_on_jump = True
        controller.tracker_max_step_m = {"right": 0.12, "left": 0.12}
        controller.tracker_max_rotation_deg = {"right": 35.0, "left": 35.0}
        controller.tracker_reacquire_after_rejects = {"right": 12, "left": 12}
        controller.tracker_position_deadband_m = {"right": 0.0, "left": 0.0}
        controller.tracker_rotation_deadband_deg = {"right": 0.0, "left": 0.0}
        controller.tracker_smoothing_alpha = {"right": 1.0, "left": 1.0}
        controller.tracker_reset_after_persistent_jump = False
        first_live_pose = np.eye(4)
        first_live_pose[:3, 3] = [0.299, 0.510, -0.425]
        self.assertTrue(
            controller._accept_tracker_matrix("right", first_live_pose)
        )
        np.testing.assert_allclose(
            controller._latest_tracker["right"],
            first_live_pose,
        )

    def test_episode_request_releases_recovery_and_acks_exact_first_packet(
        self,
    ) -> None:
        np = __import__("numpy")

        class FakeControlSocket:
            def __init__(self) -> None:
                self.requests = [
                    (
                        bridge.TRACKER_EPISODE_BASELINE_REQUEST,
                        ("127.0.0.1", 41001),
                    )
                ]
                self.sent: list[tuple[bytes, tuple[str, int]]] = []

            def recvfrom(self, _size):
                if not self.requests:
                    raise BlockingIOError
                return self.requests.pop(0)

            def sendto(self, packet, target):
                self.sent.append((packet, target))

        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        controller.hands = ("right", "left")
        controller._calib_enabled = False
        controller._calib_phase = "done"
        controller._latest_tracker = {
            "right": np.eye(4),
            "left": np.eye(4),
        }
        controller._tracker_rebase_correction = {
            side: bridge.tracker_rebase_correction(np.eye(4), np.eye(4))
            for side in controller.hands
        }
        controller._tracker_jump_reject_count = {"right": 5, "left": 7}
        controller._tracker_missing_since = {"right": 1.0, "left": 1.0}
        controller._tracker_recovery_active = {"right": True, "left": True}
        controller._tracker_recovery_samples = {
            "right": [(1.0, np.eye(4))],
            "left": [(1.0, np.eye(4))],
        }
        controller._tracker_debug_start_raw = {
            "right": np.eye(4),
            "left": np.eye(4),
        }
        controller._tracker_debug_start_out = {
            "right": np.eye(4),
            "left": np.eye(4),
        }
        controller._pending_tracker_episode_ack_targets = []
        controller._tracker_guard_armed = True
        controller.tracker_reconnect_stable_sec = 0.35
        controller.tracker_reconnect_max_position_spread_m = 0.008
        controller.tracker_reconnect_max_rotation_spread_deg = 2.0
        controller.tracker_control_socket = FakeControlSocket()
        controller.get_logger = lambda: types.SimpleNamespace(
            info=lambda _message: None,
            warn=lambda _message: None,
        )

        controller._consume_tracker_episode_baseline_requests()

        self.assertTrue(
            all(controller._latest_tracker[side] is None for side in controller.hands)
        )
        self.assertTrue(all(controller._tracker_recovery_active.values()))
        self.assertTrue(controller._tracker_episode_baseline_pending)
        self.assertFalse(controller._tracker_guard_armed)
        self.assertEqual(
            controller._pending_tracker_episode_ack_targets,
            [("127.0.0.1", 41001)],
        )

        baseline_pose = np.eye(4)
        baseline_pose[0, 3] = 0.45
        for side in controller.hands:
            self.assertTrue(
                controller._tracker_recovery_candidate_ready(
                    side,
                    baseline_pose,
                    now=0.0,
                )
            )
            self.assertTrue(
                controller._tracker_recovery_candidate_ready(
                    side,
                    baseline_pose,
                    now=0.01,
                )
            )

        first_packet = np.arange(24 * 8, dtype=np.uint8).tobytes()
        controller._acknowledge_tracker_episode_baseline(first_packet)

        self.assertEqual(
            controller.tracker_control_socket.sent,
            [
                (
                    bridge.TRACKER_EPISODE_BASELINE_ACK + first_packet,
                    ("127.0.0.1", 41001),
                )
            ],
        )
        self.assertEqual(controller._pending_tracker_episode_ack_targets, [])
        self.assertTrue(controller._tracker_guard_armed)

    def test_episode_guard_release_allows_idle_repositioning(self) -> None:
        np = __import__("numpy")

        class FakeControlSocket:
            def __init__(self) -> None:
                self.requests = [
                    (
                        bridge.TRACKER_EPISODE_GUARD_RELEASE_REQUEST,
                        ("127.0.0.1", 41002),
                    )
                ]
                self.sent: list[tuple[bytes, tuple[str, int]]] = []

            def recvfrom(self, _size):
                if not self.requests:
                    raise BlockingIOError
                return self.requests.pop(0)

            def sendto(self, packet, target):
                self.sent.append((packet, target))

        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        controller.hands = ("right", "left")
        controller._calib_enabled = False
        controller._calib_phase = "done"
        controller._tracker_guard_armed = True
        controller._tracker_episode_baseline_pending = False
        controller._latest_tracker = {
            "right": np.eye(4),
            "left": np.eye(4),
        }
        controller._tracker_rebase_correction = {
            side: bridge.identity_tracker_pose_correction()
            for side in controller.hands
        }
        controller._tracker_jump_reject_count = {"right": 5, "left": 7}
        controller._tracker_missing_since = {"right": 1.0, "left": 1.0}
        controller._tracker_recovery_active = {"right": True, "left": True}
        controller._tracker_recovery_samples = {
            "right": [(1.0, np.eye(4))],
            "left": [(1.0, np.eye(4))],
        }
        controller._tracker_debug_start_raw = {
            "right": np.eye(4),
            "left": np.eye(4),
        }
        controller._tracker_debug_start_out = {
            "right": np.eye(4),
            "left": np.eye(4),
        }
        controller._pending_tracker_episode_ack_targets = [
            ("127.0.0.1", 41999)
        ]
        controller.tracker_control_socket = FakeControlSocket()
        controller.get_logger = lambda: types.SimpleNamespace(
            info=lambda _message: None,
            warn=lambda _message: None,
        )

        controller._consume_tracker_episode_baseline_requests()

        self.assertFalse(controller._tracker_guard_armed)
        self.assertFalse(controller._tracker_episode_baseline_pending)
        self.assertTrue(
            all(controller._latest_tracker[side] is None for side in controller.hands)
        )
        self.assertFalse(any(controller._tracker_recovery_active.values()))
        self.assertEqual(controller._pending_tracker_episode_ack_targets, [])
        self.assertEqual(
            controller.tracker_control_socket.sent,
            [
                (
                    bridge.TRACKER_EPISODE_GUARD_RELEASE_ACK,
                    ("127.0.0.1", 41002),
                )
            ],
        )

    def test_idle_tracker_jump_is_accepted_before_collection_baseline(self) -> None:
        np = __import__("numpy")
        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        previous = np.eye(4)
        controller._tracker_guard_armed = False
        controller._latest_tracker = {"right": previous.copy()}
        controller._tracker_rebase_correction = {
            "right": bridge.identity_tracker_pose_correction()
        }
        controller._tracker_jump_reject_count = {"right": 0}
        controller._last_tracker_jump_log = {"right": float("-inf")}
        controller.tracker_hold_on_jump = True
        controller.tracker_max_step_m = {"right": 0.12}
        controller.tracker_max_rotation_deg = {"right": 35.0}
        controller.tracker_reacquire_after_rejects = {"right": 12}
        controller.tracker_position_deadband_m = {"right": 0.0}
        controller.tracker_rotation_deadband_deg = {"right": 0.0}
        controller.tracker_smoothing_alpha = {"right": 1.0}
        controller.tracker_reset_after_persistent_jump = False

        placed_on_table = np.eye(4)
        placed_on_table[0, 3] = 0.40

        self.assertTrue(
            controller._accept_tracker_matrix("right", placed_on_table)
        )
        np.testing.assert_allclose(
            controller._latest_tracker["right"],
            placed_on_table,
        )

    def test_dual_tracker_packet_is_suppressed_when_one_tf_is_missing(self) -> None:
        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        controller.args = types.SimpleNamespace(send_tracker=True)
        controller.hands = ("right", "left")
        controller._calib_enabled = False
        controller._calib_phase = "done"
        controller._latest_tracker = {
            "right": __import__("numpy").eye(4),
            "left": __import__("numpy").eye(4),
        }
        controller._last_tracker_missing_log = float("inf")
        controller._lookup_tracker_matrix = lambda side: (
            __import__("numpy").eye(4) if side == "right" else None
        )
        controller._accept_tracker_matrix = lambda side, matrix: True
        controller.get_logger = lambda: types.SimpleNamespace(warn=lambda _message: None)
        controller.tracker_socket = types.SimpleNamespace(
            sendto=lambda *_args: self.fail("stale tracker packet was sent")
        )

        controller._send_tracker()

    def test_brief_dual_tracker_gap_does_not_enter_strict_recovery(self) -> None:
        np = __import__("numpy")
        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        controller.args = types.SimpleNamespace(send_tracker=True)
        controller.hands = ("right", "left")
        controller._calib_enabled = False
        controller._calib_phase = "done"
        controller._latest_tracker = {
            "right": np.eye(4),
            "left": np.eye(4),
        }
        controller._tracker_missing_since = {"right": None, "left": None}
        controller._tracker_recovery_active = {"right": False, "left": False}
        controller._tracker_recovery_samples = {"right": [], "left": []}
        controller._tracker_guard_armed = True
        controller.tracker_reacquire_reset_after_sec = 0.15
        controller._last_tracker_missing_log = float("inf")
        controller._lookup_tracker_matrix = lambda side: (
            np.eye(4) if side == "right" else None
        )
        recovery_starts: list[str] = []
        controller._begin_tracker_recovery = lambda side, **_kwargs: (
            recovery_starts.append(side)
        )
        controller.get_logger = lambda: types.SimpleNamespace(
            warn=lambda _message: None,
            info=lambda _message: None,
        )
        packets: list[bytes] = []
        controller.tracker_socket = types.SimpleNamespace(
            sendto=lambda packet, _target: packets.append(packet)
        )
        controller.tracker_target = ("127.0.0.1", 5012)
        controller.swap_arms = False
        controller._tracker_sent_count = 0
        controller._log_tracker_delta_debug = lambda _outgoing: None
        controller._accept_tracker_batch = lambda _matrices, **_kwargs: True

        original_monotonic = bridge.time.monotonic
        try:
            bridge.time.monotonic = lambda: 1.0
            controller._send_tracker()
            self.assertEqual(recovery_starts, [])
            self.assertEqual(packets, [])

            controller._lookup_tracker_matrix = lambda _side: np.eye(4)
            bridge.time.monotonic = lambda: 1.10
            controller._send_tracker()
        finally:
            bridge.time.monotonic = original_monotonic

        self.assertEqual(recovery_starts, [])
        self.assertEqual(len(packets), 1)
        self.assertEqual(len(packets[0]), 192)
        self.assertEqual(
            controller._tracker_missing_since,
            {"right": None, "left": None},
        )

    def test_one_frame_discontinuity_does_not_latch_pair_recovery(self) -> None:
        np = __import__("numpy")
        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        controller.args = types.SimpleNamespace(send_tracker=True)
        controller.hands = ("right", "left")
        controller._calib_enabled = False
        controller._calib_phase = "done"
        controller._latest_tracker = {
            "right": np.eye(4),
            "left": np.eye(4),
        }
        controller._tracker_missing_since = {"right": None, "left": None}
        controller._tracker_recovery_active = {"right": False, "left": False}
        controller._tracker_recovery_samples = {"right": [], "left": []}
        controller._lookup_tracker_matrix = lambda _side: np.eye(4)
        controller._consume_tracker_episode_baseline_requests = lambda: None
        recovery_starts: list[str] = []
        controller._begin_tracker_recovery = lambda side, **_kwargs: (
            recovery_starts.append(side)
        )
        accept_results = iter((False, True))
        controller._accept_tracker_batch = lambda _matrices, **_kwargs: next(
            accept_results
        )
        packets: list[bytes] = []
        controller.tracker_socket = types.SimpleNamespace(
            sendto=lambda packet, _target: packets.append(packet)
        )
        controller.tracker_target = ("127.0.0.1", 5012)
        controller.swap_arms = False
        controller._tracker_sent_count = 0
        controller._log_tracker_delta_debug = lambda _outgoing: None
        controller._acknowledge_tracker_episode_baseline = lambda _packet: None

        controller._send_tracker()
        controller._send_tracker()

        self.assertEqual(recovery_starts, [])
        self.assertEqual(len(packets), 1)
        self.assertEqual(len(packets[0]), 192)

    def test_idle_persistent_gap_does_not_freeze_future_tracker_packets(self) -> None:
        np = __import__("numpy")
        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        controller.args = types.SimpleNamespace(send_tracker=True)
        controller.hands = ("right", "left")
        controller._calib_enabled = False
        controller._calib_phase = "done"
        controller._tracker_guard_armed = False
        controller._tracker_episode_baseline_pending = False
        controller._latest_tracker = {
            "right": np.eye(4),
            "left": np.eye(4),
        }
        controller._tracker_missing_since = {"right": None, "left": None}
        controller._tracker_recovery_active = {"right": False, "left": False}
        controller._tracker_recovery_samples = {"right": [], "left": []}
        controller.tracker_reacquire_reset_after_sec = 0.15
        controller._last_tracker_missing_log = float("inf")
        controller._lookup_tracker_matrix = lambda side: (
            np.eye(4) if side == "right" else None
        )
        recovery_starts: list[str] = []
        controller._begin_tracker_recovery = lambda side, **_kwargs: (
            recovery_starts.append(side)
        )
        controller.get_logger = lambda: types.SimpleNamespace(
            warn=lambda _message: None,
            info=lambda _message: None,
        )
        packets: list[bytes] = []
        controller.tracker_socket = types.SimpleNamespace(
            sendto=lambda packet, _target: packets.append(packet)
        )
        controller.tracker_target = ("127.0.0.1", 5012)
        controller.swap_arms = False
        controller._tracker_sent_count = 0
        controller._log_tracker_delta_debug = lambda _outgoing: None
        controller._accept_tracker_batch = lambda _matrices, **_kwargs: True

        original_monotonic = bridge.time.monotonic
        try:
            bridge.time.monotonic = lambda: 1.0
            controller._send_tracker()
            bridge.time.monotonic = lambda: 1.30
            controller._send_tracker()
            controller._lookup_tracker_matrix = lambda _side: np.eye(4)
            bridge.time.monotonic = lambda: 1.31
            controller._send_tracker()
        finally:
            bridge.time.monotonic = original_monotonic

        self.assertEqual(recovery_starts, [])
        self.assertEqual(len(packets), 1)
        self.assertEqual(len(packets[0]), 192)

    def test_persistent_jump_rebase_keeps_world_translation_axes(self) -> None:
        np = __import__("numpy")
        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        previous = np.eye(4)
        controller._latest_tracker = {"right": previous.copy()}
        controller._tracker_rebase_correction = {
            "right": bridge.identity_tracker_pose_correction()
        }
        controller._tracker_jump_reject_count = {"right": 0}
        controller._last_tracker_jump_log = {"right": float("inf")}
        controller._tracker_guard_armed = True
        controller.tracker_hold_on_jump = True
        controller.tracker_max_step_m = {"right": 0.12}
        controller.tracker_max_rotation_deg = {"right": 35.0}
        controller.tracker_reacquire_after_rejects = {"right": 2}
        controller.tracker_position_deadband_m = {"right": 0.0}
        controller.tracker_rotation_deadband_deg = {"right": 0.0}
        controller.tracker_smoothing_alpha = {"right": 1.0}
        controller.tracker_reset_after_persistent_jump = True
        controller.get_logger = lambda: types.SimpleNamespace(
            warn=lambda _message: None
        )

        discontinuity = np.eye(4)
        discontinuity[:3, :3] = np.diag([1.0, -1.0, -1.0])
        discontinuity[0, 3] = 1.0
        self.assertFalse(controller._accept_tracker_matrix("right", discontinuity))
        self.assertTrue(
            controller._accept_tracker_matrix(
                "right",
                discontinuity,
                allow_persistent_rebase=True,
            )
        )

        np.testing.assert_allclose(controller._latest_tracker["right"], previous)

        moved_up_and_left = discontinuity.copy()
        moved_up_and_left[:3, 3] += [0.0, 0.05, 0.05]
        self.assertTrue(
            controller._accept_tracker_matrix("right", moved_up_and_left)
        )
        np.testing.assert_allclose(
            controller._latest_tracker["right"][:3, 3],
            [0.0, 0.05, 0.05],
        )

    def test_transient_persistent_jump_cannot_accumulate_rebase_offset(self) -> None:
        """A long bad OpenVR burst must not become the new tracker origin."""

        np = __import__("numpy")
        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        previous = np.eye(4)
        previous[0, 3] = 0.7
        controller._latest_tracker = {"right": previous.copy()}
        original_correction = bridge.identity_tracker_pose_correction()
        controller._tracker_rebase_correction = {
            "right": original_correction
        }
        controller._tracker_jump_reject_count = {"right": 0}
        controller._last_tracker_jump_log = {"right": float("inf")}
        controller._tracker_guard_armed = True
        controller.tracker_hold_on_jump = True
        controller.tracker_max_step_m = {"right": 0.12}
        controller.tracker_max_rotation_deg = {"right": 35.0}
        controller.tracker_reacquire_after_rejects = {"right": 12}
        controller.tracker_position_deadband_m = {"right": 0.0}
        controller.tracker_rotation_deadband_deg = {"right": 0.0}
        controller.tracker_smoothing_alpha = {"right": 1.0}
        controller.tracker_reset_after_persistent_jump = True
        controller.get_logger = lambda: types.SimpleNamespace(
            warn=lambda _message: None
        )

        # The production fault lasted long enough to hit the old 12-frame
        # fallback, then snapped back to the original OpenVR coordinate.  The
        # old code rebased on frame 12 and shifted every later output by metres.
        bad_pose = previous.copy()
        bad_pose[0, 3] += 2.2
        for _ in range(20):
            self.assertFalse(
                controller._accept_tracker_matrix("right", bad_pose)
            )

        self.assertIs(
            controller._tracker_rebase_correction["right"],
            original_correction,
        )
        np.testing.assert_allclose(controller._latest_tracker["right"], previous)

        returned = previous.copy()
        returned[1, 3] += 0.01
        self.assertTrue(controller._accept_tracker_matrix("right", returned))
        np.testing.assert_allclose(controller._latest_tracker["right"], returned)

    def test_reconnect_waits_for_stable_return_without_changing_baseline(self) -> None:
        np = __import__("numpy")
        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        previous = np.eye(4)
        original_correction = bridge.identity_tracker_pose_correction()
        controller.hands = ("right",)
        controller._latest_tracker = {"right": previous.copy()}
        controller._tracker_rebase_correction = {
            "right": original_correction
        }
        controller._tracker_jump_reject_count = {"right": 0}
        controller._last_tracker_jump_log = {"right": float("-inf")}
        controller._tracker_missing_since = {"right": 0.0}
        controller._tracker_recovery_active = {"right": False}
        controller._tracker_recovery_samples = {"right": []}
        controller.tracker_reconnect_stable_sec = 0.35
        controller.tracker_reconnect_max_position_spread_m = 0.008
        controller.tracker_reconnect_max_rotation_spread_deg = 2.0
        controller.tracker_reconnect_max_position_delta_m = 0.03
        controller.tracker_reconnect_max_rotation_delta_deg = 10.0
        controller.tracker_preserve_baseline_on_reconnect = True
        controller.tracker_max_step_m = {"right": 0.12}
        controller.tracker_max_rotation_deg = {"right": 35.0}
        controller.tracker_position_deadband_m = {"right": 0.0}
        controller.tracker_rotation_deadband_deg = {"right": 0.0}
        controller.tracker_smoothing_alpha = {"right": 1.0}
        controller.tracker_hold_on_jump = True
        controller.tracker_reacquire_after_rejects = {"right": 12}
        controller.tracker_reacquire_reset_on_return = False
        controller.tracker_reset_after_persistent_jump = False
        controller.get_logger = lambda: types.SimpleNamespace(
            warn=lambda _message: None,
            info=lambda _message: None,
        )

        controller._begin_tracker_recovery(
            "right",
            reason="test dropout",
        )
        false_pose = np.eye(4)
        false_pose[0, 3] = 0.10
        self.assertFalse(
            controller._tracker_recovery_candidate_ready(
                "right",
                false_pose,
                now=0.0,
            )
        )
        # A stable but displaced SteamVR recovery pose must not replace the
        # pre-dropout baseline.
        self.assertFalse(
            controller._tracker_recovery_candidate_ready(
                "right",
                false_pose,
                now=0.36,
            )
        )
        np.testing.assert_allclose(controller._latest_tracker["right"], previous)
        self.assertIs(
            controller._tracker_rebase_correction["right"],
            original_correction,
        )

        # Once the original pose returns for the short confirmation window,
        # normal streaming may resume without any rebase.  Motion within the
        # trusted envelope must not be mistaken for instability.
        self.assertFalse(
            controller._tracker_recovery_candidate_ready(
                "right",
                previous,
                now=0.37,
            )
        )
        moving_near_previous = previous.copy()
        moving_near_previous[0, 3] = 0.02
        self.assertTrue(
            controller._tracker_recovery_candidate_ready(
                "right",
                moving_near_previous,
                now=0.73,
            )
        )
        self.assertTrue(
            controller._accept_tracker_matrix("right", moving_near_previous)
        )
        controller._finish_tracker_recovery("right")

        np.testing.assert_allclose(
            controller._latest_tracker["right"], moving_near_previous
        )
        self.assertIs(
            controller._tracker_rebase_correction["right"],
            original_correction,
        )
        self.assertFalse(controller._tracker_recovery_active["right"])

    def test_reconnect_discards_relocated_input_and_preserves_output_continuity(
        self,
    ) -> None:
        np = __import__("numpy")
        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        previous = np.eye(4)
        previous[0, 3] = 1.0
        controller.hands = ("right",)
        controller._latest_tracker = {"right": previous.copy()}
        controller._tracker_rebase_correction = {
            "right": bridge.identity_tracker_pose_correction()
        }
        controller._tracker_jump_reject_count = {"right": 0}
        controller._last_tracker_jump_log = {"right": float("-inf")}
        controller._tracker_missing_since = {"right": 0.0}
        controller._tracker_recovery_active = {"right": False}
        controller._tracker_recovery_samples = {"right": []}
        controller.tracker_reconnect_stable_sec = 0.35
        controller.tracker_reconnect_max_position_spread_m = 0.008
        controller.tracker_reconnect_max_rotation_spread_deg = 2.0
        controller.tracker_reconnect_max_position_delta_m = 0.03
        controller.tracker_reconnect_max_rotation_delta_deg = 10.0
        controller.tracker_preserve_baseline_on_reconnect = False
        controller.tracker_max_step_m = {"right": 0.12}
        controller.tracker_max_rotation_deg = {"right": 35.0}
        controller.tracker_position_deadband_m = {"right": 0.0}
        controller.tracker_rotation_deadband_deg = {"right": 0.0}
        controller.tracker_smoothing_alpha = {"right": 1.0}
        controller.tracker_hold_on_jump = True
        controller.tracker_reacquire_after_rejects = {"right": 12}
        controller.tracker_reacquire_reset_on_return = True
        controller.tracker_reset_after_persistent_jump = False
        controller.get_logger = lambda: types.SimpleNamespace(
            warn=lambda _message: None,
            info=lambda _message: None,
        )

        controller._begin_tracker_recovery(
            "right",
            reason="test relocated reconnect",
        )
        returned_pose = np.eye(4)
        self.assertFalse(
            controller._tracker_recovery_candidate_ready(
                "right",
                returned_pose,
                now=0.0,
            )
        )
        self.assertTrue(
            controller._tracker_recovery_candidate_ready(
                "right",
                returned_pose,
                now=0.36,
            )
        )
        self.assertTrue(
            controller._accept_tracker_matrix(
                "right",
                returned_pose,
                reacquired=True,
            )
        )
        controller._finish_tracker_recovery("right")

        # A one-metre OpenVR relocation is discarded: the first recovered
        # output remains exactly where output froze.
        np.testing.assert_allclose(controller._latest_tracker["right"], previous)
        self.assertFalse(controller._tracker_recovery_active["right"])

        # Subsequent real motion continues from the preserved output pose.
        moved_after_return = returned_pose.copy()
        moved_after_return[1, 3] = 0.05
        self.assertTrue(
            controller._accept_tracker_matrix("right", moved_after_return)
        )
        np.testing.assert_allclose(
            controller._latest_tracker["right"][:3, 3],
            [1.0, 0.05, 0.0],
        )

    def test_persistent_relocated_reconnect_rebases_after_stable_window(self) -> None:
        """A post-dropout SteamVR origin shift must not freeze both wrists."""

        np = __import__("numpy")
        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        controller.args = types.SimpleNamespace(send_tracker=True)
        controller.hands = ("right", "left")
        controller._calib_enabled = False
        controller._calib_phase = "done"
        controller._tracker_guard_armed = True
        controller._tracker_episode_baseline_pending = False
        previous = {side: np.eye(4) for side in controller.hands}
        controller._latest_tracker = {
            side: matrix.copy() for side, matrix in previous.items()
        }
        controller._tracker_rebase_correction = {
            side: bridge.identity_tracker_pose_correction()
            for side in controller.hands
        }
        controller._tracker_jump_reject_count = {side: 0 for side in controller.hands}
        controller._last_tracker_jump_log = {
            side: float("-inf") for side in controller.hands
        }
        controller._tracker_missing_since = {
            side: None for side in controller.hands
        }
        controller._tracker_recovery_active = {
            side: False for side in controller.hands
        }
        controller._tracker_recovery_samples = {
            side: [] for side in controller.hands
        }
        controller._tracker_recovery_rebase_ready = {
            side: False for side in controller.hands
        }
        controller.tracker_reconnect_stable_sec = 0.05
        controller.tracker_reacquire_reset_after_sec = 0.25
        controller.tracker_reconnect_max_position_spread_m = 0.008
        controller.tracker_reconnect_max_rotation_spread_deg = 2.0
        controller.tracker_reconnect_max_position_delta_m = 0.12
        controller.tracker_reconnect_max_rotation_delta_deg = 35.0
        controller.tracker_preserve_baseline_on_reconnect = True
        controller.tracker_reacquire_after_rejects = {
            side: 2 for side in controller.hands
        }
        controller.tracker_reacquire_reset_on_return = False
        controller.tracker_reset_after_persistent_jump = True
        controller.tracker_hold_on_jump = True
        controller.tracker_max_step_m = {side: 0.12 for side in controller.hands}
        controller.tracker_max_rotation_deg = {
            side: 35.0 for side in controller.hands
        }
        controller.tracker_position_deadband_m = {
            side: 0.0 for side in controller.hands
        }
        controller.tracker_rotation_deadband_deg = {
            side: 0.0 for side in controller.hands
        }
        controller.tracker_smoothing_alpha = {
            side: 1.0 for side in controller.hands
        }
        controller._last_tracker_missing_log = float("inf")
        controller.get_logger = lambda: types.SimpleNamespace(
            warn=lambda _message: None,
            info=lambda _message: None,
        )
        controller._consume_tracker_episode_baseline_requests = lambda: None
        controller._log_tracker_delta_debug = lambda _outgoing: None
        controller._acknowledge_tracker_episode_baseline = lambda _packet: None
        controller.tracker_target = ("127.0.0.1", 5012)
        controller.swap_arms = False
        controller._tracker_sent_count = 0
        packets: list[bytes] = []
        controller.tracker_socket = types.SimpleNamespace(
            sendto=lambda packet, _target: packets.append(packet)
        )

        displaced = {side: np.eye(4) for side in controller.hands}
        for matrix in displaced.values():
            matrix[0, 3] = 1.0
        live = {side: None for side in controller.hands}
        controller._lookup_tracker_matrix = lambda side: live[side]

        original_monotonic = bridge.time.monotonic
        try:
            # First let the paired TF stream become a persistent dropout.
            bridge.time.monotonic = lambda: 0.0
            controller._send_tracker()
            bridge.time.monotonic = lambda: 0.30
            controller._send_tracker()

            # The relocated pair is held until both the reject count and the
            # stable displaced window are satisfied.
            live.update(displaced)
            for now in (0.31, 0.40, 0.55):
                bridge.time.monotonic = lambda now=now: now
                controller._send_tracker()
            self.assertEqual(packets, [])
            bridge.time.monotonic = lambda: 0.56
            controller._send_tracker()
            self.assertEqual(len(packets), 1)
            self.assertEqual(len(packets[0]), 192)
            for side in controller.hands:
                np.testing.assert_allclose(
                    controller._latest_tracker[side], previous[side]
                )
                self.assertFalse(controller._tracker_recovery_active[side])

            # Motion after the rebase is relative to the frozen output, not
            # the relocated SteamVR origin.
            for matrix in live.values():
                matrix[1, 3] = 0.05
            bridge.time.monotonic = lambda: 0.57
            controller._send_tracker()
        finally:
            bridge.time.monotonic = original_monotonic

        for side in controller.hands:
            np.testing.assert_allclose(
                controller._latest_tracker[side][:3, 3],
                [0.0, 0.05, 0.0],
            )

    def test_bimanual_acceptance_does_not_advance_only_one_trusted_side(self) -> None:
        np = __import__("numpy")
        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        controller.hands = ("right", "left")
        original_right = np.eye(4)
        original_left = np.eye(4)
        controller._latest_tracker = {
            "right": original_right.copy(),
            "left": original_left.copy(),
        }
        controller._tracker_rebase_correction = {
            "right": bridge.identity_tracker_pose_correction(),
            "left": bridge.identity_tracker_pose_correction(),
        }

        moved_right = np.eye(4)
        moved_right[0, 3] = 0.01
        controller._accept_tracker_matrix = lambda side, matrix, **_kwargs: (
            controller._latest_tracker.__setitem__(side, matrix.copy())
            or side == "right"
        )

        accepted = controller._accept_tracker_batch(
            {
                "right": moved_right,
                "left": np.eye(4),
            }
        )

        self.assertFalse(accepted)
        np.testing.assert_allclose(
            controller._latest_tracker["right"],
            original_right,
        )
        np.testing.assert_allclose(
            controller._latest_tracker["left"],
            original_left,
        )

    def test_live_wrist_rotation_does_not_change_absolute_displacement(self) -> None:
        np = __import__("numpy")
        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        controller.tracker_relative = True
        controller._calib_enabled = False
        controller._calib_solved = {"right": True}
        controller._tracker_anchor_raw = {"right": np.eye(4)}
        controller.sim_anchor = {"right": np.eye(4)}
        controller.tracker_sim_correction = {"right": np.eye(3)}
        controller.tracker_raw_to_sim = {"right": np.eye(3)}
        controller.tracker_motion_correction = {
            "right": {
                "invert_world_z": False,
                "invert_palm_normal_rotation": False,
            }
        }
        controller._last_tracker_map_log = {"right": float("inf")}

        translated = np.eye(4)
        translated[:3, 3] = [0.12, -0.08, 0.20]
        translated_and_rotated = translated.copy()
        translated_and_rotated[:3, :3] = np.diag([1.0, -1.0, -1.0])

        plain = controller._map_tracker_matrix("right", translated)
        rotated = controller._map_tracker_matrix("right", translated_and_rotated)

        np.testing.assert_allclose(plain[:3, 3], [0.12, -0.08, 0.20])
        np.testing.assert_allclose(rotated[:3, 3], plain[:3, 3])

    def test_world_z_motion_correction_flips_only_vertical_displacement(self) -> None:
        np = __import__("numpy")
        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        controller.sim_anchor = {"right": np.eye(4)}
        controller.tracker_motion_correction = {
            "right": {
                "invert_world_z": True,
                "invert_palm_normal_rotation": False,
            }
        }
        controller._last_motion_correction_log = {"right": float("inf")}

        pose = np.eye(4)
        pose[:3, 3] = [0.12, -0.08, 0.20]
        corrected = controller._apply_tracker_motion_correction("right", pose)

        np.testing.assert_allclose(corrected[:3, 3], [0.12, -0.08, -0.20])
        np.testing.assert_allclose(corrected[:3, :3], pose[:3, :3])

    def test_raw_udp_backpressure_is_counted_and_exposes_sequence_gap(self) -> None:
        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        topic = "/manus_glove_0"
        controller._raw_stats_lock = threading.Lock()
        controller._raw_source_sequences = {topic: 0}
        controller._raw_sent_counts = {topic: 0}
        controller._raw_error_counts = {topic: 0}
        controller._raw_last_received = {topic: None}
        controller._raw_max_work_sec = 0.0
        controller._last_raw_recording_error_log = 0.0
        controller.raw_recording_target = ("127.0.0.1", 5018)
        errors = []
        controller.get_logger = lambda: types.SimpleNamespace(error=errors.append)

        def fail_send(*_args):
            raise BlockingIOError("send buffer full")

        controller.raw_recording_socket = types.SimpleNamespace(sendto=fail_send)
        message = types.SimpleNamespace(
            glove_id=1, side="Right", raw_node_count=0, raw_nodes=[],
            ergonomics_count=0, ergonomics=[], raw_sensor_count=0, raw_sensor=[],
            raw_sensor_orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        controller._raw_glove_cb(message, topic)
        packets = []
        controller.raw_recording_socket.sendto = lambda packet, _target: packets.append(packet)
        controller._raw_glove_cb(message, topic)

        from handson.data.manus_raw_recording import decode_manus_raw_event

        self.assertEqual(controller._raw_source_sequences[topic], 2)
        self.assertEqual(controller._raw_sent_counts[topic], 1)
        self.assertEqual(controller._raw_error_counts[topic], 1)
        self.assertEqual(decode_manus_raw_event(packets[0])["source_sequence"], 1)
        self.assertIn("BlockingIOError", errors[0])

    def test_stale_hand_input_suppresses_cached_hand_packets(self) -> None:
        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        controller.args = types.SimpleNamespace(
            send_tracker=False,
            hand_topic="/hand_input",
        )
        controller.hands = ("right", "left")
        controller._lock = threading.Lock()
        controller._latest_hands = {
            "right": __import__("numpy").ones((21, 3)),
            "left": __import__("numpy").ones((21, 3)),
        }
        controller._last_hand_msg_time = time.monotonic() - 2.0
        controller.hand_input_max_age_sec = 0.5
        controller._last_hand_stale_log = float("inf")
        controller._msg_count = 1
        controller._sent_counts = {"right": 0, "left": 0}
        controller._tracker_sent_count = 0
        controller._last_log_time = float("inf")
        controller._timer_count = 0
        controller._last_timer_start = None
        controller._timer_max_gap_sec = 0.0
        controller._timer_max_work_sec = 0.0
        controller._send_hand = lambda *_args: self.fail("stale hand packet was sent")
        controller._send_tracker = lambda: None
        controller.get_logger = lambda: types.SimpleNamespace(
            warn=lambda _message: None,
            info=lambda _message: None,
        )

        controller._timer_cb()

    def test_fresh_hand_input_still_sends_requested_hands(self) -> None:
        controller = object.__new__(bridge.ManusWujiHandsOnBridge)
        controller.args = types.SimpleNamespace(
            send_tracker=False,
            hand_topic="/hand_input",
        )
        controller.hands = ("right", "left")
        controller._lock = threading.Lock()
        controller._latest_hands = {
            "right": __import__("numpy").ones((21, 3)),
            "left": __import__("numpy").ones((21, 3)),
        }
        controller._last_hand_msg_time = time.monotonic()
        controller.hand_input_max_age_sec = 0.5
        controller._last_hand_stale_log = 0.0
        controller._msg_count = 1
        controller._sent_counts = {"right": 0, "left": 0}
        controller._tracker_sent_count = 0
        controller._last_log_time = float("inf")
        controller._timer_count = 0
        controller._last_timer_start = None
        controller._timer_max_gap_sec = 0.0
        controller._timer_max_work_sec = 0.0
        sent: list[str] = []
        controller._send_hand = lambda side, _kpts: sent.append(side)
        controller._send_tracker = lambda: None
        controller.get_logger = lambda: types.SimpleNamespace(
            warn=lambda _message: None,
            info=lambda _message: None,
        )

        controller._timer_cb()

        self.assertEqual(sent, ["right", "left"])


if __name__ == "__main__":
    unittest.main()
