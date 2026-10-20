from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT_NODE_SCRIPT = (
    ROOT
    / "teleoperation"
    / "manus"
    / "vendor"
    / "wuji-hand-teleop"
    / "src"
    / "input_devices"
    / "manus_input"
    / "manus_input_py"
    / "manus_input_py"
    / "manus_input_node.py"
)


def load_input_node_module():
    ament_index_python = types.ModuleType("ament_index_python")
    ament_index_python.packages = types.ModuleType("ament_index_python.packages")
    ament_index_python.packages.get_package_share_directory = lambda _name: ""

    rclpy = types.ModuleType("rclpy")
    rclpy.node = types.ModuleType("rclpy.node")
    rclpy.qos = types.ModuleType("rclpy.qos")
    rclpy.utilities = types.ModuleType("rclpy.utilities")
    rclpy.node.Node = type("Node", (), {})
    rclpy.qos.QoSProfile = type("QoSProfile", (), {})
    rclpy.qos.QoSReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT=1)
    rclpy.qos.QoSHistoryPolicy = types.SimpleNamespace(KEEP_LAST=1)
    rclpy.utilities.remove_ros_args = lambda args: args

    class Float32MultiArray:
        def __init__(self) -> None:
            self.data = []

    std_msgs = types.ModuleType("std_msgs")
    std_msgs.msg = types.ModuleType("std_msgs.msg")
    std_msgs.msg.Float32MultiArray = Float32MultiArray

    manus_ros2_msgs = types.ModuleType("manus_ros2_msgs")
    manus_ros2_msgs.msg = types.ModuleType("manus_ros2_msgs.msg")
    manus_ros2_msgs.msg.ManusGlove = type("ManusGlove", (), {})

    stubs = {
        "ament_index_python": ament_index_python,
        "ament_index_python.packages": ament_index_python.packages,
        "rclpy": rclpy,
        "rclpy.node": rclpy.node,
        "rclpy.qos": rclpy.qos,
        "rclpy.utilities": rclpy.utilities,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs.msg,
        "manus_ros2_msgs": manus_ros2_msgs,
        "manus_ros2_msgs.msg": manus_ros2_msgs.msg,
    }
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in stubs}
    try:
        sys.modules.update(stubs)
        spec = importlib.util.spec_from_file_location("manus_input_freshness_node", INPUT_NODE_SCRIPT)
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


input_node = load_input_node_module()


class RecordingPublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


class ManusInputFreshnessTest(unittest.TestCase):
    def make_node(self, *, include_right: bool, include_left: bool):
        node = object.__new__(input_node.ManusInputNode)
        node.config = types.SimpleNamespace(
            include_right_hand=include_right,
            include_left_hand=include_left,
            allow_missing_hand=False,
            require_all_requested_updates=True,
            max_hand_update_age_sec=0.10,
            swap_left_right=False,
        )
        node._right_fingers = np.ones((21, 3), dtype=np.float32)
        node._left_fingers = np.full((21, 3), 2.0, dtype=np.float32)
        node._right_updated = False
        node._left_updated = False
        node._right_updated_at = None
        node._left_updated_at = None
        node.hand_publisher = RecordingPublisher()
        return node

    def test_config_defaults_to_independent_hand_freshness(self) -> None:
        config = input_node.ManusInputConfig()

        self.assertTrue(config.allow_missing_hand)
        self.assertFalse(config.require_all_requested_updates)
        self.assertEqual(config.max_hand_update_age_sec, 0.0)

    def test_dual_hand_frame_waits_for_each_glove_to_update(self) -> None:
        node = self.make_node(include_right=True, include_left=True)
        node._right_updated = True
        node._right_updated_at = 1.00

        with mock.patch.object(input_node.time, "monotonic", return_value=1.01):
            node._publish_latest_frame()

        self.assertEqual(node.hand_publisher.messages, [])
        self.assertTrue(node._right_updated)

        node._left_updated = True
        node._left_updated_at = 1.02
        with mock.patch.object(input_node.time, "monotonic", return_value=1.02):
            node._publish_latest_frame()

        self.assertEqual(len(node.hand_publisher.messages), 1)
        self.assertEqual(len(node.hand_publisher.messages[0].data), 126)
        self.assertFalse(node._right_updated)
        self.assertFalse(node._left_updated)

    def test_single_requested_hand_publishes_without_other_glove(self) -> None:
        node = self.make_node(include_right=True, include_left=False)
        node._left_fingers = None
        node._right_updated = True
        node._right_updated_at = 1.00

        with mock.patch.object(input_node.time, "monotonic", return_value=1.01):
            node._publish_latest_frame()

        self.assertEqual(len(node.hand_publisher.messages), 1)
        self.assertEqual(len(node.hand_publisher.messages[0].data), 63)
        self.assertFalse(node._right_updated)

    def test_independent_dual_hand_mode_publishes_live_side_with_zero_missing_side(self) -> None:
        node = self.make_node(include_right=True, include_left=True)
        node.config.allow_missing_hand = True
        node.config.require_all_requested_updates = False
        node.config.max_hand_update_age_sec = 0.0
        node._left_updated = True
        node._left_updated_at = 1.00
        node._left_fingers = np.full((21, 3), 3.0, dtype=np.float32)
        node._right_fingers = None

        with mock.patch.object(input_node.time, "monotonic", return_value=1.01):
            node._publish_latest_frame()

        self.assertEqual(len(node.hand_publisher.messages), 1)
        payload = np.asarray(node.hand_publisher.messages[0].data, dtype=np.float32)
        self.assertEqual(payload.size, 126)
        np.testing.assert_allclose(payload[:63], 0.0)
        np.testing.assert_allclose(payload[63:], 3.0)

    def test_independent_dual_hand_mode_does_not_republish_disconnected_cached_side(self) -> None:
        node = self.make_node(include_right=True, include_left=True)
        node.config.allow_missing_hand = True
        node.config.require_all_requested_updates = False
        node.config.max_hand_update_age_sec = 0.0
        node._right_updated = False
        node._left_updated = True
        node._left_updated_at = 1.00

        with mock.patch.object(input_node.time, "monotonic", return_value=1.01):
            node._publish_latest_frame()

        payload = np.asarray(node.hand_publisher.messages[0].data, dtype=np.float32)
        np.testing.assert_allclose(payload[:63], 0.0)
        np.testing.assert_allclose(payload[63:], 2.0)

    def test_unpaired_update_expires_before_other_glove_recovers(self) -> None:
        for stale_side, recovering_side in (("left", "right"), ("right", "left")):
            with self.subTest(stale_side=stale_side):
                node = self.make_node(include_right=True, include_left=True)
                setattr(node, f"_{stale_side}_updated", True)
                setattr(node, f"_{stale_side}_updated_at", 1.00)

                with mock.patch.object(input_node.time, "monotonic", return_value=1.05):
                    node._publish_latest_frame()

                self.assertEqual(node.hand_publisher.messages, [])
                self.assertTrue(getattr(node, f"_{stale_side}_updated"))

                setattr(node, f"_{recovering_side}_updated", True)
                setattr(node, f"_{recovering_side}_updated_at", 2.00)
                with mock.patch.object(input_node.time, "monotonic", return_value=2.00):
                    node._publish_latest_frame()

                self.assertEqual(node.hand_publisher.messages, [])
                self.assertFalse(getattr(node, f"_{stale_side}_updated"))
                self.assertTrue(getattr(node, f"_{recovering_side}_updated"))

                setattr(node, f"_{stale_side}_updated", True)
                setattr(node, f"_{stale_side}_updated_at", 2.05)
                with mock.patch.object(input_node.time, "monotonic", return_value=2.05):
                    node._publish_latest_frame()

                self.assertEqual(len(node.hand_publisher.messages), 1)


if __name__ == "__main__":
    unittest.main()
