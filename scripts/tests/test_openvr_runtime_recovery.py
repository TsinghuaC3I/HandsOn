from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OPENVR_PACKAGE = (
    ROOT
    / "teleoperation"
    / "manus"
    / "vendor"
    / "wuji-hand-teleop"
    / "src"
    / "input_devices"
    / "openvr_input"
    / "openvr_input"
)
WRAPPER_PATH = OPENVR_PACKAGE / "openvr_tracker_wrapper.py"
NODE_PATH = OPENVR_PACKAGE / "openvr_input_node.py"


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakePoseMatrix:
    def __init__(self, x: float) -> None:
        self.m = [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]


class FakeTrackedPose:
    def __init__(self, x: float = 0.0, valid: bool = False) -> None:
        self.bPoseIsValid = valid
        self.mDeviceToAbsoluteTracking = FakePoseMatrix(x)


class FakeVRSystem:
    def __init__(self, max_devices: int = 8) -> None:
        self.max_devices = max_devices
        self.devices: dict[int, tuple[str, float, bool]] = {}
        self.references: dict[int, tuple[str, float, bool]] = {}
        self.fail_scan = False
        self.fail_pose = False

    def set_devices(self, devices: dict[int, tuple[str, float, bool]]) -> None:
        self.devices = dict(devices)

    def set_references(self, references: dict[int, tuple[str, float, bool]]) -> None:
        self.references = dict(references)

    def getTrackedDeviceClass(self, index: int) -> int:
        if self.fail_scan:
            raise RuntimeError("scan IPC failed")
        if index in self.devices:
            return 3
        if index in self.references:
            return 4
        return 0

    def getStringTrackedDeviceProperty(self, index: int, _property: int) -> str:
        return (self.devices | self.references)[index][0]

    def getDeviceToAbsoluteTrackingPose(
        self,
        _universe: int,
        _prediction: int,
        max_devices: int,
    ) -> list[FakeTrackedPose]:
        if self.fail_pose:
            raise RuntimeError("pose IPC failed")
        poses = [FakeTrackedPose() for _ in range(max_devices)]
        for index, (_serial, x, valid) in (self.devices | self.references).items():
            poses[index] = FakeTrackedPose(x=x, valid=valid)
        return poses


class FakeOpenVR(types.ModuleType):
    VRApplication_Other = 3
    TrackedDeviceClass_GenericTracker = 3
    TrackedDeviceClass_TrackingReference = 4
    TrackingUniverseStanding = 1
    Prop_SerialNumber_String = 1000

    def __init__(self, system: FakeVRSystem) -> None:
        super().__init__("openvr")
        self.system = system
        self.k_unMaxTrackedDeviceCount = system.max_devices
        self.init_calls = 0
        self.shutdown_calls = 0

    def init(self, _application_type: int) -> None:
        self.init_calls += 1

    def VRSystem(self) -> FakeVRSystem:  # noqa: N802 - mirrors OpenVR API
        return self.system

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def load_wrapper(fake_openvr: FakeOpenVR):
    module_name = f"test_openvr_wrapper_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, WRAPPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"openvr": fake_openvr, module_name: module}):
        spec.loader.exec_module(module)
    return module


class OpenVRTrackerWrapperRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.system = FakeVRSystem()
        self.fake_openvr = FakeOpenVR(self.system)
        self.module = load_wrapper(self.fake_openvr)

    def make_wrapper(
        self,
        *,
        base_station_serials=None,
        base_station_recovery_stable_s=0.0,
    ):
        return self.module.OpenVRTrackerWrapper(
            {"right_wrist": "RIGHT", "left_wrist": "LEFT"},
            base_station_serials=base_station_serials,
            base_station_recovery_stable_s=base_station_recovery_stable_s,
            rescan_interval_s=0.5,
            reconnect_interval_s=2.0,
            clock=self.clock,
        )

    def test_extra_base_station_only_freezes_when_fixed_pair_is_lost(self) -> None:
        self.system.set_devices(
            {1: ("RIGHT", 1.0, True), 2: ("LEFT", 2.0, True)}
        )
        self.system.set_references(
            {
                3: ("OWN-A", 0.0, True),
                4: ("OWN-B", 0.0, True),
            }
        )
        wrapper = self.make_wrapper(
            base_station_serials=("OWN-A", "OWN-B"),
            base_station_recovery_stable_s=0.25,
        )
        self.assertTrue(all(value is None for value in wrapper.get_poses().values()))
        self.clock.advance(0.25)
        initial = wrapper.get_poses()
        self.assertIsNotNone(initial["right_wrist"])

        self.system.set_references(
            {
                3: ("OWN-A", 0.0, True),
                4: ("OWN-B", 0.0, True),
                5: ("FOREIGN", 0.0, True),
            }
        )
        self.clock.advance(0.5)

        continued = wrapper.get_poses()

        self.assertIsNotNone(continued["right_wrist"])
        self.assertIsNotNone(continued["left_wrist"])

        self.system.set_references(
            {
                3: ("OWN-A", 0.0, True),
                4: ("OWN-B", 0.0, False),
                5: ("FOREIGN", 0.0, True),
            }
        )
        frozen = wrapper.get_poses()
        self.assertTrue(all(value is None for value in frozen.values()))

        self.system.set_references(
            {
                3: ("OWN-A", 0.0, True),
                4: ("OWN-B", 0.0, True),
                5: ("FOREIGN", 0.0, True),
            }
        )
        recovering = wrapper.get_poses()
        self.assertTrue(all(value is None for value in recovering.values()))
        self.clock.advance(0.25)
        recovered = wrapper.get_poses()
        self.assertIsNotNone(recovered["right_wrist"])
        self.assertIsNotNone(recovered["left_wrist"])

    def test_missing_fixed_base_station_freezes_until_stable_recovery(self) -> None:
        self.system.set_devices(
            {1: ("RIGHT", 1.0, True), 2: ("LEFT", 2.0, True)}
        )
        self.system.set_references(
            {3: ("OWN-A", 0.0, True), 4: ("OWN-B", 0.0, True)}
        )
        wrapper = self.make_wrapper(
            base_station_serials=("OWN-A", "OWN-B"),
            base_station_recovery_stable_s=0.25,
        )

        self.assertTrue(all(value is None for value in wrapper.get_poses().values()))
        self.clock.advance(0.25)
        initial_ready = wrapper.get_poses()
        self.assertIsNotNone(initial_ready["right_wrist"])

        self.system.set_references(
            {3: ("OWN-A", 0.0, True), 4: ("OWN-B", 0.0, False)}
        )
        frozen = wrapper.get_poses()
        self.assertTrue(all(value is None for value in frozen.values()))

        self.system.set_references(
            {3: ("OWN-A", 0.0, True), 4: ("OWN-B", 0.0, True)}
        )
        self.clock.advance(0.5)
        waiting = wrapper.get_poses()
        self.assertTrue(all(value is None for value in waiting.values()))
        self.clock.advance(0.24)
        self.assertTrue(all(value is None for value in wrapper.get_poses().values()))
        self.clock.advance(0.01)
        recovered = wrapper.get_poses()
        self.assertIsNotNone(recovered["right_wrist"])
        self.assertIsNotNone(recovered["left_wrist"])

    def test_configured_roles_are_returned_when_one_tracker_is_missing(self) -> None:
        self.system.set_devices({1: ("RIGHT", 1.25, True)})
        wrapper = self.make_wrapper()

        poses = wrapper.get_poses()

        self.assertEqual(set(poses), {"right_wrist", "left_wrist"})
        self.assertAlmostEqual(float(poses["right_wrist"][0, 3]), 1.25)
        self.assertIsNone(poses["left_wrist"])

    def test_trackers_that_power_on_after_client_start_are_hot_bound(self) -> None:
        wrapper = self.make_wrapper()
        initial = wrapper.get_poses()
        self.assertTrue(all(value is None for value in initial.values()))

        self.system.set_devices(
            {
                4: ("RIGHT", 4.0, True),
                6: ("LEFT", 6.0, True),
            }
        )
        self.clock.advance(0.5)
        recovered = wrapper.get_poses()

        self.assertAlmostEqual(float(recovered["right_wrist"][0, 3]), 4.0)
        self.assertAlmostEqual(float(recovered["left_wrist"][0, 3]), 6.0)
        self.assertEqual(wrapper._detected_trackers, {4: "right_wrist", 6: "left_wrist"})

    def test_periodic_rescan_recovers_missing_serial_and_changed_index(self) -> None:
        self.system.set_devices(
            {
                1: ("RIGHT", 1.0, True),
                2: ("LEFT", 2.0, True),
            }
        )
        wrapper = self.make_wrapper()
        initial = wrapper.get_poses()
        self.assertAlmostEqual(float(initial["left_wrist"][0, 3]), 2.0)

        self.system.set_devices({5: ("LEFT", 5.0, True)})
        self.clock.advance(0.5)
        missing = wrapper.get_poses()
        self.assertIsNone(missing["right_wrist"])
        self.assertAlmostEqual(float(missing["left_wrist"][0, 3]), 5.0)

        self.system.set_devices(
            {
                4: ("RIGHT", 4.0, True),
                5: ("LEFT", 5.0, True),
            }
        )
        self.clock.advance(0.5)
        recovered = wrapper.get_poses()
        self.assertAlmostEqual(float(recovered["right_wrist"][0, 3]), 4.0)
        self.assertEqual(wrapper._detected_trackers, {4: "right_wrist", 5: "left_wrist"})

    def test_runtime_api_failure_reconnects_only_after_backoff(self) -> None:
        self.system.set_devices(
            {
                1: ("RIGHT", 1.0, True),
                2: ("LEFT", 2.0, True),
            }
        )
        wrapper = self.make_wrapper()
        self.assertEqual(self.fake_openvr.init_calls, 1)

        self.system.fail_pose = True
        failed = wrapper.get_poses()
        self.assertTrue(all(value is None for value in failed.values()))
        self.assertEqual(self.fake_openvr.shutdown_calls, 1)

        wrapper.get_poses()
        self.clock.advance(1.99)
        wrapper.get_poses()
        self.assertEqual(self.fake_openvr.init_calls, 1)

        self.system.fail_pose = False
        self.clock.advance(0.01)
        recovered = wrapper.get_poses()
        self.assertEqual(self.fake_openvr.init_calls, 2)
        self.assertIsNotNone(recovered["right_wrist"])
        self.assertIsNotNone(recovered["left_wrist"])

    def test_initial_api_failure_recovers_without_restarting_process(self) -> None:
        self.system.set_devices(
            {
                1: ("RIGHT", 1.0, True),
                2: ("LEFT", 2.0, True),
            }
        )
        self.system.fail_scan = True
        wrapper = self.make_wrapper()
        self.assertFalse(wrapper.is_connected())
        self.assertEqual(self.fake_openvr.init_calls, 1)

        unavailable = wrapper.get_poses()
        self.assertTrue(all(value is None for value in unavailable.values()))
        self.assertEqual(self.fake_openvr.init_calls, 1)

        self.system.fail_scan = False
        self.clock.advance(2.0)
        recovered = wrapper.get_poses()
        self.assertEqual(self.fake_openvr.init_calls, 2)
        self.assertTrue(wrapper.is_connected())
        self.assertIsNotNone(recovered["right_wrist"])
        self.assertIsNotNone(recovered["left_wrist"])

    def test_shutdown_disables_future_reconnects(self) -> None:
        self.system.set_devices({1: ("RIGHT", 1.0, True)})
        wrapper = self.make_wrapper()
        wrapper.shutdown()
        self.clock.advance(10.0)

        poses = wrapper.get_poses()

        self.assertTrue(all(value is None for value in poses.values()))
        self.assertEqual(self.fake_openvr.init_calls, 1)


class FakeLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, _message: str) -> None:
        pass


class FakeNodeClock:
    class _Now:
        @staticmethod
        def to_msg() -> object:
            return object()

    def now(self) -> "FakeNodeClock._Now":
        return self._Now()


class SequencedTrackerWrapper:
    def __init__(self) -> None:
        self.results = [
            {"right_wrist": None, "left_wrist": None},
            {
                "right_wrist": np.eye(4, dtype=np.float64),
                "left_wrist": np.eye(4, dtype=np.float64),
            },
        ]

    def get_poses(self):
        return self.results.pop(0)


def load_node_module():
    package_name = f"test_openvr_node_package_{uuid.uuid4().hex}"
    module_name = f"{package_name}.openvr_input_node"
    package = types.ModuleType(package_name)
    package.__path__ = [str(OPENVR_PACKAGE)]

    class ExternalShutdownException(Exception):
        pass

    class BaseNode:
        pass

    rclpy = types.ModuleType("rclpy")
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = BaseNode
    rclpy_executors = types.ModuleType("rclpy.executors")
    rclpy_executors.ExternalShutdownException = ExternalShutdownException
    rclpy_utilities = types.ModuleType("rclpy.utilities")
    rclpy_utilities.remove_ros_args = lambda argv: list(argv)

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.TransformStamped = type("TransformStamped", (), {})
    tf2_ros = types.ModuleType("tf2_ros")
    tf2_ros.TransformBroadcaster = type("TransformBroadcaster", (), {})

    ament = types.ModuleType("ament_index_python")
    ament_packages = types.ModuleType("ament_index_python.packages")
    ament_packages.get_package_share_directory = lambda _name: str(OPENVR_PACKAGE.parent)

    wrapper_module = types.ModuleType(f"{package_name}.openvr_tracker_wrapper")
    wrapper_module.OpenVRTrackerWrapper = type("OpenVRTrackerWrapper", (), {})

    injected = {
        package_name: package,
        module_name: None,
        f"{package_name}.openvr_tracker_wrapper": wrapper_module,
        "rclpy": rclpy,
        "rclpy.node": rclpy_node,
        "rclpy.executors": rclpy_executors,
        "rclpy.utilities": rclpy_utilities,
        "geometry_msgs": geometry_msgs,
        "geometry_msgs.msg": geometry_msgs_msg,
        "tf2_ros": tf2_ros,
        "ament_index_python": ament,
        "ament_index_python.packages": ament_packages,
    }
    spec = importlib.util.spec_from_file_location(module_name, NODE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    injected[module_name] = module
    with patch.dict(sys.modules, injected):
        spec.loader.exec_module(module)
    return module, rclpy, ExternalShutdownException


class OpenVRInputNodeLifecycleTest(unittest.TestCase):
    def test_stream_transition_logs_missing_then_ready_once(self) -> None:
        module, _rclpy, _shutdown_error = load_node_module()
        node = object.__new__(module.OpenVRInputNode)
        node.config = module.OpenVRInputConfig(
            tracker_serials={"right_wrist": "RIGHT", "left_wrist": "LEFT"}
        )
        node.tracker_wrapper = SequencedTrackerWrapper()
        node.tf_broadcaster = object()
        node._last_missing_trackers = None
        logger = FakeLogger()
        node.get_logger = lambda: logger
        node.get_clock = lambda: FakeNodeClock()
        node._broadcast_tf = lambda *_args, **_kwargs: None

        node._publish_callback()
        node._publish_callback()

        self.assertEqual(len(logger.warnings), 1)
        self.assertIn("right_wrist, left_wrist", logger.warnings[0])
        self.assertEqual(len(logger.infos), 3)
        self.assertEqual(logger.infos[0], "[OpenVR Input] Tracker stream state: ")
        self.assertIn("right_wrist, left_wrist", logger.infos[1])
        self.assertIn("Tracker stream ready", logger.infos[2])

    def test_external_shutdown_does_not_call_rclpy_shutdown_twice(self) -> None:
        module, rclpy, shutdown_error = load_node_module()
        calls = {"init": 0, "shutdown": 0, "node_shutdown": 0, "destroy": 0}
        context_ok = {"value": True}

        def init(*_args, **_kwargs) -> None:
            calls["init"] += 1

        def spin(_node) -> None:
            context_ok["value"] = False
            raise shutdown_error()

        def shutdown() -> None:
            calls["shutdown"] += 1

        rclpy.init = init
        rclpy.spin = spin
        rclpy.ok = lambda: context_ok["value"]
        rclpy.shutdown = shutdown

        class RuntimeNode:
            def shutdown(self) -> None:
                calls["node_shutdown"] += 1

            def destroy_node(self) -> None:
                calls["destroy"] += 1

        module.OpenVRInputNode = lambda _config: RuntimeNode()

        with tempfile.TemporaryDirectory() as temporary_directory:
            config = Path(temporary_directory) / "openvr_input.yaml"
            config.write_text(
                "tracker_serials:\n  right_wrist: RIGHT\n  left_wrist: LEFT\n",
                encoding="utf-8",
            )
            module.main(["-c", str(config)])

        self.assertEqual(calls["init"], 1)
        self.assertEqual(calls["node_shutdown"], 1)
        self.assertEqual(calls["destroy"], 1)
        self.assertEqual(calls["shutdown"], 0)


if __name__ == "__main__":
    unittest.main()
