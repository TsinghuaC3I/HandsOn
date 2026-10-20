from __future__ import annotations

import io
import importlib.util
import stat
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "configure_openvr_trackers.py"
SPEC = importlib.util.spec_from_file_location("configure_openvr_trackers", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
tracker_binding = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tracker_binding
SPEC.loader.exec_module(tracker_binding)


class OpenVRTrackerBindingTest(unittest.TestCase):
    def test_lowercase_environment_names_take_precedence(self) -> None:
        requests = tracker_binding.environment_requests(
            {
                "tracker_left": "7",
                "TRACKER_LEFT": "8",
                "tracker_right": "LHR-RIGHT",
                "TRACKER_RIGHT": "LHR-OTHER",
            }
        )
        self.assertEqual(
            requests,
            {"left_wrist": "7", "right_wrist": "LHR-RIGHT"},
        )

    def test_auto_binding_uses_openvr_x_position(self) -> None:
        devices = [
            tracker_binding.TrackerDevice(8, "LHR-LEFT", -0.42),
            tracker_binding.TrackerDevice(3, "LHR-RIGHT", 0.37),
        ]
        bindings = tracker_binding.resolve_bindings({}, {}, devices)
        self.assertEqual(bindings["right_wrist"], "LHR-RIGHT")
        self.assertEqual(bindings["left_wrist"], "LHR-LEFT")

    def test_auto_binding_refuses_to_guess_without_two_valid_poses(self) -> None:
        devices = [
            tracker_binding.TrackerDevice(9, "LHR-SECOND", None),
            tracker_binding.TrackerDevice(4, "LHR-FIRST", float("nan")),
        ]
        with self.assertRaisesRegex(
            tracker_binding.TrackerBindingError,
            "two valid OpenVR tracker poses",
        ):
            tracker_binding.resolve_bindings({}, {}, devices)

    def test_auto_binding_refuses_more_than_two_trackers(self) -> None:
        devices = [
            tracker_binding.TrackerDevice(2, "LHR-RIGHT", 0.3),
            tracker_binding.TrackerDevice(3, "LHR-LEFT", -0.3),
            tracker_binding.TrackerDevice(4, "LHR-EXTRA", 0.0),
        ]
        with self.assertRaisesRegex(
            tracker_binding.TrackerBindingError,
            "exactly two unassigned",
        ):
            tracker_binding.resolve_bindings({}, {}, devices)

    def test_one_auto_role_binds_only_unique_unassigned_tracker(self) -> None:
        configured = {"right_wrist": "LHR-RIGHT", "left_wrist": "auto"}
        devices = [
            tracker_binding.TrackerDevice(2, "LHR-RIGHT", None),
            tracker_binding.TrackerDevice(3, "LHR-LEFT", None),
        ]
        bindings = tracker_binding.resolve_bindings(configured, {}, devices)
        self.assertEqual(bindings["left_wrist"], "LHR-LEFT")

        devices.append(tracker_binding.TrackerDevice(4, "LHR-EXTRA", None))
        with self.assertRaisesRegex(
            tracker_binding.TrackerBindingError,
            "exactly one unassigned",
        ):
            tracker_binding.resolve_bindings(configured, {}, devices)

    def test_enumeration_retries_until_two_poses_are_valid(self) -> None:
        def pose(valid: bool, x: float = 0.0):
            matrix = [[0.0] * 4 for _ in range(3)]
            matrix[0][3] = x
            return types.SimpleNamespace(
                bPoseIsValid=valid,
                mDeviceToAbsoluteTracking=types.SimpleNamespace(m=matrix),
            )

        class FakeSystem:
            calls = 0

            def getDeviceToAbsoluteTrackingPose(self, *_args):
                self.calls += 1
                if self.calls == 1:
                    return [pose(False), pose(False), pose(False)]
                return [pose(False), pose(True, 0.4), pose(True, -0.4)]

            def getTrackedDeviceClass(self, index):
                return 1 if index in {1, 2} else 0

            def getStringTrackedDeviceProperty(self, index, _property):
                return {1: "LHR-RIGHT", 2: "LHR-LEFT"}[index]

        system = FakeSystem()
        fake_openvr = types.SimpleNamespace(
            VRApplication_Other=1,
            TrackingUniverseStanding=1,
            k_unMaxTrackedDeviceCount=3,
            TrackedDeviceClass_GenericTracker=1,
            Prop_SerialNumber_String=1,
            init=lambda _application: None,
            shutdown=lambda: None,
            VRSystem=lambda: system,
        )
        with mock.patch.dict(sys.modules, {"openvr": fake_openvr}):
            devices = tracker_binding.enumerate_openvr_trackers(
                wait_for_poses=True,
                pose_wait_seconds=0.1,
                pose_poll_seconds=0.0,
            )

        self.assertEqual(system.calls, 2)
        self.assertEqual([device.x for device in devices], [0.4, -0.4])

    def test_environment_indices_override_persisted_bindings(self) -> None:
        configured = {
            "right_wrist": "LHR-OLD-RIGHT",
            "left_wrist": "LHR-OLD-LEFT",
        }
        requests = {"right_wrist": "index=6", "left_wrist": "#2"}
        devices = [
            tracker_binding.TrackerDevice(2, "LHR-NEW-LEFT", -0.2),
            tracker_binding.TrackerDevice(6, "LHR-NEW-RIGHT", 0.2),
        ]
        bindings = tracker_binding.resolve_bindings(configured, requests, devices)
        self.assertEqual(
            bindings,
            {
                "right_wrist": "LHR-NEW-RIGHT",
                "left_wrist": "LHR-NEW-LEFT",
            },
        )

    def test_environment_serials_do_not_require_openvr_enumeration(self) -> None:
        requests = {
            "right_wrist": "LHR-EXPLICIT-RIGHT",
            "left_wrist": "LHR-EXPLICIT-LEFT",
        }
        self.assertFalse(tracker_binding.bindings_need_devices({}, requests))
        self.assertEqual(
            tracker_binding.resolve_bindings({}, requests, []),
            {
                "right_wrist": "LHR-EXPLICIT-RIGHT",
                "left_wrist": "LHR-EXPLICIT-LEFT",
            },
        )

    def test_base_station_selection_uses_balanced_opposite_side_pair(self) -> None:
        trackers = [
            tracker_binding.TrackerDevice(1, "LHR-RIGHT", 0.2, 1.0, 0.0),
            tracker_binding.TrackerDevice(2, "LHR-LEFT", -0.2, 1.0, 0.0),
        ]
        references = [
            tracker_binding.TrackingReferenceDevice(
                4, "LHB-OWN-LEFT", -0.8, 2.0, 0.1
            ),
            tracker_binding.TrackingReferenceDevice(
                5, "LHB-OWN-RIGHT", 0.82, 2.0, 0.1
            ),
            # This neighboring station is closer on the left, but pairing it
            # with the right station would be strongly asymmetric.
            tracker_binding.TrackingReferenceDevice(
                6, "LHB-FOREIGN", -0.15, 2.0, 0.1
            ),
        ]

        pair = tracker_binding.choose_nearest_base_station_pair(
            {"right_wrist": "LHR-RIGHT", "left_wrist": "LHR-LEFT"},
            trackers,
            references,
        )

        self.assertEqual(pair, ("LHB-OWN-RIGHT", "LHB-OWN-LEFT"))

    def test_base_station_selection_requires_left_and_right_station(self) -> None:
        trackers = [
            tracker_binding.TrackerDevice(1, "LHR-RIGHT", 0.2, 1.0, 0.0),
            tracker_binding.TrackerDevice(2, "LHR-LEFT", -0.2, 1.0, 0.0),
        ]
        references = [
            tracker_binding.TrackingReferenceDevice(
                4, "LHB-LEFT-A", -0.8, 2.0, 0.1
            ),
            tracker_binding.TrackingReferenceDevice(
                5, "LHB-LEFT-B", -0.4, 2.0, 0.2
            ),
        ]

        with self.assertRaisesRegex(
            tracker_binding.TrackerBindingError,
            "one pose-ready station on each side",
        ):
            tracker_binding.choose_nearest_base_station_pair(
                {"right_wrist": "LHR-RIGHT", "left_wrist": "LHR-LEFT"},
                trackers,
                references,
            )

    def test_configure_writes_private_machine_local_yaml_without_touching_template(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            template = root / "repository-template.yaml"
            output = root / "machine" / "openvr_input.yaml"
            original_template = (
                "tracker_serials:\n"
                "  right_wrist: auto\n"
                "  left_wrist: auto\n"
                "wrist_offset: [0.0, -0.00275, 0.0535]\n"
                "publish_rate_hz: 120.0\n"
            )
            template.write_text(original_template, encoding="utf-8")
            devices = [
                tracker_binding.TrackerDevice(5, "LHR-LEFT", -0.5),
                tracker_binding.TrackerDevice(7, "LHR-RIGHT", 0.5),
            ]

            with mock.patch.object(
                tracker_binding, "enumerate_openvr_trackers", return_value=devices
            ):
                bindings = tracker_binding.configure(template, output, {})

            self.assertEqual(bindings["right_wrist"], "LHR-RIGHT")
            self.assertEqual(template.read_text(encoding="utf-8"), original_template)
            payload = yaml.safe_load(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["tracker_serials"]["right_wrist"], "LHR-RIGHT")
            self.assertEqual(payload["tracker_serials"]["left_wrist"], "LHR-LEFT")
            self.assertEqual(payload["wrist_offset"], [0.0, -0.00275, 0.0535])
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)

    def test_persisted_fixed_bindings_do_not_reenumerate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            template = root / "template.yaml"
            output = root / "openvr_input.yaml"
            template.write_text(
                "tracker_serials:\n  right_wrist: auto\n  left_wrist: auto\n",
                encoding="utf-8",
            )
            output.write_text(
                "tracker_serials:\n"
                "  right_wrist: LHR-FIXED-RIGHT\n"
                "  left_wrist: LHR-FIXED-LEFT\n"
                "wrist_offset: [1.0, 2.0, 3.0]\n",
                encoding="utf-8",
            )

            with mock.patch.object(
                tracker_binding,
                "enumerate_openvr_trackers",
                side_effect=AssertionError("fixed bindings must not enumerate"),
            ):
                tracker_binding.configure(template, output, {})

            payload = yaml.safe_load(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["wrist_offset"], [1.0, 2.0, 3.0])
            self.assertEqual(
                payload["tracker_serials"]["left_wrist"], "LHR-FIXED-LEFT"
            )

    def test_persisted_fixed_base_station_pair_is_never_reselected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            template = root / "template.yaml"
            output = root / "openvr_input.yaml"
            template.write_text(
                "tracker_serials:\n  right_wrist: auto\n  left_wrist: auto\n"
                "base_station_serials: []\n",
                encoding="utf-8",
            )
            output.write_text(
                "tracker_serials:\n"
                "  right_wrist: LHR-FIXED-RIGHT\n"
                "  left_wrist: LHR-FIXED-LEFT\n"
                "base_station_serials:\n"
                "- LHB-OWN-RIGHT\n"
                "- LHB-OWN-LEFT\n",
                encoding="utf-8",
            )

            with mock.patch.object(
                tracker_binding,
                "enumerate_openvr_tracking_references",
                side_effect=AssertionError("fixed base stations must not be reselected"),
            ):
                tracker_binding.configure(template, output, {})

            payload = yaml.safe_load(output.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["base_station_serials"],
                ["LHB-OWN-RIGHT", "LHB-OWN-LEFT"],
            )

    def test_failed_auto_binding_does_not_rewrite_existing_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            template = root / "template.yaml"
            output = root / "openvr_input.yaml"
            template.write_text(
                "tracker_serials:\n  right_wrist: auto\n  left_wrist: auto\n",
                encoding="utf-8",
            )
            original = (
                "# keep this exact content\n"
                "tracker_serials:\n  right_wrist: auto\n  left_wrist: auto\n"
                "wrist_offset: [1.0, 2.0, 3.0]\n"
            )
            output.write_text(original, encoding="utf-8")
            devices = [
                tracker_binding.TrackerDevice(1, "LHR-LEFT", None),
                tracker_binding.TrackerDevice(2, "LHR-RIGHT", None),
            ]

            with mock.patch.object(
                tracker_binding,
                "enumerate_openvr_trackers",
                return_value=devices,
            ):
                with self.assertRaises(tracker_binding.TrackerBindingError):
                    tracker_binding.configure(template, output, {})

            self.assertEqual(output.read_text(encoding="utf-8"), original)

    def test_refuses_to_overwrite_repository_template(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            template = Path(temporary_directory) / "openvr_input.yaml"
            template.write_text("tracker_serials: {}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                tracker_binding.TrackerBindingError, "refusing to overwrite"
            ):
                tracker_binding.configure(template, template, {})

    def test_write_oserror_becomes_concise_binding_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "openvr_input.yaml"
            with mock.patch.object(
                tracker_binding.tempfile,
                "mkstemp",
                side_effect=PermissionError("read-only directory"),
            ):
                with self.assertRaisesRegex(
                    tracker_binding.TrackerBindingError,
                    "cannot write machine-local OpenVR config",
                ):
                    tracker_binding.write_machine_config(
                        output,
                        {
                            "tracker_serials": {
                                "right_wrist": "LHR-RIGHT",
                                "left_wrist": "LHR-LEFT",
                            }
                        },
                    )

            original = "tracker_serials: {right_wrist: OLD-R, left_wrist: OLD-L}\n"
            output.write_text(original, encoding="utf-8")
            with mock.patch.object(
                tracker_binding.os,
                "replace",
                side_effect=PermissionError("rename denied"),
            ):
                with self.assertRaises(tracker_binding.TrackerBindingError):
                    tracker_binding.write_machine_config(
                        output,
                        {
                            "tracker_serials": {
                                "right_wrist": "LHR-RIGHT",
                                "left_wrist": "LHR-LEFT",
                            }
                        },
                    )
            self.assertEqual(output.read_text(encoding="utf-8"), original)
            self.assertEqual(
                [path for path in output.parent.iterdir() if path != output],
                [],
            )

        stderr = io.StringIO()
        with mock.patch.object(
            tracker_binding,
            "configure",
            side_effect=tracker_binding.TrackerBindingError("write denied"),
        ):
            with redirect_stderr(stderr):
                return_code = tracker_binding.main(
                    ["--template", "template.yaml", "--output", "output.yaml"]
                )
        self.assertEqual(return_code, 1)
        self.assertIn("[ERROR] OpenVR tracker binding: write denied", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
