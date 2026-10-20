from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from handson.data.camera_info import (
    DEFAULT_FISHEYE_INTRINSICS_PATH,
    _scale_camera_matrix,
    load_fisheye_intrinsics,
    write_video_camera_info,
)
from handson.sim.isaaclab.bimanual_assembly_env import (
    BimanualAssemblyIsaacLabEnv,
)
from handson.sim.isaaclab.recording_cameras import (
    CAMERA_PROFILE_ENV,
    CAMERA_PROFILE_VERSION,
    DEFAULT_CAMERA_PROFILE_NAME,
    HEAD_RECORDING_CAMERA_EYE,
    HEAD_RECORDING_CAMERA_FOVY_DEG,
    HEAD_RECORDING_CAMERA_QUAT_WXYZ_OPENGL,
    LEGACY_CAMERA_PROFILE_NAME,
    OPERATOR_VIEWPORT_CAMERA_EYE,
    OPERATOR_VIEWPORT_CAMERA_TARGET,
    REAL_V1_CAMERA_PROFILE_NAME,
    RECORDING_OBJECT_XY_BOUNDS,
    WRIST_RECORDING_CAMERA_OUTPUT_SIZE,
    WRIST_REAL_V1_INTRINSICS_BY_SIDE,
    WRIST_UNDISTORT_K_NEW_848X480,
    _camera_profile_name_from_env,
    constrain_recording_object_xy,
    recording_camera_profile,
)
from handson.sim.isaaclab.abc_collection_env import _quat_wxyz_to_matrix


def test_bundled_realtek_fleet_mean_calibration_is_loadable() -> None:
    intrinsics = load_fisheye_intrinsics(DEFAULT_FISHEYE_INTRINSICS_PATH)

    assert intrinsics.path == DEFAULT_FISHEYE_INTRINSICS_PATH
    assert (intrinsics.image_width, intrinsics.image_height) == (848, 480)
    np.testing.assert_allclose(
        intrinsics.K,
        [
            [271.112092500, 0.0, 423.901563333],
            [0.0, 272.657822222, 243.054800000],
            [0.0, 0.0, 1.0],
        ],
    )
    np.testing.assert_allclose(
        intrinsics.D.reshape(-1),
        [-0.185015, 0.014974, -0.000507, 0.0],
    )
    np.testing.assert_allclose(
        intrinsics.new_K,
        [
            [355.778243619, 0.0, 423.5],
            [0.0, 355.778243619, 239.5],
            [0.0, 0.0, 1.0],
        ],
    )


def test_handson_is_the_default_camera_profile(monkeypatch) -> None:
    monkeypatch.delenv(CAMERA_PROFILE_ENV, raising=False)

    assert DEFAULT_CAMERA_PROFILE_NAME == "handson"
    assert _camera_profile_name_from_env() == DEFAULT_CAMERA_PROFILE_NAME


def test_removed_camera_profile_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setenv(CAMERA_PROFILE_ENV, LEGACY_CAMERA_PROFILE_NAME)

    with pytest.raises(ValueError, match="no longer selects"):
        assert _camera_profile_name_from_env() == LEGACY_CAMERA_PROFILE_NAME


def test_recording_camera_profile_is_json_safe_and_marks_wrist_as_rectified() -> None:
    profile = recording_camera_profile()

    json.dumps(profile)
    assert profile["version"] == CAMERA_PROFILE_VERSION
    assert profile["default_for_new_collection"] is (
        profile["name"] == DEFAULT_CAMERA_PROFILE_NAME
    )
    assert profile["legacy_compatibility_only"] is (
        profile["name"] == LEGACY_CAMERA_PROFILE_NAME
    )
    assert profile["head"]["fisheye_undistortion"] is False
    assert profile["wrist"]["output_resolution"] == [848, 480]
    expected_hfov = (
        np.mean(
            [
                values["horizontal_fov_deg"]
                for values in WRIST_REAL_V1_INTRINSICS_BY_SIDE.values()
            ]
        )
        if profile["name"] == DEFAULT_CAMERA_PROFILE_NAME
        else 100.0
    )
    assert profile["wrist"]["horizontal_fov_deg"] == pytest.approx(
        expected_hfov
    )
    assert profile["wrist"]["rendered_frames_are_already_rectified"] is True


def test_operator_viewport_is_behind_robot_and_looks_toward_table() -> None:
    eye = np.asarray(OPERATOR_VIEWPORT_CAMERA_EYE, dtype=np.float64)
    target = np.asarray(OPERATOR_VIEWPORT_CAMERA_TARGET, dtype=np.float64)

    assert eye[0] < -0.8
    assert target[0] > eye[0]
    assert 0.89 <= target[2] <= 1.08
    np.testing.assert_allclose(eye[1], target[1])


def test_randomized_object_workspace_is_visible_and_arm_reachable() -> None:
    rotation = _quat_wxyz_to_matrix(
        np.asarray(HEAD_RECORDING_CAMERA_QUAT_WXYZ_OPENGL)
    )
    eye = np.asarray(HEAD_RECORDING_CAMERA_EYE)
    tan_half_fovy = math.tan(
        math.radians(HEAD_RECORDING_CAMERA_FOVY_DEG) / 2.0
    )
    corners = np.asarray(
        [
            [x, y, z]
            for z in (0.89, 1.08)
            for x in RECORDING_OBJECT_XY_BOUNDS[:, 0]
            for y in RECORDING_OBJECT_XY_BOUNDS[:, 1]
        ],
        dtype=np.float64,
    )
    camera_points = (rotation.T @ (corners - eye).T).T
    depth = -camera_points[:, 2]
    # The production head stream is 16:9. Horizontal coverage therefore uses
    # the vertical tangent scaled by the native aspect ratio.
    projection_scale = np.asarray(
        [tan_half_fovy * (16.0 / 9.0), tan_half_fovy],
        dtype=np.float64,
    )
    ndc = camera_points[:, :2] / (depth[:, None] * projection_scale)

    assert np.all(depth > 0.0)
    assert np.max(np.abs(ndc)) <= 0.85

    reachability_path = (
        Path(__file__).resolve().parents[1]
        / "handson"
        / "sim"
        / "envs"
        / "c01_rokae_wuji_reachability.npz"
    )
    with np.load(reachability_path) as reachability:
        point_clouds = (
            reachability["right_points"],
            reachability["left_points"],
        )
    for x in RECORDING_OBJECT_XY_BOUNDS[:, 0]:
        for y in RECORDING_OBJECT_XY_BOUNDS[:, 1]:
            query = np.asarray([x, y, 1.03], dtype=np.float64)
            nearest = min(
                float(np.linalg.norm(points - query, axis=1).min())
                for points in point_clouds
            )
            assert nearest <= 0.025

    np.testing.assert_allclose(
        constrain_recording_object_xy([-0.8, 0.8]),
        RECORDING_OBJECT_XY_BOUNDS[[0, 1], [0, 1]],
    )


def test_fisheye_intrinsic_scaling_uses_pixel_center_compensation() -> None:
    scaled = _scale_camera_matrix(
        np.asarray(
            [
                [613.8387, 0.0, 960.4092],
                [0.0, 613.4801, 547.4983],
                [0.0, 0.0, 1.0],
            ]
        ),
        (1920, 1080),
        (848, 480),
    )

    np.testing.assert_allclose(
        scaled,
        [
            [271.112092500, 0.0, 423.901563333],
            [0.0, 272.657822222, 243.054800000],
            [0.0, 0.0, 1.0],
        ],
        atol=1e-9,
    )


def test_wrist_camera_sidecar_marks_saved_frame_as_already_rectified(
    tmp_path: Path,
) -> None:
    write_video_camera_info(
        videos_dir=tmp_path,
        camera_keys=["wrist_left"],
        first_frames={
            "wrist_left": np.zeros((480, 848, 3), dtype=np.uint8)
        },
        intrinsics_yaml=DEFAULT_FISHEYE_INTRINSICS_PATH,
        camera_profile=recording_camera_profile(),
    )

    info = json.loads(
        (tmp_path / "wrist_left_camera_info.json").read_text(
            encoding="utf-8"
        )
    )
    assert info["recording_camera_profile_version"] == CAMERA_PROFILE_VERSION
    assert info["saved_projection"]["model"] == "pinhole"
    expected_k = (
        WRIST_REAL_V1_INTRINSICS_BY_SIDE["left"]["K_new_848x480"]
        if recording_camera_profile()["name"] == DEFAULT_CAMERA_PROFILE_NAME
        else WRIST_UNDISTORT_K_NEW_848X480
    )
    assert info["saved_projection"]["K"] == [list(row) for row in expected_k]
    assert info["saved_projection"]["must_not_rectify_again"] is True
    calibration = info["physical_fisheye_intrinsics"]
    assert calibration["model"] == "fisheye"
    assert (calibration["image_width"], calibration["image_height"]) == (848, 480)
    assert np.asarray(calibration["K"]).shape == (3, 3)
    assert np.isfinite(calibration["D"]).all()


def test_real_v1_sidecar_uses_serial_specific_rectified_intrinsics(
    tmp_path: Path,
) -> None:
    profile = recording_camera_profile()
    profile["name"] = "real_v1"
    profile["version"] = "camera_pose_v7_real_v1"
    profile["wrist"]["fleet_mean_intrinsics"] = False

    write_video_camera_info(
        videos_dir=tmp_path,
        camera_keys=["wrist_left", "wrist_right"],
        first_frames={
            "wrist_left": np.zeros((480, 848, 3), dtype=np.uint8),
            "wrist_right": np.zeros((480, 848, 3), dtype=np.uint8),
        },
        intrinsics_yaml=DEFAULT_FISHEYE_INTRINSICS_PATH,
        camera_profile=profile,
    )

    for side in ("left", "right"):
        info = json.loads(
            (tmp_path / f"wrist_{side}_camera_info.json").read_text(
                encoding="utf-8"
            )
        )
        expected = WRIST_REAL_V1_INTRINSICS_BY_SIDE[side]
        assert info["recording_camera_profile_version"] == "camera_pose_v7_real_v1"
        assert info["saved_projection"]["horizontal_fov_deg"] == expected[
            "horizontal_fov_deg"
        ]
        assert info["saved_projection"]["K"] == [
            list(row) for row in expected["K_new_848x480"]
        ]


def test_standalone_bimanual_collection_uses_native_wrist_resolution() -> None:
    env = BimanualAssemblyIsaacLabEnv(
        launch_isaac=False,
        image_obs=True,
    )

    wrist_width, wrist_height = WRIST_RECORDING_CAMERA_OUTPUT_SIZE
    assert env.observation_space["images"]["ego"].shape == (480, 854, 3)
    assert env.observation_space["images"]["wrist_left"].shape == (
        wrist_height,
        wrist_width,
        3,
    )
    assert env.observation_space["images"]["wrist_right"].shape == (
        wrist_height,
        wrist_width,
        3,
    )


def test_accurate_visibility_uses_per_camera_exposure_and_render(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr("handson.sim.isaaclab.default_appearance.configure_exposure", lambda name="head_camera": events.append("exposure:"+name))

    class FakeSimulation:
        def render(self) -> None:
            events.append("render")

    class FakeCamera:
        def __init__(self, name: str) -> None:
            self.name = name

        def update(self, _dt: float, *, force_recompute=False) -> None:
            events.append(f"update:{self.name}")

    env = BimanualAssemblyIsaacLabEnv(launch_isaac=False, image_obs=False)
    env._sim = FakeSimulation()
    env._scene = {
        name: FakeCamera(name)
        for name in (
            "head_camera",
            "wrist_left_camera",
            "wrist_right_camera",
        )
    }
    env._accurate_camera_visibility = True
    env._update_recording_cameras = lambda: None
    env._set_wrist_obstructions_hidden = (
        lambda hidden: events.append(f"hidden:{hidden}")
    )
    env._camera_rgb_frame = lambda _name, _fallback: np.zeros(
        (2, 2, 3),
        dtype=np.uint8,
    )

    frames = env.render()

    assert len(frames) == 3
    assert events == [
        "hidden:False",
        "exposure:head_camera",
        "render",
        "update:head_camera",
        "hidden:True",
        "exposure:wrist_left_camera",
        "render",
        "update:wrist_left_camera",
        "exposure:wrist_right_camera",
        "render",
        "update:wrist_right_camera",
        "hidden:False",
        "exposure:head_camera",
    ]


def test_low_latency_visibility_keeps_single_shared_render_group() -> None:
    events: list[str] = []

    class FakeSimulation:
        def render(self) -> None:
            events.append("render")

    class FakeCamera:
        def __init__(self, name: str) -> None:
            self.name = name

        def update(self, _dt: float) -> None:
            events.append(f"update:{self.name}")

    env = BimanualAssemblyIsaacLabEnv(launch_isaac=False, image_obs=False)
    env._sim = FakeSimulation()
    env._scene = {
        name: FakeCamera(name)
        for name in (
            "head_camera",
            "wrist_left_camera",
            "wrist_right_camera",
        )
    }
    env._accurate_camera_visibility = False
    env._update_recording_cameras = lambda: None
    env._set_wrist_obstructions_hidden = (
        lambda hidden: events.append(f"hidden:{hidden}")
    )
    env._camera_rgb_frame = lambda _name, _fallback: np.zeros(
        (2, 2, 3),
        dtype=np.uint8,
    )

    env.render()

    assert events == [
        "hidden:True",
        "render",
        "update:head_camera",
        "update:wrist_left_camera",
        "update:wrist_right_camera",
    ]
