"""Exercise paper camera construction without starting Kit or a GPU."""
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from handson.sim.isaaclab import calibrated_cameras, compat


@pytest.fixture(params=[False, True], ids=['lab2-wxyz', 'lab3-xyzw'])
def camera_type(monkeypatch, request):
    monkeypatch.setattr(compat, 'isaaclab_uses_xyzw', lambda: request.param)
    class Camera:
        def __init__(self, cfg):
            self.cfg = cfg
            self.device = "cpu"
            self._view = SimpleNamespace(prim_paths=[cfg.prim_path], count=1)

        def _update_intrinsic_matrices(self, env_ids):
            pass

        def set_world_poses(self, positions=None, orientations=None, env_ids=None, convention="ros"):
            self.last_position = np.asarray(positions)
            self.last_orientation = np.asarray(orientations)

        def set_world_poses_from_view(self, eyes, targets, env_ids=None):
            pass

    sensors = ModuleType("isaaclab.sensors")
    sensors.Camera = Camera
    simulation = ModuleType("isaaclab.sim")
    simulation.find_matching_prims = lambda path: []
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", sensors)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", simulation)

    class Launcher:
        def __init__(self, args=None, **kwargs):
            pass

    calibrated_cameras.install(Launcher)
    Launcher({})
    return Camera


def configuration(sensor):
    return SimpleNamespace(
        prim_path="/World/" + sensor, width=640, height=640,
        spawn=SimpleNamespace(vertical_aperture=20.955, clipping_range=(0.1, 10.0)),
        offset=SimpleNamespace(),
    )


def test_all_three_camera_projections_match_frozen_contract(camera_type):
    path = Path(calibrated_cameras.__file__).parents[2] / "evaluation/camera_contract.json"
    specs = json.loads(path.read_text())["cameras"]
    for sensor, role in [("HeadRecordCamera", "main"),
                         ("LeftWristRecordCamera", "wrist_left"),
                         ("RightWristRecordCamera", "wrist_right"),
                         ("wrist_left_camera", "wrist_left"),
                         ("wrist_right_camera", "wrist_right")]:
        camera = camera_type(configuration(sensor))
        cfg, spec = camera.cfg, specs[role]
        assert [cfg.width, cfg.height] == spec["native_width_height"]
        effective_fy = cfg.spawn.focal_length * cfg.height / cfg.spawn.vertical_aperture
        assert effective_fy == pytest.approx(spec["K_effective"][1][1])
        if role != "main":
            assert cfg.spawn.clipping_range[0] == spec["near_clip_m"]
        else:
            expected = calibrated_cameras.profile()['quaternion_wxyz_opengl']
            np.testing.assert_allclose(compat.quat_isaaclab_to_wxyz(cfg.offset.rot), expected)


def test_evaluation_can_apply_sampled_head_pose_without_changing_preview_mount(camera_type):
    pytest.importorskip("torch")
    camera = camera_type(configuration("head_camera"))
    nominal = np.asarray([calibrated_cameras.profile()["position_m"]])
    sampled = nominal + [[0.001, -0.001, 0.0005]]
    camera.set_world_poses(sampled)
    np.testing.assert_allclose(camera.last_position, nominal, atol=1e-7)
    np.testing.assert_allclose(
        compat.quat_isaaclab_to_wxyz(camera.last_orientation[0]),
        calibrated_cameras.profile()['quaternion_wxyz_opengl'], atol=1e-7)
    camera._handson_sampled_pose = True
    camera.set_world_poses(sampled)
    np.testing.assert_array_equal(camera.last_position, sampled)
    camera._handson_sampled_pose = False
    camera.set_world_poses(sampled)
    np.testing.assert_allclose(camera.last_position, nominal, atol=1e-7)
