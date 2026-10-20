from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validate_isaac_deferred_camera_roundtrip import (
    controller_snapshot, diagnostic_right_pose, require_world_pose_dls_backend,
)


def test_diagnostic_accepts_actual_normalized_runtime_backend(monkeypatch):
    from handson.tasks.mappings import CONFIG_MAPPING
    monkeypatch.setenv('HANDSON_ISAACLAB_ARM_IK_BACKEND', 'dls')
    env = CONFIG_MAPPING['t35_rotate_propeller']().get_native_environment(
        policy_mode=True, launch_isaac=False, image_obs=False)
    try:
        assert require_world_pose_dls_backend(env.unwrapped) == 'dls'
    finally:
        env.close()
    with pytest.raises(RuntimeError, match="backend='haorui'"):
        require_world_pose_dls_backend(SimpleNamespace(_arm_ik_backend='haorui'))


def test_diagnostic_reads_the_controller_ee_body_instead_of_reset_cache():
    poses = np.asarray([[[1, 2, 3, 1, 0, 0, 0], [4, 5, 6, 1, 0, 0, 0]]], dtype=float)
    robot = SimpleNamespace(body_names=['left_tool', 'right_tool'],
        data=SimpleNamespace(body_pose_w=poses, joint_pos=np.zeros((1, 4))))
    raw = SimpleNamespace(_arm_ik_backend='dls', _control_robot=lambda side: robot,
        _ee_body_indices={'right': 1, 'left': 0},
        _joint_targets={'right': np.zeros((1, 4)), 'left': np.zeros((1, 4))})
    for side in ('right', 'left'):
        setattr(raw, f'_{side}_tcp_pose', np.full(7, 99.0))
        setattr(raw, f'_{side}_pose_commanded', False)
        setattr(raw, f'_{side}_hand', np.zeros(2))
    snapshot = controller_snapshot(raw)
    np.testing.assert_array_equal(snapshot['right']['actual_ee_pose'], poses[0, 1])
    assert snapshot['right']['ee_body_name'] == 'right_tool'
    assert snapshot['right']['cached_tcp_command'][0] == 99


def test_thirty_frame_diagnostic_has_five_baseline_and_twentyfive_actual_ee_commands():
    actual = np.asarray([0.12, -0.3, 1.1, 1.0, 0.0, 0.0, 0.0])
    anchor = None
    poses = []
    for step in range(30):
        measured = actual.copy()
        # Later measurement drift must not accumulate into a new target anchor.
        if step > 5:
            measured[2] += 0.1
        pose, anchor = diagnostic_right_pose(step, 30, 5, measured, anchor)
        poses.append(pose)
    np.testing.assert_array_equal(poses[:5], np.zeros((5, 7)))
    np.testing.assert_array_equal(poses[5], actual)
    np.testing.assert_allclose(poses[-1], actual, atol=1e-12)
    np.testing.assert_array_equal(anchor, actual)
    assert len(poses[5:]) == 25
    assert max(pose[2] - actual[2] for pose in poses[5:]) == pytest.approx(0.005)
    np.testing.assert_array_equal(np.asarray(poses[5:])[:, 3:], np.tile(actual[3:], (25, 1)))


def test_diagnostic_rejects_invalid_actual_ee_before_cartesian_activation():
    with pytest.raises(ValueError, match='invalid measured'):
        diagnostic_right_pose(5, 30, 5, np.zeros(7), None)
