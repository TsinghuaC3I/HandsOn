from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('case', ['passed', 'readback_error', 'missing_camera', 'constant_rgb', 'wrong_length'])
def test_exact_camera_render_and_rejections_without_physics(tmp_path: Path, case: str):
    # Isolate absl flags imported by the real replay loader from recorder tests.
    program = r'''
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from scripts import isaac_exact_physx_camera_postrender as helper

destination, case = Path(sys.argv[1]), sys.argv[2]
destination.mkdir()
calls = []

class Trace:
    def validate_restore_schema(self, schema):
        assert schema == {'test_entity_layout': True}
    def restore(self, snapshot, schema):
        self.current = snapshot
        calls.append(snapshot['physx_robot_joint_position'][0])
        return {'robot_joints': 1}
    def snapshot(self):
        value = self.current['physx_robot_joint_position'].copy()
        if case == 'readback_error':
            value += 0.01
        return {'physx_robot_joint_position': value}

class Replay:
    schema = {'test_entity_layout': True}
    step_count = 3
    def snapshot(self, index, *, after_action):
        assert after_action is False
        return {'physx_robot_joint_position': np.asarray([index * 0.1]),
                'physx_robot_joint_valid': np.ones(1, dtype=bool)}

checker = ((np.indices((32, 48)).sum(axis=0) // 4) % 2).astype(np.uint8) * 255
pattern = np.stack([checker, 255-checker, checker], axis=-1)
images = {'ego': pattern, 'wrist_left': np.roll(pattern, 1, axis=0),
          'wrist_right': np.roll(pattern, 2, axis=0)}
if case == 'constant_rgb':
    images['ego'] = np.broadcast_to(np.asarray([255, 0, 0], dtype=np.uint8), pattern.shape)
trace = Trace()
cameras = {name: object() for name in ('head_camera', 'wrist_left_camera', 'wrist_right_camera')}
if case == 'missing_camera':
    cameras.pop('wrist_left_camera')
def forbidden_step(*args, **kwargs):
    raise AssertionError('exact visual reconstruction must never advance physics or execute actions')
raw = SimpleNamespace(_physx_ground_truth_trace=trace, _cameras=cameras,
    _isaac_warnings=[], _compute_observation=lambda: {'images': images}, step=forbidden_step)
env = SimpleNamespace(unwrapped=raw, observation=lambda obs: obs, step=forbidden_step)
replacements = []
try:
    output, quality = helper._write_exact_physx_videos(destination, env, Replay(),
        recording={}, fps=30, intrinsics_yaml='', before_replace=lambda: replacements.append(True))
except RuntimeError as exc:
    markers = {'readback_error': 'readback exceeded tolerance',
               'missing_camera': 'did not construct live cameras',
               'constant_rgb': 'constant frame'}
    assert case in markers and markers[case] in str(exc), (case, exc)
    assert not replacements
    assert not (destination/'videos').exists()
else:
    assert case in {'passed', 'wrong_length'}
    assert calls == [0.0, 0.1, 0.2] and replacements == [True]
    assert quality['status'] == 'passed'
    reconstruction = json.loads((output/'camera_reconstruction.json').read_text())
    assert reconstruction['physics_steps_per_frame'] == 0
    assert reconstruction['actions_reexecuted'] is False
    assert reconstruction['maximum_restore_errors'] == {'physx_robot_joint_position': 0.0}
    assert [item['frame_count'] for item in quality['videos']] == [3, 3, 3]
    if case == 'wrong_length':
        try:
            helper._validate_encoded_video(output/'main.mp4', expected_frames=4, expected_shape=pattern.shape)
        except RuntimeError as exc:
            assert '3 decoded frames, expected 4' in str(exc)
        else:
            raise AssertionError('wrong frame count was accepted')
print('exact-camera-check-passed')
'''
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['HANDSON_APPLY_FISHEYE_DISTORTION'] = '0'
    result = subprocess.run([sys.executable, '-c', program, str(tmp_path/'episode'), case],
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'exact-camera-check-passed' in result.stdout


def test_exact_camera_trace_is_enabled_before_environment_construction(tmp_path: Path):
    (tmp_path/'replay.zarr').mkdir()
    program = r'''
import os
from pathlib import Path
import sys
from scripts import isaac_exact_physx_camera_postrender as helper
class TraceConfigured(Exception):
    pass
class Config:
    def get_native_environment(self, **kwargs):
        assert os.environ['HANDSON_RECORD_PHYSX_GROUND_TRUTH'] == '1'
        assert os.environ['HANDSON_ISAACLAB_DEFER_CAMERA_RENDER'] == '0'
        assert os.environ['HANDSON_ISAACLAB_REQUIRE_REAL_RECORDING_CAMERAS'] == '1'
        assert kwargs['image_obs'] and kwargs['seed'] == 23
        raise TraceConfigured
helper.CONFIG_MAPPING = {'test_task': lambda: Config()}
helper._recorded_render_settings = lambda path: ({}, True, 23, 17, 'rokae', 'wuji1')
helper._load_episode_actions = lambda path: [object()]
helper._load_physx_replay_trajectory = lambda *args, **kwargs: object()
try:
    helper.render_exact_physx_episode(Path(sys.argv[1]), 'test_task', fps=30,
                                      intrinsics_yaml='', before_replace=lambda: None)
except TraceConfigured:
    print('trace-before-construction')
'''
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['HANDSON_RECORD_PHYSX_GROUND_TRUTH'] = '0'
    result = subprocess.run([sys.executable, '-c', program, str(tmp_path)],
                            env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'trace-before-construction' in result.stdout
