"""Release assets must be portable and fail closed on incomplete copies."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from handson.sim.isaaclab import render_assets
from handson.sim.isaaclab.calibrated_cameras import profile
from handson.sim.isaaclab.recording_cameras import WRIST_RECORDING_CAMERA_LOCAL_POSES
from handson.sim.isaaclab.bimanual_assembly_env import _pose_vector_to_matrix


def test_packaged_rendering_dependencies_are_complete():
    root = render_assets.validate_assets()
    manifest = json.loads((root / 'manifest.json').read_text())
    assert len(manifest['files']) >= 98
    robot = json.loads((root / 'robot/manifest.json').read_text())
    assert len(robot['links']) == 60
    for item in [*robot['links'], *robot['dependencies']]:
        assert (root / 'robot' / item['asset']).is_file()


def test_paper_robot_and_renderer_are_bundled():
    from handson.robots.assets import require_benchmark_assets
    require_benchmark_assets()


@pytest.fixture
def portable_bundle(tmp_path, monkeypatch):
    root = tmp_path / 'relocated'
    root.mkdir()
    (root / 'mesh.usdc').write_bytes(b'a verified geometry fixture')
    manifest = {'files': {'mesh.usdc': hashlib.sha256((root / 'mesh.usdc').read_bytes()).hexdigest()}}
    data = json.dumps(manifest).encode()
    (root / 'manifest.json').write_bytes(data)
    monkeypatch.setattr(render_assets, 'MANIFEST_SHA256', hashlib.sha256(data).hexdigest())
    monkeypatch.setenv('HANDSON_RENDER_ASSETS', str(root))
    return root


def test_assets_resolve_after_relocation(portable_bundle):
    assert render_assets.validate_assets() == portable_bundle


def test_missing_dependency_cannot_fall_back(portable_bundle):
    (portable_bundle / 'mesh.usdc').unlink()
    with pytest.raises(FileNotFoundError, match='Required rendering asset'):
        render_assets.validate_assets()


def test_modified_dependency_invalidates_previous_verification(portable_bundle):
    render_assets.validate_assets()
    (portable_bundle / 'mesh.usdc').write_bytes(b'changed geometry')
    with pytest.raises(ValueError, match='checksum mismatch'):
        render_assets.validate_assets()


def test_manifest_tampering_is_rejected(portable_bundle):
    (portable_bundle / 'manifest.json').write_text('{"files": {}}')
    with pytest.raises(ValueError, match='manifest checksum'):
        render_assets.validate_assets()


def test_wrist_pose_and_published_calibration_agree():
    for side, matrix in profile()['wrist_camera_to_palm_opengl'].items():
        np.testing.assert_allclose(_pose_vector_to_matrix(WRIST_RECORDING_CAMERA_LOCAL_POSES[side]), matrix, atol=1e-12)


def test_image_observations_enable_live_cameras_by_default(monkeypatch):
    from handson.sim.isaaclab.d_collection_env import HandsOnDIsaacLabEnv
    monkeypatch.delenv('HANDSON_ISAACLAB_DEFER_CAMERA_RENDER', raising=False)
    monkeypatch.delenv('HANDSON_ISAACLAB_PREVIEW_LIVE_CAMERA', raising=False)
    env = HandsOnDIsaacLabEnv(task_name='t12_load_and_start_microwave', launch_isaac=False,
                            render_mode='rgb_array', image_obs=True)
    assert not env._deferred_camera_render
    assert env.observation_space['images']['front'].shape == (480, 854, 3)
    assert env.observation_space['images']['wrist_left'].shape == (480, 848, 3)
