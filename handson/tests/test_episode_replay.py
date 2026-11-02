"""Reject mismatched initialization and preserve failure/terminal semantics."""
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from handson.replay import bundle
from handson.replay.runtime import should_stop


def test_bundled_initial_robot_state_matches_released_eef52():
    episode = bundle.load_episode('e015401')
    assert episode.actions.shape == (209, 52)
    np.testing.assert_allclose(episode.task_state['ball_pose'][:3],
        [-0.3497619927, -0.1877793521, 0.9230204821], atol=1e-9)
    assert episode.manifest['historical_far_side_threshold_m'] == episode.manifest['current_far_side_threshold_m']


def test_device_cover_bundle_binds_old_environment_and_explicit_richer_input():
    episode = bundle.load_episode('e013673')
    assert episode.actions.shape == (720, 52)
    assert episode.manifest['paper_variant'] == 'stabilize_body'
    episode.require_control('eef52')
    episode.require_control('recorded_joint_targets')
    assert episode.joint_targets['next_physx_substep_robot_joint_target'].shape == (720, 4, 2, 27)
    assert episode.reference['next_physx_task_rigid_body_state'].shape == (720, 2, 13)
    with pytest.raises(ValueError, match='no bundled recorded joint targets'):
        bundle.load_episode('e015401').require_control('recorded_joint_targets')


def test_replay_rejects_wrong_source_cohort_before_any_state_write():
    episode = bundle.load_episode('e013673')
    env = SimpleNamespace(task_name=episode.manifest['task'], paper_variant=None, env_step=0,
                          restore_recorded_initial_state=Mock())
    with pytest.raises(ValueError, match='source cohort mismatch'):
        bundle.restore_initial_state(env, episode)
    env.restore_recorded_initial_state.assert_not_called()


@pytest.fixture
def copied_bundle(tmp_path):
    shutil.copytree(bundle.EPISODES / 'e015401', tmp_path / 'e015401')
    return tmp_path


def test_modified_initialization_payload_is_rejected(copied_bundle):
    path = copied_bundle / 'e015401/initial_snapshot.npz'
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        bundle.load_episode('e015401', root=copied_bundle)


def test_wrong_episode_cannot_reuse_known_initialization():
    with pytest.raises(ValueError, match='No bundled initialization'):
        bundle.load_episode('e015402')


def test_mismatched_public_state_is_rejected_even_with_updated_payload_hash(copied_bundle):
    path = copied_bundle / 'e015401/commands.npz'
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    arrays['states'][0, 0] += .1
    np.savez_compressed(path, **arrays)
    mp = path.with_name('manifest.json')
    manifest = json.loads(mp.read_text())
    manifest['files']['commands.npz'] = hashlib.sha256(path.read_bytes()).hexdigest()
    mp.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='snapshot and public episode do not match'):
        bundle.load_episode('e015401', root=copied_bundle)


def test_changed_runtime_is_not_silently_qualified(copied_bundle):
    mp = copied_bundle / 'e015401/manifest.json'
    manifest = json.loads(mp.read_text())
    manifest['runtime_files']['sim/isaaclab/ycb01_semantics.py'] = '0' * 64
    mp.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Unvalidated runtime'):
        bundle.load_episode('e015401', root=copied_bundle)


def test_restore_refuses_writes_after_actions_or_a_prior_restore():
    episode = bundle.load_episode('e015401')
    env = SimpleNamespace(task_name=episode.manifest['task'], env_step=1,
                          restore_recorded_initial_state=Mock())
    with pytest.raises(ValueError, match='after action execution'):
        bundle.restore_initial_state(env, episode)
    env.env_step = 0
    env._episode_replay_restore_started = True
    with pytest.raises(ValueError, match='already attempted'):
        bundle.restore_initial_state(env, episode)
    env.restore_recorded_initial_state.assert_not_called()


@pytest.mark.parametrize('info', [{'succeed': False}, {'succeed': True, 'failed': True},
                                  {'succeed': True, 'manual_reset': True}])
def test_diagnostic_continuation_cannot_ignore_failure(info):
    assert should_stop(True, False, info, True)


def test_success_continuation_is_explicit_and_timeout_always_stops():
    assert should_stop(True, False, {'succeed': True}, False)
    assert not should_stop(True, False, {'succeed': True}, True)
    assert should_stop(True, True, {'succeed': True}, True)


def test_silent_kit_exit_cannot_report_completed_replay(tmp_path, monkeypatch):
    from handson.replay import __main__ as cli
    worker = SimpleNamespace(pid=1, wait=lambda **kw: 0)
    monkeypatch.setattr(cli.subprocess, 'Popen', lambda *a, **kw: worker)
    out = tmp_path / 'replay'
    with pytest.raises(SystemExit) as exc:
        cli.main(['--episode', 'e015401', '--output', str(out)])
    assert exc.value.code == 1
    assert not (out / 'episode/RESULT.json').exists()
    assert json.loads((out / 'ERROR.json').read_text())['formal_benchmark_trial'] is False


def test_all_frozen_origin_and_trace_identities_match_packaged_source():
    root = bundle.PACKAGE
    for kind in ('origin', 'trace'):
        records = json.loads((root / f'evaluation/{kind}_implementations.json').read_text())
        for entry in records.values():
            path = root.parent / Path(entry['module'].replace('.', '/')).with_suffix('.py')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['packaged_sha256']


def test_cable_identity_matches_robot_visual_manifest():
    root = bundle.PACKAGE / 'assets/rendering'
    cables = json.loads((root / 'cables.json').read_text())
    assert cables['visual_bundle_sha256'] == hashlib.sha256((root / 'robot/manifest.json').read_bytes()).hexdigest()
