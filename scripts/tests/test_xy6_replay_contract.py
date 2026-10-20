"""Exercise XY6 through the ordinary Zarr loader and boundary comparator."""

import json
import ast
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import zarr

from handson.sim.isaaclab.b320_bimanual_env import HandsOnB320BimanualIsaacLabEnv
from handson.sim.isaaclab.eval_comb06_season_and_stir_env import (
    HandsOnXY6IsaacLabEnv, TASK_NAME, XY6_HIDDEN_STATE_FIELDS,
)
from scripts.certify_physx_replay import CertificationError, compare_snapshot, load_demo
from scripts.tests.test_certify_physx_replay import _write_demo


@pytest.fixture
def demo_dir(tmp_path, monkeypatch):
    directory = _write_demo(tmp_path)
    manifest_path = directory / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["task"]["name"] = TASK_NAME
    manifest["recording"]["stable_recording_origin"]["task_name"] = TASK_NAME
    manifest_path.write_text(json.dumps(manifest))
    schema_path = directory / "recording_schema.json"
    schema = json.loads(schema_path.read_text())
    monkeypatch.setattr(HandsOnB320BimanualIsaacLabEnv, "get_recording_schema",
                        lambda self: schema["environment_schema"])
    env = HandsOnXY6IsaacLabEnv(launch_isaac=False, image_obs=False)
    try:
        obs, _ = env.reset(seed=11)
        environment = env.get_recording_schema()
        schema["environment_schema"] = environment
        semantics = environment["fields"]["xy6_hidden_state"]
        data = zarr.open_group(str(directory / "replay.zarr"), mode="a")["data"]
        for name in ("xy6_hidden_state", "next_xy6_hidden_state"):
            schema["fields"][name] = {
                "shape_per_step": semantics["shape"], "dtype": "float64",
                "semantics": dict(semantics),
            }
            data.array(name, np.stack([obs["xy6_hidden_state"]] * 2))
        schema_path.write_text(json.dumps(schema))
    finally:
        env.close()
    return directory


@pytest.mark.parametrize("field", ["phase", "salt_hit_seconds", "accumulated_rad", "anchor_0"])
def test_xy6_loader_compares_hidden_history_without_restoring_it(demo_dir, field):
    demo = load_demo(demo_dir)
    spec = next(s for s in demo.field_specs if s.name == "xy6_hidden_state")
    assert spec.replay_restore_supported is False
    recorded = {s.name: np.asarray(demo.next_fields[s.name][0]) for s in demo.field_specs}
    actual = {n: v.copy() for n, v in recorded.items()}
    index = XY6_HIDDEN_STATE_FIELDS.index(field)
    actual["xy6_hidden_state"][index] += .01
    _, divergence = compare_snapshot(
        recorded, actual, demo.field_specs, phase="post_action", action_index=0,
        recorded_time_origin=0., actual_time_origin=0.,
    )
    assert divergence["field"] == "xy6_hidden_state"
    assert divergence["index"] == [index]


def test_xy6_loader_requires_current_next_continuity(demo_dir):
    data = zarr.open_group(str(demo_dir / "replay.zarr"), mode="a")["data"]
    data["xy6_hidden_state"][1, 0] = 2.
    with pytest.raises(CertificationError, match="continuity is broken for xy6_hidden_state"):
        load_demo(demo_dir)


def _script_functions(filename, *names):
    """Load the exact helper without registering a second CLI's Abseil flags."""
    path = Path(__file__).resolve().parents[1] / filename
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    namespace = {'np':np, 'zarr':zarr, 'Path':Path, 'json':json, 'Mapping':Mapping}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def _camera_replay_function(name):
    return _script_functions('replay_demos_zarr.py', name)[name]


def test_xy6_camera_replay_uses_native_action_order():
    from handson.tasks.eval_comb06_season_and_stir.config import TaskConfig
    _step_with_recorded_action = _camera_replay_function('_step_with_recorded_action')
    env = TaskConfig().get_native_environment(policy_mode=True, launch_isaac=False, image_obs=False)
    try:
        env.reset(seed=0)
        raw = env.unwrapped
        right = raw._right_tcp_pose.copy()
        left = raw._left_tcp_pose.copy()
        right[2] += .02
        left[2] += .03
        hands = np.arange(40, dtype=np.float64) / 100 + .1
        action = np.r_[right, left, hands]
        _step_with_recorded_action(
            env, action,
            action_storage_layout="right_pose_left_pose_right_hand_left_hand",
        )
        np.testing.assert_allclose(raw._right_tcp_pose, right)
        np.testing.assert_allclose(raw._left_tcp_pose, left)
        np.testing.assert_allclose(raw._right_hand, hands[:20])
        np.testing.assert_allclose(raw._left_hand, hands[20:])
    finally:
        env.close()


@pytest.mark.parametrize("task_name,native,legacy", [
    (TASK_NAME, False, False),
    (TASK_NAME, True, False),
    (TASK_NAME, True, True),
    (TASK_NAME, False, True),
    ("t30_pour_seasonings", False, False),
    ("t30_pour_seasonings", True, False),
    ("t30_pour_seasonings", False, True),
])
def test_teleop_recorder_zarr_replay_preserves_both_robot_commands(
    tmp_path, monkeypatch, task_name, native, legacy,
):
    """Exercise the real wrapper and env; only tracker/socket I/O is replaced."""
    import handson.tasks.sim_teleop as teleop
    from handson.tasks.mappings import CONFIG_MAPPING

    recorder = _script_functions(
        'record_demos_zarr.py', '_flatten_action_for_storage',
        '_action_storage_layout_for_action', '_action_storage_layout',
    )
    replay = _script_functions(
        'replay_demos_zarr.py', '_load_episode_actions', '_load_replay_manifest',
        '_manifest_recording_parameters', '_recorded_action_storage_layout',
        '_step_with_recorded_action',
    )
    config = CONFIG_MAPPING[task_name]()
    env = config.get_native_environment(policy_mode=True, launch_isaac=False, image_obs=False)
    env.reset(seed=0)
    raw = env.unwrapped
    # Start no receiver threads and bind no real tracker/glove UDP sockets.
    monkeypatch.setattr(teleop, '_create_udp_socket', lambda *a, **k: Mock())
    monkeypatch.setattr(teleop.threading.Thread, 'start', lambda self: None)
    wrapper = teleop.DualArmViveHandTeleopWrapper(raw, config.teleop)
    try:
        right, left = raw._right_tcp_pose.copy(), raw._left_tcp_pose.copy()
        right[2] += .02
        left[2] += .03
        monkeypatch.setattr(wrapper, '_vive_action', lambda: {'right':right, 'left':left})
        _, _, _, _, info = wrapper.step(np.r_[np.zeros(14), np.arange(40)/100+.1])
        executed = info['intervene_action']
        assert isinstance(executed, dict)
        assert executed['right'].shape == executed['left'].shape == (27,)
        expected = [raw._right_tcp_pose.copy(), raw._left_tcp_pose.copy(),
                    raw._right_hand.copy(), raw._left_hand.copy()]
        # Native diagnostics bypass teleop; formal episodes store its exact dict.
        action = np.concatenate(expected) if native else executed
        recording = {
            'action_storage_layout': recorder['_action_storage_layout_for_action'](action),
        }
        if legacy:
            recording = ({'recording_purpose':'xy6_camera_diagnostic',
                          'recorder':'validate_xy6_live.py'} if native else {})
        (tmp_path/'replay_manifest.json').write_text(json.dumps({'recording':recording}))
        zarr_path = tmp_path/'replay.zarr'
        data = zarr.open_group(str(zarr_path), mode='w').create_group('data')
        data.array('action', recorder['_flatten_action_for_storage'](action)[None])
        manifest = replay['_load_replay_manifest'](tmp_path)
        layout = replay['_recorded_action_storage_layout'](
            replay['_manifest_recording_parameters'](manifest), task_id=task_name,
        )
        env.reset(seed=0)
        stored = replay['_load_episode_actions'](zarr_path)[0]
        replay['_step_with_recorded_action'](env, stored, action_storage_layout=layout)
        for actual, wanted in zip(
            (raw._right_tcp_pose, raw._left_tcp_pose, raw._right_hand, raw._left_hand), expected,
        ):
            np.testing.assert_array_equal(actual, wanted)
    finally:
        wrapper.close()


def test_explicit_layout_overrides_legacy_probe_identity_and_rejects_unknown():
    resolve = _camera_replay_function('_recorded_action_storage_layout')
    recording = {'recording_purpose':'xy6_camera_diagnostic',
                 'recorder':'validate_xy6_live.py',
                 'action_storage_layout':'right_pose_hand_then_left_pose_hand'}
    assert resolve(recording, task_id=TASK_NAME) == recording['action_storage_layout']
    recording['action_storage_layout'] = 'unknown_layout'
    with pytest.raises(ValueError, match='Unsupported recorded action storage layout'):
        resolve(recording, task_id=TASK_NAME)


@pytest.mark.parametrize('layout', [
    'right_pose_hand_then_left_pose_hand', 'right_pose_left_pose_right_hand_left_hand',
])
def test_bimanual_layout_rejects_bad_shape_before_stepping(layout):
    from types import SimpleNamespace
    step = _camera_replay_function('_step_with_recorded_action')
    with pytest.raises(ValueError, match='invalid shape'):
        step(SimpleNamespace(unwrapped=None), np.zeros(27), action_storage_layout=layout)


def test_xy6_visual_restore_returns_all_recorded_objects_and_wuji_joints(tmp_path):
    _load_xy6_recording_state_extras = _camera_replay_function('_load_xy6_recording_state_extras')
    env = HandsOnXY6IsaacLabEnv(launch_isaac=False, image_obs=False)
    try:
        obs, _ = env.reset(seed=0)
        poses = obs['xy6_object_poses'].copy()
        poses[3:, 0] += .08  # Lid and spoon are absent from the policy vector.
        data = zarr.open_group(str(tmp_path/'replay.zarr'), mode='w').create_group('data')
        data.array('xy6_object_poses', np.stack([poses]*2))
        data.array('xy6_hidden_state', np.stack([obs['xy6_hidden_state']]*2))
        extras = _load_xy6_recording_state_extras(tmp_path/'replay.zarr', expected_steps=2)
        restored = env.restore_recorded_initial_state({k:v[0] for k,v in extras.items()})
        np.testing.assert_array_equal(restored['xy6_object_poses'], poses)
        hands = np.arange(40,dtype=np.float64)/100
        env._restore_recorded_robot_command_state({'gripper_pose':hands})
        np.testing.assert_array_equal(env._right_hand, hands[:20])
        np.testing.assert_array_equal(env._left_hand, hands[20:])
        with pytest.raises(ValueError,match='finite shape'):
            _load_xy6_recording_state_extras(tmp_path/'replay.zarr',expected_steps=3)
        del data['xy6_object_poses']
        with pytest.raises(ValueError,match='requires xy6_object_poses'):
            _load_xy6_recording_state_extras(tmp_path/'replay.zarr',expected_steps=2)
    finally:
        env.close()
