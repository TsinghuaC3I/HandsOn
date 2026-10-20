from __future__ import annotations

import json

import pytest

import repair_and_upload_wrist_cameras as camera_worker


def test_worker_discovers_state_only_ycb_episode(tmp_path) -> None:
    demo_dir = tmp_path / "t47_trigger_drill_on_block_demo_1__physx_v3"
    (demo_dir / "replay.zarr").mkdir(parents=True)
    (demo_dir / "collection_meta.json").write_text(
        json.dumps(
            {
                "task_name": "t47_trigger_drill_on_block",
                "sim_backend": "isaacsim",
                "engine_code": "E02",
                "step_count": 42,
                "video_fps": 30,
                "isaac_camera_render_deferred": True,
            }
        ),
        encoding="utf-8",
    )
    (demo_dir / "replay_manifest.json").write_text(
        json.dumps(
            {
                "task": {"name": "t47_trigger_drill_on_block"},
                "recording": {
                    "sim_backend": "isaacsim",
                    "step_count": 42,
                    "video_fps": 30,
                    "isaac_camera_render_deferred": True,
                },
                "software": {
                    "git": {
                        "commit": "1" * 40,
                        "dirty": False,
                        "changed_files": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    episode = camera_worker._episode_from_dir(demo_dir)

    assert episode is not None
    assert episode.task == "t47_trigger_drill_on_block"
    assert episode.expected_frames == 42
    assert not (demo_dir / "videos").exists()


def test_fresh_worker_explicitly_enables_postrender_cameras(monkeypatch) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_REQUIRE_REAL_RECORDING_CAMERAS", "0")
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "1")
    monkeypatch.setenv("HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY", "1")

    camera_worker._configure_camera_postrender_environment()

    assert camera_worker.os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] == "0"
    assert camera_worker.os.environ["HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY"] == "0"


@pytest.mark.parametrize("modern", [False, True])
def test_fresh_supervisor_accepts_declared_state_only_source(tmp_path, modern):
    from scripts import fresh_process_certified_collection as supervisor
    declaration = (
        {"canonical_camera_postrender_required": True,
         "online_camera_capture_scope": "all_canonical_postrender"}
        if modern else {"isaac_camera_render_deferred": True}
    )
    for name in ('recording_schema.json', 'collection_meta.json'):
        (tmp_path / name).write_text(json.dumps(declaration))
    manifest = {'task': {'name': 't47_trigger_drill_on_block'},
                'recording': {'sim_backend': 'isaacsim', **declaration}}
    (tmp_path / 'replay_manifest.json').write_text(json.dumps(manifest))
    assert supervisor._episode_integrity_error(tmp_path) is None
    (tmp_path / 'collection_meta.json').write_text('{}')
    assert 'videos/' in supervisor._episode_integrity_error(tmp_path)


def test_fresh_supervisor_enables_cameras_only_after_source_exit(monkeypatch, tmp_path):
    from scripts import fresh_process_certified_collection as supervisor
    from types import SimpleNamespace
    config = SimpleNamespace(recorder_args=('--exp_name=t47_trigger_drill_on_block', '--sim_backend=isaacsim'),
                             startup_timeout_seconds=180)
    monkeypatch.setenv('HANDSON_ISAACLAB_DEFER_ARM_RENDER', '0')
    env = supervisor._recorder_environment(config, startup_ready_file=tmp_path / 'ready', startup_status_file=tmp_path / 'status')
    assert env['HANDSON_ISAACLAB_DEFER_CAMERA_RENDER'] == '1'
    assert env['HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY'] == '0'
    assert env['HANDSON_ISAAC_DEFERRED_ARM_WORKER'] == '1'


def test_postrender_rejects_constant_rgb_even_with_different_channels():
    import numpy as np
    frame = np.broadcast_to(np.asarray([255, 0, 0], dtype=np.uint8), (32, 32, 3))
    assert camera_worker._frame_is_housing_obstruction(frame)


def test_camera_compatibility_patch_preserves_native_task_restore(monkeypatch):
    import handson.sim.isaaclab.abc_collection_env as module

    class NativeRuntime:
        _recorded_state_restore_includes_task_state = True

        def restore_recorded_initial_state(self, state):
            self.trigger_dof = state['trigger_dof']
            self.hidden_latch = state['hidden_latch']

        def _restore_recorded_table_height(self, delta):
            self.delta = delta

        def _set_task_body_world_pose(self, view, pose):
            self.pose = pose

    names = ('restore_recorded_initial_state', '_restore_recorded_table_height',
             '_set_task_body_world_pose')
    methods = {name: getattr(NativeRuntime, name) for name in names}
    monkeypatch.setattr(module, 'HandsOnABCIsaacLabEnv', NativeRuntime)
    camera_worker._install_approved_camera_replay_patch(task_name='t47_trigger_drill_on_block')
    assert all(getattr(NativeRuntime, name) is method for name, method in methods.items())
    instance = NativeRuntime()
    instance.restore_recorded_initial_state({'trigger_dof': 0.002, 'hidden_latch': True})
    assert instance.trigger_dof == 0.002 and instance.hidden_latch is True
    assert callable(instance.render)


def test_camera_compatibility_patch_keeps_other_tasks_on_legacy_restore(monkeypatch):
    import handson.sim.isaaclab.abc_collection_env as module

    class HistoricalRuntime:
        _recorded_state_restore_includes_task_state = True

        def restore_recorded_initial_state(self, state):
            raise AssertionError('other tasks must retain their historical repair')

    original = HistoricalRuntime.restore_recorded_initial_state
    monkeypatch.setattr(module, 'HandsOnABCIsaacLabEnv', HistoricalRuntime)
    camera_worker._install_approved_camera_replay_patch(task_name='a01_pick_place')
    assert callable(HistoricalRuntime.restore_recorded_initial_state)
    assert HistoricalRuntime.restore_recorded_initial_state is not original
    assert callable(HistoricalRuntime._restore_recorded_table_height)
    assert callable(HistoricalRuntime._set_task_body_world_pose)


def test_camera_compatibility_patch_requires_real_rgb(monkeypatch):
    import handson.sim.isaaclab.abc_collection_env as module
    import pytest

    class HistoricalRuntime:
        _camera_aliases = {}
        _cameras = {}
        _render_width, _render_height = 32, 32

        def _fallback_camera_frame(self, key):
            pytest.fail('a fallback frame must never reach strict post-rendering')

    monkeypatch.setattr(module, 'HandsOnABCIsaacLabEnv', HistoricalRuntime)
    monkeypatch.setenv('HANDSON_ISAACLAB_REQUIRE_REAL_RECORDING_CAMERAS', '1')
    camera_worker._install_approved_camera_replay_patch()
    with pytest.raises(RuntimeError, match='real recording camera RGB unavailable'):
        HistoricalRuntime()._camera_rgb_frame('head_camera', 'ego')
