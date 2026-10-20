#!/usr/bin/env python3
"""Live contact/reset/camera deferral regression, without a success trajectory."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import traceback
import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2305, 17, 991])
    parser.add_argument('--steps', type=int, default=120)
    args = parser.parse_args()
    version = importlib.metadata.version('isaacsim')
    assert version.startswith('5.1.')
    os.environ['HANDSON_ISAACLAB_DEFER_CAMERA_RENDER'] = '1'
    os.environ['HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY'] = '0'
    os.environ['HANDSON_RECORD_PHYSX_GROUND_TRUTH'] = '1'
    from handson.tasks.t47_trigger_drill_on_block.config import TaskConfig
    from handson.sim.isaaclab.ycb05_semantics import YCB05_CONTACT_CHANNELS
    from handson.sim.isaaclab.recording_origin import stabilize_recording_origin, stable_recording_origin_kwargs_for_task
    root = Path(__file__).resolve().parents[1]
    sources = ['handson/handson/sim/isaaclab/abc_collection_env.py', 'handson/handson/sim/isaaclab/bimanual_assembly_env.py', 'handson/handson/sim/isaaclab/ycb05_semantics.py', 'handson/handson/tasks/t47_trigger_drill_on_block/config.py', 'scripts/record_demos_zarr.py', 'repair_and_upload_wrist_cameras.py']
    report = {'pass': False, 'isaacsim': version, 'success_demonstration': False,
              'source_sha256': {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
              'seeds': []}
    env = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        env = TaskConfig().get_native_environment(policy_mode=True, launch_isaac=True,
                image_obs=False, randomize=True, render_mode='human')
        raw = env.unwrapped
        report['camera_products'] = list(raw._cameras)
        assert raw.control_dt == 1 / 30 and raw.physics_dt == 1 / 120 and raw._n_substeps == 4
        report['timing'] = {'control_dt': raw.control_dt, 'physics_dt': raw.physics_dt, 'substeps': raw._n_substeps}
        assert raw._deferred_camera_render and not raw._cameras and not raw.image_obs
        sampled = []
        for seed in args.seeds:
            obs, info = env.reset(seed=seed)
            sampled.append({k: v.copy() for k, v in raw._ycb05_sampled_reset_poses.items()})
            obs, info = stabilize_recording_origin(env, obs, info,
                **stable_recording_origin_kwargs_for_task(raw.task_name))
            origin = info['physical_recording_origin']
            assert origin['initial_geometry_gate_passed'], origin
            assert origin['gate_robot_velocity_stable'], origin
            assert origin['gate_task_velocity_stable'] or origin['gate_task_static_pose_stable'], origin
            previous_sim_time = float(np.asarray(obs['physx_sim_time']).reshape(-1)[0])
            frame_dts = []
            for step in range(args.steps):
                obs, reward, terminated, truncated, info = raw.step_with_current_joint_targets()
                sim_time = float(np.asarray(obs['physx_sim_time']).reshape(-1)[0])
                frame_dts.append(sim_time - previous_sim_time)
                assert abs(frame_dts[-1] - 1 / 30) < 1e-6, frame_dts[-1]
                previous_sim_time = sim_time
                metrics = raw._last_ycb05_metrics
                assert metrics['finite'] and metrics['contacts_valid'], metrics
                assert not raw._compute_failure() and not raw._compute_success_now(), info
                assert not terminated and not truncated, info
                public_obs = env.observation(obs)
                assert env.observation_space.contains(public_obs), (public_obs, env.observation_space)
            report['seeds'].append({'seed': seed, 'steps': args.steps, 'sampled_poses': sampled[-1],
                'initialization': origin, 'last_metrics': metrics,
                'actual_frame_dt_range': [min(frame_dts), max(frame_dts)],
                'contact_view_counts': {k: [int(raw._ycb05_contact_views[k].sensor_count), int(raw._ycb05_contact_views[k].filter_count)] for k in YCB05_CONTACT_CHANNELS}})
            print('YCB05_CONTACT_SEED_PASS', seed, args.steps, flush=True)
        view = raw._ycb05_contact_views.pop('wood_hand')
        try:
            raw.step_with_current_joint_targets()
            assert raw._compute_failure() and not raw._compute_success_now()
            assert raw._ycb05_state['failure_reason'] == 'invalid_contact_state'
            assert raw._ycb05_state['invalid_contact_channels'] == ['wood_hand']
        finally:
            raw._ycb05_contact_views['wood_hand'] = view
        raw.step_with_current_joint_targets()
        assert raw._compute_failure(), 'sensor recovery cannot clear first terminal reason'
        obs, info = env.reset(seed=args.seeds[0])
        for key, value in sampled[0].items():
            np.testing.assert_array_equal(raw._ycb05_sampled_reset_poses[key], value)
        obs, info = stabilize_recording_origin(env, obs, info,
            **stable_recording_origin_kwargs_for_task(raw.task_name))
        raw.step_with_current_joint_targets()
        assert raw._last_ycb05_metrics['contacts_valid'] and not raw._compute_failure()
        assert not raw._ycb05_state['invalid_contact_channels']
        report['post_fault_reset_initialization'] = info['physical_recording_origin']
        report['invalid_sensor_rejected_then_reset_recovered'] = True
        report['pass'] = True
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2, default=json_value) + '\n')
        print('YCB05_CONTACT_REPORT', str(args.output), report['pass'], flush=True)
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
