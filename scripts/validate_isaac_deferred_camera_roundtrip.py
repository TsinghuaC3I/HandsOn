#!/usr/bin/env python3
"""Two-process diagnostic of real state/action capture and shared camera rendering.

Run collect and render in separate, exclusively reserved Isaac 5.1 processes.
The diagnostic Zarr has no claimed successful task outcome or replay certificate.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=json_value) + '\n')


def controller_snapshot(raw):
    def array(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        return np.asarray(value).copy()

    result = {'ik_backend': raw._arm_ik_backend}
    for side in ('right', 'left'):
        robot = raw._control_robot(side)
        index = raw._ee_body_indices[side]
        result[side] = {
            'ee_body_name': robot.body_names[index],
            'actual_ee_pose': array(robot.data.body_pose_w[0, index, :7]),
            'cached_tcp_command': array(getattr(raw, f'_{side}_tcp_pose')),
            'pose_commanded': bool(getattr(raw, f'_{side}_pose_commanded')),
            'joint_position': array(robot.data.joint_pos),
            'joint_target': array(raw._joint_targets[side]),
            'hand_command': array(getattr(raw, f'_{side}_hand')),
        }
    return result


def require_world_pose_dls_backend(raw):
    backend = str(raw._arm_ik_backend)
    # The runtime normalizes its enum to "dls"; "isaaclab_dls" is a log label.
    if backend != 'dls':
        raise RuntimeError(f'diagnostic world-EE action anchor requires DLS; got backend={backend!r}')
    return backend


def diagnostic_right_pose(step, steps, baseline_steps, actual_ee_pose, anchor):
    if step < baseline_steps:
        return np.zeros(7), anchor
    if anchor is None:
        anchor = np.asarray(actual_ee_pose, dtype=float).copy()
        if anchor.shape != (7,) or not np.all(np.isfinite(anchor)) or np.linalg.norm(anchor[3:]) <= 1e-8:
            raise ValueError('invalid measured DLS end-effector pose')
    pose = anchor.copy()
    motion_step = step - baseline_steps
    pose[2] += 0.005 * math.sin(math.pi * motion_step / max(1, steps - baseline_steps - 1))
    return pose, anchor


def collect(env, config, args, report):
    import zarr
    from handson.data.episode_store import ZarrEpisodeStore
    from handson.data.recording_fields import (
        declared_recording_field_names, resolve_environment_recording_schema,
        select_current_observation_for_recording, select_next_observation_for_recording,
        stack_transition_numeric_observation_fields, write_recording_schema,
    )
    from handson.sim.isaaclab.recording_origin import stabilize_recording_origin, stable_recording_origin_kwargs_for_task
    raw = env.unwrapped
    obs, info = env.reset(seed=args.seed)
    obs, info = stabilize_recording_origin(env, obs, info,
        **stable_recording_origin_kwargs_for_task(args.task))
    assert not raw.image_obs and not raw._cameras
    assert os.environ['HANDSON_ISAACLAB_DEFER_CAMERA_RENDER'] == '1'
    assert os.environ['HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY'] == '0'
    report['initialization'] = info['physical_recording_origin']
    report['timing'] = {'control_dt': raw.control_dt, 'physics_dt': raw.physics_dt, 'substeps': raw._n_substeps}
    schema = resolve_environment_recording_schema(env)
    fields = declared_recording_field_names(schema)
    report['ik_backend'] = str(raw._arm_ik_backend)
    require_world_pose_dls_backend(raw)
    initial_controller = controller_snapshot(raw)
    report['initial_controller'] = initial_controller
    report['controller_frames'] = []
    baseline_steps = min(5, args.steps // 2)
    report['baseline_steps'] = baseline_steps
    report['control_command_mode'] = 'hold_baseline_then_actual_dls_ee_5mm_cartesian_motion'
    right_anchor = None
    left = np.zeros(7)
    right_hand = np.asarray(raw._right_hand).copy()
    left_hand = np.asarray(raw._left_hand).copy()
    transitions = []
    timestamps = []
    frame_dts = []
    for step in range(args.steps):
        before = controller_snapshot(raw)
        # A reset TCP cache is a command, not measured FK.  First retain the
        # accepted joint target using the existing zero-pose sentinel; then
        # anchor world-frame DLS commands to its actual end-effector body.
        right_pose, right_anchor = diagnostic_right_pose(step, args.steps, baseline_steps,
            before['right']['actual_ee_pose'], right_anchor)
        if right_anchor is not None:
            report['cartesian_anchor'] = right_anchor.copy()
        if int(np.prod(env.action_space.shape)) == 7 + right_hand.size:
            action = np.concatenate([right_pose, right_hand])
            recorded_action = action
        else:
            action = np.concatenate([right_pose, left, right_hand, left_hand])
            # Policy wrapper groups poses first; the shared recording/replay
            # contract stores complete raw right and then left side commands.
            recorded_action = np.concatenate([right_pose, right_hand, left, left_hand])
        current_time = float(np.asarray(obs['physx_sim_time']).reshape(-1)[0])
        timestamps.append(current_time)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        after = controller_snapshot(raw)
        frame_debug = {
            'step': step, 'phase': 'baseline' if step < baseline_steps else 'cartesian',
            'policy_action': action.copy(), 'raw_recorded_action': recorded_action.copy(),
            'before': before, 'after': after,
            'failed': bool(step_info.get('failed', False)),
            'terminal_reason': step_info.get('terminal_reason'),
            'joint_target_delta_max': {
                side: float(np.max(np.abs(after[side]['joint_target'] - before[side]['joint_target'])))
                for side in ('right', 'left')
            },
        }
        report['controller_frames'].append(frame_debug)
        if step < baseline_steps:
            assert all(value <= 1e-6 for value in frame_debug['joint_target_delta_max'].values()), frame_debug
        else:
            assert after['right']['pose_commanded'], frame_debug
        next_time = float(np.asarray(next_obs['physx_sim_time']).reshape(-1)[0])
        frame_dts.append(next_time - current_time)
        assert abs(frame_dts[-1] - raw.control_dt) < 1e-6, frame_dts[-1]
        assert not terminated and not truncated and not step_info.get('succeed', False), {
            'frame': step, 'terminal_reason': step_info.get('terminal_reason'),
            'failed': step_info.get('failed'), 'succeed': step_info.get('succeed')}
        transitions.append(copy.deepcopy({
            'observations': select_current_observation_for_recording(obs, required_fields=fields, include_required_fields=not transitions),
            'next_observations': select_next_observation_for_recording(next_obs, required_fields=fields),
            'actions': recorded_action, 'dones': False, 'infos': step_info,
        }))
        obs = next_obs
    motion_frames = report['controller_frames'][baseline_steps:]
    actual_ee = np.stack([item['after']['right']['actual_ee_pose'] for item in motion_frames])
    drive_targets = np.stack([item['after']['right']['joint_target'] for item in motion_frames])
    ik_errors = [str(item) for item in raw._isaac_warnings if 'ik failed' in str(item).lower()]
    report['cartesian_response'] = {
        'actual_ee_z_span_m': float(np.ptp(actual_ee[:, 2])),
        'actual_ee_position_max_frame_step_m': float(np.max(np.linalg.norm(np.diff(actual_ee[:, :3], axis=0), axis=1))),
        'right_drive_target_span_max_rad': float(np.max(np.ptp(drive_targets, axis=0))),
        'first_pose_activation_target_delta_max_rad': motion_frames[0]['joint_target_delta_max']['right'],
        'ik_errors': ik_errors,
    }
    assert not ik_errors, report['cartesian_response']
    assert report['cartesian_response']['actual_ee_z_span_m'] > 1e-5, 'Cartesian command produced no measured EE response'
    assert report['cartesian_response']['right_drive_target_span_max_rad'] > 1e-5, 'Cartesian command did not change drive targets'
    data = {
        'action': np.stack([t['actions'] for t in transitions]),
        'state': np.stack([t['observations']['state'] for t in transitions]),
        'next_state': np.stack([t['next_observations']['state'] for t in transitions]),
        'terminal': np.zeros(args.steps, dtype=np.uint8),
        'succeed': np.zeros(args.steps, dtype=np.uint8),
        'timestamp': np.asarray(timestamps, dtype=np.float64),
    }
    data.update(stack_transition_numeric_observation_fields(transitions,
        required_fields=fields, environment_schema=schema))
    episode = args.episode.resolve()
    episode.mkdir(parents=True, exist_ok=False)
    store = ZarrEpisodeStore.create_empty(storage=zarr.DirectoryStore(str(episode/'replay.zarr')))
    store.append_episode(data, compressors='disk')
    write_recording_schema(episode, store.root, data, environment_schema=schema)
    write_json(episode/'replay_manifest.json', {
        'task': {'name': args.task},
        'recording': {
            'sim_backend': 'isaacsim', 'randomize': True,
            'collection_run_seed': args.seed, 'episode_reset_seed': args.seed,
            'arm_type': raw.arm_type, 'hand_type': raw.hand_type,
            'step_count': args.steps, 'video_fps': 30,
            'isaac_camera_render_deferred': True,
        },
    })
    write_json(episode/'DIAGNOSTIC.json', {
        'scope': 'non_successful_real_state_action_camera_regression',
        'task': args.task, 'seed': args.seed, 'frames': args.steps,
        'collector_pid': os.getpid(), 'task_success': False,
        'formal_physical_certification': False, 'camera_products_online': [],
        'source_sha256': report['source_sha256'],
    })
    assert not (episode/'videos').exists()
    assert not np.any(store.root['data']['succeed'][:])
    report.update(episode=str(episode), frames=args.steps, actual_frame_dt_range=[min(frame_dts), max(frame_dts)],
                  state_fields=sorted(data), camera_products_online=[])


def render(args, report):
    from scripts import isaac_render_arm_replay as renderer
    from scripts import isaac_exact_physx_camera_postrender as exact
    diagnostic = json.loads((args.episode/'DIAGNOSTIC.json').read_text())
    assert diagnostic['task'] == args.task
    assert diagnostic['collector_pid'] != os.getpid(), 'postrender requires a fresh process'
    assert diagnostic['source_sha256'] == report['source_sha256'], 'collect/render source mismatch'
    renderer.FLAGS(['validate_isaac_deferred_camera_roundtrip'])
    output = renderer._render_and_publish(args.episode.resolve(), args.task)
    reconstruction = json.loads((output/'camera_reconstruction.json').read_text())
    assert reconstruction['frame_count'] == diagnostic['frames']
    assert reconstruction['physics_steps_per_frame'] == 0
    assert reconstruction['actions_reexecuted'] is False
    from PIL import Image
    import av
    frames = {}
    videos = []
    for name in ('main', 'wrist_left', 'wrist_right'):
        with av.open(str(output/f'{name}.mp4')) as video:
            frame = next(video.decode(video=0)).to_ndarray(format='rgb24')
            frames[name] = frame
            Image.fromarray(frame).save(args.episode/f'{name}.png')
        videos.append(exact._validate_encoded_video(output/f'{name}.mp4',
            expected_frames=diagnostic['frames'], expected_shape=frame.shape))
    assert not np.array_equal(frames['wrist_left'], frames['wrist_right'])
    report.update(output=str(output), camera_quality={'status': 'passed',
        'state_reconstruction': reconstruction, 'videos': videos},
        collector_pid=diagnostic['collector_pid'], renderer_pid=os.getpid(),
        scope='fresh_process_production_exact_physx_camera_renderer_on_real_non_successful_diagnostic')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('collect','render'), required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=2305)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--source-path', type=Path, action='append', default=[],
                   help='Additional task runtime source, relative to the repository root.')
    args=p.parse_args()
    if args.steps < 5:
        p.error('--steps must be at least 5 for a baseline and a Cartesian round trip')
    version=importlib.metadata.version('isaacsim')
    assert version.startswith('5.1.')
    os.environ['HANDSON_ISAACLAB_DEFER_CAMERA_RENDER']='1' if args.phase=='collect' else '0'
    os.environ['HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY']='0'
    os.environ['HANDSON_ISAACLAB_REQUIRE_REAL_RECORDING_CAMERAS']='0' if args.phase=='collect' else '1'
    os.environ['HANDSON_RECORD_PHYSX_GROUND_TRUTH']='1' if args.phase=='collect' else '0'
    sources={'handson/handson/sim/isaaclab/abc_collection_env.py',
             'handson/handson/sim/isaaclab/bimanual_assembly_env.py',
             'handson/handson/sim/isaaclab/physx_ground_truth.py',
             'scripts/isaac_render_arm_replay.py',
             'scripts/isaac_exact_physx_camera_postrender.py',
             'scripts/validate_isaac_deferred_camera_roundtrip.py'}
    # Include task semantics and specialized runtime subclasses as well as the
    # shared base.  A custom package can supply additional source paths.
    task_directory = ROOT/'handson'/'handson'/'tasks'/args.task
    sources.update(str(path.relative_to(ROOT)) for path in task_directory.rglob('*.py'))
    runtime_directory = ROOT/'handson'/'handson'/'sim'/'isaaclab'
    prefix = args.task.split('_', 1)[0]
    sources.update(str(path.relative_to(ROOT)) for path in runtime_directory.glob(f'{prefix}_*.py'))
    sources.update(str(path) for path in args.source_path)
    report={'pass':False, 'phase':args.phase, 'isaacsim':version,
            'success_demonstration':False, 'formal_physical_certification':False,
            'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(sources)}}
    env=None
    try:
        if args.phase == 'collect':
            from handson.tasks.mappings import CONFIG_MAPPING
            config=CONFIG_MAPPING[args.task]()
            env=config.get_native_environment(policy_mode=True, sim_backend='isaacsim',
                launch_isaac=True, render_mode='human', randomize=True,
                image_obs=False, seed=args.seed, arm_type='rokae', hand_type='wuji1')
            collect(env,config,args,report)
        else:
            render(args,report)
        report['pass']=True
    except Exception:
        report['error']=traceback.format_exc()
        raise
    finally:
        write_json(args.output,report)
        print('CAMERA_ROUNDTRIP_REPORT',str(args.output),report['pass'],flush=True)
        if env is not None: env.close()
        if args.phase == 'render':
            # The production renderer owns its fresh Kit process.  As in its
            # CLI entrypoint, exit only after the report and videos are saved.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0 if report['pass'] else 1)


if __name__=='__main__':
    main()
