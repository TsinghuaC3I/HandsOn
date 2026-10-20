"""Bounded GEN7 engineering trials through public TCP/native-hand actions.

The bottle, trigger and robot state are never written by this probe. Only
ordinary environment reset and env.step(controller.action()) advance a trial.
This is neither operator acceptance nor a saved collection/replay episode.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def emit(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda v: v.tolist()) + "\n")


def query_matrix(raw):
    """Read-only virtual nozzle queries against the current live scene.

    Virtual nozzle poses are query inputs only, never actor poses. This checks
    source-surface/occlusion geometry independently of robot manipulation.
    """
    from handson.sim.isaaclab import eval_gen06_spray_shoe_scene as scene
    from handson.sim.isaaclab.gen7_physics_geometry import PhysicsGeometry
    geometry = getattr(raw, '_gen7_physics_geometry', None) or PhysicsGeometry(raw)
    snapshot = geometry.prepare(raw)
    shoe = raw._pose_for_role('receptacle').copy()
    rotation = Rotation.from_quat(shoe[3:], scalar_first=True)
    centers = scene.upper_triangles().mean(axis=1)
    point = centers[np.argmin(np.linalg.norm(centers - [0., -.075, .055], axis=1))]
    target = shoe[:3] + rotation.apply(point)
    palm_shape, palm_inverse = next((shape, inv) for shape, inv in snapshot.shapes
                                   if '/right_palm_link/' in shape['path'] and 'vertices' in shape)
    palm_world = np.linalg.inv(palm_inverse)
    palm_center = palm_world[:3, :3] @ palm_shape['vertices'].mean(axis=0) + palm_world[:3, 3]
    cases = {
        'source_upper': (target + [0., 0., .15], np.array([0., 0., -1.])),
        'through_table': (target - [0., 0., .25], np.array([0., 0., 1.])),
        'away_from_shoe': (target + [0., 0., .15], np.array([0., 0., 1.])),
        'outside_silhouette': (target + [.25, 0., .15], np.array([0., 0., -1.])),
        'inside_actual_palm_occluder': (palm_center, (target-palm_center)/np.linalg.norm(target-palm_center)),
    }
    rows = {}
    for name, (origin, forward) in cases.items():
        nozzle_rotation = Rotation.align_vectors([forward], [[0., 1., 0.]])[0]
        bottle = np.r_[origin - nozzle_rotation.apply(scene.asset_manifest()['nozzle_local']),
                       nozzle_rotation.as_quat(scalar_first=True)]
        proxy = SimpleNamespace(_pose_for_role=lambda role: bottle if role == 'target' else shoe,
            _stage=None, _gen7_raycast_closest=snapshot.raycast_closest,
            _c04_spray_sample_directions=raw._c04_spray_sample_directions)
        rows[name] = {'metrics': scene.spray_aim_metrics(proxy),
                      'central_closest_hit': snapshot.raycast_closest(origin, forward, 2.),
                      'nozzle_xyz': origin.tolist()}
    return {'kind': 'virtual_nozzle_read_only_live_collision_queries',
            'actor_state_writes': 0, 'cases': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--stabilize', action='store_true')
    parser.add_argument('--profile', choices=('side_grasp', 'front_trigger'), default='side_grasp')
    args = parser.parse_args()
    if not 1 <= args.seeds <= 3:
        raise ValueError('The operability probe is bounded to at most three trials.')
    if os.environ.get('HANDSON_ISAACLAB_DEFER_CAMERA_RENDER') != '1':
        raise RuntimeError('This probe requires deferred recording cameras.')
    version = importlib.metadata.version('isaacsim')
    if not version.startswith('5.1.'):
        raise RuntimeError(f'Expected Isaac Sim 5.1, received {version}')
    from handson.runtime.primitive_control import HandPrimitiveController
    from handson.tasks.eval_gen06_spray_shoe.config import TaskConfig
    from handson.sim.isaaclab.gen7_physics_geometry import physical_body_matrices
    from handson.sim.isaaclab.recording_origin import (
        stabilize_recording_origin, stable_recording_origin_kwargs_for_task,
    )
    task = Path(__file__).resolve().parent
    result = {'isaacsim': version, 'trial_kind': 'public_tcp_hand_actions',
              'profile': args.profile,
              'full_acceptance_pass': False, 'operator_acceptance': False,
              'recording_replay_acceptance': False, 'runs': [],
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    env = None
    try:
        env = TaskConfig().get_environment(policy_mode=True, sim_backend='isaacsim',
            render_mode='rgb_array', headless=True, image_obs=False, randomize=True)
        raw = env.unwrapped
        raw._fast_shutdown_on_close = True
        for seed in range(args.seeds):
            obs, info = env.reset(seed=seed)
            if args.stabilize:
                obs, info = stabilize_recording_origin(env, obs, info,
                    **stable_recording_origin_kwargs_for_task(raw.task_name))
            controller = HandPrimitiveController(hand_type='wuji1')
            controller.sync_from_env(raw)
            frames = []
            bottle_start = raw._pose_for_role('target').copy()
            emit(args.output / f'seed_{seed}_live_query_matrix.json', query_matrix(raw))
            terminated = truncated = False

            def step(label):
                nonlocal terminated, truncated
                controller.advance(raw._control_dt)
                action = controller.action()
                _, _, terminated, truncated, info = env.step(action)
                matrices = physical_body_matrices(raw)
                palm = next(matrix for path, matrix in matrices.items()
                            if path.endswith('/right_palm_link'))
                row = {'stage': label, 'frame': len(frames), 'action': action.tolist(),
                       'bottle_pose': raw._pose_for_role('target').tolist(),
                       'palm_xyz': palm[:3, 3].tolist(),
                       'tcp_error_m': float(np.linalg.norm(palm[:3, 3] - action[:3])),
                       'trigger_rad': float(raw._physics_dof_position('joint_0')[0]),
                       'grasp_force_n': raw._c04_grasp_contact_force(),
                       'trigger_force_n': raw._c04_trigger_contact_force(),
                       'armed': bool(raw._c04_trigger_armed),
                       'emitting': bool(raw._c04_water_emitting),
                       'succeed': bool(info.get('succeed', False)),
                       'failed': bool(info.get('failed', False)),
                       'terminated': bool(terminated), 'truncated': bool(truncated),
                       'hit_count': int(info.get('gen7_spray_hit_count', 0))}
                frames.append(row)
                if len(frames) % 30 == 0:
                    print('[GEN7 operability]', seed, json.dumps(row), flush=True)
                if not np.all(np.isfinite(row['bottle_pose'])):
                    raise RuntimeError('Non-finite bottle pose during public action trial.')
                return not (terminated or truncated)

            def hold(label, count):
                for _ in range(count):
                    if not step(label):
                        return False
                return True

            def move(label, pose, count):
                start = controller.poses['right'].copy()
                rotations = Rotation.from_quat(np.asarray([start[3:], pose[3:]]), scalar_first=True)
                slerp = Slerp([0., 1.], rotations)
                for fraction in np.linspace(0., 1., count + 1)[1:]:
                    command = np.r_[start[:3] + fraction * (pose[:3] - start[:3]),
                                    slerp([fraction]).as_quat(scalar_first=True)[0]]
                    controller.set_pose(command)
                    if not step(label):
                        return False
                return True

            bottle_rotation = Rotation.from_quat(bottle_start[3:], scalar_first=True)
            # Side approach: palm normal toward +X, index finger above the
            # other fingers and finger length toward -Y, in the bottle frame.
            hand_rotation = bottle_rotation * (
                Rotation.from_euler('x', 90, degrees=True) if args.profile == 'side_grasp'
                else Rotation.from_euler('x', -90, degrees=True) * Rotation.from_euler('y', 90, degrees=True))
            hand_q = hand_rotation.as_quat(scalar_first=True)
            def pose(offset):
                return np.r_[bottle_start[:3] + bottle_rotation.apply(offset), hand_q]
            height = (.115, .135, .155)[seed]
            pregrasp = [-.10, .16, .32] if args.profile == 'side_grasp' else [-.24, .12, .32]
            lowered = [-.075, .12, height] if args.profile == 'side_grasp' else [-.21, .10, .22]
            contact = [-.035, .10, height] if args.profile == 'side_grasp' else [-.17, .065, .22]
            controller.set_grasp('open')
            trial_ok = move('clear_above_home', np.r_[controller.poses['right'][:2],
                                bottle_start[2] + .36, controller.poses['right'][3:]], 60)
            if trial_ok:
                trial_ok = move('pregrasp', pose(pregrasp), 90)
            if trial_ok:
                trial_ok = move('lower_clear', pose(lowered), 75)
            if trial_ok:
                trial_ok = move('approach', pose(contact), 60)
            if trial_ok:
                controller.set_grasp('cylindrical')
                controller.set_fingers({'index': 0.})
                trial_ok = hold('grasp_without_index', 60)
            if trial_ok:
                lift = controller.poses['right'].copy()
                if args.profile == 'side_grasp':
                    lift[2] += .035
                trial_ok = move('lift_test', lift, 60)
            if trial_ok:
                controller.set_fingers({'index': 1.}, base_preset='cylindrical')
                trial_ok = hold('index_pull', 60)
            if trial_ok:
                controller.set_fingers({'index': 0.})
                trial_ok = hold('index_release', 60)
            if trial_ok:
                controller.set_grasp('open')
                trial_ok = hold('open_release', 45)
            emit(args.output / f'seed_{seed}_frames.json', frames)
            trace = getattr(raw, '_physx_ground_truth_trace', None)
            if trace is not None:
                np.savez_compressed(args.output / f'seed_{seed}_final_snapshot.npz', **trace.snapshot())
            row = {'seed': seed, 'reset_parameters': raw._gen7_reset_parameters.tolist(),
                   'initial_bottle_pose': bottle_start.tolist(), 'frames': len(frames),
                   'max_grasp_force_n': max(f['grasp_force_n'] for f in frames),
                   'max_trigger_force_n': max(f['trigger_force_n'] for f in frames),
                   'max_trigger_rad': max(f['trigger_rad'] for f in frames),
                   'max_tcp_error_m': max(f['tcp_error_m'] for f in frames),
                   'max_bottle_rise_m': max(f['bottle_pose'][2] - bottle_start[2] for f in frames),
                   'grasp_armed': any(f['armed'] for f in frames),
                   'emission_observed': any(f['emitting'] for f in frames),
                   'success_observed': any(f['succeed'] for f in frames),
                   'completed_action_sequence': bool(trial_ok),
                   'final_frame': frames[-1]}
            result['runs'].append(row)
            emit(args.output / 'operability_result.json', result)
            print('[GEN7 operability trial result]', json.dumps(row), flush=True)
        raw._stage.Export(str(args.output / 'final_composed_scene.usdc'))
        result['completed'] = True
        emit(args.output / 'operability_result.json', result)
    except BaseException:
        result['error'] = traceback.format_exc()
        emit(args.output / 'operability_result.json', result)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
