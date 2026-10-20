"""GEN6 physical approach/grasp/turn/release diagnostic through public actions.

No object/robot state, task threshold, sensor value or latch is written after
reset. This is an engineering controller, never human or zero-shot evidence.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

TASK_DIR = Path(__file__).resolve().parent
ROOT = TASK_DIR.parents[2]
sys.path[:0] = [str(ROOT / 'handson'), str(ROOT)]
from live_scene_probe import safe, snapshot, robot_snapshot, collision_capture, TASK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--offset', nargs=3, type=float, default=(-.12, .065, .080))
    parser.add_argument('--grip-scale', type=float, default=.85)
    parser.add_argument('--collection-origin', action='store_true')
    parser.add_argument('--sides', nargs='+', choices=['left', 'right'], default=['left', 'right'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'task': TASK, 'scope': 'Physical engineering controller; no human or model-generalization claim.',
              'offset_m': args.offset, 'grip_scale': args.grip_scale,
              'started_at_unix': time.time(), 'steps': [], 'phases': []}
    env = None
    code = 1
    actions = []
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation, Slerp
        from PIL import Image
        from handson.sim.isaaclab.b320_bimanual_env import HandsOnB320BimanualIsaacLabEnv
        from handson.sim.isaaclab.abc_collection_env import _C_TOOL_WUJI_POWER_GRIP
        from handson.sim.isaaclab.gen6_audio_mixer_scene import asset_geometry
        if args.collection_origin:
            from scripts import record_demos_zarr as recorder
            recorder.FLAGS(['gen6_physical_diagnostic', '--exp_name=' + TASK,
                            '--sim_backend=isaacsim', '--randomize=False', '--seed=0'])
            os.environ[recorder.STABLE_ORIGIN_ENV] = '1'

        class ProbeEnv(HandsOnB320BimanualIsaacLabEnv):
            latest = None
            def __init__(self, *a, **kw):
                ProbeEnv.latest = self
                super().__init__(*a, **kw)
            def _configure_task_simulation_settings(self):
                super()._configure_task_simulation_settings()
                from pxr import PhysxSchema, UsdPhysics
                for prim in self._stage.Traverse():
                    if prim.IsA(UsdPhysics.Scene):
                        PhysxSchema.PhysxSceneAPI.Apply(prim).CreateEnableSceneQuerySupportAttr().Set(True)

        env = ProbeEnv(TASK, seed=0, physics_dt=1/120, control_dt=1/30, time_limit=40.,
                       render_mode='rgb_array', image_obs=True, randomize=False,
                       randomize_dynamics=False, hand_type='wuji1', arm_type='rokae',
                       launch_isaac=True, headless=True, hz=30, isaac_device='cuda:0', physics_backend='physx')
        if args.collection_origin:
            _, origin_info, _, _ = recorder._reset_recording_episode_with_stable_origin(
                env, {}, sim_backend='isaacsim', task_id=TASK, collection_run_seed=0,
                reset_attempt_index=0, seed_resets=True,
                collection_seed_source=recorder.EXPLICIT_COLLECTION_SEED_SOURCE)
            report['recording_origin_info'] = safe(origin_info)
        else:
            env.reset(seed=0)
        report['initial'] = snapshot(env)
        report['initial_robots'] = robot_snapshot(env)
        report['state_write_count_start'] = env._trajectory_state_write_count
        poses = {s: env._pose_matrix_to_vector(env._robot_link_pose_matrix(s)) for s in ('left', 'right')}
        home = {s: v.copy() for s, v in poses.items()}
        hands = {s: np.zeros(20) for s in poses}
        hands['left'][0] = hands['right'][0] = .1
        done = False
        step_id = 0
        phase = 'reset'

        def checkpoint(label):
            row = {'phase': label, 'control_step': step_id, 'state': snapshot(env),
                   'robots': robot_snapshot(env), 'shapes': collision_capture(env, label)}
            from handson.sim.isaaclab.b320_bimanual_env import _tensor_to_numpy
            row['contact_manifolds'] = {}
            for key, view in env._ka005_2_contact_views.items():
                try:
                    values = [_tensor_to_numpy(v) for v in view.get_contact_data(env._sim_dt)]
                    forces, points, normals, separation, counts, starts = values
                    contacts = []
                    for pair, (count, start) in enumerate(zip(counts.ravel(), starts.ravel())):
                        for index in range(int(start), int(start) + int(count)):
                            hand_side = 'left' if key[0] == 'L' else 'right'
                            body_paths = env._hand_body_paths(hand_side)
                            contacts.append({'sensor_filter_pair': pair, 'hand_body': body_paths[pair],
                                             'knob_body': env._b320_spawn_result.body_paths['knob_' + ('left' if key[1] == 'L' else 'right')],
                                             'point_m': points.reshape(-1, 3)[index],
                                             'normal': normals.reshape(-1, 3)[index],
                                             'normal_force_n': forces.ravel()[index],
                                             'depth_m': max(0., -float(separation.ravel()[index]))})
                    row['contact_manifolds'][key] = contacts
                except Exception as error:
                    row['contact_manifolds'][key] = {'error': str(error)}
            report['phases'].append(safe(row))
            if not args.collection_origin:
                frames = env.render()
                for camera, frame in zip(('main', 'wrist_left', 'wrist_right'), frames):
                    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(args.output / f'{len(report["phases"]):02d}_{label}_{camera}.png')
            (args.output / 'report_partial.json').write_text(json.dumps(safe(report), indent=2) + '\n')
            print('GEN6_MANIPULATION_PHASE', label, json.dumps(safe(snapshot(env))), flush=True)

        def step(label):
            nonlocal done, step_id, phase
            phase = label
            command = {s: np.concatenate((poses[s], hands[s])).astype(np.float32) for s in poses}
            actions.append(np.concatenate((poses['right'], poses['left'], hands['right'], hands['left'])))
            obs, reward, terminated, truncated, info = env.step(command)
            step_id += 1
            row = {'control_step': step_id, 'phase': label, 'q_rad': env._ka005_2_state.last_q.copy(),
                   'qd_rad_s': env._ka005_2_state.last_qd.copy(),
                   'forces_n': dict(env._ka005_2_contact_force_n), 'flags': env._ka005_2_state.condition_flags(),
                   'terminated': bool(terminated), 'truncated': bool(truncated), 'succeed': bool(info.get('succeed'))}
            report['steps'].append(row)
            if terminated or truncated:
                report['terminal'] = {'control_step': step_id, 'phase': label, 'info': safe(info)}
                done = True
                if not info.get('succeed'):
                    raise RuntimeError(f'Unexpected terminal in {label}: {info}')
            return row

        def move(side, target_pose, target_hand, count, label):
            start_pose, start_hand = poses[side].copy(), hands[side].copy()
            rotations = Rotation.from_quat(np.array([start_pose[3:], target_pose[3:]])[:, [1,2,3,0]])
            slerp = Slerp([0.,1.], rotations)
            for i in range(1, count + 1):
                if done: break
                f = i/count
                q = slerp(f).as_quat()[[3,0,1,2]]
                poses[side] = np.r_[start_pose[:3]*(1-f) + target_pose[:3]*f, q]
                hands[side] = start_hand*(1-f) + target_hand*f
                step(label)
            checkpoint(label)

        checkpoint('reset')
        geometry = asset_geometry()
        mount = Rotation.from_quat(np.array(geometry['knob_orientation_wxyz'])[[1,2,3,0]])
        axis = mount.apply([0.,0.,1.])
        for side in args.sides:
            if done: break
            center = np.array(env._physics_body_pose('knob_' + side)[:3])
            sign = 1. if side == 'left' else -1.
            grip = home[side].copy()
            grip[:3] = center + [args.offset[0], sign*args.offset[1], args.offset[2]]
            hover = grip.copy(); hover[2] += .08
            raised = poses[side].copy(); raised[2] = hover[2]
            open_hand = hands[side].copy()
            closed = _C_TOOL_WUJI_POWER_GRIP.copy() * args.grip_scale
            move(side, raised, open_hand, 40, side + '_raise')
            move(side, hover, open_hand, 40, side + '_approach')
            move(side, grip, open_hand, 40, side + '_lower')
            move(side, grip, closed, 50, side + '_grasp')
            held = []
            for _ in range(30):
                row = step(side + '_contact_hold')
                held.append(row['forces_n']['LL' if side == 'left' else 'RR'])
            checkpoint(side + '_contact_hold')
            report.setdefault('contact_holds', {})[side] = {'forces_n': held, 'duration_s': 1., 'no_terminal': not done}
            if min(held) < .15:
                raise RuntimeError(f'{side} grasp did not sustain 1 s of assigned-hand contact')
            pivot = center + axis * .0225
            original_pose = poses[side].copy()
            initial_rotation = Rotation.from_quat(original_pose[3:][[1,2,3,0]])
            index = 0 if side == 'left' else 1
            angle = 0.
            stable = 0
            def set_turn_pose(lift=0.):
                nonlocal angle
                error = env._ka005_2_state.target_q[index] - env._ka005_2_state.last_q[index]
                correction = .06 * error
                if abs(error) < np.radians(1.5):
                    correction = 0.
                angle += float(np.clip(correction, -.010, .010))
                rotation = Rotation.from_rotvec(axis * angle)
                poses[side] = np.r_[pivot + rotation.apply(original_pose[:3]-pivot) + axis * lift,
                                   (rotation*initial_rotation).as_quat()[[3,0,1,2]]]
                return error
            for i in range(250):
                if done: break
                error = set_turn_pose()
                step(side + '_turn')
                # This controls an engineering release pose. Actual target,
                # speed, release and stability acceptance stays exclusively
                # in the unchanged source state machine at every substep.
                stable = stable + 1 if abs(error) < np.radians(2.5) else 0
                if stable >= 10: break
            checkpoint(side + '_turn')
            report.setdefault('target_control', {})[side] = {'stable_control_frames': stable, 'palm_turn_deg': np.degrees(angle)}
            if stable < 10:
                raise RuntimeError(f'{side} physical feedback did not settle at the target')
            # Release by lifting along the physical spindle first. Opening the
            # fingers while still wrapped around the flutes dragged the knob.
            for i in range(1, 61):
                if done: break
                set_turn_pose(.065 * i/60)
                step(side + '_lift_release')
            checkpoint(side + '_lift_release')
            move(side, poses[side].copy(), open_hand, 30, side + '_release')
            withdraw = poses[side].copy(); withdraw[2] += .10
            move(side, withdraw, open_hand, 40, side + '_withdraw')
            move(side, home[side], open_hand, 50, side + '_return')
        for _ in range(30):
            if done: break
            step('final_stability')
        checkpoint('final')
        report['physical_success'] = bool(env._ka005_2_state.success)
        report['state_write_count_end'] = env._trajectory_state_write_count
        assert report['state_write_count_start'] == report['state_write_count_end']
        report['probe_completed'] = True
        code = 0
    except BaseException as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        traceback.print_exc()
    finally:
        report['elapsed_s'] = time.time() - report['started_at_unix']
        (args.output / 'report.json').write_text(json.dumps(safe(report), indent=2) + '\n')
        if actions:
            import numpy as np
            np.save(args.output / 'diagnostic_raw_actions.npy', np.asarray(actions, dtype=np.float32))
        print('GEN6_MANIPULATION_RESULT', json.dumps({'code': code, 'error': report.get('error')}), flush=True)
        if env is None and 'ProbeEnv' in locals(): env = ProbeEnv.latest
        if env is not None:
            env._fast_shutdown_on_close = True
            env.close()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
