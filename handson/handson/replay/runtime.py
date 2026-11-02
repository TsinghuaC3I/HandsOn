"""Replay public commands in the current native environment from recorded t0."""
import json
import hashlib
import os
from pathlib import Path
import time
import traceback

import numpy as np

from .bundle import load_episode, require, restore_initial_state


def restore_episode_appearance(env, recorded):
    """Apply the published resolved profile, without re-sampling its RNG key."""
    from handson.sim.isaaclab import room_appearance, default_appearance
    from handson.sim.isaaclab.blender_visual_layer import physics_signature
    from handson.sim.isaaclab.render_assets import validate_assets
    root = validate_assets()
    require(hashlib.sha256((root / 'materials/ASSETS.json').read_bytes()).hexdigest()
            == recorded['assets_manifest_sha256'], 'Historical appearance asset identity mismatch')
    before = physics_signature(env._stage), default_appearance.physics_material_signature(env._stage)
    report = room_appearance.apply(env._stage, env._cameras, recorded['profile'], root / 'materials')
    after = physics_signature(env._stage), default_appearance.physics_material_signature(env._stage)
    require(before == after, 'Restoring appearance changed the physical scene')
    env.default_appearance_report['room'] = report
    default_appearance.verify_environment(env)
    return dict(profile=report['profile'], source_metadata_sha256=recorded['source_metadata_sha256'],
                physics_unchanged=True, pixel_identical_certified=False)


def should_stop(terminated, truncated, info, complete_recording):
    # A reset request/failure always ends this attempt, even without Gym's done flag.
    if truncated or info.get('failed') or info.get('manual_reset'):
        return True
    return bool(terminated and not (complete_recording and info.get('succeed', False)))


def run(episode_id, output, *, render_stride=10, complete_recording=False, control_mode='eef52',
        isolated_worker=False):
    require(render_stride > 0, 'render stride must be positive')
    episode = load_episode(episode_id)
    episode.require_control(control_mode)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    from handson.evaluation.readback import plain
    from handson.evaluation.task_contracts import configure, initialize, install_native_trace
    from handson.evaluation.cameras import install_launcher, install_native_camera_creation, render_probe
    from handson.evaluation.native import construct, capture
    from handson.evaluation import camera_randomization
    from handson.eef52 import EEF52Control, live_state
    from handson.tasks.registry import TASKS

    def write(name, value):
        (out / name).write_text(json.dumps(plain(value), indent=2) + '\n')

    def phase(name, **extra):
        value = dict(phase=name, **extra)
        write('PROGRESS.json', value)
        print(json.dumps(value), flush=True)

    task = episode.manifest['task']
    variant = episode.manifest.get('paper_variant')
    contract = configure(task, variant)
    os.environ['HANDSON_BACKGROUND_MODE'] = 'off'
    os.environ['HANDSON_ISAACLAB_CACHE'] = str((out / 'scene-cache').resolve())
    os.environ['HANDSON_ISAACLAB_DEFER_CAMERA_RENDER'] = '0'
    os.environ['HANDSON_ISAACLAB_REQUIRE_REAL_RECORDING_CAMERAS'] = '1'
    original_camera_sample = camera_randomization.sample
    wrapped = None
    try:
        write('REPLAY_PROTOCOL.json', dict(episode.manifest, formal_benchmark_trial=False,
            control_mode=control_mode, uses_original_substep_targets=control_mode == 'recorded_joint_targets',
            scene_state_writes_after_action_start=False,
            historical_checker_certified=False, complete_recording=complete_recording,
            rendering='recorded resolved room appearance and camera jitter; pixel identity not certified'))
        phase('initializing_runtime')
        install_launcher()
        from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
        install_native_camera_creation(HandsOnABCIsaacLabEnv)
        adapter = TASKS[task].evaluation_adapter(variant)
        adapter.prepare_cameras()
        install_native_trace(task, out, variant)
        adapter.prepare(out)
        env, wrapped, _ = construct(task, episode.manifest['construction_seed'], paper_variant=variant)
        if not env._cameras:
            env._init_recording_cameras()
        phase('initializing_episode')
        observation, info = env.reset(seed=episode.manifest['reset_seed'])
        write('SAMPLED_RESET.json', capture(env))
        observation, info = initialize(env, wrapped, observation, info, contract, out)
        write('SAMPLED_RECORDING_ORIGIN.json', info)
        write('T0_RESTORE.json', restore_initial_state(env, episode))
        write('APPEARANCE_RESTORE.json', restore_episode_appearance(env, episode.appearance))
        initial = capture(env)
        write('INITIAL_STATE.json', initial)
        actual, _ = live_state(env)
        initial_error = float(np.max(np.abs(actual - episode.states[0])))
        require(initial_error <= 1e-5, 'Restored EEF52 differs from the public episode')
        # Scoped to this fresh worker process. render_probe also verifies that
        # rendering cannot alter measured physics.
        camera_randomization.sample = lambda *a, **kw: episode.camera_jitter
        job = dict(task=task, paper_variant=variant)
        render_probe(env, job, out, 0, episode.manifest['reset_seed'], 'initial', capture)
        if control_mode == 'recorded_joint_targets':
            from .control import RecordedJointTargetControl
            control = RecordedJointTargetControl(env, episode.physics_schema)
        else:
            control = EEF52Control(env)
        first_success = first_historical_threshold = None
        last_tick = initial['physics_tick']
        start = time.monotonic()
        physical = []
        with (out / 'trajectory.jsonl').open('w') as trajectory:
            for index, action in enumerate(episode.actions):
                before, _ = live_state(env)
                if control_mode == 'recorded_joint_targets':
                    targets = episode.joint_targets
                    returned = control.step(action, targets['next_physx_substep_robot_joint_target'][index],
                                            targets['next_physx_substep_robot_joint_target_valid'][index])
                else:
                    returned = control.step(action)
                _, reward, terminated, truncated, info = returned
                after, _ = live_state(env)
                tick = int(env._sim.current_time_step_index)
                require(tick - last_tick == 4, 'Physical action clock changed')
                last_tick = tick
                if info.get('succeed', False) and first_success is None:
                    first_success = index + 1
                if 'historical_far_side_threshold_m' in episode.manifest:
                    progress = float(info['ycb01_ball_gate_local'][1])
                    if progress >= episode.manifest['historical_far_side_threshold_m'] and first_historical_threshold is None:
                        first_historical_threshold = index + 1
                snapshot = env._physx_ground_truth_trace.snapshot()
                physical.append({k: snapshot[k] for k in ['physx_tcp_state', 'physx_task_rigid_body_state',
                                                         'physx_robot_joint_position', 'physx_robot_joint_target']})
                row = plain(dict(action_index=index, input_action=action, state_before=before,
                    state_after=after, reward=reward, terminated=terminated, truncated=truncated,
                    info=info, objects={k: v.get_transforms() for k, v in env._task_body_views.items()},
                    physics_tick=tick))
                trajectory.write(json.dumps(row) + '\n')
                trajectory.flush()
                stop = should_stop(terminated, truncated, info, complete_recording)
                if (index + 1) % render_stride == 0 or terminated or truncated or index == len(episode.actions) - 1 or stop:
                    render_probe(env, job, out, index + 1, episode.manifest['reset_seed'], 'step', capture)
                if index % 20 == 0:
                    phase('replaying', completed_actions=index + 1, requested_actions=len(episode.actions))
                if stop:
                    break
        write('FINAL_STATE.json', capture(env))
        arrays = {key: np.stack([row[key] for row in physical]) for key in physical[0]}
        np.savez_compressed(out / 'physical_readback.npz', **arrays)
        errors = None
        if episode.reference is not None:
            errors = {}
            for key in ['physx_tcp_state', 'physx_task_rigid_body_state']:
                recorded = episode.reference['next_' + key][:len(physical)]
                distance = np.linalg.norm(arrays[key][:, :, :3] - recorded[:, :, :3], axis=-1)
                errors[key] = dict(position_rms_m=float(np.sqrt(np.mean(distance ** 2))),
                                   position_max_m=float(distance.max()))
        result = dict(episode=episode_id, task=task, completed_actions=index + 1,
            paper_variant=variant, control_mode=control_mode,
            uses_original_substep_targets=control_mode == 'recorded_joint_targets',
            recorded_target_substeps=getattr(control, 'applied_substeps', 0),
            recorded_target_max_abs_error=getattr(control, 'max_target_error', None),
            historical_state_errors=errors, exact_historical_state_replay_certified=False,
            requested_actions=len(episode.actions), native_success=first_success is not None,
            first_native_success_step=first_success, first_historical_geometry_step=first_historical_threshold,
            historical_checker_certified=False, formal_benchmark_trial=False,
            continued_after_native_success=first_success is not None and index + 1 > first_success,
            initial_eef_max_abs_error=initial_error, task_state_restores=1,
            physical_control_hz=30, physics_dt=env._physics_dt, substeps_per_action=env._n_substeps,
            wall_seconds_actions_and_render=time.monotonic() - start, terminal=row)
        write('RESULT.json', result)
        phase('completed', completed_actions=index + 1, native_success=result['native_success'])
        return result
    except BaseException as exc:
        write('ERROR.json', dict(error=repr(exc), traceback=traceback.format_exc(), formal_benchmark_trial=False))
        raise
    finally:
        camera_randomization.sample = original_camera_sample
        if wrapped is not None:
            if isolated_worker and variant == 'stabilize_body':
                # This legacy base already provides an explicit Kit shutdown
                # path for RTX workers: SimulationContext.stop() can block
                # after capture. Use it only in the CLI's owned child process,
                # after writing the result/error and closing trajectory files.
                # Direct Python callers retain their normal close semantics.
                wrapped.unwrapped._fast_shutdown_on_close = True
            wrapped.close()
