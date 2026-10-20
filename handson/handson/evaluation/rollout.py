"""One shared rollout: native old executor, fresh RGB, native task outcomes."""
import json
import math
from pathlib import Path
import time

from .contracts import require, write_new
from ..eef52 import live_state, command_readback, EEF52Control
from .evidence import Videos, native_return, json_safe
from .schedule import RenderSchedule
from .policy import PolicyClient
from .history import ImageHistory
from .readback import physics_tick
from ..tasks.registry import TASKS


def terminal_reason(task, returned, c12_contact=False):
    return TASKS[task].evaluation_adapter().terminal_reason(returned, c12_contact)


def task_setting_contract(job):
    """Require an approved B setting to be applied before the native reset.

    The common rollout cannot safely mutate a task after reset.  A B task job
    therefore carries a complete selection receipt produced by the task
    builder.  This guard prevents silently falling back to a default setting.
    """
    task = job['task']
    from . import settings as b_task_config
    if task not in b_task_config.task_ids():
        return None
    selection = job.get('setting_selection')
    require(isinstance(selection, dict),
            f'B task {task} requires setting_selection before reset')
    require(selection.get('task') == task,
            f'B setting task mismatch: {selection.get("task")} != {task}')
    require(job.get('setting_applied_before_reset') is True,
            f'B task {task} setting was not applied before reset')
    readback = job.get('setting_runtime_verified', {})
    require(readback.get('verified') is True and readback.get('task') == task
            and readback.get('setting_id') == selection['setting_id'],
            'B setting has no actual environment readback')
    expected = b_task_config.select_setting(
        int(selection['master_seed']), task, int(selection['episode_index']),
        forced_setting=selection['setting_id'],
    )
    require(expected['setting_id'] == selection['setting_id'],
            f'B setting id is not approved for {task}')
    require(expected['profile'] == selection['profile'],
            f'B setting profile drift for {task}/{selection["setting_id"]}')
    return selection


def run(env, job, output, number, seed, capture):
    import numpy as np
    from PIL import Image
    from .cameras import render_probe
    adapter = TASKS[job['task']].evaluation_adapter()
    control = EEF52Control(env)
    cfg = job['policy_probe']
    spec = job['benchmark']
    horizon = int(cfg.get('action_horizon',30))
    schedule = RenderSchedule(spec.get('render_stride',30), horizon, round(1/env._control_dt))
    history = ImageHistory(cfg.get('history_frames', 1), cfg.get('history_stride', schedule.stride),
                           horizon, schedule.stride)
    require(spec['executor'] == 'legacy_native_eef' and job.get('source_execution') is None,
            'Unified benchmark must execute native EEF commands')
    bound = int(math.ceil(env._time_limit/env._control_dt)) - int(env.env_step)
    require(bound > 0 and int(env.env_step) == 0, 'Policy must start at the native recording boundary')
    smoke = int(spec.get('smoke_control_steps', 0))
    native_bound = bound
    budget = job.get('evaluation_budget', {})
    original_limit = budget.get('native_time_limit_s', env._time_limit)
    original_bound = int(math.ceil(original_limit / env._control_dt))
    if smoke:
        require(not spec.get('formal', False) and smoke >= 90,
                'A bounded runtime smoke cannot be a formal benchmark score')
        bound = min(bound, smoke)
    require(cfg['max_chunks'] >= math.ceil(bound/horizon), 'Checkpoint request budget too small')
    folder = output/f'policy_{number:03d}'; folder.mkdir(exist_ok=False)
    setting_selection = task_setting_contract(job)
    setting_seeds = (setting_selection or {}).get('seeds', {})
    camera_seed, policy_seed = setting_seeds.get('camera', seed), setting_seeds.get('policy', seed)
    frames = {role:np.asarray(Image.open(output/f'initial_{number:03d}_{role}.png').convert('RGB'))
              for role in ('main','wrist_left','wrist_right')}
    start = time.monotonic()
    timing = dict(policy_start=0., inference=0., physics=0., render=0., video=0., audit=0.)
    tick0 = physics_tick(env)
    policy, videos = PolicyClient(cfg, folder), None
    executed, frame_steps, returned, reason = 0, [0], None, None
    write_new(folder/'PROTOCOL.json', dict(**spec, cadence=schedule.receipt(),
        seed=seed, prompt=cfg['prompt'], native_time_limit=original_limit,
        evaluated_time_limit=env._time_limit, evaluation_budget=budget,
        action_path='decode_command -> env.step', control_dt=env._control_dt,
        physics_dt=env._physics_dt, no_future_states_or_reference_actions=True,
        setting_selection=setting_selection,
        setting_applied_before_reset=job.get('setting_applied_before_reset', False)))
    if setting_selection is not None:
        write_new(folder/'TASK_SETTING_RECEIPT.json', setting_selection)
    try:
        before = time.monotonic(); metadata = policy.start()
        timing['policy_start'] += time.monotonic()-before
        videos = Videos(folder, frames, fps=1/(env._control_dt*schedule.stride), ffmpeg=cfg['ffmpeg'])
        before = time.monotonic(); videos.write(frames); timing['video'] += time.monotonic()-before
        with (folder/'steps.jsonl').open('x') as events:
            for request_index in range(math.ceil(bound/horizon)):
                state, measured = live_state(env)
                before = time.monotonic()
                actions = policy.predict(state, history.views(frames, request_index), request_index, policy_seed+request_index)
                timing['inference'] += time.monotonic()-before
                for action_index, action in enumerate(actions):
                    tick = physics_tick(env); before = time.monotonic()
                    returned = native_return(control.step(action))
                    command = control.last_command
                    live_state(env)  # Validate finite measured state, including terminal actions.
                    timing['physics'] += time.monotonic()-before; executed += 1
                    require(physics_tick(env)-tick == env._n_substeps,
                            'Wrong physical ticks per command')
                    reason = adapter.terminal_reason(returned, env=env)
                    if executed == 1:
                        received = command_readback(env, command)
                        write_new(folder/'FIRST_COMMAND_RECEIVED.json', received)
                        require(all(v['position_max_abs']<1e-10 and v['hand_max_abs']<1e-10
                                    and v['rotation_geodesic_rad']<1e-10 for v in received.values()),
                                'Native high-level command readback changed')
                    before = time.monotonic()
                    events.write(json.dumps(json_safe(dict(step=executed, request=request_index,
                        action_index=action_index, action=action, native=returned)),allow_nan=False)+'\n')
                    timing['audit'] += time.monotonic()-before
                    if schedule.capture(executed) or reason or executed == bound:
                        def receive(images):
                            frames.update({k:v.copy() for k,v in images.items()})
                            history.record(executed, images)
                        before = time.monotonic()
                        render_probe(env,job,output,number,camera_seed,f'frame_{executed:05d}',capture,
                            save_png=executed==bound or bool(reason),image_sink=receive)
                        timing['render'] += time.monotonic()-before
                        before = time.monotonic(); videos.write(frames)
                        timing['video'] += time.monotonic()-before; frame_steps.append(executed)
                    if reason or executed == bound:
                        break
                events.flush()
                print(json.dumps(dict(task=job['task'],steps=executed,seconds=time.monotonic()-start,
                    succeed=returned['info'].get('succeed'),stop=reason)),flush=True)
                if reason or executed == bound:
                    break
        video_receipt = videos.close(); videos = None
        require(all(v['exit_code']==0 for v in video_receipt.values()), 'Video encoder failed')
        require(physics_tick(env)-tick0 == executed*env._n_substeps,
                'Unexpected physics advance during rendering/inference')
        # Native reset wrapper uses `actions` for its physics-tick accounting.
        # Retain `steps` for the shared benchmark result schema, with the exact
        # same executed-command count; this does not change a rollout/outcome.
        result = dict(task=job['task'], seed=seed, steps=executed, actions=executed,
            success=bool(returned['info'].get('succeed',False)),
            native_return=returned,
            stop_reason=reason or ('diagnostic_budget' if bound < native_bound else 'native_horizon'),
            complete_native_episode=bool(reason) or executed == native_bound,
            complete_evaluation_episode=bool(reason) or executed == native_bound,
            complete_original_native_episode=bool(
                returned['terminated'] and executed <= original_bound
                or returned['info'].get('time_limit_reached') and native_bound == original_bound),
            timing_seconds=timing, wall_seconds=time.monotonic()-start,
            physics_steps=physics_tick(env)-tick0,
            physical_seconds=(physics_tick(env)-tick0)*env._physics_dt,
            frame_steps=frame_steps, videos=video_receipt, policy_metadata=metadata,
            source_commit=job.get('source_revision', 'packaged'),
            counts_as_formal_trial=bool(spec.get('formal',False)),
            physics_backend=spec.get('physics_backend', 'physx'),
            paper_physics_protocol=spec.get('paper_physics_protocol', True),
            protocol='handson_native_eef52', full_alignment_accepted=False)
        write_new(folder/'RESULT.json',result)
        return result
    finally:
        if videos is not None:
            videos.close()
        policy.close()
