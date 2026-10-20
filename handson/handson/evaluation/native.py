"""Portable entry to the native factory, measured EEF52 boundary and rollout."""
from pathlib import Path
import json
import os
import shutil
import traceback

import numpy as np

from .contracts import require, write_new
from .eef52 import live_state
from .evidence import json_safe
from ..tasks.registry import TASKS


def construct(task, seed=0, *, launch=True, selection=None, physics_backend="physx"):
    from .backends import validate_task
    validate_task(task, physics_backend)
    if launch:
        from handson.robots.assets import require_benchmark_assets
        require_benchmark_assets()
    from handson.tasks import CONFIG_MAPPING
    config = CONFIG_MAPPING[task]()
    # The paper evaluator constructs the task's own policy wrapper, then steps
    # the unwrapped native environment with named right/left 27-value commands.
    from .task_contracts import load
    contract = load(task) if physics_backend == "physx" else None
    kwargs = dict(contract['constructor']) if contract else {}
    kwargs.update(seed=seed, randomize=True, image_obs=False, hand_type="wuji1",
                  headless=True, launch_isaac=launch, isaac_device="cuda:0", control_dt=1 / 30,
                  wrist_camera_output_size=(848, 480))
    if selection is not None:
        kwargs['physics_dt'] = selection['physics']['physics_dt']
    wrapped = config.get_native_environment(policy_mode=True, render_mode="rgb_array",
                                     sim_backend="isaacsim" if physics_backend == "physx" else "isaacsim_newton", **kwargs)
    env = wrapped.unwrapped
    env = TASKS[task].evaluation_adapter().native_view(env, launch)
    require(env.task_name == task, "Factory produced the wrong native task")
    require(abs(env._control_dt - 1 / 30) < 1e-12, "Control rate differs from 30 Hz")
    require(abs(env._n_substeps * env._physics_dt - 1 / 30) < 1e-12,
            "Physical action rate differs from 30 Hz")
    return env, wrapped, config


def capture(env):
    from .readback import capture as native_capture
    from .backends import capture_newton
    state, evidence = live_state(env)
    readback = capture_newton if env._physics_backend == "newton" else native_capture
    return dict(readback(env), state=state.tolist(), eef=evidence)


def check_task(task, physics_backend="physx"):
    env, wrapped, config = construct(task, launch=False, physics_backend=physics_backend)
    try:
        return dict(task=task, environment=type(env).__name__, hand_type=env.hand_type,
                    physics_backend=env._physics_backend,
                    control_dt=env._control_dt, physics_dt=env._physics_dt,
                    n_substeps=env._n_substeps, native_time_limit=env._time_limit,
                    action_contract="right26_left26_world_palm", state_dim=52,
                    cameras=["main", "wrist_left", "wrist_right"],
                    scope="CPU factory only; no reset, rendering or policy execution")
    finally:
        wrapped.close()


def run_episode(task, output, *, seed=0, policy=None, diagnostic_steps=0, render_stride=30,
                physics_backend="physx", max_policy_steps=0):
    from . import backends
    backends.validate_task(task, physics_backend)
    require(not (diagnostic_steps and max_policy_steps), "Choose hold diagnostics or bounded policy execution")
    require(not max_policy_steps or policy is not None, "Bounded policy execution requires a policy")
    from handson.robots.assets import require_benchmark_assets
    require_benchmark_assets()
    from ..sim.isaaclab.default_appearance import verify_environment
    from .cameras import render_probe
    from .rollout import run

    adapter = TASKS[task].evaluation_adapter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    require(policy is not None or diagnostic_steps >= 90,
            "Provide a policy or request a diagnostic of at least 90 actions")
    from .task_contracts import configure, initialize, install_native_trace
    task_contract = configure(task) if physics_backend == "physx" else None
    backends.configure(physics_backend)
    os.environ["HANDSON_BACKGROUND_MODE"] = "off"
    # The original evaluator gives each attempt its own conversion workspace.
    # Included MJCF files share basenames across tasks and cannot be rewritten
    # concurrently in a common cache.
    os.environ["HANDSON_ISAACLAB_CACHE"] = str((output / "scene-cache").resolve())
    os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "0"
    os.environ["HANDSON_ISAACLAB_REQUIRE_REAL_RECORDING_CAMERAS"] = "1"
    env = wrapped = None
    try:
        from .cameras import install_native_camera_creation, install_launcher
        install_launcher(physics_backend)
        from ..sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
        install_native_camera_creation(HandsOnABCIsaacLabEnv)
        if physics_backend == "physx":
            adapter.prepare_cameras()
            install_native_trace(task, output)
            adapter.prepare(output)
        from .setting_runtime import prepare, configure_solver, verify_b_setting
        selection = prepare(task, seed, output) if physics_backend == "physx" else None
        if physics_backend == "physx":
            configure_solver(selection)
        reset_seed = selection["seeds"]["reset"] if selection else seed
        env, wrapped, config = construct(task, reset_seed, selection=selection, physics_backend=physics_backend)
        require(env._sim is not None, "Isaac Sim did not start")
        if not env._cameras:
            env._init_recording_cameras()
        observation, reset_info = env.reset(seed=reset_seed)
        if physics_backend == "physx":
            observation, reset_info = initialize(
                env, wrapped, observation, reset_info, task_contract, output)
        else:
            backends.record_newton_reset(env, output, reset_info)
        write_new(output / "PHYSICS_BACKEND.json", backends.runtime_receipt(env, physics_backend))
        if task_contract is not None:
            from .readback import verify_native_control
            write_new(output / 'CONTROL_CONTRACT.json',
                      verify_native_control(env, task_contract['runtime_control']))
        require(env.env_step == 0, "Policy must start at the native recording boundary")
        write_new(output / "RECORDING_ORIGIN.json", json_safe(reset_info))
        from .goals import bind
        instruction = bind(env, task, seed)
        prompt = instruction["prompt"]
        write_new(output / "PROMPT.json", instruction)
        budgets = json.loads(Path(__file__).with_name("budgets.json").read_text())
        require(task in budgets, "Task has no published evaluation budget")
        budget = budgets[task]
        original_limit = float(env._time_limit)
        env._time_limit = int(budget["steps"]) * env._control_dt
        write_new(output / "BUDGET.json", dict(budget, native_time_limit=original_limit,
                   evaluated_time_limit=env._time_limit, control_dt=env._control_dt,
                   physics_dt=env._physics_dt, n_substeps=env._n_substeps))
        ffmpeg = shutil.which("ffmpeg")
        require(ffmpeg is not None, "Install ffmpeg to record evaluation videos")
        cfg = dict(policy or {"transport": "hold"})
        cfg.update(prompt=prompt, action_horizon=int(cfg.get("action_horizon", 30)),
                   max_chunks=10000, ffmpeg=ffmpeg)
        job = dict(task=task, policy_probe=cfg, benchmark=dict(
            executor="legacy_native_eef", formal=policy is not None and not max_policy_steps, render_stride=render_stride,
            physics_backend=physics_backend, paper_physics_protocol=physics_backend == "physx",
            smoke_control_steps=diagnostic_steps or max_policy_steps), source_revision="packaged")
        job["evaluation_budget"] = dict(budget, native_time_limit_s=original_limit)
        if selection is not None:
            job.update(setting_selection=selection, setting_applied_before_reset=True,
                       setting_runtime_verified=verify_b_setting(env, selection))
        adapter.configure_job(job)
        write_new(output / "JOB.json", job)
        initial = capture(env)
        write_new(output / "STATE_CONTRACT.json", initial)
        camera_seed = selection["seeds"]["camera"] if selection else seed
        render_probe(env, job, output, 0, camera_seed, "initial", capture)
        result = adapter.run(run, env, job, output, 0, seed, capture)
        write_new(output / "FINAL_STATE.json", capture(env))
        verify_environment(env)
        write_new(output / "RENDERING.json", env.default_appearance_report)
        result.update(diagnostic_only=policy is None, complete_evaluation_episode=(
            policy is not None and result["complete_native_episode"]), evaluation_budget=budget)
        if policy is None:
            result["native_success_during_hold"] = result.pop("success")
            result["success"] = None
            result["counts_as_formal_trial"] = False
        write_new(output / "RESULT.json", result)
        return result
    except Exception as exc:
        write_new(output / "ERROR.json", dict(error=repr(exc), traceback=traceback.format_exc(),
                                              counts_as_model_failure=False,
                                              diagnostics=json_safe(getattr(exc, "diagnostics", {}))))
        raise
    finally:
        if wrapped is not None:
            wrapped.close()
