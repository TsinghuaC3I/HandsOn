"""Explicit physics capabilities for the shared paper-camera/EEF52 evaluator."""
from __future__ import annotations

import os

from .contracts import require, write_new


NEWTON_TASKS = frozenset({"t09_insert_peg", "t26_place_fruit_in_bucket"})
BACKENDS = ("physx", "newton")


def supported_tasks(backend):
    from ..tasks.registry import TASKS
    if backend not in BACKENDS:
        raise ValueError(f"Unknown physics backend: {backend}")
    return tuple(sorted(TASKS if backend == "physx" else NEWTON_TASKS))


def validate_task(task, backend):
    if task not in supported_tasks(backend):
        raise ValueError(f"{task} is not supported by the {backend} evaluator; "
                         "use --list --physics-backend " + backend)


def configure(backend):
    """Select tracing before construction; Newton uses its own native reset."""
    if backend == "newton":
        os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "0"
        # Evaluation audits the live manager state below. The collection trace
        # records every substep and assumes a separate recording lifecycle.
        os.environ["HANDSON_RECORD_NEWTON_GROUND_TRUTH"] = "0"
        os.environ["HANDSON_STABLE_RECORDING_ORIGIN"] = "0"
        from .newton_runtime import pin_warp
        pin_warp()
    else:
        os.environ["HANDSON_RECORD_NEWTON_GROUND_TRUTH"] = "0"
        os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
        os.environ["HANDSON_STABLE_RECORDING_ORIGIN"] = "1"


def capture_newton(env):
    """Read actual Newton state; never reuse a PhysX tensor/reset snapshot."""
    import numpy as np
    from .readback import plain, physics_tick
    manager = env._sim.physics_manager
    state, model = manager.get_state_0(), manager.get_model()
    require(state is not None and model is not None, "Newton state was not initialized")
    arrays = {}
    for name in ("body_q", "body_qd", "joint_q", "joint_qd", "particle_q", "particle_qd"):
        value = getattr(state, name, None)
        if value is not None:
            value = value.numpy().copy()
            require(np.isfinite(value).all(), "Nonfinite Newton state: " + name)
            arrays[name] = value.tolist()
    require("body_q" in arrays and "joint_q" in arrays, "Incomplete Newton state")
    return plain(dict(
        physics_backend="newton", step=env.env_step,
        physics_tick=physics_tick(env),
        control_dt=env._control_dt, physics_dt=env._physics_dt,
        substeps=env._n_substeps,
        native_state=arrays, body_names=list(model.body_label),
        state_source="live_newton_manager",
        solver_settings=env._newton_solver_settings,
        recorded_state_used=False,
    ))


def record_newton_reset(env, output, reset_info):
    from .eef52 import live_state
    from .readback import plain
    state, measured = live_state(env)
    require(env.env_step == 0, "Newton reset consumed policy actions")
    receipt = dict(physics_backend="newton", reset="native_environment_reset",
                   paper_physx_origin_applied=False, state=state.tolist(),
                   measured=measured, native_reset_info=plain(reset_info))
    write_new(output / "NEWTON_RESET.json", receipt)
    return receipt


def runtime_receipt(env, backend):
    import importlib.metadata
    names = ("isaacsim", "isaaclab", "isaaclab-newton", "newton", "warp-lang", "mujoco-warp")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    require(env._physics_backend == backend, "Requested and actual physics backends differ")
    return dict(physics_backend=backend, hand_type=env.hand_type,
                rendering="handson", action_contract="right26_left26_world_palm",
                runtime_quaternion_order="xyzw" if getattr(env, "_isaaclab_uses_xyzw", False) else "wxyz",
                graphics_transform_sync="newton_fabric_cpu_hierarchy" if backend == "newton" else "native_physx",
                versions=versions, paper_physics_protocol=backend == "physx",
                control_dt=env._control_dt, physics_dt=env._physics_dt,
                n_substeps=env._n_substeps)
