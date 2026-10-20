#!/usr/bin/env python3
"""Report which task DOF views a live Isaac task exposes (diagnostic only)."""

from __future__ import annotations

import argparse
import json
import os
import threading

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("HANDSON_ENABLE_ISAACSIM_NEWTON", "0")
os.environ.setdefault("HANDSON_RECORD_PHYSX_GROUND_TRUTH", "1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--variant", default="baseline")
    args = parser.parse_args()
    from handson.sim.isaaclab.d_long_horizon_scene import D_LONG_TASK_NAMES

    if args.task in D_LONG_TASK_NAMES:
        from handson.sim.isaaclab.d_long_horizon_env import (
            HandsOnDLongHorizonIsaacLabEnv as Env,
        )
    else:
        from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv as Env
    # Diagnostic bisection variants (D20 only): disable one D20-specific
    # pre-PhysX customization at a time to find what breaks the tongs
    # articulation.
    import handson.sim.isaaclab.abc_collection_env as abc_mod
    import handson.sim.isaaclab.d_long_horizon_env as dl_mod
    if args.variant == "no_stage_config":
        dl_mod.HandsOnDLongHorizonIsaacLabEnv._d20_configure_stage = lambda self, sim_utils: None
    elif args.variant == "no_fixtures":
        original = abc_mod.HandsOnABCIsaacLabEnv._c_tool_fixture_names
        def _no_fixtures(task_name, *a, **k):
            return () if task_name == args.task else original(task_name, *a, **k)
        abc_mod.HandsOnABCIsaacLabEnv._c_tool_fixture_names = staticmethod(_no_fixtures)
    elif args.variant == "no_stability":
        dl_mod._DLONG_RESET_STABILITY_BODIES.pop(dl_mod._D20, None)
    elif args.variant == "no_complete_hand":
        abc_mod._COMPLETE_EXTERNAL_HAND_COLLISION_TASKS = frozenset(
            abc_mod._COMPLETE_EXTERNAL_HAND_COLLISION_TASKS - {args.task}
        )
    elif args.variant == "no_runtime_init":
        dl_mod.HandsOnDLongHorizonIsaacLabEnv._d20_init_runtime = lambda self, hand_paths: None
    print("DOF_PROBE_VARIANT", args.variant, flush=True)
    env = Env(
        args.task,
        seed=3,
        randomize=False,
        image_obs=False,
        render_mode="rgb_array",
        hz=30,
        physics_dt=1.0 / 120.0,
        physics_backend="physx",
        headless=True,
    )
    try:
        env.reset(seed=3)
        report = {
            "task": args.task,
            "variant": args.variant,
            "dof_views": sorted(getattr(env, "_task_dof_views", {}) or {}),
            "articulation_states": [
                str(item[0]) for item in getattr(env, "_task_reset_articulation_states", [])
            ],
            "body_views": sorted(getattr(env, "_task_body_views", {}) or {}),
            "warnings": list(getattr(env, "_isaac_warnings", []) or []),
        }
        print("DOF_PROBE_RESULT " + json.dumps(report), flush=True)
    finally:
        threading.Timer(60.0, lambda: os._exit(0)).start()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
