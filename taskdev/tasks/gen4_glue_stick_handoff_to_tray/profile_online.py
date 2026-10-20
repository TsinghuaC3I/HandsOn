#!/usr/bin/env python3
"""Measure the ordinary full-physics, state-only GEN4 control loop."""
import argparse
import ast
import cProfile
import io
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys
import time
import traceback
import types

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    sys.argv = [sys.argv[0], '--portable-root', str(root / '.gen4-local/kit'),
                '--/log/file=' + str((args.output / 'kit.log').resolve())]
    os.environ.update(HANDSON_ENABLE_ISAACSIM_NEWTON='0',
                      HANDSON_ISAACLAB_DEFER_CAMERA_RENDER='1',
                      HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY='0',
                      HANDSON_STABLE_RECORDING_ORIGIN='1',
                      HANDSON_RECORD_PHYSX_GROUND_TRUTH='1')
    from handson.sim.isaaclab.gen4_env import HandsOnGEN4IsaacLabEnv
    env = HandsOnGEN4IsaacLabEnv(seed=args.seed, randomize=True, image_obs=False,
                                headless=True, render_mode='rgb_array',
                                isaac_device='cuda:0', arm_type='rokae', hand_type='wuji1')
    try:
        assert not env._cameras, 'Online profiling must not create canonical cameras'
        from handson.sim.isaaclab.recording_origin import (
            stabilize_recording_origin, stable_recording_origin_kwargs_for_task,
        )
        observation, info = env.reset(seed=args.seed)
        observation, info = stabilize_recording_origin(
            env, observation, info, **stable_recording_origin_kwargs_for_task(env.task_name),
        )
        assert info['physical_recording_origin']['final_boundary_physics_gate_passed']
        (args.output / 'origin.json').write_text(json.dumps(
            info['physical_recording_origin'], indent=2, default=lambda value: value.tolist(),
        ) + '\n')
        action = np.zeros(54)
        for _ in range(30):
            env.step(action)
        profile = cProfile.Profile()
        profile.enable()
        for _ in range(60):
            _, _, terminated, truncated, _ = env.step(action)
            assert not terminated and not truncated
        profile.disable()
        profile.dump_stats(str(args.output / 'online.prof'))
        for sort in ('cumulative', 'tottime'):
            output = io.StringIO()
            pstats.Stats(profile, stream=output).strip_dirs().sort_stats(sort).print_stats(60)
            (args.output / f'{sort}.txt').write_text(output.getvalue())
        # Interleave the unchanged release calculation and current calculation
        # in this diagnostic only. Neither changes any physics input or state.
        from handson.sim.isaaclab import physx_ground_truth as module
        source_path = 'handson/handson/sim/isaaclab/physx_ground_truth.py'
        baseline_source = subprocess.check_output(
            ['git', 'show', '282390f0fec08daca53e2d6b03f6ad8772b609f4:' + source_path],
            cwd=root, text=True,
        )
        tree = ast.parse(baseline_source)
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'PhysxGroundTruthTrace')
        method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '_canonicalize_contact_fields')
        namespace = dict(vars(module))
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<baseline-contact>', 'exec'), namespace)
        functions = {'baseline': namespace['_canonicalize_contact_fields'],
                     'current': module.PhysxGroundTruthTrace._canonicalize_contact_fields}
        trace = env._physx_ground_truth_trace
        windows = []
        for label in ('baseline', 'current', 'current', 'baseline', 'baseline', 'current'):
            trace._canonicalize_contact_fields = types.MethodType(functions[label], trace)
            start_time, start_sim = time.monotonic(), env._sim.current_time
            for _ in range(90):
                _, _, terminated, truncated, _ = env.step(action)
                assert not terminated and not truncated
            elapsed = time.monotonic() - start_time
            windows.append({'implementation': label, 'steps': 90, 'wall_seconds': elapsed,
                            'wall_hz': 90 / elapsed, 'simulation_seconds': env._sim.current_time - start_sim})
        del trace._canonicalize_contact_fields
        report = {'task': 'GEN4', 'completed': True, 'ordinary_origin_established': True,
                  'canonical_cameras': list(env._cameras), 'windows': windows,
                  'median_wall_hz': {label: float(np.median([x['wall_hz'] for x in windows if x['implementation'] == label]))
                                     for label in functions}}
        (args.output / 'profile.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    except BaseException:
        failure = traceback.format_exc()
        (args.output / 'failure.txt').write_text(failure)
        print(failure, flush=True)
        raise
    finally:
        env._fast_shutdown_on_close = True
        env.close()


if __name__ == '__main__':
    main()
