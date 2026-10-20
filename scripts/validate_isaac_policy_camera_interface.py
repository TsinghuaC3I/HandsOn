#!/usr/bin/env python3
"""Fresh-process three-camera preview of the public TaskConfig interface.

This checks real camera construction and exported frames. It is not a claim of
successful collection, deferred video completion, or physical replay.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import traceback
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2303)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert importlib.metadata.version('isaacsim').startswith('5.1.')
    os.environ['HANDSON_ISAACLAB_DEFER_CAMERA_RENDER'] = '0'
    os.environ['HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY'] = '0'
    os.environ['HANDSON_ENABLE_ISAACSIM_NEWTON'] = '0'
    config = importlib.import_module(f'handson.tasks.{args.task}.config').TaskConfig()
    report = {'scope': 'fresh_process_real_policy_camera_preview', 'task': args.task,
              'seed': args.seed, 'pass': False, 'success_demonstration': False,
              'isaacsim': importlib.metadata.version('isaacsim')}
    env = None
    try:
        env = config.get_environment(policy_mode=True, launch_isaac=True,
                                     image_obs=True, randomize=True, render_mode='human')
        obs, info = env.reset(seed=args.seed)
        raw = env.unwrapped
        assert not raw._deferred_camera_render and not raw._live_main_camera_only
        report['camera_handles'] = sorted(raw._cameras)
        assert set(raw._cameras) == {'head_camera', 'wrist_left_camera', 'wrist_right_camera'}
        for _ in range(20):
            raw_obs, _, terminated, truncated, info = raw.step_with_current_joint_targets()
            assert not terminated and not truncated, str(info)
            obs = env.observation(raw_obs)
        images = {k: np.asarray(v) for k, v in obs.items() if np.asarray(v).ndim == 3 and np.asarray(v).shape[-1] == 3}
        main = 'front'
        assert env.action_space.shape == (52,) and obs['state'].shape == (52,)
        assert {main, 'wrist_left', 'wrist_right'} <= images.keys(), sorted(images)
        from PIL import Image
        report['images'] = {}
        for name in (main, 'wrist_left', 'wrist_right'):
            frame = images[name]
            assert frame.dtype == np.uint8 and min(frame.shape[:2]) > 32
            assert float(frame.std()) > 2.0 and np.unique(frame.reshape(-1,3),axis=0).shape[0] > 32
            assert env.observation_space[name].contains(frame), (name, frame.shape)
            Image.fromarray(frame).save(args.output_dir / f'{name}.png')
            report['images'][name] = {'shape': list(frame.shape), 'std': float(frame.std()),
                                      'sha256': hashlib.sha256(frame.tobytes()).hexdigest()}
        assert len({v['sha256'] for v in report['images'].values()}) == 3, 'camera views must differ'
        np.savez_compressed(args.output_dir / 'policy_images.npz', **{k:images[k] for k in report['images']})
        report['pass'] = True
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        (args.output_dir / 'camera_report.json').write_text(json.dumps(report, indent=2)+'\n')
        print('POLICY_CAMERA_REPORT', report['pass'], args.output_dir, flush=True)
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
