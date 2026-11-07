"""CPU checks for per-task selection and interrupted-frame recovery."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

CODE = Path(__file__).resolve().parent


class ReleaseTests(unittest.TestCase):
    def test_selection_is_per_task_complete_and_tie_stable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inventory = []
            for task in range(1, 51):
                # Reverse inventory order must not affect the tie at the cutoff.
                for name, steps in [('c', 90), ('b', 60), ('a', 60)]:
                    episode = f'episode_{name}'
                    p = root / 'tasks' / f'{task:02d}' / episode
                    p.mkdir(parents=True)
                    meta = dict(duration_seconds=steps/30, step_count=steps, data_fps=30,
                        hand_type='wuji2', sim_backend='isaacsim_newton', task_name=f'task{task}',
                        dataset_generation='nvidia_newton_cross_gpu_v2' if task == 9 else 'nvidia_newton_v1',
                        task_outcome='success')
                    (p / 'collection_meta.json').write_text(json.dumps(meta))
                    with zipfile.ZipFile(p / 'replay.zarr.zip', 'w') as z:
                        z.writestr('data/action/.zarray', json.dumps({'shape': [steps, 27]}))
                    inventory.append(dict(task_id=f'{task:02d}', prefix=f'cloud/{episode}/', bytes=1, object_count=2))
            (root / 'episodes.json').write_text(json.dumps(inventory))
            subprocess.run([sys.executable, str(CODE / 'select_shortest.py'), '--source', str(root),
                '--output', str(root / 'out'), '--per-task', '2'], check=True, capture_output=True)
            rows = [json.loads(s) for s in (root / 'out/selection.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 100)
            for task in range(1, 51):
                selected = [r for r in rows if r['task_id'] == f'{task:02d}']
                self.assertEqual([r['episode_id'] for r in selected], ['episode_a', 'episode_b'])
                self.assertEqual([r['step_count'] for r in selected], [60, 60])

    def test_interruption_recovers_exact_frame_order_and_rejects_changed_contract(self):
        import numpy as np
        actions = np.arange(37 * 2).reshape(37, 2)
        states = np.arange(37 * 3).reshape(37, 3)
        calls = []
        attempt = [0]
        def capture(a, initial, s, *args, **kw):
            start = int(a[0, 0] // 2)
            calls.append(start)
            self.assertTrue(np.array_equal(initial, s[0]))
            result = []
            for offset, action in enumerate(a):
                if attempt[0] == 0 and offset == 7:
                    attempt[0] += 1
                    raise InterruptedError('injected preemption')
                t = {'actions': action, 'observations': {'rgb': np.full((2, 2, 3), start+offset, dtype=np.uint8)},
                     'infos': {'frame': start+offset}}
                kw['transition_callback'](t, {'main': 'rgb', 'wrist_left': 'rgb', 'wrist_right': 'rgb'})
                result.append(t)
            return False, result, {}, {}, {}
        renderer = types.ModuleType('scripts.isaac_render_arm_replay')
        renderer._replay_single_demo = capture
        renderer.app = types.SimpleNamespace(run=lambda fn: None)
        renderer.main = lambda: None
        renderer.CANONICAL_CAMERA_KEYS = ('main', 'wrist_left', 'wrist_right')
        renderer.observation_image = lambda obs, key: obs[key]
        renderer._safe_rgb = lambda x: x
        renderer.postprocess_canonical_image = lambda x, *a: x
        scripts = types.ModuleType('scripts')
        scripts.isaac_render_arm_replay = renderer
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sample = root / 'episode'
            sample.mkdir()
            (sample / 'replay.zarr.zip').write_bytes(b'fixture')
            (root / 'assets/glove').mkdir(parents=True)
            (root / 'assets/glove/manifest.json').write_text('{}')
            (root / 'profile.json').write_text('{}')
            (root / 'cloth.py').write_text('# fixture')
            env = dict(HANDSON_WUJI2_RUNTIME=str(root), HANDSON_WUJI2_EPISODE=str(sample),
                HANDSON_RENDER_MAX_FRAMES='0', WUJI2_VISUAL_BUNDLE='glove',
                WUJI2_CLOTH_PROFILE='profile.json', WUJI2_CLOTH_MODULE='cloth', DEXJOCO_V16_PREVIEW_SPP='128')
            with patch.dict(os.environ, env), patch.dict(sys.modules, {'scripts': scripts, 'scripts.isaac_render_arm_replay': renderer}):
                replay = runpy.run_path(str(CODE / 'render_checkpointed.py'))['replay']
                with self.assertRaises(InterruptedError):
                    replay(actions, states[0], states, desc='test')
                result = replay(actions, states[0], states, desc='test')[1]
                self.assertEqual([t['infos']['frame'] for t in result], list(range(37)))
                self.assertEqual(calls, [0, 7])
                self.assertEqual(len(replay(actions, states[0], states, desc='test')[1]), 37)
                self.assertEqual(calls, [0, 7])
                (root / 'profile.json').write_text('{"changed": true}')
                with self.assertRaises(AssertionError):
                    replay(actions, states[0], states, desc='test')


if __name__ == '__main__':
    unittest.main()
