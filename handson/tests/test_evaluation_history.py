"""CPU rollout tests with a real HTTP transport and episode-owned history."""
import http.server
import io
import json
from pathlib import Path
import socketserver
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(CODE), str(Path(__file__).resolve().parent)]
from handson.evaluation import policy as benchmark_policy
from handson.evaluation import rollout as benchmark_rollout
from handson.evaluation import cameras as live_camera_probe
from handson.evaluation import history as entry

META = dict(model='hy', step=120000, checkpoint='/ckpt/fixture')


class Server:
    def __init__(self, drops=0, nan=False):
        self.requests, self.drops, self.nan = [], drops, nan
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.reply(META)

            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                if outer.drops:
                    outer.drops -= 1
                    self.close_connection = True
                    self.connection.close()
                    return
                d = np.load(io.BytesIO(body), allow_pickle=False)
                outer.requests.append({k: d[k] for k in d.files})
                actions = np.zeros((30, 52)) + int(d['seed'])
                if outer.nan and int(d['request']) == 1:
                    actions[3, 7] = np.nan
                self.reply(dict(actions=[[None if np.isnan(v) else v for v in row] for row in actions.tolist()]))

            def reply(self, value):
                data = json.dumps(value).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = socketserver.ThreadingTCPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.httpd.server_address[1]}'
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def value(step):
    return 200 if step == 0 else step // 10


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.saved = live_camera_probe.render_probe
        retry = patch.object(benchmark_policy, "RETRY_WAIT_S", 0.01)
        retry.start()
        self.addCleanup(retry.stop)

    def tearDown(self):
        live_camera_probe.render_probe = self.saved

    def trial(self, frames=6, stride=10, wire='front', drops=0, nan=False, seconds=4.0):
        server = Server(drops=drops, nan=nan)
        env = SimpleNamespace(_control_dt=1/30, _physics_dt=1/120, _time_limit=seconds, env_step=0,
                              _n_substeps=4, _sim=SimpleNamespace(current_time_step_index=0))

        def step(command):
            env.env_step += 1
            env._sim.current_time_step_index += 4
            return None, 0, env.env_step == 110, False, {'succeed': env.env_step == 110}
        env.step = step
        rendered = []

        def render(env_, job, output, number, seed, phase, capture, *, save_png=True, image_sink=None):
            rendered.append(phase)
            v = value(env_.env_step)
            images = {k: np.full((4, 5, 3), v + i, dtype=np.uint8) for i, k in enumerate(entry.RENDER_VIEWS)}
            image_sink(images)
        live_camera_probe.render_probe = render

        class Video:
            def __init__(self, *a, **kw): self.n = 0
            def write(self, frames): self.n += 1
            def close(self): return {k: dict(exit_code=0, frames=self.n) for k in entry.RENDER_VIEWS}
        render_stride = stride if frames > 1 else 30
        job = dict(task='t24_place_ball_in_bowl', source_execution=None,
                   reference_provenance={'recorded_git': {'commit': 'fixture'}},
                   benchmark={'executor': 'legacy_native_eef', 'render_stride': render_stride},
                   policy_probe={'max_chunks': 120, 'prompt': 'Pick the golf ball and place it into the bowl.',
                                 'ffmpeg': 'fixture', 'transport': 'http_physical_eef52', 'endpoint': server.url,
                                 'required_metadata': META, 'front_wire_key': wire, 'action_horizon': 30,
                                 'history_frames': frames, 'history_stride': stride})
        try:
            with tempfile.TemporaryDirectory() as td:
                folder = Path(td)
                for i, k in enumerate(entry.RENDER_VIEWS):
                    Image.fromarray(np.full((4, 5, 3), 200 + i, dtype=np.uint8)).save(folder/f'initial_000_{k}.png')
                with patch.object(benchmark_rollout, 'Videos', Video), \
                     patch.object(benchmark_rollout, 'live_state', lambda e: (np.full(52, e.env_step, np.float32), {})), \
                     patch.object(benchmark_rollout, 'command_readback', lambda e, c: {}):
                    try:
                        result = benchmark_rollout.run(env, job, folder, 0, 77, lambda e: {})
                        error = None
                    except Exception as exc:
                        result, error = None, exc
                files = sorted(p.name for p in (folder/'policy_000').iterdir())
            return dict(result=result, error=error, requests=server.requests, files=files, rendered=rendered)
        finally:
            server.close()

    def test_history_ring(self):
        t = self.trial()
        self.assertIsNone(t['error'])
        self.assertEqual(t['result']['steps'], 110)
        self.assertTrue(t['result']['success'])
        self.assertEqual(t['result']['frame_steps'], [0] + list(range(10, 111, 10)))
        req = t['requests']
        self.assertEqual([int(r['request']) for r in req], [0, 1, 2, 3])
        self.assertEqual([int(r['seed']) for r in req], [77, 78, 79, 80])
        self.assertEqual([float(r['state'][0]) for r in req], [0, 30, 60, 90])
        expect = {0: [0] * 6, 1: [0, 0, 0, 10, 20, 30], 2: [10, 20, 30, 40, 50, 60], 3: [40, 50, 60, 70, 80, 90]}
        for r in req:
            k = int(r['request'])
            for i, role in enumerate(('front', 'wrist_left', 'wrist_right')):
                self.assertEqual(r[role].shape, (6, 4, 5, 3))
                self.assertEqual(r[role].dtype, np.uint8)
                self.assertEqual([int(f[0, 0, 0]) for f in r[role]], [value(s) + i for s in expect[k]], (k, role))
            self.assertEqual(str(r['prompt']), 'Pick the golf ball and place it into the bowl.')
        self.assertFalse([f for f in t['files'] if f.endswith(('.npz', '.npy'))], t['files'])

    def test_single_frame_front_and_base(self):
        for wire in ('front', 'base'):
            t = self.trial(frames=1, stride=0, wire=wire)
            self.assertIsNone(t['error'])
            self.assertEqual(t['result']['frame_steps'], [0, 30, 60, 90, 110])
            for r in t['requests']:
                k = int(r['request'])
                self.assertEqual(r[wire].shape, (4, 5, 3))
                self.assertEqual(int(r[wire][0, 0, 0]), value(30 * k))
                self.assertEqual(int(r['wrist_right'][0, 0, 0]), value(30 * k) + 2)
                self.assertNotIn('front' if wire == 'base' else 'base', r)
            self.assertFalse([f for f in t['files'] if f.endswith(('.npz', '.npy'))])

    def test_retry_after_dropped_connections(self):
        clean, retried = self.trial(frames=1, stride=0), self.trial(frames=1, stride=0, drops=2)
        self.assertIsNone(retried['error'])
        self.assertEqual(clean['result']['steps'], retried['result']['steps'])
        self.assertEqual(len(retried['requests']), 4)

    def test_nonfinite_is_classifiable(self):
        t = self.trial(frames=1, stride=0, nan=True)
        self.assertIsNotNone(t['error'])
        self.assertIn("Model must return finite physical world EEF52 commands", repr(t['error']))

    def test_history_needs_matching_render_stride(self):
        with self.assertRaises(Exception):
            entry.ImageHistory(frames=6, stride=10, horizon=30, render_stride=30)


if __name__ == '__main__':
    unittest.main()
