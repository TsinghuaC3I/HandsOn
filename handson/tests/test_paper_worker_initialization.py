"""Compare initialization with the recovered paper worker and scene helper."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from handson.evaluation import task_contracts
from handson.tasks.registry import TASKS


ORACLE = json.loads((Path(__file__).parent / 'fixtures' /
                     'paper_worker_initialization.json').read_text())


class PaperWorkerInitializationTests(unittest.TestCase):
    def test_task_environment_matches_archived_worker_with_external_overrides(self):
        original = ORACLE['environment']['statements']
        original = original.replace('DEXJOCO_', 'HANDSON_')
        contaminated = {
            'PATH': '/retained/system/path',
            'CUDA_VISIBLE_DEVICES': 'retained-device',
            'WORKSPACE_DIR': '/retained/workspace',
            'HANDSON_ISAACLAB_ARM_IK_BACKEND': 'haorui_new',
            'HANDSON_PHYSX_MAX_POSITION_ITERS': '1',
            'HANDSON_BACKGROUND_MODE': 'random',
            'HANDSON_ISAACLAB_ARM_WORKSPACE_MAX_RADIUS': '0.01',
            'HANDSON_UNRECORDED_OVERRIDE': 'must not leak',
        }
        for task in TASKS:
            with self.subTest(task=task):
                contract = task_contracts.load(task)
                expected = dict(contaminated)
                exec(original, {'os': SimpleNamespace(environ=expected),
                                'job': {'recorded_control_environment':
                                        (contract or {}).get('control_environment', {})}})
                with patch.dict(os.environ, contaminated, clear=True):
                    task_contracts.configure(task)
                    self.assertEqual(dict(os.environ), expected)

    def test_external_ik_override_cannot_change_native_factory(self):
        from handson.tasks import CONFIG_MAPPING
        for task in ['t34_roll_ball_through_gate', 't50_water_plant',
                     'eval_gen06_spray_shoe']:
            with self.subTest(task=task), patch.dict(os.environ, {
                    'HANDSON_ISAACLAB_ARM_IK_BACKEND': 'haorui_new'}):
                task_contracts.configure(task)
                env = CONFIG_MAPPING[task]().get_environment(
                    policy_mode=True, launch_isaac=False, image_obs=False)
                try:
                    self.assertEqual(env.unwrapped._arm_ik_backend, 'dls')
                finally:
                    env.close()

    def test_spray_initial_visibility_preserves_archived_function(self):
        import handson
        original = ORACLE['spray_visual']['function']
        original = original.replace('_is_c04_water_plant_task', '_is_50_water_plant_task')
        original = original.replace('gen7_spray_shoe_care', 'eval_gen06_spray_shoe')
        source = Path(handson.__file__).parent / 'sim/isaaclab/task_preview.py'
        actual = next(n for n in ast.parse(source.read_text()).body
                      if isinstance(n, ast.FunctionDef) and n.name == '_hide_c04_spray_visual')
        self.assertEqual(ast.dump(actual), ast.dump(ast.parse(original).body[0]))


if __name__ == '__main__':
    unittest.main()
