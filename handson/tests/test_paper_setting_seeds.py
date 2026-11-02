"""Golden values produced by the archived, hash-verified campaign selector."""
import json
from pathlib import Path
import unittest

from handson.evaluation import settings


class PaperSettingSeedTests(unittest.TestCase):
    def test_public_names_preserve_historical_random_streams_and_profile_draws(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures' /
                              'paper_setting_seed_vectors.json').read_text())
        self.assertEqual({r['task'] for r in fixture['cases']}, set(settings.task_ids()))
        for row in fixture['cases']:
            with self.subTest(task=row['task']):
                actual = settings.select_setting(row['master_seed'], row['task'],
                                                 row['episode_index'])
                for key, expected in row['expected'].items():
                    self.assertEqual(actual[key], expected, key)
                self.assertEqual(actual['task'], row['task'])

    def test_forced_profile_keeps_historical_random_streams(self):
        task = 't12_load_and_start_microwave'
        original = settings.select_setting(42, task, 17)
        for profile in settings.task_spec(task)['profiles']:
            forced = settings.select_setting(42, task, 17,
                                             forced_setting=profile['setting_id'])
            self.assertEqual(forced['seeds'], original['seeds'])
            self.assertEqual(forced['setting_draw'], original['setting_draw'])
            self.assertEqual(forced['setting_id'], profile['setting_id'])
            self.assertTrue(forced['forced_for_validation'])


if __name__ == '__main__':
    unittest.main()
