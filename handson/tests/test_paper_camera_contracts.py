"""Camera inputs recovered from hash-verified paper jobs, independent of scores."""
import json
from pathlib import Path
import unittest

from handson.evaluation import camera_contracts, paper_trials


class PaperCameraContractTests(unittest.TestCase):
    def test_every_source_cohort_uses_its_archived_projection(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/paper_camera_contracts.json').read_text())
        for case in fixture['cases']:
            with self.subTest(task=case['task'], variant=case['paper_variant']):
                trial = paper_trials.load(case['model'], case['task'], case['index'])
                self.assertEqual(trial['provenance'], case['provenance'])
                contract, receipt = camera_contracts.select(case['task'], case['paper_variant'])
                self.assertEqual(paper_trials.digest(contract), case['camera_contract_sha256'])
                self.assertEqual(receipt['profile_sha256'], case['camera_contract_sha256'])

    def test_early_wrists_and_later_wrists_remain_distinct(self):
        old, _ = camera_contracts.select('t06_fit_device_cover', 'stabilize_body')
        new, _ = camera_contracts.select('t06_fit_device_cover')
        self.assertEqual(old['cameras']['wrist_left']['K_effective'], old['cameras']['wrist_right']['K_effective'])
        self.assertNotEqual(new['cameras']['wrist_left']['K_effective'], new['cameras']['wrist_right']['K_effective'])
        self.assertEqual(old['cameras']['main'], new['cameras']['main'])
        self.assertEqual(old['photometry'], new['photometry'])

    def test_scoped_contract_restores_even_on_error_and_does_not_leak_mutation(self):
        default = camera_contracts.current()
        with self.assertRaisesRegex(RuntimeError, 'test interruption'):
            with camera_contracts.use('t05_fill_cup') as receipt:
                self.assertEqual(paper_trials.digest(camera_contracts.current()), receipt['profile_sha256'])
                self.assertNotEqual(camera_contracts.current(), default)
                damaged = camera_contracts.current()
                damaged['cameras'].clear()
                self.assertTrue(camera_contracts.current()['cameras'])
                raise RuntimeError('test interruption')
        self.assertEqual(camera_contracts.current(), default)

    def test_newton_keeps_existing_camera_contract(self):
        self.assertEqual(camera_contracts.select('t05_fill_cup', physics_backend='newton')[0], camera_contracts.current())


if __name__ == '__main__':
    unittest.main()
