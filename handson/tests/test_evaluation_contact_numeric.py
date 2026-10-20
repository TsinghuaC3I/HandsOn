import unittest

from handson.evaluation.contact_numeric import (CHECKED_SOURCE_PAIRS, checked_source_pair,
                                   contiguous_checker_source)


class ContactSourceGuardTest(unittest.TestCase):
    def test_only_exact_pairs_accepted(self):
        pairs = list(CHECKED_SOURCE_PAIRS)
        self.assertEqual(len(pairs), 4)
        for pair in pairs:
            self.assertEqual(len(checked_source_pair(*pair)['candidate_sha256']), 64)
        with self.assertRaisesRegex(ValueError, 'Unreviewed'):
            checked_source_pair(pairs[0][0], pairs[1][1])
        with self.assertRaisesRegex(ValueError, 'Unreviewed'):
            checked_source_pair('unknown-B01-version', pairs[0][1])

    def test_numeric_patch_preserves_exact_comparison(self):
        source = '\n'.join((
            'normal = row[3:6]', 'impulse = row[6:9]',
            'n = float(np.linalg.norm(row[3:6]))',
            'valid = np.array_equal(actual, expected)', 'limit = 0.005'))
        candidate = contiguous_checker_source(source)
        self.assertEqual(candidate.count('copy=True'), 3)
        self.assertIn('np.array_equal(actual, expected)', candidate)
        self.assertIn('limit = 0.005', candidate)
        with self.assertRaisesRegex(ValueError, 'Unreviewed checker'):
            contiguous_checker_source(source.replace('normal = row[3:6]', 'normal = None'))


if __name__ == '__main__':
    unittest.main()
