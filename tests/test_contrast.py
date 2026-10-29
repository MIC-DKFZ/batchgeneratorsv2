import unittest
from itertools import product

import numpy as np

from batchgeneratorsv2.transforms.intensity.contrast import BGContrast


class TestBGContrast(unittest.TestCase):
    """
    Ranges that end at or below one must be sampled from the range, as in batchgenerators' contrast, gamma and scale
    sampling. A range ending at one used to return exactly one for half of the draws.
    """

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())

    def test_ranges_up_to_one_are_respected(self):
        for value_range, seed in product(((0.7, 0.9), (0.8, 0.8), (0.7, 1)), range(50)):
            with self.subTest(value_range=value_range, seed=seed):
                np.random.seed(seed)
                factor = BGContrast(value_range)()
                self.assertGreaterEqual(factor, value_range[0])
                self.assertLessEqual(factor, value_range[1])
                # uniform never returns its upper bound, so a factor of exactly one comes from the old upper branch
                self.assertLess(factor, 1)

    def test_number_of_draws_is_unchanged(self):
        # a branch choice and the factor
        for value_range in ((0.7, 0.9), (0.8, 0.8), (0.75, 1.25), (0.7, 1.5), (0.5, 2), (1.1, 1.5), (0.7, 1), (1, 1)):
            with self.subTest(value_range=value_range):
                expected = np.random.RandomState(0)
                expected.random_sample(2)
                np.random.seed(0)
                BGContrast(value_range)()
                actual_state, expected_state = np.random.get_state(), expected.get_state()
                np.testing.assert_array_equal(actual_state[1], expected_state[1])
                self.assertEqual(actual_state[2:], expected_state[2:])


if __name__ == '__main__':
    unittest.main()
