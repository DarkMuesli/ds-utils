import unittest

import pandas as pd

from ds_utils_darkmuesli.imputation import vectorized_subsequence_distances


class TestSubsequenceMatcher(unittest.TestCase):

    def test_vectorized_subsequence_distances_one_distance(self):
        s = pd.Series([5, 5, 5], index=pd.date_range("2021-01-01", periods=3, freq="h"))
        subseq = s.copy()

        actual = vectorized_subsequence_distances(s, subseq)

        expected = pd.DataFrame({
            'distance': [0.0],
            'start_idx': [s.index[0]],
            'end_idx': [s.index[-1]],
        }, index=[s.index[0]])

        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))

    def test_vectorized_subsequence_distances_single_value(self):
        s = pd.Series([5, 5, 5], index=pd.date_range("2021-01-01", periods=3, freq="h"))
        subseq = pd.Series([5], index=pd.date_range("2021-01-01", periods=1, freq="h"))

        actual = vectorized_subsequence_distances(s, subseq)

        expected = pd.DataFrame({
            'distance': [0.0, 0.0, 0.0],
            'start_idx': s.index,
            'end_idx': s.index
        }, index=s.index)

        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))

    def test_vectorized_subsequence_distances_window_too_long(self):
        s = pd.Series([5, 5, 5], index=pd.date_range("2021-01-01", periods=3, freq="h"))
        long_subseq = pd.Series([5, 5, 5, 5], index=pd.date_range("2021-01-01", periods=4, freq="h"))
        self.assertRaises(ValueError, vectorized_subsequence_distances, s, long_subseq)
