import unittest

import pandas as pd

from imputation import identify_missing_data_gaps_with_count


class TestGapDetection(unittest.TestCase):
    def test_identify_missing_data_gaps_with_count(self):
        df_3_gaps = pd.DataFrame({'a': [1, 2, 3, None, None, 6, 7, 8, None, 10, 11, 12, 13, 14, 15, 16, 17, 18, None,
                                        None, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]})
        df_1_gap = pd.DataFrame({'a': [1, 2, 3, None, None, None, 7, 8, 9, 10]})
        df_no_gaps = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        gaps = identify_missing_data_gaps_with_count(df_3_gaps)
        self.assertEqual(3, len(gaps))
        self.assertEqual((3, 4, 2), gaps[0])
        self.assertEqual((8, 8, 1), gaps[1])
        self.assertEqual((18, 19, 2), gaps[2])

        gaps = identify_missing_data_gaps_with_count(df_1_gap)
        self.assertEqual(1, len(gaps))
        self.assertEqual((3, 5, 3), gaps[0])

        gaps = identify_missing_data_gaps_with_count(df_no_gaps)
        self.assertEqual(0, len(gaps))

    def test_gaps_at_end_with_different_lengths(self):
        time_index1 = pd.date_range(start='2021-01-01 00:00:00', periods=6, freq='1min')
        df1 = pd.DataFrame({'Value': [1, 2, 3, 4, 5, pd.NA]}, index=time_index1)
        expected1 = [(df1.index[5], df1.index[5], 1)]
        self.assertEqual(identify_missing_data_gaps_with_count(df1), expected1)

        time_index2 = pd.date_range(start='2021-01-01 00:00:00', periods=8, freq='1min')
        df2 = pd.DataFrame({'Value': [1, 2, 3, 4, pd.NA, pd.NA, pd.NA, 8]}, index=time_index2)
        expected2 = [(df2.index[4], df2.index[6], 3)]
        self.assertEqual(identify_missing_data_gaps_with_count(df2), expected2)
