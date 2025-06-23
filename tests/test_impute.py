import unittest

import numpy as np
import pandas as pd

from imputation import fill_missing_with_mean, subsequence_imputation


class TestImputation(unittest.TestCase):

    def test_subsequence_imputation_multiple_values(self):
        expected = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] * 2,
                             index=pd.date_range("2021-01-01", periods=18, freq="h"))
        series = expected.copy()
        series.iloc[3:6] = np.nan

        pre_imputed = fill_missing_with_mean(series)
        actual = subsequence_imputation(pre_imputed)

        self.assertEqual(len(expected), len(actual))
        self.assertTrue(expected.equals(actual))

    def test_subsequence_imputation_single_value(self):
        expected = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] * 2,
                             index=pd.date_range("2021-01-01", periods=18, freq="h"))
        series = expected.copy()
        series.iloc[3] = np.nan

        pre_imputed = fill_missing_with_mean(series)
        actual = subsequence_imputation(pre_imputed)

        self.assertEqual(len(expected), len(actual))
        self.assertTrue(expected.equals(actual))

    def test_missing_frequency_raises_error(self):
        series = pd.Series([1, 2, 3])
        self.assertRaises(ValueError, subsequence_imputation, series)

        irregular_index = pd.to_datetime([
            "2021-01-01 00:00:00", "2021-01-01 00:01:05", "2021-01-01 00:02:08"
        ])
        irregular_series = pd.Series([1, 2, 3], index=irregular_index)
        self.assertRaises(ValueError, subsequence_imputation, irregular_series)

    def test_subsequence_imputation_multiple_sources(self):
        complete_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                         1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 19.0, 9.0,
                         1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        missing_value_list = [1.0, 2.0, 3.0, np.nan, np.nan, np.nan, 7.0, 8.0, 9.0,
                              1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 19.0, 9.0,
                              1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        series = pd.Series(missing_value_list,
                           index=pd.date_range(start='2021-01-01 00:00:00',
                                               periods=len(missing_value_list),
                                               freq='h'))
        imputed_series = fill_missing_with_mean(series)
        actual = subsequence_imputation(imputed_series)
        expected = pd.Series(complete_list,
                             index=pd.date_range(start='2021-01-01 00:00:00',
                                                 periods=len(complete_list),
                                                 freq='h'))

        res = pd.DataFrame({'input_series': series, 'imputed': actual, 'expected': expected})
        print('\n', res)

        self.assertEqual(len(expected), len(actual), "The length of the resulting series is not correct.")
        self.assertTrue(expected.equals(actual), "The values of the resulting series are not correct.")

    def test_subsequence_imputation_front_edge(self):
        complete_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] * 2
        missing_value_list = [1.0, 2.0, np.nan, np.nan, np.nan, 6.0, 7.0, 8.0, 9.0,
                              1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        series = pd.Series(missing_value_list,
                           index=pd.date_range(start='2021-01-01 00:00:00',
                                               periods=len(missing_value_list),
                                               freq='h'))
        imputed_series = fill_missing_with_mean(series)
        actual = subsequence_imputation(imputed_series)
        expected = pd.Series(complete_list,
                             index=pd.date_range(start='2021-01-01 00:00:00',
                                                 periods=len(complete_list),
                                                 freq='h'))

        res = pd.DataFrame({'input_series': series, 'imputed': actual, 'expected': expected})
        print('\n', res)

        self.assertEqual(len(expected), len(actual), "The length of the resulting series is not correct.")
        self.assertTrue(expected.equals(actual), "The values of the resulting series are not correct.")

    def test_subsequence_imputation_back_edge(self):
        complete_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] * 2
        missing_value_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                              1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.nan, np.nan, 9.0]
        series = pd.Series(missing_value_list,
                           index=pd.date_range(start='2021-01-01 00:00:00',
                                               periods=len(missing_value_list),
                                               freq='h'))
        imputed_series = fill_missing_with_mean(series)
        actual = subsequence_imputation(imputed_series)
        expected = pd.Series(complete_list,
                             index=pd.date_range(start='2021-01-01 00:00:00',
                                                 periods=len(complete_list),
                                                 freq='h'))

        res = pd.DataFrame({'input_series': series, 'imputed': actual, 'expected': expected})
        print('\n', res)

        self.assertEqual(len(expected), len(actual), "The length of the resulting series is not correct.")
        self.assertTrue(expected.equals(actual), "The values of the resulting series are not correct.")
