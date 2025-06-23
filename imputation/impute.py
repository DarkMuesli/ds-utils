import pandas as pd

from .gap_detection import identify_missing_data_gaps_with_count
from .subsequence_matcher import vectorized_subsequence_distances, series_mean


def fill_missing_with_mean(series: pd.Series) -> pd.Series:
    """
    Fill single missing values in a series with the mean of the neighboring values.
    Only fills gaps of length 1.

    :param series: The series to fill missing values in.
    :return: A copy of the given series with missing values filled.
    """

    if not (isinstance(series, pd.Series)):
        raise ValueError("The parameter must be a Pandas Series.")

    series_ = series.copy()
    missing_mask = series_.isna()
    valid_mask = ~missing_mask
    for i in range(1, len(series_) - 1):
        if missing_mask.iloc[i] and valid_mask.iloc[i - 1] and valid_mask.iloc[i + 1]:
            series_.iloc[i] = (series_.iloc[i - 1] + series_.iloc[i + 1]) / 2
    return series_


def subsequence_imputation(series: pd.Series,
                           distance_func=vectorized_subsequence_distances,
                           weighting_func=series_mean) -> pd.Series:
    """
    Impute missing values in a series using Partial Subsequence Matching (PSM).

    This implementation of the PSM algorithm first detects all gaps in the dataframe.
    For each gap of size n, subsequences of the same length of the gap are extracted, both on the left and right side.
    The distance between these two subsequences and every window of size n is computed using distance_func.
    The window with the smallest distance is selected for each side.
    The values of the left and right subsequences are then processed using weighting_func.
    The resulting values are then used to fill the gap.
    Currently, this imputation function does not work with single value gaps.
    Use fill_missing_with_mean before using this function to impute single value gaps.

    See https://doi.org/10.1007/s11269-022-03408-6 for more information.

    :param series: The pd.Series to impute missing data into.
    :param distance_func: The distance function to use for calculating distances between subsequences.
        Must return a DataFrame with columns 'start_idx', 'end_idx' and 'distance'. Defaults to vectorized function.
    :param weighting_func: The weighting function to use for combining the left and right subsequences.
        Must return a numpy array with the same length as the input arrays. Defaults to mean.
    :return: A copy of the given series with imputed values.
    """

    if not isinstance(series, pd.Series):
        raise ValueError("The parameter must be a Pandas Series.")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("The series index must be a DatetimeIndex.")

    if series.index.freq is None:
        raise ValueError("The series must have a frequency.")

    series = series.copy()
    freq = series.index.freq

    gaps = identify_missing_data_gaps_with_count(series)
    gaps = sorted(gaps, key=lambda x: x[2])

    for gap in gaps:
        start_idx = gap[0]
        end_idx = gap[1]
        missing_count = gap[2]

        l_start_idx = series.index.get_loc(start_idx) - missing_count
        r_end_idx = series.index.get_loc(end_idx) + missing_count

        l_s = None
        r_s = None

        if l_start_idx >= 0:
            l_subseq = series[l_start_idx:series.index.get_loc(start_idx)]
            distances = distance_func(series, l_subseq)
            sorted_distances = distances.sort_values(by='distance', ascending=True)

            for idx, distance in sorted_distances.iterrows():
                lower = (series.index.get_loc(distance['end_idx'])) + 1
                upper = (series.index.get_loc(distance['end_idx'])) + missing_count + 1
                l_s = series[lower:upper]

                if (
                        not l_s.isna().any().any()
                        and (len(l_s.values) == missing_count)
                ):
                    break

        if r_end_idx + 1 <= len(series):
            r_subseq = series[series.index.get_loc(end_idx + freq):r_end_idx + 1]
            distances = distance_func(series, r_subseq)
            sorted_distances = distances.sort_values(by='distance', ascending=True)

            for idx, distance in sorted_distances.iterrows():
                if (series.index.get_loc(distance['start_idx'])) - missing_count < 0:
                    continue
                lower = (series.index.get_loc(distance['start_idx'])) - missing_count
                upper = (series.index.get_loc(distance['start_idx']))
                r_s = series[lower:upper]

                if not r_s.isna().any().any():
                    break

        if l_s is None and r_s is None:
            continue
        elif l_s is None:
            l_s = r_s
        elif r_s is None:
            r_s = l_s

        impute_values = weighting_func(l_s.values, r_s.values)
        series.loc[start_idx:end_idx] = impute_values
    return series
