import numpy as np
import pandas as pd


def vectorized_subsequence_distances(series: pd.Series, subsequence: pd.Series) -> pd.DataFrame:
    """Compute Euclidean distance between `subsequence` and all windows in `series`."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("The series index must be a DatetimeIndex.")

    series_array = np.array(series).flatten()
    subsequence_array = np.array(subsequence).flatten()
    subsequence_len = len(subsequence)

    if subsequence_len > len(series_array):
        raise ValueError("Subsequence is longer than the series.")

    s_subsequences = np.lib.stride_tricks.sliding_window_view(series_array, subsequence_len)

    distances = np.linalg.norm(s_subsequences - subsequence_array, axis=1)

    df = pd.DataFrame(index=series.index[:len(series) - subsequence_len + 1])
    df['distance'] = distances
    df['start_idx'] = df.index
    df['end_idx'] = df.index + df.index.freq * (subsequence_len - 1)

    df.dropna(inplace=True)

    return df


def series_mean(one: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Element-wise mean between two arrays."""
    return (one + other) / 2
