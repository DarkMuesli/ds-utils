import pandas as pd


def identify_missing_data_gaps_with_count(df: pd.DataFrame | pd.Series) -> list[tuple]:
    """
    Identify gaps of missing data in a DataFrame and count the number of missing data points in each gap.

    param df: Pandas DataFrame with missing values.
    return: List of tuples, each tuple representing the start index, end index, and the count of missing data points in a gap.
    """
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    gaps = []
    in_gap = False
    start_idx = None
    missing_count = 0
    prev_idx = None

    for idx, row in df.iterrows():
        if row.isna().any():
            if not in_gap:
                start_idx = idx  # Start of a new gap
                in_gap = True
                missing_count = 1  # Initialize missing count for the new gap
            else:
                missing_count += 1  # Increment missing count within the existing gap
        else:
            if in_gap:
                end_idx = prev_idx  # End of the gap
                gaps.append((start_idx, end_idx, missing_count))
                in_gap = False
        prev_idx = idx

    # Check if the last rows are part of an open gap
    if in_gap:
        end_idx = df.index[-1]
        gaps.append((start_idx, end_idx, missing_count))

    return gaps
