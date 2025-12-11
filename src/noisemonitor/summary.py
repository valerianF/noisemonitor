import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

from datetime import time
from typing import Optional, List, Union

from .util import filter
from .util import core

def harmonica_periodic(
    df: pd.DataFrame,
    column: Optional[Union[int, str]] = 0,
    use_chunks: bool = True,
    day1: Optional[str] = None,
    day2: Optional[str] = None
) -> pd.DataFrame:
    """Compute the average HARMONICA indicators for each hour of a day.
    Optionally, can return the average values for specific days of the week.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with a datetime index and sound level values.
    column: int or str, default 0
        column index (int) or column name (str) to use for calculations. 
        If None, the first column of the DataFrame will be used.
    use_chunks: bool default True
        whether to process the data in chunks for large datasets.
    day1: Optional[str], default None
        First day of the week to include in the calculation.
    day2: Optional[str], default None
        Last day of the week to include in the calculation.

    Returns
    ----------
    pd.DataFrame: DataFrame with time index and 24 BGN, EVT, and HARMONICA 
    values.
    """
    column = core._column_to_index(df, column)

    # Compute hourly HARMONICA indicators
    temp_df = filter._days(df, day1, day2)
    harmonica_df = core.harmonica(temp_df, column, use_chunks)

    # Compute the average values for each hour of the day
    daily_avg = harmonica_df.groupby(harmonica_df.index.hour).mean()

    # Create a time index for the result
    time_index = [time(hour=h) for h in range(24)]
    daily_avg.index = time_index

    if harmonica_df['HARMONICA'].isna().any():
        warnings.warn(
            "Insufficient data coverage detected when computing HARMONICA "
            "indices. Some hours were filtered out before averaging.",
            core.CoverageWarning,
            stacklevel=2
        )

    return daily_avg

def periodic(
    df: pd.DataFrame,
    freq: str='D',
    column: Optional[Union[int, str]] = 0,
    values: bool=False,
    coverage_check: bool = False,
    coverage_threshold: float = 0.5
) -> pd.DataFrame:
    """Compute Leq and Lden on a periodic basis.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with a datetime index and sound level values.
    freq: str, default 'D'
        frequency for the computation. 'D' for daily, 'W' for weekly, and
        'MS' for monthly.
    column: int or str, default 0
        column index (int) or column name (str) to use for calculations. 
        If None, the first column of the DataFrame will be used.
    values: bool, default False
        if set to True, the function will return individual day, evening
        and night values in addition to the lden.
    coverage_check: bool, default False
        if set to True, assess data coverage and automatically filter periods
        with insufficient data coverage and emit warnings.
    coverage_threshold: float, default 0.5
        minimum data coverage ratio required (0.0 to 1.0).

    Returns
    ----------
    pd.DataFrame: DataFrame with Leq and Lden values for each day or week.
    """
    column = core._column_to_index(df, column)

    results = []

    if freq == 'D':
        resampled = df.resample('D')
    elif freq == 'MS':
        resampled = df.resample('MS')
    elif freq == 'W':
        resampled = df.resample('W-MON')
    else:
        raise ValueError("Invalid frequency. Use 'D' for daily, 'W' for" 
                            " weekly or 'MS' for monthly.")
    
    coverage_warning_issued = False

    for period, group in resampled:
        if len(group) > 0:
            array = group.iloc[:, column]

            if coverage_check:
                passes_threshold = core.check_coverage(
                    array,
                    coverage_threshold,
                    emit_warning = not coverage_warning_issued
                )

                if passes_threshold: 
                    leq_value = core.equivalent_level(array)
                else:
                    leq_value = np.nan
                    coverage_warning_issued = True
            else:
                leq_value = core.equivalent_level(array)

            lden_values = core.lden(
                group,
                column=column,
                values=values,
                coverage_check=coverage_check,
                coverage_threshold=coverage_threshold
            )
            result = {
                'Period': period,
                'Leq': leq_value,
                'Lden': lden_values['Lden'][0]
            }
            if values:
                result.update({
                    'Lday': lden_values['Lday'][0],
                    'Levening': lden_values['Levening'][0],
                    'Lnight': lden_values['Lnight'][0]
                })
            results.append(result)

    result_df = pd.DataFrame(results).set_index('Period')
    return result_df

def freq_periodic(
    df: pd.DataFrame, 
    freq: str = 'D', 
    values: bool = False,
    coverage_check: bool = False,
    coverage_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Compute periodic levels for each frequency band (e.g. octave 
    band or third octave bands) in the input DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing octave or third-octave frequency bands as 
        columns.
    freq: str, default 'D'
        Frequency for the computation. 'D' for daily, 'W' for weekly, 
        and 'MS' for monthly.
    values: bool, default False
        If set to True, the function will return individual day, 
        evening, and night values in addition to the Lden.
    coverage_check: bool, default False
        if set to True, assess data coverage and automatically filter periods
        with insufficient data coverage and emit warnings.
    coverage_threshold: float, default 0.5
        minimum data coverage ratio required (0.0 to 1.0).

    Returns
    ----------
    pd.DataFrame: DataFrame with weekly or daily levels for each 
        frequency band.
    """
    results = {
        col: periodic(df, column=col, freq=freq, values=values,
                    coverage_check=coverage_check, coverage_threshold=coverage_threshold)
        for col in df.columns
    }

    combined_results = pd.concat(results, axis=1, keys=results.keys())

    combined_results.columns = pd.MultiIndex.from_tuples(
        [(indicator, band) for band in combined_results.columns.levels[0] 
                        for indicator in combined_results[band].columns],
        names=["Indicator", "Frequency Band"]
    )

    return combined_results

def lden(
    df: pd.DataFrame, 
    day1: Optional[str] = None, 
    day2: Optional[str] = None, 
    column: Optional[Union[int, str]] = 0,
    values: bool=True,
    coverage_check: bool = False,
    coverage_threshold: float = 0.5
) -> pd.DataFrame:
    """Return the Lden, an indicator of noise level based on Leq over
    a whole day with a penalty for evening (19h-23h) and night (23h-7h)
    time noise. By default, an average Lden encompassing
    all the dataset is computed. Can return a Lden value
    corresponding to specific days of the week.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with a datetime index and sound level values.
    day1 (optional): str, a day of the week in english, case-insensitive
        first day of the week included in the Lden computation.
    day2 (optional): str, a day of the week in english, case-insensitive
        last (included) day of the week in the Lden computation. If day2 
        happens later in the week than day1 the average will be computed 
        outside of these days.
    column: int or str, default 0
        column index (int) or column name (str) to use for calculations. 
        If None, the first column of the DataFrame will be used.
    values: bool, default False
        If set to True, the function will return individual day, evening
        and night values in addition to the lden.
    coverage_check: bool, default False
        if set to True, assess data coverage and automatically filter periods
        with insufficient data coverage.
    coverage_threshold: float, default 0.5
        minimum data coverage ratio required (0.0 to 1.0).

    Returns
    ---------- 
    pd.DataFrame: dataframe with daily or weekly lden rounded to two decimals. 
        Associated day, evening and night values are returned if values 
        is set to True.
    """
    column = core._column_to_index(df, column)
    temp = filter._days(df, day1, day2)

    return core.lden(
        temp, 
        column, 
        values=values,
        coverage_check=coverage_check,
        coverage_threshold=coverage_threshold  
        )

def leq(
    df: pd.DataFrame, 
    hour1: int, 
    hour2: int, 
    day1: Optional[str] = None, 
    day2: Optional[str] = None, 
    column: Optional[Union[int, str]] = 0,
    stats: bool = True,
    coverage_check: bool = False,
    coverage_threshold: float = 0.5
) -> pd.DataFrame:
    """Return the equivalent level (and optionally statistical indicators)
    between two hours of the day. Can return a value corresponding to 
    specific days of the week.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with a datetime index and sound level values.
    hour1: int, between 0 and 24
        hour for the starting time of the daily average.
    hour2: int, between 0 and 24
        hour for the ending time of the daily average. If hour2 > hour1 
        the average will be computed outside of these hours.
    day1 (optional): str, a day of the week in english, case-insensitive
        first day of the week included in the Lden computation.
    day2 (optional): str, a day of the week in english, case-insensitive
        last (included) day of the week in the Lden computation. If day2 
        happens later in the week than day1 the average will be computed 
        outside of these days.
    column: int or str, default 0
        column index (int) or column name (str) to use for calculations. 
        If None, the first column of the DataFrame will be used.
    stats: bool, default True
        If set to True, the function will return L10, L50 and L90 
        together with the Leq.
    coverage_check: bool, default False
        if set to True, assess data coverage and automatically filter periods
        with insufficient data coverage and emit warnings.
    coverage_threshold: float, default 0.5
        minimum data coverage ratio required (0.0 to 1.0).

    Returns
    ---------- 
    float: daily or weekly equivalent level rounded to two decimals. 
        Statistical indicators are included if stats is set to True.
    """
    column = core._column_to_index(df, column)

    temp = filter._days(df, day1, day2)
    array = filter._hours(temp, hour1, hour2).iloc[:, column]

    if coverage_check:
        passes_threshold = core.check_coverage(
            array,
            coverage_threshold,
            emit_warning=True
        )
        if not passes_threshold:
            if stats:
                return pd.DataFrame({
                    'Leq': [np.nan],
                    'L10': [np.nan],
                    'L50': [np.nan],
                    'L90': [np.nan]
                })
            return pd.DataFrame({'Leq': [np.nan]})

    if stats:
        interval = core.get_interval(df)
        if interval > 1:
            warnings.warn(
                "Computing the L10, L50, and L90 should be done with "
                "an integration time equal to or below 1s. Results"
                " might not be valid for this indicator.\n"
            )
        return pd.DataFrame({
            'Leq': [np.round(core.equivalent_level(array),2)],
            'L10': [np.round(np.nanpercentile(array, 90), 2)],
            'L50': [np.round(np.nanpercentile(array, 50), 2)],
            'L90': [np.round(np.nanpercentile(array, 10), 2)]
        })
    return pd.DataFrame({
        'Leq': [np.round(core.equivalent_level(array), 2)]
    })

def freq_indicators(
    df: pd.DataFrame,
    hour1: Optional[int] = 0,
    hour2: Optional[int] = 24,
    day1: Optional[str] = None,
    day2: Optional[str] = None,
    stats: bool = False,
    values: bool = True,
    coverage_check: bool = False,
    coverage_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Compute overall Leq and Lden for each frequency band.

    Parameters
    ----------
    hour1 (optional): int, default 0
        Starting hour for the daily Leq average (0-24).
    hour2 (optional): int, default 24
        Ending hour for the daily Leq average (0-24).
    day1 (optional): str, default None
        First day of the week in English, case-insensitive,
        to include in the computation.
    day2 (optional): str, default None
        Last day of the week in English, case-insensitive,
        to include in the computation.
    stats: bool, default False
        If set to True, the function will include statistical
        indicators (L10, L50, L90) in addition to Leq.
    values: bool, default True
        If set to True, the function will include individual
        day, evening, and night values in addition to Lden.
    coverage_check: bool, default False
        if set to True, assess data coverage and automatically filter periods
        with insufficient data coverage and emit warnings.
    coverage_threshold: float, default 0.5
        minimum data coverage ratio required (0.0 to 1.0).

    Returns
    ----------
    pd.DataFrame:
        DataFrame with rows corresponding to indicators
        (e.g., Leq, Lden, etc.) and columns corresponding to
        frequency bands.
    """
    # Initialize a dictionary to store results for each frequency band
    results = {}

    for col_idx in range(df.shape[1]):
        # Compute overall Leq
        leq_result = leq(
            df,
            hour1=hour1,
            hour2=hour2,
            day1=day1,
            day2=day2,
            column=col_idx,
            stats=stats,
            coverage_check=coverage_check,
            coverage_threshold=coverage_threshold
        )

        # Compute overall Lden
        lden_result = lden(
            df,
            day1=day1,
            day2=day2,
            column=col_idx,
            values=values,
            coverage_check=coverage_check,
            coverage_threshold=coverage_threshold
        )

        # Combine results for this frequency band
        combined_result = {
            'Leq': leq_result['Leq'][0],
            'Lden': lden_result['Lden'][0]
        }

        if stats:
            combined_result.update({
                'L10': leq_result['L10'][0],
                'L50': leq_result['L50'][0],
                'L90': leq_result['L90'][0]
            })

        if values:
            combined_result.update({
                'Lday': lden_result['Lday'][0],
                'Levening': lden_result['Levening'][0],
                'Lnight': lden_result['Lnight'][0]
            })

        # Store the combined result for this frequency band
        results[df.columns[col_idx]] = combined_result

    # Convert the results dictionary into a DataFrame
    results_df = pd.DataFrame(results)

    return results_df

def nday(
    df: pd.DataFrame,
    indicator: str = 'Leq',
    bins: Optional[List[int]] = None,
    freq: str = 'D',
    column: Optional[Union[int, str]] = 0,
    coverage_check: bool = False,
    coverage_threshold: float = 0.5
) -> pd.DataFrame:
    """Compute the number of days in a dataset for which the indicators 
    are between given values of decibels.

    Parameters
    ----------
    indicator: str, default 'Leq'
        Indicator to use for the computation. Options are 'Leq', 'Lden', 
        'Lday', 'Levening', and 'Lnight'.
    bins: list of int, optional
        List of decibel values to define the bins. By default: <40 dBA 
        and every 5 dBA until >=80 dBA.
    freq: str, default 'D'
        Frequency for the computation. 'D' for daily and 'W' for weekly.
    column: int or str, default 0
        column index (int) or column name (str) to use for calculations. 
        If None, the first column of the DataFrame will be used.
    coverage_check: bool, default False
        if set to True, assess data coverage and automatically filter periods
        with insufficient data coverage and emit warnings.
    coverage_threshold: float, default 0.5
        minimum data coverage ratio required (0.0 to 1.0).

    Returns
    ----------
    DataFrame : DataFrame with the number of days for each decibel range.
    """
    column = core._column_to_index(df, column)

    if bins is None:
        bins = [40, 45, 50, 55, 60, 65, 70, 75, 80]

    # Compute daily or weekly indicators
    indicators_df = periodic(
        df,
        freq=freq,
        column=column,
        values=True,
        coverage_check=coverage_check,
        coverage_threshold=coverage_threshold
    )

    # Validate the indicator
    if indicator not in indicators_df.columns:
        raise ValueError("Invalid indicator. Choose from "
            f"{indicators_df.columns.tolist()}")

    # Count the number of days for each decibel range
    counts = pd.cut(
        indicators_df[indicator].dropna(), 
        bins=[-float('inf')] + bins + [float('inf')], 
        right=False
        ).value_counts().sort_index()

    # Create a DataFrame for the counts
    counts_df = pd.DataFrame(counts).reset_index()
    counts_df.columns = ['Decibel Range', 'Number of Days']

    return counts_df, bins   

