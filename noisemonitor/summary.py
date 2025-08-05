import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

from datetime import time
from typing import Optional, List

from .util import filter
from .util import core

def harmonica_periodic(
    df: pd.DataFrame,
    column: Optional[int] = 0,  
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
    column: int, default 0
        column index to use for calculations. If None, the first column
        of the DataFrame will be used.
    use_chunks: bool default True
        whether to process the data in chunks for large datasets.
    day1: Optional[str], default None
        First day of the week to include in the calculation.
    day2: Optional[str], default None
        Last day of the week to include in the calculation.

    Returns
    ----------
    DataFrame: DataFrame with time index and 24 BGN, EVT, and HARMONICA 
    values.
    """

    # Compute hourly HARMONICA indicators
    temp_df = filter._days(df, day1, day2)
    harmonica_df = core.harmonica(temp_df, column, use_chunks)

    # Compute the average values for each hour of the day
    daily_avg = harmonica_df.groupby(harmonica_df.index.hour).mean()

    # Create a time index for the result
    time_index = [time(hour=h) for h in range(24)]
    daily_avg.index = time_index

    return daily_avg

def periodic(
    df: pd.DataFrame,
    freq: str='D',
    column: Optional[int] = 0,
    values: bool=False
) -> pd.DataFrame:
    """Compute Leq,24h and Lden on a periodic basis.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with a datetime index and sound level values.
    freq: str, default 'D'
        frequency for the computation. 'D' for daily, 'W' for weekly, and
        'M' for monthly.
    column: int, default 0
        column index to use for calculations. If None, the first column
        of the DataFrame will be used.
    values: bool, default False
        if set to True, the function will return individual day, evening
        and night values in addition to the lden.

    Returns
    ----------
    DataFrame: DataFrame with Leq,24h and Lden values for each day or week.
    """

    results = []

    if freq == 'D':
        resampled = df.resample('D')
    elif freq == 'M':
        resampled = df.resample('MS')
    elif freq == 'W':
        resampled = df.resample('W-MON')
    else:
        raise ValueError("Invalid frequency. Use 'D' for daily, 'W' for" 
                            " weekly or 'M' for monthly.")

    for period, group in resampled:
        if len(group) > 0:
            leq_value = core.equivalent_level(group.iloc[:, column])
            lden_values = core.lden(group, column, values=values)
            result = {
                'Period': period,
                'Leq,24h': leq_value,
                'Lden': lden_values['lden'][0]
            }
            if values:
                result.update({
                    'Lday': lden_values['lday'][0],
                    'Levening': lden_values['levening'][0],
                    'Lnight': lden_values['lnight'][0]
                })
            results.append(result)

    result_df = pd.DataFrame(results).set_index('Period')
    return result_df

def freq_periodic(
    df: pd.DataFrame, 
    freq: str = 'D', 
    values: bool = False
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
        and 'M' for monthly.
    values: bool, default False
        If set to True, the function will return individual day, 
        evening, and night values in addition to the Lden.

    Returns
    ----------
    pd.DataFrame
        DataFrame with weekly or daily levels for each frequency band.
    """
    results = {
        col: periodic(df, column=col, freq=freq, values=values)
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
    column: Optional[int] = 0,
    values: bool=True
) -> pd.DataFrame:
    """Return the Lden, a descriptor of noise level based on Leq over
    a whole day with a penalty for evening (19h-23h) and night (23h-7h)
    time noise. By default, an average Lden is computed that is 
    representative of all the sound level data. Can return a Lden value
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
    column: int, default 0
        column name to use for calculations. If None, the first column
        of the DataFrame will be used.
    values: bool, default False
        If set to True, the function will return individual day, evening
        and night values in addition to the lden.

    Returns
    ---------- 
    dict: daily or weekly lden rounded to two decimals. Associated day,
        evening and night values are returned if values is set to True.
    """

    temp = filter._days(df, day1, day2)

    return core.lden(temp, column, values=values)

def leq(
    df: pd.DataFrame, 
    hour1: int, 
    hour2: int, 
    day1: Optional[str] = None, 
    day2: Optional[str] = None, 
    column: Optional[int] = 0,
    stats: bool = True
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
    column: int, default 0
        column index to use for calculations. If None, the first column
        of the DataFrame will be used.
    stats: bool, default True
        If set to True, the function will return L10, L50 and L90 
        together with the Leq.

    Returns
    ---------- 
    float: daily or weekly equivalent level rounded to two decimals. 
    Statistical indicators are included if stats is set to True.
    """

    temp = filter._days(df, day1, day2)
    array = filter._hours(temp, hour1, hour2).iloc[:, column]

    if stats:
        interval = core.get_interval(df)
        if interval > 1:
            warnings.warn("Computing the L10, L50, and L90 should be done with "
                            "an integration time equal to or below 1s. Results"
                            " might not be valid for this descriptor.\n")
        return pd.DataFrame({
            'leq': [np.round(core.equivalent_level(array), 2)],
            'l10': [np.round(np.nanpercentile(array, 90), 2)],
            'l50': [np.round(np.nanpercentile(array, 50), 2)],
            'l90': [np.round(np.nanpercentile(array, 10), 2)]
        })
    return pd.DataFrame({'leq': [np.round(core.equivalent_level(array), 2)]})

def freq_descriptors(
    df: pd.DataFrame,
    hour1: Optional[int] = 0,
    hour2: Optional[int] = 24,
    day1: Optional[str] = None,
    day2: Optional[str] = None,
    stats: bool = False,
    values: bool = True
) -> pd.DataFrame:
    """
    Compute overall Leq and Lden for each frequency band in the NoiseMonitor DataFrame.

    Parameters
    ----------
    hour1 (optional): int, default 0
        Starting hour for the daily Leq average (0-24).
    hour2 (optional): int, default 24
        Ending hour for the daily Leq average (0-24).
    day1 (optional): str, default None
        First day of the week in English, case-insensitive, to include in the computation.
    day2 (optional): str, default None
        Last day of the week in English, case-insensitive, to include in the computation.
    stats: bool, default False
        If set to True, the function will include statistical indicators (L10, L50, L90) 
        in addition to Leq.
    values: bool, default True
        If set to True, the function will include individual day, evening, and night 
        values in addition to Lden.

    Returns
    ----------
    pd.DataFrame
        DataFrame with rows corresponding to indicators (e.g., Leq, Lden, etc.) and 
        columns corresponding to frequency bands.
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
            stats=stats
        )

        # Compute overall Lden
        lden_result = lden(
            df,
            day1=day1,
            day2=day2,
            column=col_idx,
            values=values
        )

        # Combine results for this frequency band
        combined_result = {
            'Leq': leq_result['leq'][0],
            'Lden': lden_result['lden'][0]
        }

        if stats:
            combined_result.update({
                'L10': leq_result['l10'][0],
                'L50': leq_result['l50'][0],
                'L90': leq_result['l90'][0]
            })

        if values:
            combined_result.update({
                'Lday': lden_result['lday'][0],
                'Levening': lden_result['levening'][0],
                'Lnight': lden_result['lnight'][0]
            })

        # Store the combined result for this frequency band
        results[df.columns[col_idx]] = combined_result

    # Convert the results dictionary into a DataFrame
    results_df = pd.DataFrame(results)

    return results_df

def nday(
    df: pd.DataFrame,
    indicator: str = 'Leq,24h',
    bins: Optional[List[int]] = None,
    freq: str = 'D',
    column: Optional[int] = 0
) -> pd.DataFrame:
    """Compute the number of days in a dataset for which the indicators 
    are between given values of decibels.

    Parameters
    ----------
    indicator: str, default 'Leq,24h'
        Indicator to use for the computation. Options are 'Leq,24h', 'Lden', 
        'Lday', 'Levening', and 'Lnight'.
    bins: list of int, optional
        List of decibel values to define the bins. By default: <40 dBA 
        and every 5 dBA until >=80 dBA.
    freq: str, default 'D'
        Frequency for the computation. 'D' for daily and 'W' for weekly.
    column: int, default 0
        column name to use for calculations. If None, the first column
        of the DataFrame will be used.

    Returns
    ----------
    DataFrame : DataFrame with the number of days for each decibel range.
    """

    if bins is None:
        bins = [40, 45, 50, 55, 60, 65, 70, 75, 80]

    # Compute daily or weekly indicators
    indicators_df = periodic(
        df,
        freq=freq,
        column=column,
        values=True
    )

    # Validate the indicator
    if indicator not in indicators_df.columns:
        raise ValueError("Invalid indicator. Choose from "
            f"{indicators_df.columns.tolist()}")

    # Count the number of days for each decibel range
    counts = pd.cut(
        indicators_df[indicator], 
        bins=[-float('inf')] + bins + [float('inf')], 
        right=False
        ).value_counts().sort_index()

    # Create a DataFrame for the counts
    counts_df = pd.DataFrame(counts).reset_index()
    counts_df.columns = ['Decibel Range', 'Number of Days']

    return counts_df, bins   

