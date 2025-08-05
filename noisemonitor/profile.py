import pandas as pd
import numpy as np
import warnings

from datetime import time
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from .util import filter
from .util import core

def periodic(
    df: pd.DataFrame,
    hour1: int,
    hour2: int,
    day1: Optional[str] = None,
    day2: Optional[str] = None,
    column: Optional[int] = 0,
    win: int = 3600,
    step: int = 0,
    traffic_noise_indicators: bool = False,
    roughness_indicators: bool = False
) -> pd.DataFrame:
    """Compute daily or weekly rolling averages of the sound level, in 
    terms of equivalent level (Leq), and percentiles (L10, L50 and L90).

    Parameters
    ---------- 
    hour1: int, between 0 and 23
        hour for the starting time of the daily average.
    hour2: int, between 0 and 23
        hour (included) for the ending time of the daily average. 
        If hour2 > hour1 the average will be computed outside of 
        these hours.
    day1: Optional[str], default None
        First day of the week included in the weekly average.
    day2: Optional[str], default None
        Last day of the week included in the weekly average.
    column: int, default 0
        column index to use for calculations. If None, the first column
        of the DataFrame will be used.
    win: int, default 3600
        window size for the averaging function, in seconds.
    step: int, default 0
        step size to compute a rolling average. If set to 0 (default 
        value), the function will compute non-rolling averages.
    traffic_noise_indicators: bool, default False
        if set to True, the function will compute traffic noise indicators 
        Traffic Noise Index (Griffiths and Langdon, 1968) as well as the 
        Noise Pollution Level (Robinson, 1971) in addition to the 
        equivalent levels and percentiles.
    roughness_indicators: bool, default False
        if set to True, the function will compute roughness indicators 
        based on difference between consecutive LAeq,1s values according 
        to (DeFrance et al., 2010).

    Returns
    ---------- 
    DataFrame: dataframe containing time index and daily averaged
        Leq, L10, L50 and L90 at the corresponding columns

    """

    if core.get_interval(df) > 1:
        warnings.warn(f"Computing the L10, L50, L90, traffic, or "
            "roughness noise indicators should be done with "
            "an integration time equal to or below 1s. Results"
            " might not be valid for these descriptors.\n")
    
    if step == 0:
        step = win

    temp = filter._days(df, day1, day2)

    NLim = ((hour2-hour1)%24*3600)//step + 1

    averages = {
        'Leq': np.zeros(NLim),
        'L10': np.zeros(NLim),
        'L50': np.zeros(NLim),
        'L90': np.zeros(NLim)
    }

    if traffic_noise_indicators:
        averages.update({
            'TNI': np.zeros(NLim),
            'NPL': np.zeros(NLim)
        })
    if roughness_indicators:
        averages.update({
            'dLav': np.zeros(NLim),
            'dLmax,1': np.zeros(NLim),
            'dLmin,90': np.zeros(NLim)
        })

    times = []

    for i in range(0, NLim):
        t = hour1*3600 + i*step + win//2
        t1 = hour1*3600 + i*step
        t2 = hour1*3600 + i*step + win

        temp_slice = temp.between_time(
            time(
                hour=(t1//3600)%24, 
                minute=(t1%3600)//60, 
                second=(t1%3600)%60
            ), 
            time(
                hour=(t2//3600)%24, 
                minute=(t2%3600)//60, 
                second=(t2%3600)%60
        ))

        arr = temp_slice.iloc[:, column]
        averages['Leq'][i] = core.equivalent_level(arr)
        averages['L10'][i] = np.nanpercentile(arr, 90)
        averages['L50'][i] = np.nanpercentile(arr, 50)
        averages['L90'][i] = np.nanpercentile(arr, 10)

        if traffic_noise_indicators:
            L10 = averages['L10'][i]
            L90 = averages['L90'][i]
            Leq = averages['Leq'][i]
            sigma_Leq = arr.std()

            # Traffic Noise Index (TNI)
            averages['TNI'][i] = 4 * (L10 - L90) + L90 - 30

            # Noise Pollution Level (NPL)
            averages['NPL'][i] = Leq + 2.56 * sigma_Leq

        if roughness_indicators:
            dL = np.abs(np.diff(arr.dropna()))
            averages['dLav'][i] = np.mean(dL)
            averages['dLmax,1'][i] = np.mean(dL[dL >= np.percentile(dL, 99)])
            averages['dLmin,90'][i] = np.mean(dL[dL <= np.percentile(dL, 10)])

        times.append(time(
            hour=(t//3600)%24, 
            minute=(t%3600)//60, 
            second=(t%3600)%60
            ))

    return pd.DataFrame(index=times, data=averages)
    
def freq_periodic(
    df: pd.DataFrame,
    hour1: int,
    hour2: int,
    day1: Optional[str] = None,
    day2: Optional[str] = None,
    win: int = 3600,
    step: int = 0,
    chunks: bool = True
) -> pd.DataFrame:
    """
    Compute weekly levels for each frequency band in the NoiseMonitor DataFrame.

    Parameters
    ----------
    df: pd.DataFrame,
        DataFrame with a datetime index and sound level values for each 
        frequency band.
    hour1: int, between 0 and 23
        Hour for the starting time of the daily average.
    hour2: int, between 0 and 23
        Hour (included) for the ending time of the daily average.
    day1: Optional[str], default None
        First day of the week included in the weekly average.
    day2: Optional[str], default None
        Last day of the week included in the weekly average.
    win: int, default 3600
        Window size for the averaging function, in seconds.
    step: int, default 0
        Step size to compute a sliding average. If set to 0 (default value),
        the function will compute non-sliding averages.
    chunks: bool, default True
        If set to True, the function will use parallel processing to compute
        weekly levels for each frequency band.

    Returns
    ----------
    pd.DataFrame
        DataFrame with weekly levels for each frequency band.
    """
    warnings.simplefilter("ignore")

    if chunks:
        results = {}
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    periodic,
                    df,
                    hour1,
                    hour2,
                    day1,
                    day2,
                    col,
                    win,
                    step
                ): col
                for col in df.columns
            }
            for future in as_completed(futures):
                col = futures[future]
                results[col] = future.result()
    else:
        results = {
            col: periodic(
                df,
                hour1,
                hour2,
                day1,
                day2,
                col,
                win,
                step
            )
            for col in df.columns
        }

    warnings.resetwarnings()

    float_columns = {col: float(col) for col in results.keys()}
    results = {float_columns[col]: df for col, df in results.items()}

    combined_results = pd.concat(results, axis=1, keys=results.keys())
    combined_results.columns = pd.MultiIndex.from_tuples(
        [(indicator, band) for band in combined_results.columns.levels[0]
                            for indicator in combined_results[band].columns],
        names=["Indicator", "Frequency Band"]
    )

    combined_results = combined_results.sort_index(
        axis=1,
        level="Frequency Band"
    )

    return combined_results

def nne(
    df: pd.DataFrame,
    hour1: int,
    hour2: int,
    background_type: str = 'leq',
    exceedance: int = 5,
    min_gap: int = 3,
    win: int = 3600,
    step: int = 0,
    column: Optional[int] = 0,
    day1: Optional[str] = None,
    day2: Optional[str] = None
) -> pd.DataFrame:
    """Compute the Number of Noise Events (NNE) following the algorithm 
    proposed in (Brown and De Coensel, 2018). The function computes the 
    average NNE using sliding windows, computing daily or weekly profiles.
    Note that this function is computaionally expensive as noise NNEs are
    separately computed for each individual day and then averaged since
    background levels are relative to each day.

    Parameters
    ----------
    hour1: int, between 0 and 23
        hour for the starting time of the daily average.
    hour2: int, between 0 and 23
        hour for the ending time of the daily average. If hour1 > hour2 
        the average will be computed outside of these hours.
    background_type: str
        Type of background level descriptor for computing the threshold to 
        use for defining a noise event. Can be 'leq', 'l50', 'l90' or int 
        for a constant value.
    exceedance: int, default 5
        Exceedance value in dB to add to the background level to define the
        detection threshold, when the background level is adaptive.
    min_gap: int
        Minimum time gap in seconds between successive noise events.
    win: int, default 3600
        Window size for the averaging function, in seconds.
    step: int, default 0
        Step size to compute a sliding average. If set to 0 (default value), 
        the function will compute non-sliding averages.
    column: int, default 0
        column index to use for calculations. If None, the first column
        of the DataFrame will be used.
    day1: Optional[str], default None
        First day of the week to include in the calculation.
    day2: Optional[str], default None
        Last day of the week to include in the calculation.

    Returns
    ----------
    DataFrame: DataFrame with the number of noise events for each sliding 
    window.
    """
    if core.get_interval(df) > 1:
        warnings.warn(f"Computing the average Number of "
            "Noise Events should be done with "
            "an integration time equal to or below 1s. Results"
            " might not be valid for these descriptors.\n")

    if column is None:
        column = 0

    temp_df = filter._days(df, day1, day2)
    temp_df = filter._hours(temp_df, hour1, hour2)

    if step == 0:
        step = win
    
    NLim = ((hour2-hour1)%24*3600)//step - win//step + 1

    event_times = []
    daily_event_counts = []

    # Compute the expected number of values for each day
    freq = (temp_df.index[2] - temp_df.index[1])
    expected_intervals = pd.date_range(
        start=temp_df.index.min().normalize(),
        end=temp_df.index.min().normalize() + (
            pd.Timedelta(hours=(hour2-hour1)%24)),
        freq=freq
    )

    expected_intervals_count = len(expected_intervals)

    for day, group in temp_df.groupby(temp_df.index.date):
        # Check if the day is missing data
        if len(group) != expected_intervals_count:
            continue

        day_event_counts = np.zeros(NLim)
        for i in range(0, NLim):
            t1 = hour1*3600 + i*step
            t2 = hour1*3600 + i*step + win

            temp = group.between_time(
                time(
                    hour=(t1//3600)%24, 
                    minute=(t1%3600)//60, 
                    second=(t1%3600)%60
                ), 
                time(
                    hour=(t2//3600)%24, 
                    minute=(t2%3600)//60, 
                    second=(t2%3600)%60
            ))

            arr = temp.iloc[:, column]
            if background_type == 'leq':
                threshold = core.equivalent_level(arr) + exceedance
            elif background_type == 'l50':
                threshold = np.nanpercentile(arr, 50) + exceedance
            elif background_type == 'l90':
                threshold = np.nanpercentile(arr, 10) + exceedance
            elif isinstance(background_type, (int)):
                threshold = background_type
            else:
                raise ValueError("Invalid background type. Use 'leq', "
                                "'l50', 'l90', or an int value.")
            day_event_counts[i] = core.noise_events(
                temp, column, threshold, min_gap)
        daily_event_counts.append(day_event_counts)

    daily_event_counts = np.array(daily_event_counts)
    average_event_counts = np.mean(daily_event_counts, axis=0)

    for i in range(0, NLim):
        t = hour1*3600 + i*step + win//2
        event_times.append(time(
            hour=(t//3600)%24, 
            minute=(t%3600)//60, 
            second=(t%3600)%60
            ))

    return pd.DataFrame(
        index=event_times, 
        data={'Average NNEs': average_event_counts}
    )
    
def series(
    df: pd.DataFrame,
    win: int = 3600,
    step: int = 0,
    column: Optional[int] = 0,
    start_at_midnight: bool = False
) -> pd.DataFrame:
    """Sliding average of the entire sound level array, in terms of
    equivalent level (LEQ), and percentiles (L10, L50 and L90).

    Parameters
    ---------- 
    win: int, default 3600
        window size (in seconds) for the averaging function. For averages
        at a daily or weekly window, we recommend using the
        daily_weekly_indicators function instead.
    step: int, default 0
        step size (in seconds) to compute a sliding average. If set to 0
        (default value), the function will compute non-sliding averages.
    column: int, default 0
        column index to use for calculations. If None, the first column
        of the DataFrame will be used.
    start_at_midnight: bool, default False
        if set to True, the computation will start at midnight.

    Returns
    ---------- 
    DataFrame: dataframe containing datetime index and averaged
        Leq, L10, L50 and L90 at the corresponding columns

    """
    interval = core.get_interval(df)

    if interval > 1:
        warnings.warn(f"Computing the L10, L50, and L90 should be done with "
            "an integration time equal to or below 1s. Results"
            " might not be valid for these descriptors.\n")

    step = step // interval
    win = win // interval

    N = len(df)

    if step == 0:
        step = win

    if start_at_midnight:
        start_time = df.index[0].replace(
            hour=0, minute=0, second=0, microsecond=0)
        if df.index[0] > start_time:
            start_time += pd.Timedelta(days=1)
        start_index = df.index.get_indexer(
            [start_time], method='nearest')[0]
    else:
        start_index = 0

    NLim = max((N - start_index - win) // step + 1, 1)

    overallmean = np.zeros(NLim)
    overallL10 = np.zeros(NLim)
    overallL50 = np.zeros(NLim)
    overallL90 = np.zeros(NLim)
    overalltime = []

    nan_warning_issued = False

    for i in range(0, NLim):
        arr = df.iloc[
            int(start_index + i * step):int(start_index + i * step + win)
        ].iloc[:, column]
        overallmean[i] = core.equivalent_level(arr)
        if np.isnan(arr).all():
            if not nan_warning_issued:
                warnings.warn(f"All-NaN slice(s) encountered. "
                            "Check for gaps in the data.")
                nan_warning_issued = True
            overallL10[i] = np.nan
            overallL50[i] = np.nan
            overallL90[i] = np.nan
        else:
            overallL10[i] = np.nanpercentile(arr, 90)
            overallL50[i] = np.nanpercentile(arr, 50)
            overallL90[i] = np.nanpercentile(arr, 10)
        overalltime.append(df.index[int(start_index + i*step + win/2)])

    meandf = pd.DataFrame(
        index=overalltime,
        data={
            'Leq': overallmean,
            'L10': overallL10,
            'L50': overallL50,
            'L90': overallL90
        })
    return meandf