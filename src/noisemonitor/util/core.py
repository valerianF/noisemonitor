"""Core functions."""

import pandas as pd
import numpy as np
import warnings
import sys
import multiprocessing

from datetime import time
from typing import Union, Callable, Tuple, Optional, Dict

from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool


class CoverageWarning(UserWarning):
    """Warning emitted when data coverage is insufficient."""
    pass


def _column_to_index(df: pd.DataFrame, column: Union[int, str]) -> int:
    """Convert string column name to integer index if needed."""
    if isinstance(column, str):
        return df.columns.get_loc(column)
    return column

def get_interval(df):
    """Compute the interval in seconds between rows 2 and 3.
    
    In some datasets, the first row index may not be representative
    of the interval.
    """
    if len(df.index) < 3:
        raise ValueError(
            "DataFrame index must have at least three entries "
            "to compute interval."
        )
    return (df.index[2] - df.index[1]).seconds

def check_coverage(
    array: np.array,
    threshold: float = 0.5,
    emit_warning: bool = False
) -> bool:
    """Check if data coverage meets the specified threshold.
    
    Assesses the proportion of valid (non-NaN) values in an array and
    determines if it meets the minimum coverage requirement.
    
    Parameters
    ----------
    array: np.array
        Input array to assess for data coverage.
    threshold: float, default 0.5
        Minimum data coverage ratio required (0.0 to 1.0).
    emit_warning: bool, default False
        If True, emit a warning when coverage is insufficient.
    
    Returns
    -------
    bool: True if coverage meets threshold, False otherwise
    """
    if len(array) == 0:
        return False
    
    valid_mask = ~np.isnan(array)
    valid_count = np.sum(valid_mask)
    total_count = len(array)
    coverage_ratio = valid_count / total_count
    
    passes_threshold = bool(coverage_ratio >= threshold)
    
    if emit_warning and not passes_threshold:
        warnings.warn(
            f"Insufficient data coverage detected < "
            f"Some periods will be filtered and return NaN.",
            CoverageWarning,
            stacklevel=3
        )
    
    return passes_threshold

def equivalent_level(array: np.array) -> float:
    """Compute the equivalent sound level from the input array.

    Parameters
    ----------
    array: np.array
        Input array of sound levels in decibels.

    Returns
    ----------
    float
        Equivalent sound level in decibels.
    """

    if len(array) == 0 or np.isnan(array).all():
        return np.nan
            
    return 10*np.log10(np.nanmean(np.power(np.full(len(array), 10), array/10))) 

def harmonica(
        df: pd.DataFrame, 
        column: int,
        use_chunks: bool = True
        ) -> pd.DataFrame:
    """Compute the HARMONICA indicator and return a DataFrame with EVT, BGN, 
    and HARMONICA indicators as proposed in (Mietlicki et al., 2015).

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the LAeq,1s values with a time or datetime index.
    column: int
        Column index containing the LAeq,1s values.
    use_chunks: bool default True
        whether to process the data in chunks for large datasets.

    Returns
    ----------
    DataFrame: DataFrame with EVT, BGN, and HARMONICA indicators.
    """
    # Check interval
    interval = (df.index[1] - df.index[0]).total_seconds()
    if interval > 1:
        raise ValueError("Computing the HARMONICA indicator requires "
                    "an integration time equal to or below 1s. "
                    f"Current interval is {interval}s.")
        
    results = []
    previous_data = pd.DataFrame() 

    if use_chunks:
        try:
            with ProcessPoolExecutor() as executor:
                futures = []
                for hour, group in df.resample('h'):
                    futures.append(executor.submit(
                        hourly_harmonica, 
                        hour, 
                        group, 
                        column, 
                        interval, 
                        previous_data
                    ))
                    previous_data = group

                for future in as_completed(futures):
                    results.append(future.result())
        except (BrokenProcessPool, RuntimeError, OSError) as e:
            # If multiprocessing fails (e.g., in testing environments),
            # fall back to single-threaded processing
            warnings.warn(
                f"Parallel processing unavailable ({type(e).__name__}), "
                "using single-threaded processing instead.",
                RuntimeWarning,
                stacklevel=2
            )
            results = []
            previous_data = pd.DataFrame()
            for hour, group in df.resample('h'):
                results.append(hourly_harmonica(
                    hour, 
                    group, 
                    column, 
                    interval, 
                    previous_data
                ))
                previous_data = group
    else:
        for hour, group in df.resample('h'):
            results.append(hourly_harmonica(
                hour, 
                group, 
                column, 
                interval, 
                previous_data
            ))
            previous_data = group

    return pd.DataFrame(results).set_index('hour')

def hourly_harmonica(hour, group, column, interval, previous_data):
    """Compute a single hour of data to compute HARMONICA indicators."""
    # Filter previous_data to include only the last 10 minutes
    if not previous_data.empty:
        start_time = group.index[0] - pd.Timedelta(minutes=10)
        previous_data = previous_data.loc[previous_data.index >= start_time]

    # Combine the current hour's data with the filtered previous data
    combined_data = pd.concat([previous_data, group])

    if (len(group) != 3600 // interval) or \
    (group.iloc[:, column].isna().sum().sum() / len(group) > 0.2):
        return {
            'hour': hour, 
            'EVT': np.nan, 
            'BGN': np.nan, 
            'HARMONICA': np.nan
            }

    # Compute LAeq for the hour
    laeq = equivalent_level(group.iloc[:, column])

    # Compute LA95eq for the hour using a rolling window
    la95 = combined_data.iloc[:, column].rolling(
        window=int(600 // interval),
        step=int(max(1, 1//interval))
    ).apply(lambda x: np.nanpercentile(x, 5), raw=True)

    la95 = la95.loc[group.index]
    la95eq = equivalent_level(la95.dropna())

    # Compute EVT, BGN, and HARMONICA
    evt = 0.2 * (la95eq - 30)
    bgn = 0.25 * (laeq - la95eq)
    harmonica = bgn + evt

    return {
        'hour': hour, 
        'EVT': evt, 
        'BGN': bgn, 
        'HARMONICA': harmonica
        }

def lden(
    df: pd.DataFrame, 
    column: Union[int, str], 
    values: bool = False,
    coverage_check: bool = False,
    coverage_threshold: float = 0.5
    ) -> pd.DataFrame:
    """Compute the Lden value for a given DataFrame.

    Parameters
    ----------
    df: DataFrame
        DataFrame with a datetime index and sound level values.
    column: int or str
        column index (int) or column name (str) to use for calculations. 
        Should contain LAeq values.
    values: bool, default False
        If set to True, the function will return individual day, evening
        and night values in addition to the lden.
    coverage_check: bool, default False
        if set to True, assess data coverage and automatically filter periods
        with insufficient data coverage and emit warnings.
    coverage_threshold: float, default 0.5
        minimum data coverage ratio required (0.0 to 1.0).

    Returns
    ----------
    DataFrame: Lden value and optionally day, evening, and night levels.
    """
    column = _column_to_index(df, column)
    
    day = df.between_time(
        time(hour=7), time(hour=19)).iloc[:, column]
    evening = df.between_time(
        time(hour=19), time(hour=23)).iloc[:, column]
    night = df.between_time(
        time(hour=23), time(hour=7)).iloc[:, column]
    
    lday = np.round(equivalent_level(day), 2)
    levening = np.round(equivalent_level(evening), 2)
    lnight = np.round(equivalent_level(night), 2)

    lden = np.round(10 * np.log10(
        (
            12 * np.power(10, lday / 10)
            + 4 * np.power(10, (levening + 5) / 10)
            + 8 * np.power(10, (lnight + 10) / 10)
        ) / 24), 2)
    
    if coverage_check:
        passes_threshold_day = check_coverage(
            day,
            coverage_threshold
        )
        passes_threshold_evening = check_coverage(
            evening,
            coverage_threshold
        )
        passes_threshold_night = check_coverage(
            night,
            coverage_threshold
        )

        if not (passes_threshold_day and passes_threshold_evening and 
                    passes_threshold_night):
            warnings.warn(
                f"Insufficient data coverage detected for Lden computation. "
                "Some periods will be filtered and return NaN.",
                CoverageWarning,
                stacklevel=3
                )
            lden = np.nan
            if not passes_threshold_day:
                lday = np.nan
            if not passes_threshold_evening:
                levening = np.nan
            if not passes_threshold_night:
                lnight = np.nan

    if values:
        return pd.DataFrame({
            'Lden': [lden],
            'Lday': [lday],
            'Levening': [levening],
            'Lnight': [lnight]
        }, dtype='float64')
    return pd.DataFrame({'Lden': [lden]}, dtype='float64')

def noise_events(
    df: pd.DataFrame,
    column: int,
    threshold: float,
    min_gap: int
) -> int:
    """Compute the Number of Noise Events (NNE) in a DataFrame slice.
    
    According to the algorithm proposed in (Brown and De Coensel, 2018).
    Please note that this indicator is highly dependent on the refresh
    rate of the 
    data. Usually, NNEs are computed with LAeq,1s for traffic noise.

    Parameters
    ----------
    df: DataFrame
        DataFrame with a datetime index and sound level values.
    column: int
        Column index to use for calculations.
    threshold: float
        Threshold level in dB to define a noise event.
    min_gap: int
        Minimum time gap in seconds between successive noise events.

    Returns
    ----------
    int: Number of noise events.
    """
    if len(df) == 0:
        return 0
        
    events = 0
    in_event = False
    last_event_end = df.iloc[:, column].index[0]

    for timestamp, value in df.iloc[:, column].items():
        if value > threshold:
            if not in_event:
                in_event = True
                if (timestamp - last_event_end).total_seconds() >= min_gap:
                    events += 1
        else:
            if in_event:
                in_event = False
                last_event_end = timestamp

    return events


