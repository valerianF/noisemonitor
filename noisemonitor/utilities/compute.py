""" Various functions used across noisemonitor modules to compute indicators. 
"""

import pandas as pd
import numpy as np
import warnings

from datetime import time

from concurrent.futures import ProcessPoolExecutor, as_completed

def equivalent_level(array: np.array) -> float:
    """Compute the equivalent sound level from the input array."""
    if len(array) == 0 or np.isnan(array).all():
        return np.nan
    return 10*np.log10(np.mean(np.power(np.full(len(array), 10), array/10))) 

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
        Column name containing the LAeq,1s values.
    use_chunks: bool default True
        whether to process the data in chunks for large datasets.

    Returns
    ----------
    DataFrame: DataFrame with EVT, BGN, and HARMONICA indicators.
    """
    # Check interval
    interval = (df.index[1] - df.index[0]).total_seconds()
    if interval > 1:
        warnings.warn("Computing the HARMONICA indicator should be done with "
                      "an integration time equal to or below 1s. Results might"
                      " not be valid.\n")
        
    results = []
    previous_data = pd.DataFrame() 

    if use_chunks:
        with ProcessPoolExecutor() as executor:
            futures = []
            for hour, group in df.resample('H'):
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
    else:
        for hour, group in df.resample('H'):
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
    (group[column].isna().sum().sum() / len(group) > 0.2):
        return {
            'hour': hour, 
            'EVT': np.nan, 
            'BGN': np.nan, 
            'HARMONICA': np.nan
            }

    # Compute LAeq for the hour
    laeq = equivalent_level(group[column])

    # Compute LA95eq for the hour using a rolling window
    la95 = combined_data[column].rolling(
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
    column: str, 
    values: bool = False
    ) -> pd.DataFrame:
    """Compute the Lden value for a given DataFrame.

    Parameters
    ----------
    df: DataFrame
        DataFrame with a datetime index and sound level values.
    column: str
        column name to use for calculations. Should contain LAeq values.
    values: bool, default False
        If set to True, the function will return individual day, evening
        and night values in addition to the lden.

    Returns
    ----------
    DataFrame: Lden value and optionally day, evening, and night levels.
    """
    lday = equivalent_level(df.between_time(
        time(hour=7), time(hour=19))[column])
    levening = equivalent_level(df.between_time(
        time(hour=19), time(hour=23))[column])
    lnight = equivalent_level(df.between_time(
        time(hour=23), time(hour=7))[column])

    lden = 10 * np.log10(
        (
            12 * np.power(10, lday / 10)
            + 4 * np.power(10, (levening + 5) / 10)
            + 8 * np.power(10, (lnight + 10) / 10)
        ) / 24)

    if values:
        return pd.DataFrame({
            'lden': [np.round(lden, 2)],
            'lday': [np.round(lday, 2)],
            'levening': [np.round(levening, 2)],
            'lnight': [np.round(lnight, 2)]
        })
    return pd.DataFrame({'lden': [np.round(lden, 2)]})

def noise_events(
    df: pd.DataFrame,
    column: str,
    threshold: float,
    min_gap: int
) -> int:
    """Compute the Number of Noise Events (NNE) in a DataFrame slice, 
    according to the algorithm proposed in (Brown and De Coensel, 2018). Please 
    note that this indicator is highly dependent on the refresh rate of the 
    data. Usually, NNEs are computed with LAeq,1s for traffic noise.

    Parameters
    ----------
    df: DataFrame
        DataFrame with a datetime index and sound level values.
    column: str
        Column name to use for calculations.
    threshold: float
        Threshold level in dB to define a noise event.
    min_gap: int
        Minimum time gap in seconds between successive noise events.

    Returns
    ----------
    int: Number of noise events.
    """
    events = 0
    in_event = False
    last_event_end = df[column].index[0]

    for timestamp, value in df[column].items():
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