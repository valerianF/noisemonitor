"""Core functions."""

import pandas as pd
import numpy as np
import warnings

from datetime import time
from typing import Union, Callable, Tuple, Optional, Dict

from concurrent.futures import ProcessPoolExecutor, as_completed


def _column_to_index(df: pd.DataFrame, column: Union[int, str]) -> int:
    """Convert string column name to integer index if needed."""
    if isinstance(column, str):
        return df.columns.get_loc(column)
    return column


def _assess_coverage(
    df: pd.DataFrame, period: str = 'D', threshold: float = 0.5
) -> pd.Series:
    """Assess data coverage using pandas groupby operations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index
    period : str, default 'D'
        Resampling period (e.g., 'D' for daily, 'W' for weekly,
        'h' for hourly, '1800s' for 30min windows)
    threshold : float, default 0.5
        Minimum coverage ratio required
    
    Returns
    -------
    pd.Series
        Boolean Series indicating which periods meet coverage threshold
    """
    # Count non-null values per period using resample
    valid_counts = df.resample(period).count().iloc[:, 0]
    total_counts = df.resample(period).size()
    
    # Calculate coverage ratio using vectorized operations 
    coverage_ratio = valid_counts / total_counts
    
    # Return boolean mask for periods meeting threshold
    return coverage_ratio >= threshold


def _assess_lden_coverage(
    df: pd.DataFrame, period: str = 'D', threshold: float = 0.5
) -> Dict[str, pd.Series]:
    """Assess coverage for Lden periods (day/evening/night).
    
    Parameters
    ---------- 
    df : pd.DataFrame
        Input DataFrame with datetime index
    period : str, default 'D'
        Resampling period for grouping (e.g., 'D' for daily, 'W' for weekly)
    threshold : float, default 0.5
        Minimum coverage ratio required
        
    Returns
    -------
    Dict[str, pd.Series]
        Dictionary with coverage assessment for each Lden period
    """
    
    # Define period masks using vectorized datetime operations
    hour = df.index.hour
    day_mask = (hour >= 7) & (hour < 19)
    evening_mask = (hour >= 19) & (hour < 23) 
    night_mask = (hour >= 23) | (hour < 7)
    
    coverage = {}
    for period_name, mask in [
        ('day', day_mask), ('evening', evening_mask), ('night', night_mask)
    ]:
        period_data = df[mask]
        if len(period_data) > 0:
            # Use the specified period for coverage assessment (not just daily)
            valid_counts = period_data.resample(period).count().iloc[:, 0]
            total_counts = period_data.resample(period).size()
            coverage[period_name] = (valid_counts / total_counts) >= threshold
        else:
            # Create empty series with proper datetime index
            # to avoid concatenation issues
            coverage[period_name] = pd.Series(
                [], dtype=bool, name=period_name
            )
    
    return coverage


def with_coverage_check(
    coverage_threshold: float = 0.5,
    period_type: str = 'auto'
):
    """Concise decorator for adding coverage checking to core functions.
    
    Uses pandas built-in functions for efficiency - no row iteration.
    When coverage_check=True, filtering and warnings are automatically applied.
    """
    def decorator(func):
        def wrapper(
            df: Union[pd.DataFrame, np.ndarray],
            column: Union[int, str] = 0,
            coverage_check: bool = True,
            coverage_threshold: float = coverage_threshold, 
                   freq: Optional[str] = None,
                   win: Optional[int] = None,
                   **kwargs):
            
            # Handle arrays and Series - check for sufficient non-NaN values
            if not isinstance(df, pd.DataFrame):
                if coverage_check:
                    # For arrays/Series, check if we have enough valid data
                    if isinstance(df, pd.Series):
                        data = df.values
                    else:
                        data = np.asarray(df)
                    
                    valid_count = np.sum(~np.isnan(data))
                    total_count = len(data)
                    
                    if total_count > 0:
                        coverage_ratio = valid_count / total_count
                        if coverage_ratio < coverage_threshold:
                            warnings.warn(
                                f"Coverage check: {valid_count}/{total_count} "
                                f"valid values ({coverage_ratio*100:.1f}%) below "
                                f"{coverage_threshold*100}% threshold",
                                UserWarning
                            )
                
                return func(df, column, **kwargs)
            
            # Skip coverage assessment if disabled
            if not coverage_check:
                col_idx = _column_to_index(df, column)
                original_kwargs = {
                    k: v for k, v in kwargs.items()
                    if k not in [
                        'coverage_check', 'coverage_threshold', 'freq', 'win'
                    ]
                }
                return func(df, col_idx, **original_kwargs)
            
            # Convert column to index if needed
            col_idx = _column_to_index(df, column)
            
            # Determine period for coverage assessment
            if period_type == 'lden':
                # Determine resampling period from function context
                if freq:
                    period = freq
                else:
                    period = 'D'  # Default to daily
                
                # For Lden, assess day/evening/night periods with the
                # specified frequency
                coverage = _assess_lden_coverage(
                    df, period=period, threshold=coverage_threshold
                )
                # Filter out empty series before concatenation
                non_empty_coverage = {
                    k: v for k, v in coverage.items() if not v.empty
                }
                if non_empty_coverage:
                    meets_threshold = pd.concat(
                        non_empty_coverage.values()
                    ).groupby(level=0).all()
                else:
                    # If all periods are empty, create empty boolean series
                    meets_threshold = pd.Series([], dtype=bool)
            else:
                # Determine resampling period from function context
                if freq:
                    # From summary.periodic - use freq parameter
                    period = freq
                elif win:
                    # From profile.periodic - convert window size
                    # to resampling period
                    period = f'{win}s'
                else:
                    # Default to daily
                    period = 'D'
                
                meets_threshold = _assess_coverage(
                    df, period=period, threshold=coverage_threshold
                )
            
            # Apply filtering automatically when coverage check is enabled
            if len(meets_threshold) > 0 and not meets_threshold.all():
                filtered_df = df.copy()
                # Set periods that don't meet threshold to NaN
                for date in meets_threshold[~meets_threshold].index:
                    if (period_type == 'lden' or
                        period in ['D', 'W', 'ME', 'W-MON', 'MS']):
                        # For daily/weekly/monthly periods,
                        # filter entire periods
                        if period == 'D':
                            # Use pd.Timestamp.normalize() instead of
                            # deprecated .date()
                            date_normalized = pd.Timestamp(date).normalize()
                            next_day = (
                                date_normalized + pd.Timedelta(days=1)
                            )
                            mask = (
                                (filtered_df.index >= date_normalized) &
                                (filtered_df.index < next_day)
                            )
                        elif period in ['W', 'W-MON']:
                            # For weekly, filter the entire week
                            week_start = pd.Timestamp(date)
                            week_end = (
                                week_start + pd.Timedelta(weeks=1)
                            )
                            mask = (
                                (filtered_df.index >= week_start) &
                                (filtered_df.index < week_end)
                            )
                        elif period in ['ME', 'MS']:
                            # For monthly, filter the entire month
                            month_start = pd.Timestamp(date)
                            month_end = (
                                month_start + pd.DateOffset(months=1)
                            )
                            mask = (
                                (filtered_df.index >= month_start) &
                                (filtered_df.index < month_end)
                            )
                    else:
                        # For window-based periods (e.g., '3600s')
                        date_start = pd.Timestamp(date)
                        
                        # Window-based (e.g., '3600s')
                        if period.endswith('s'):
                            try:
                                seconds = int(period[:-1])
                                date_end = (
                                    date_start +
                                    pd.Timedelta(seconds=seconds)
                                )
                            except ValueError:
                                # Fallback for invalid period format
                                date_end = (
                                    date_start + pd.Timedelta(hours=1)
                                )
                        else:
                            date_end = date_start + pd.Timedelta(days=1)
                        
                        mask = (
                            (filtered_df.index >= date_start) &
                            (filtered_df.index < date_end)
                        )
                    
                    filtered_df.loc[mask, :] = np.nan
                
                # Emit warning automatically
                n_filtered = (~meets_threshold).sum()
                n_total = len(meets_threshold)
                pct = (n_filtered / n_total * 100) if n_total > 0 else 0
                warnings.warn(
                    f"Coverage filter: {n_filtered}/{n_total} periods "
                    f"({pct:.1f}%) below {coverage_threshold*100}% threshold",
                    UserWarning
                )
            else:
                filtered_df = df
            
            # Call original function with original parameters
            # (excluding coverage ones)
            original_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in [
                    'coverage_check', 'coverage_threshold', 'freq', 'win'
                ]
            }
            result = func(filtered_df, col_idx, **original_kwargs)
            
            # Store coverage info as result attribute
            if hasattr(result, 'attrs'):
                result.attrs['coverage_check'] = meets_threshold
                
            return result
            
        # Preserve original function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


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

@with_coverage_check()
def equivalent_level(
    data: Union[np.ndarray, pd.DataFrame],
    column: Union[int, str] = 0
) -> float:
    """Compute the equivalent sound level from the input.
    
    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data - can be a numpy array or DataFrame with datetime index
    column : int or str, default 0
        Column index or name if data is a DataFrame. Ignored for arrays.
        
    Returns
    -------
    float
        Equivalent sound level in dB
    """
    # Handle DataFrame or array input
    if isinstance(data, pd.DataFrame):
        col_idx = _column_to_index(data, column)
        array = data.iloc[:, col_idx].values
    else:
        array = data
    
    # Compute equivalent level
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

@with_coverage_check(period_type='lden')
def lden(
    df: pd.DataFrame, 
    column: Union[int, str], 
    values: bool = False,
    coverage_check: bool = True,
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

    Returns
    ----------
    DataFrame: Lden value and optionally day, evening, and night levels.
    """
    column = _column_to_index(df, column)
    
    lday = equivalent_level(df.between_time(
        time(hour=7), time(hour=19)).iloc[:, column])
    levening = equivalent_level(df.between_time(
        time(hour=19), time(hour=23)).iloc[:, column])
    lnight = equivalent_level(df.between_time(
        time(hour=23), time(hour=7)).iloc[:, column])

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
        }, dtype='float64')
    return pd.DataFrame({'lden': [np.round(lden, 2)]}, dtype='float64')





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


