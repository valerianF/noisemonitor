"""Core functions."""

import pandas as pd
import numpy as np
import warnings
import sys
import multiprocessing
import re

from datetime import time
from typing import Union, Callable, Tuple, Optional, Dict

from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool


# Mantissas of the preferred one-third-octave frequencies (ISO 266, Table 1)
_PREFERRED_MANTISSAS = (1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0)

# Largest relative deviation of a frequency from the preferred frequency of a
# one-third-octave band for the frequency to be read as that band: 1.59 %,
# between preferred and base-2 frequencies from 20 Hz to 20 kHz (ISO 266)
_BAND_TOLERANCE = 0.0159

# Frequency weightings A and C of IEC 61672-1:2013, Table 3, in dB, by nominal
# one-third-octave mid-band frequency in Hz.
_A_WEIGHTING = {
    10.0: -70.4, 12.5: -63.4, 16.0: -56.7, 20.0: -50.5, 25.0: -44.7,
    31.5: -39.4, 40.0: -34.6, 50.0: -30.2, 63.0: -26.2, 80.0: -22.5,
    100.0: -19.1, 125.0: -16.1, 160.0: -13.4, 200.0: -10.9, 250.0: -8.6,
    315.0: -6.6, 400.0: -4.8, 500.0: -3.2, 630.0: -1.9, 800.0: -0.8,
    1000.0: 0.0, 1250.0: 0.6, 1600.0: 1.0, 2000.0: 1.2, 2500.0: 1.3,
    3150.0: 1.2, 4000.0: 1.0, 5000.0: 0.5, 6300.0: -0.1, 8000.0: -1.1,
    10000.0: -2.5, 12500.0: -4.3, 16000.0: -6.6, 20000.0: -9.3
}
_C_WEIGHTING = {
    10.0: -14.3, 12.5: -11.2, 16.0: -8.5, 20.0: -6.2, 25.0: -4.4,
    31.5: -3.0, 40.0: -2.0, 50.0: -1.3, 63.0: -0.8, 80.0: -0.5,
    100.0: -0.3, 125.0: -0.2, 160.0: -0.1, 200.0: 0.0, 250.0: 0.0,
    315.0: 0.0, 400.0: 0.0, 500.0: 0.0, 630.0: 0.0, 800.0: 0.0,
    1000.0: 0.0, 1250.0: 0.0, 1600.0: -0.1, 2000.0: -0.2, 2500.0: -0.3,
    3150.0: -0.5, 4000.0: -0.8, 5000.0: -1.3, 6300.0: -2.0, 8000.0: -3.0,
    10000.0: -4.4, 12500.0: -6.2, 16000.0: -8.5, 20000.0: -11.2
}

# Low-frequency criteria on the difference between the C- and A-weighted
# levels of a range of one-third-octave bands (nominal Hz, both included).
# Low-frequency content is flagged when the difference exceeds the threshold,
# or reaches it when 'inclusive'.
_LF_METHODS = {
    # WHO (Berglund et al., 1999), after Botsford, on the whole spectrum
    # (20 Hz to 20 kHz, as in Montano & Gushiken, 2023)
    'botsford': {'bands': (20.0, 20000.0), 'threshold': 10.0,
                'inclusive': False},
    # Montano & Gushiken (2023), Equation 3, on the low-frequency range of
    # ISO 1996-2:2017 (16 Hz to 200 Hz)
    'montano': {'bands': (16.0, 200.0), 'threshold': 15.0,
                'inclusive': False},
    # Decree 176/2009 of Catalonia, Annex 3, 3.2 (also Barcelona's
    # environmental bylaw): no significant low frequency below 20 dB
    'barcelona': {'bands': (20.0, 160.0), 'threshold': 20.0,
                'inclusive': True},
}


class CoverageWarning(UserWarning):
    """Warning emitted when data coverage is insufficient."""
    pass


class IntegrationTimeWarning(UserWarning):
    """Warning emitted when an indicator needs a finer integration time
    (1 s or below) than the one of the data, e.g. percentiles or noise
    events computed from LAeq,1min."""
    pass


def _column_to_index(df: pd.DataFrame, column: Union[int, str]) -> int:
    """Convert string column name to integer index if needed."""
    if isinstance(column, str):
        return df.columns.get_loc(column)
    return column


def _parse_freq_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert frequency column names to float values (Hz), e.g. 63, '125 Hz',
    '1.25k' or '1 kHz'.

    A frequency within 1.59 % of the preferred frequency of a one-third-octave
    band (ISO 266, Table 1) is given that frequency: e.g., 31.62 (exact,
    1000 * 10**(n/10) Hz with n = -15), 31.6 and 31.25 (base-2) all give
    31.5. Other frequencies keep their parsed value.

    If any column cannot be parsed, the original dataframe is returned
    unchanged and a warning is issued. Two columns giving the same frequency
    raise a ValueError.
    """

    def nominal(value):
        if value <= 0:
            return value
        # Band n of ISO 266 (exact frequency 1000 * 10**(n/10) Hz) and its
        # preferred frequency, a mantissa of Table 1 times a power of ten
        # (rounded to remove float errors, e.g. 3.15 * 10 = 31.499999...)
        n = int(round(10 * np.log10(value / 1000)))
        preferred = round(_PREFERRED_MANTISSAS[n % 10] * 10.0 ** (n // 10 + 3),
                          10)
        if abs(value - preferred) > _BAND_TOLERANCE * preferred:
            return value
        return preferred

    def parse_band(col):
        if isinstance(col, (int, float, np.integer, np.floating)) \
                and not isinstance(col, bool):
            return nominal(float(col))

        if isinstance(col, str):
            s = col.strip().lower().replace(" ", "")

            # Match number + optional 'k' + optional 'hz'
            match = re.fullmatch(r"(\d*\.?\d+)(k)?hz?", s)
            if not match:
                match = re.fullmatch(r"(\d*\.?\d+)(k)?", s)

            if match:
                value = float(match.group(1))
                if match.group(2):  # 'k' present
                    value *= 1000
                return nominal(value)

        raise ValueError(f"Could not parse column name: {col}. "
                        "Frequency columns headers are left unchanged.")

    try:
        new_columns = pd.Index(df.columns).map(parse_band)
    except ValueError as e:
        warnings.warn(str(e))
        return df

    if new_columns.has_duplicates:
        duplicated = [
            f"{', '.join(repr(c) for c in df.columns[new_columns == band])}"
            f" ({band:g} Hz)"
            for band in new_columns[new_columns.duplicated()].unique()
        ]
        raise ValueError(
            "Several columns give the same frequency band: "
            + "; ".join(duplicated) + ".")

    return df.set_axis(new_columns, axis=1)


def _lf_levels(
    df: pd.DataFrame,
    method: Union[str, list]
) -> pd.DataFrame:
    """A- and C-weighted levels of each sample over the band range of each
    low-frequency method (see _LF_METHODS), from Z-weighted one-third-octave
    band levels. A sample is NaN for a method if any of its bands is NaN.

    Returns a DataFrame with the index of df and MultiIndex columns
    (Indicator 'LA' or 'LC', Method). Raises a ValueError for an unknown
    method, a column name that is not a frequency, or a missing band.
    """
    methods = [method] if isinstance(method, str) else list(method)
    unknown = [m for m in methods if m not in _LF_METHODS]
    if not methods or unknown:
        raise ValueError(
            f"Unknown low-frequency method(s): {unknown}. Use one or several "
            f"of {list(_LF_METHODS)}.")
    methods = list(dict.fromkeys(methods))

    with warnings.catch_warnings():
        # A column name that is not a frequency raises below instead
        warnings.simplefilter('ignore')
        bands = _parse_freq_columns(df)
        if not all(isinstance(c, float) for c in bands.columns):
            names = [c for c in df.columns if not isinstance(
                _parse_freq_columns(pd.DataFrame(columns=[c])).columns[0],
                float)]
            raise ValueError(
                "All columns must be frequency bands (e.g. '20', '25 Hz' or "
                f"'1.25k'); could not read {names} as frequencies.")

    levels = {}
    for m in methods:
        f_min, f_max = _LF_METHODS[m]['bands']
        required = [f for f in _A_WEIGHTING if f_min <= f <= f_max]
        missing = [f for f in required if f not in bands.columns]
        if missing:
            raise ValueError(
                f"The '{m}' method needs all one-third-octave bands from "
                f"{f_min:g} Hz to {f_max:g} Hz; missing: "
                f"{', '.join(f'{f:g}' for f in missing)} Hz.")
        z = bands[required].to_numpy(dtype=float)
        for indicator, weighting in (('LA', _A_WEIGHTING),
                                    ('LC', _C_WEIGHTING)):
            weights = np.array([weighting[f] for f in required])
            levels[(indicator, m)] = 10 * np.log10(
                np.sum(np.power(10, (z + weights) / 10), axis=1))

    result = pd.DataFrame(levels, index=df.index)
    result.columns.names = ['Indicator', 'Method']
    return result


def _lf_apply(
    func: Callable,
    levels: pd.DataFrame,
    flag: bool = False,
    **kwargs
) -> pd.DataFrame:
    """Low-frequency indicators from the output of _lf_levels: func, a
    single-column function of noisemonitor returning an 'Leq' column (e.g.,
    profile.series), is run on each level column with kwargs, giving 'LA' and
    'LC' per window or period, then 'C-A' and, if flag, 'Lowfreq_Flag'
    (nullable boolean, <NA> where C-A is NaN).

    Statistical levels and the Lden of func are not used: their warnings
    (IntegrationTimeWarning, numpy RuntimeWarning, CoverageWarning of the
    Lden) are dropped, and other warnings are emitted once.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        leq = pd.concat(
            [func(levels, column=i, **kwargs)['Leq']
            for i in range(levels.shape[1])],
            axis=1)
    leq.columns = levels.columns

    emitted = set()
    for w in caught:
        if (issubclass(w.category, (IntegrationTimeWarning, RuntimeWarning))
                or (issubclass(w.category, CoverageWarning)
                    and 'Lden' in str(w.message))):
            continue
        key = (w.category, str(w.message))
        if key not in emitted:
            emitted.add(key)
            warnings.warn(w.message, stacklevel=3)

    methods = levels.columns.get_level_values('Method').unique()
    results = {}
    for indicator in ('LA', 'LC'):
        for m in methods:
            results[(indicator, m)] = leq[(indicator, m)]
    for m in methods:
        results[('C-A', m)] = leq[('LC', m)] - leq[('LA', m)]
    if flag:
        for m in methods:
            difference = results[('C-A', m)]
            threshold = _LF_METHODS[m]['threshold']
            if _LF_METHODS[m]['inclusive']:
                flagged = difference >= threshold
            else:
                flagged = difference > threshold
            results[('Lowfreq_Flag', m)] = (
                flagged.astype('boolean').mask(difference.isna()))

    result = pd.DataFrame(results)
    result.columns.names = ['Indicator', 'Method']
    return result

def get_interval(df):
    """Compute the interval in seconds between rows 2 and 3, as a float
    (e.g., 0.5 for 500 ms data, 86400 for daily data).

    In some datasets, the first row index may not be representative
    of the interval.
    """
    if len(df.index) < 3:
        raise ValueError(
            "DataFrame index must have at least three entries "
            "to compute interval."
        )
    return (df.index[2] - df.index[1]).total_seconds()

def _laeq_1s(levels: pd.Series) -> pd.Series:
    """Average sub-second levels energetically over each whole second
    (LAeq,1s), labelled by the start of the second. NaN values are
    ignored; a second without any value gives NaN. The sampling interval is
    assumed to divide one second (e.g. 100, 125, 250 or 500 ms)."""
    energy = np.power(10, levels.astype(float) / 10).resample('1s').mean()
    return 10 * np.log10(energy)

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
            f"Insufficient data coverage detected: "
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
    """Compute the hourly HARMONICA index and its EVT and BGN components, 
    as proposed by Bruitparif (Mietlicki et al., 2015):
    BGN = 0.2 (LA95eq - 30), EVT = 0.25 (LAeq - LA95eq) and
    HARMONICA = BGN + EVT, where LAeq is the equivalent level of the hour
    and LA95eq the equivalent level of its LA95 values. One LA95 is computed
    every second t of the hour over the 10 min before it, [t - 600 s, t),
    as the 5th percentile taken as the next higher sample (numpy method
    'higher'), from the values available in the window; a window holding
    fewer than 50% of its expected values gives no LA95, so an hour without
    data before it uses the LA95 values from 5 min into the hour. An hour
    with less than 80% of its expected values returns NaN. HARMONICA is not
    capped: values above 10 are possible.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing LAeq values at 1 s or finer with a datetime
        index. The index was designed on LAeq,1s: sub-second levels are 
        first averaged energetically over each whole second (LAeq,1s), 
        assuming that their interval divides one second (e.g., 100, 125, 
        250 or 500 ms).
    column: int
        Column index containing the LAeq values.
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
    if interval < 1:
        df = _laeq_1s(df.iloc[:, column]).to_frame()
        column = 0
        interval = 1.0

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
    """Compute the HARMONICA indicators of one hour (see harmonica)."""
    # Return NaN only when fewer than 80% of the expected data points
    # (both missing rows and NaN values) are present for the hour.
    expected_count = int(round(3600 / interval))
    nan_count = group.iloc[:, column].isna().sum()
    missing_count = max(0, expected_count - len(group))
    valid_ratio = (expected_count - missing_count - nan_count) / expected_count

    if valid_ratio < 0.8:
        return {
            'hour': hour, 
            'EVT': np.nan, 
            'BGN': np.nan, 
            'HARMONICA': np.nan
            }

    # Levels of the hour, preceded by those of the previous hour (if any)
    levels = group.iloc[:, column]
    if not previous_data.empty:
        levels = pd.concat([previous_data.iloc[:, column], levels])
    times = levels.index
    values = levels.to_numpy(dtype=float)

    # One LA95 per second t of the hour, over the 10 min before it,
    # [t - 600 s, t), from the values available in this window. LA95 is the
    # 5th percentile taken as the next higher sample (numpy method 'higher').
    # A window holding fewer than half of its expected values gives no LA95.
    window_count = int(round(600 / interval))
    ends = group.index[::max(1, int(round(1.0 / interval)))]
    starts = times.searchsorted(ends - pd.Timedelta(seconds=600))
    stops = times.searchsorted(ends)
    la95 = np.full(len(ends), np.nan)
    for i, (start, stop) in enumerate(zip(starts, stops)):
        window = values[start:stop]
        window = window[~np.isnan(window)]
        if len(window) >= window_count / 2:
            la95[i] = np.percentile(window, 5, method='higher')

    laeq = equivalent_level(group.iloc[:, column])
    la95eq = equivalent_level(la95)

    # Compute EVT, BGN, and HARMONICA
    bgn = 0.2 * (la95eq - 30)
    evt = 0.25 * (laeq - la95eq)
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
    coverage_threshold: float = 0.5,
    start_day: int = 7,
    start_evening: int = 19,
    start_night: int = 23
    ) -> pd.DataFrame:
    """Compute the Lden value for a given DataFrame, following the
    EU Directive 2002/49/EC: Lden = 10 lg[(t_day 10^(Lday/10) +
    t_evening 10^((Levening+5)/10) + t_night 10^((Lnight+10)/10)) / 24],
    where t_day, t_evening and t_night are the durations of the periods in
    hours (ISO 1996-1:2016). The Directive defines Lden over a year;
    here it is computed over all the data in df. Defaults are the
    Directive's 07:00-19:00, 19:00-23:00 and 23:00-07:00 (12, 4 and 8
    hours); other start hours may be chosen. Periods are half-open, e.g.,
    [07:00, 19:00).

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
    start_day: int, default 7
        hour (0-23) at which the day period starts.
    start_evening: int, default 19
        hour (0-23) at which the evening period starts.
    start_night: int, default 23
        hour (0-23) at which the night period starts.

    Returns
    ----------
    DataFrame: Lden value and optionally day, evening, and night levels.
    """
    column = _column_to_index(df, column)

    starts = {
        'start_day': start_day,
        'start_evening': start_evening,
        'start_night': start_night
    }
    for name, value in starts.items():
        if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
                or not 0 <= value <= 23):
            raise ValueError(
                f"{name} must be an integer between 0 and 23, got {value!r}.")

    t_day = (start_evening - start_day) % 24
    t_evening = (start_night - start_evening) % 24
    t_night = (start_day - start_night) % 24

    if 0 in (t_day, t_evening, t_night) or t_day + t_evening + t_night != 24:
        raise ValueError(
            "start_day, start_evening and start_night must be distinct and "
            "follow each other in this order over 24 hours, got "
            f"{start_day}, {start_evening} and {start_night}.")
    
    day = df.between_time(
        time(hour=start_day), time(hour=start_evening),
        inclusive='left').iloc[:, column]
    evening = df.between_time(
        time(hour=start_evening), time(hour=start_night),
        inclusive='left').iloc[:, column]
    night = df.between_time(
        time(hour=start_night), time(hour=start_day),
        inclusive='left').iloc[:, column]
    
    lday = equivalent_level(day)
    levening = equivalent_level(evening)
    lnight = equivalent_level(night)

    lden = 10 * np.log10(
        (
            t_day * np.power(10, lday / 10)
            + t_evening * np.power(10, (levening + 5) / 10)
            + t_night * np.power(10, (lnight + 10) / 10)
        ) / 24)
    
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
    Events are detected on the sound level time series as given, at its
    own resolution (e.g. LAeq,1s, or sub-second levels). Please note that
    the number of events depends on the time weighting and on the sampling
    rate of the data.

    An exceedance is a run of samples strictly above the threshold (NaN
    values are not above it). The slice is treated as a record of its own:
    its first exceedance is always an event, even when the slice starts
    within it. Each following exceedance is a new event if the time from
    the end of the previous exceedance (its first sample at or below the
    threshold) to its first sample is at least min_gap; otherwise it
    extends the current event (Baclet and Rumpler, 2024).

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
    levels = df.iloc[:, column]
    above = levels.to_numpy(dtype=float, na_value=np.nan) > threshold
    if not above.any():
        return 0

    # Exceedance i covers the samples starts[i] to ends[i] - 1
    edges = np.diff(above.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    gaps = (levels.index[starts[1:]] - levels.index[ends[:-1]]).total_seconds()

    return int(1 + np.count_nonzero(gaps >= min_gap))


