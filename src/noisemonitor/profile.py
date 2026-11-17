import pandas as pd
import numpy as np
import warnings

from datetime import time
from typing import Optional, Union

from .util import filter
from .util import core

def periodic(
    df: pd.DataFrame,
    hour1: int,
    hour2: int,
    day1: Optional[str] = None,
    day2: Optional[str] = None,
    column: Optional[Union[int, str]] = 0,
    win: int = 3600,
    step: int = 0,
    traffic_noise_indicators: bool = False,
    roughness_indicators: bool = False,
    stat: Optional[Union[int, list]] = None,
    coverage_check: bool = False,
    coverage_threshold: float = 0.5
) -> pd.DataFrame:
    """Compute daily or weekly rolling averages of the sound level, in 
    terms of equivalent level (Leq), statistical levels (L10, L50 and 
    L90), and optionally other indicators (e.g., Traffic Noise Index). 
    Percentiles are computed by linear interpolation between samples
    (numpy's default method).

    For statistical and other percentile-based levels, data should be sampled
    at 1 s or finer (e.g., 100 ms) although they are computed
    from the samples as given, so their values depend on the integration time
    of the data.

    Each window pools the samples of all the selected days whose time of
    day falls in it: its levels are computed over these pooled samples, not
    as the mean of daily values (e.g., the L90 of the 08:00-09:00 window is
    the 10th percentile of all the samples between 08:00 and 09:00, and its
    Leq their energetic mean).

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with a datetime index and sound level values.
    hour1: int, between 0 and 23
        hour at which the first window starts.
    hour2: int, between 0 and 23
        hour at which the last window starts (included): with win=3600,
        hour1=0 and hour2=23 give the 24 hours of the day. If
        hour1 > hour2, the windows wrap past midnight.
    day1: Optional[str], default None
        First day of the week included in the weekly average.
    day2: Optional[str], default None
        Last day of the week included in the weekly average.
    column: int or str, default 0
        column index (int) or column name (str) to use for calculations. 
        If None, the first column of the DataFrame will be used.
    win: int, default 3600
        window size for the averaging function, in whole seconds. Each
        window covers the half-open interval [start, start + win): a
        timestamp marks the start of its integration period.
    step: int, default 0
        step size to compute a rolling average, in whole seconds. If set to
        0 (default value), the function will compute non-rolling averages.
    traffic_noise_indicators: bool, default False
        if set to True, the function will compute traffic noise indicators
        Traffic Noise Index (Griffiths and Langdon, 1968) as well as the
        Noise Pollution Level (Robinson, 1971) in addition to the
        equivalent levels and percentiles, for each window:
        TNI = 4 (L10 - L90) + L90 - 30 and NPL = Leq + 2.56 σ, σ being the
        standard deviation of the levels of the window (sample standard
        deviation, n - 1). Griffiths and Langdon defined TNI on the means of
        24 hourly L10 and L90 values; the formula being linear, their daily
        TNI is the mean of the 24 hourly values of 1 h windows over one day.
    roughness_indicators: bool, default False
        if set to True, the function will compute roughness indicators
        based on difference between consecutive LAeq,1s values according
        to (DeFrance et al., 2010): from the absolute differences |dL|,
        dLav is their mean, dLmax,1 the mean of the largest 1 % and dLmin,90
        the mean of the lowest 90 % (percentiles by linear interpolation,
        values at the percentile included). Sub-second data are first averaged
        energetically over each whole second (LAeq,1s), assuming that their
        interval divides one second (e.g., 100, 125, 250 or 500 ms). Only
        samples one sampling interval apart are differenced: pairs with a
        missing value, split by a gap, or across the junction between days
        pooled in a window are left out. A window without any such pair
        gives NaN.
    stat: int, list of ints, or None, default None
        statistical level(s) to compute (between 1 and 99). If an int is provided,
        computes the corresponding percentile (e.g., stat=10 computes L10). If a list
        is provided, computes all specified percentiles. Results are added to the output
        DataFrame with column names like 'L10', 'L5', etc.
    coverage_check: bool, default False
        if set to True, assess data coverage and automatically filter periods
        with insufficient data coverage and emit warnings. Coverage counts
        the samples present in the data: days outside the data range are
        not counted as missing. Windows without any value (all NaN, or no
        sample) are NaN in any case, with a CoverageWarning.
    coverage_threshold: float, default 0.5
        minimum data coverage ratio required (0.0 to 1.0).

    Returns
    ----------
    DataFrame: dataframe containing time index and daily averaged
        Leq, L10, L50 and L90 at the corresponding columns

    """
    column = core._column_to_index(df, column)

    for name, value in {'hour1': hour1, 'hour2': hour2}.items():
        if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
                or not 0 <= value <= 23):
            raise ValueError(
                f"{name} must be an integer between 0 and 23 (hour at which "
                f"the first or last window starts), got {value!r}.")

    if core.get_interval(df) > 1:
        warnings.warn("Computing the L10, L50, L90, traffic, "
            "roughness noise indicators or other statistical levels "
            "should be done with an integration time equal to or below 1s. "
            "Results might not be valid for these indicators.\n",
            core.IntegrationTimeWarning, stacklevel=2)

    if step == 0:
        step = win

    temp = filter._days(df, day1, day2)

    NLim = ((hour2-hour1)%24*3600)//step + 1

    # Validate and process stat argument
    stat_levels = []
    if stat is not None:
        if isinstance(stat, int):
            stat_levels = [stat]
        elif isinstance(stat, list):
            stat_levels = stat
        else:
            raise ValueError("stat must be an int, a list of ints, or None")
        
        # Validate that all values are between 1 and 99
        for s in stat_levels:
            if not isinstance(s, int) or s < 1 or s > 99:
                raise ValueError(f"stat values must be integers between 1 and 99, got {s}")

    averages = {
        'Leq': np.full(NLim, np.nan),
        'L10': np.full(NLim, np.nan),
        'L50': np.full(NLim, np.nan),
        'L90': np.full(NLim, np.nan)
    }

    for s in stat_levels:
        averages[f'L{s}'] = np.full(NLim, np.nan)

    if traffic_noise_indicators:
        averages.update({
            'TNI': np.full(NLim, np.nan),
            'NPL': np.full(NLim, np.nan)
        })
    if roughness_indicators:
        averages.update({
            'dLav': np.full(NLim, np.nan),
            'dLmax,1': np.full(NLim, np.nan),
            'dLmin,90': np.full(NLim, np.nan)
        })

    if roughness_indicators:
        interval = df.index[2] - df.index[1]
        rough_levels = None
        if interval < pd.Timedelta(seconds=1):
            rough_levels = core._laeq_1s(temp.iloc[:, column])
            interval = pd.Timedelta(seconds=1)

    times = []
    coverage_warning_issued = False

    for i in range(0, NLim):
        t = hour1*3600 + i*step + win//2
        t1 = hour1*3600 + i*step
        t2 = hour1*3600 + i*step + win

        start = time(
            hour=(t1//3600)%24,
            minute=(t1%3600)//60,
            second=(t1%3600)%60
        )
        end = time(
            hour=(t2//3600)%24,
            minute=(t2%3600)//60,
            second=(t2%3600)%60
        )
        temp_slice = temp.between_time(start, end, inclusive='left')

        times.append(time(
            hour=(t//3600)%24, 
            minute=(t%3600)//60, 
            second=(t%3600)%60
            ))

        arr = temp_slice.iloc[:, column]

        if coverage_check:
            passes_threshold = core.check_coverage(
                arr,
                coverage_threshold,
                emit_warning = not coverage_warning_issued
            )

            if not passes_threshold:
                coverage_warning_issued = True
                continue

        # A window without any value (all NaN, or no sample) stays NaN
        if not arr.notna().any():
            if not coverage_warning_issued:
                warnings.warn("No data in some windows: they return NaN.",
                            core.CoverageWarning, stacklevel=2)
                coverage_warning_issued = True
            continue

        averages['Leq'][i] = core.equivalent_level(arr)
        averages['L10'][i] = np.nanpercentile(arr, 90)
        averages['L50'][i] = np.nanpercentile(arr, 50)
        averages['L90'][i] = np.nanpercentile(arr, 10)
        
        for s in stat_levels:
            averages[f'L{s}'][i] = np.nanpercentile(arr, 100 - s)

        if traffic_noise_indicators:
            L10 = averages['L10'][i]
            L90 = averages['L90'][i]
            Leq= averages['Leq'][i]
            sigma_Leq= arr.std()

            # Traffic Noise Index (TNI)
            averages['TNI'][i] = 4 * (L10 - L90) + L90 - 30

            # Noise Pollution Level (NPL)
            averages['NPL'][i] = Leq+ 2.56 * sigma_Leq

        if roughness_indicators:
            levels = arr if rough_levels is None else rough_levels.between_time(
                start, end, inclusive='left')
            # Differences between samples one interval apart only
            consecutive = (levels.index[1:] - levels.index[:-1]) == interval
            dL = np.abs(np.diff(levels.to_numpy(dtype=float)))[consecutive]
            dL = dL[~np.isnan(dL)]
            if dL.size:
                averages['dLav'][i] = np.mean(dL)
                averages['dLmax,1'][i] = np.mean(dL[dL >= np.percentile(dL, 99)])
                averages['dLmin,90'][i] = np.mean(dL[dL <= np.percentile(dL, 90)])

    return pd.DataFrame(index=times, data=averages)
    
def freq_periodic(
    df: pd.DataFrame,
    hour1: int,
    hour2: int,
    day1: Optional[str] = None,
    day2: Optional[str] = None,
    win: int = 3600,
    step: int = 0,
    chunks: bool = True,
    stat: Optional[Union[int, list]] = None,
    coverage_check: bool = False,
    coverage_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Compute daily or weekly profiles for each frequency band in the
    DataFrame, as noisemonitor.profile.periodic. For statistical levels,
    data should be sampled at 1 s or finer.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with a datetime index and sound level values for each
        frequency band, columns named by their frequency as in
        noisemonitor.summary.freq_periodic.
    hour1: int, between 0 and 23
        Hour at which the first window starts.
    hour2: int, between 0 and 23
        Hour at which the last window starts (included), as in
        noisemonitor.profile.periodic. If hour1 > hour2, the windows wrap
        past midnight.
    day1: Optional[str], default None
        First day of the week included in the weekly average.
    day2: Optional[str], default None
        Last day of the week included in the weekly average.
    win: int, default 3600
        Window size for the averaging function, in whole seconds. Each
        window covers the half-open interval [start, start + win).
    step: int, default 0
        Step size to compute a sliding average, in whole seconds. If set to
        0 (default value), the function will compute non-sliding averages.
    chunks: bool, default True
        If set to True, the function will use parallel processing to compute
        weekly levels for each frequency band (one after the other, with a
        RuntimeWarning, if parallel processing is unavailable).
    stat: Optional[Union[int, list]], default None
        statistical level(s) to compute (between 1 and 99). If provided,
        computes the corresponding percentile(s) (e.g., stat=5 computes L5).
    coverage_check: bool, default False
        if set to True, assess data coverage and automatically filter periods
        with insufficient data coverage and emit warnings.
    coverage_threshold: float, default 0.5
        minimum data coverage ratio required (0.0 to 1.0).

    Returns
    ----------
    pd.DataFrame
        DataFrame with the levels of each window for each frequency band.
    """

    df = core._parse_freq_columns(df).sort_index(axis=1)

    results = core._map_bands(
        periodic, df, chunks, coverage_check, hour1=hour1, hour2=hour2,
        day1=day1, day2=day2, win=win, step=step, stat=stat,
        coverage_threshold=coverage_threshold)

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

def freq_series(
    df: pd.DataFrame,
    win: int = 3600,
    step: int = 0,
    chunks: bool = True,
    start_at_midnight: bool = False,
    coverage_check: bool = False,
    coverage_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Compute time series of sound levels for each frequency band in the
    DataFrame, as noisemonitor.profile.series. For statistical levels, 
    data should be sampled at 1 s or finer.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with a datetime index and sound level values for each
        frequency band, columns named by their frequency as in
        noisemonitor.summary.freq_periodic.
    win: int, default 3600
        Window size for the averaging function, in whole seconds.
    step: int, default 0
        Step size to compute a rolling average, in whole seconds. If set to
        0 (default value), the function will compute non-rolling averages.
    chunks: bool, default True
        If set to True, the function will use parallel processing to compute
        the series for each frequency band (one after the other, with a
        RuntimeWarning, if parallel processing is unavailable).
    start_at_midnight: bool, default False
        if True, the first window starts at the first midnight of the data,
        as in noisemonitor.profile.series.
    coverage_check: bool, default False
        if set to True, assess data coverage and automatically filter periods
        with insufficient data coverage and emit warnings.
    coverage_threshold: float, default 0.5
        minimum data coverage ratio required (0.0 to 1.0).

    Returns
    -------
    pd.DataFrame
        DataFrame with time series for each frequency band.
    """

    df = core._parse_freq_columns(df).sort_index(axis=1)

    results = core._map_bands(
        series, df, chunks, coverage_check, win=win, step=step,
        start_at_midnight=start_at_midnight,
        coverage_threshold=coverage_threshold)

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

def lf_periodic(
    df: pd.DataFrame,
    hour1: int,
    hour2: int,
    day1: Optional[str] = None,
    day2: Optional[str] = None,
    method: Union[str, list] = 'montano',
    win: int = 3600,
    step: int = 0,
    coverage_check: bool = False,
    coverage_threshold: float = 0.5
) -> pd.DataFrame:
    """Daily or weekly profile of low-frequency criteria: the A- and
    C-weighted equivalent levels of a range of one-third-octave bands and
    their difference C-A, in windows pooled over the days as in
    noisemonitor.profile.periodic. Input bands and methods are those of
    noisemonitor.summary.lf_periodic (see its description).

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with a datetime index and Z-weighted one-third-octave band
        levels as columns, as in noisemonitor.summary.lf_periodic.
    hour1: int, between 0 and 23
        hour at which the first window starts.
    hour2: int, between 0 and 23
        hour at which the last window starts (included), as in
        noisemonitor.profile.periodic. If hour1 > hour2, the windows wrap
        past midnight.
    day1: Optional[str], default None
        First day of the week included in the weekly average.
    day2: Optional[str], default None
        Last day of the week included in the weekly average.
    method: str or list of str, default 'montano'
        low-frequency criterion or criteria: 'botsford', 'montano' or
        'barcelona' (see noisemonitor.summary.lf_periodic).
    win: int, default 3600
        window size, in whole seconds, as in noisemonitor.profile.periodic.
    step: int, default 0
        step size of sliding windows, in whole seconds (0: consecutive
        windows).
    coverage_check: bool, default False
        if set to True, assess data coverage and automatically filter periods
        with insufficient data coverage and emit warnings.
    coverage_threshold: float, default 0.5
        minimum data coverage ratio required (0.0 to 1.0).

    Returns
    ----------
    pd.DataFrame: DataFrame with the time of day of each window and
        MultiIndex columns (Indicator, Method): 'LA' and 'LC', the A- and
        C-weighted equivalent levels of the method's bands, and 'C-A', their
        difference, in dB.
    """
    levels = core._lf_levels(df, method)
    return core._lf_apply(
        periodic, levels, hour1=hour1, hour2=hour2, day1=day1, day2=day2,
        win=win, step=step, coverage_check=coverage_check,
        coverage_threshold=coverage_threshold)

def lf_series(
    df: pd.DataFrame,
    method: Union[str, list] = 'montano',
    win: int = 3600,
    step: int = 0,
    start_at_midnight: bool = False,
    coverage_check: bool = False,
    coverage_threshold: float = 0.5
) -> pd.DataFrame:
    """Time series of low-frequency criteria: the A- and C-weighted
    equivalent levels of a range of one-third-octave bands and their
    difference C-A, over consecutive or sliding windows of the whole data,
    as noisemonitor.profile.series. Input bands and methods are those of
    noisemonitor.summary.lf_periodic (see its description).

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with a datetime index and Z-weighted one-third-octave band
        levels as columns, as in noisemonitor.summary.lf_periodic.
    method: str or list of str, default 'montano'
        low-frequency criterion or criteria: 'botsford', 'montano' or
        'barcelona' (see noisemonitor.summary.lf_periodic).
    win: int, default 3600
        window size, in whole seconds, as in noisemonitor.profile.series.
    step: int, default 0
        step size of sliding windows, in whole seconds (0: consecutive
        windows).
    start_at_midnight: bool, default False
        if True, the first window starts at the first midnight of the data,
        as in noisemonitor.profile.series.
    coverage_check: bool, default False
        if set to True, assess data coverage and automatically filter periods
        with insufficient data coverage and emit warnings.
    coverage_threshold: float, default 0.5
        minimum data coverage ratio required (0.0 to 1.0).

    Returns
    ----------
    pd.DataFrame: DataFrame with a datetime index (centre of each window) and
        MultiIndex columns (Indicator, Method): 'LA' and 'LC', the A- and
        C-weighted equivalent levels of the method's bands, and 'C-A', their
        difference, in dB.
    """
    levels = core._lf_levels(df, method)
    return core._lf_apply(
        series, levels, win=win, step=step,
        start_at_midnight=start_at_midnight, coverage_check=coverage_check,
        coverage_threshold=coverage_threshold)

def nne(
    df: pd.DataFrame,
    hour1: int,
    hour2: int,
    background_type: Union[int, str] = 'Leq',
    exceedance: int = 5,
    min_gap: int = 3,
    win: int = 3600,
    step: int = 0,
    column: Optional[Union[int, str]] = 0,
    day1: Optional[str] = None,
    day2: Optional[str] = None,
    coverage_check: bool = True,
    coverage_threshold: float = 0.8
) -> pd.DataFrame:
    """Compute the Number of Noise Events (NNE) following the algorithm 
    proposed in (Brown and De Coensel, 2018). The function computes the 
    average NNE using sliding windows, computing daily or weekly profiles.
    Note that this function is computationally expensive as noise NNEs are
    separately computed for each individual day and then averaged since
    background levels are relative to each day. For noise events, data
    should be sampled at 1 s or finer (e.g., 125 ms). Events are detected on
    the samples as given, so their number depends on the time weighting and
    sampling rate of the data. The days are taken from hour1, so that windows
    past midnight belong to the night that starts at hour1 and hold
    contiguous data (e.g., a 22-06 window on Mondays will run
    from Mondays 22h to Tuesdays 6h). Each window of each day is counted as a
    record of its own, with its own threshold: its first exceedance is an
    event even when the window starts within it, so an exceedance spanning
    two windows counts in both. Following exceedances are new events if
    they start at least min_gap seconds after the end of the previous one.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with a datetime index and sound level values.
    hour1: int, between 0 and 23
        hour at which the first window starts.
    hour2: int, between 0 and 24
        hour at which the last window ends (excluded): the windows cover
        [hour1, hour2), and hour1=0, hour2=24 give the full day. If
        hour1 > hour2, the windows wrap past midnight.
    background_type: int or str, default 'Leq'
        Type of background level indicator for computing the threshold to 
        use for defining a noise event. Can be 'Leq', 'L50', 'L90' or int 
        for a constant value.
    exceedance: int, default 5
        Exceedance value in dB to add to the background level to define the
        detection threshold, when the background level is adaptive.
    min_gap: int
        Minimum time gap in seconds between successive noise events: the
        time spent at or below the threshold between two exceedances.
    win: int, default 3600
        Window size for the averaging function, in whole seconds. Each
        window covers the half-open interval [start, start + win).
    step: int, default 0
        Step size to compute a sliding average, in whole seconds. If set to
        0 (default value), the function will compute non-sliding averages.
    column: int or str, default 0
        column index (int) or column name (str) to use for calculations. 
        If None, the first column of the DataFrame will be used.
    day1: Optional[str], default None
        First day of the week to include in the calculation: the day at
        whose hour1 the windows start (e.g., with hour1=22, hour2=6 and
        day1=day2='friday', the night from Friday 22:00 to Saturday 06:00).
    day2: Optional[str], default None
        Last day of the week to include in the calculation, as day1.
    coverage_check: bool, default True
        if set to True, assess data coverage and automatically filter periods
        with insufficient data coverage and emit warnings. We recommand to use
        coverage checks for NNEs computation, as gaps in data significantly affect
        the counts of noise events. Samples of a window missing from the
        data (window cut by the start or end of the data) count as missing.
        Windows without any value (all NaN, or no sample) have no count in
        any case (left out of the average), with a CoverageWarning.
    coverage_threshold: float, default 0.8
        minimum data coverage ratio required (0.0 to 1.0).

    Returns
    ----------
    DataFrame: DataFrame with the number of noise events for each sliding 
    window.
    """
    column = core._column_to_index(df, column)

    for name, value, last in (('hour1', hour1, 23), ('hour2', hour2, 24)):
        if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
                or not 0 <= value <= last):
            raise ValueError(
                f"{name} must be an integer between 0 and {last}, "
                f"got {value!r}.")

    if step == 0:
        step = win

    # Hours spanned by [hour1, hour2): 24 for hour1=0, hour2=24
    span = (hour2 - hour1) % 24 or (24 if (hour1, hour2) == (0, 24) else 0)
    NLim = (span*3600)//step - win//step + 1

    if NLim < 1:
        raise ValueError(
            f"No window of {win} s fits between hour1={hour1} and "
            f"hour2={hour2}.")

    # Samples in a full window, on the time grid of the data
    window_count = int(round(win / core.get_interval(df)))

    if core.get_interval(df) > 1:
        warnings.warn("Computing the average Number of "
            "Noise Events should be done with "
            "an integration time equal to or below 1s. Results"
            " might not be valid for these indicators.\n",
            core.IntegrationTimeWarning, stacklevel=2)

    # Each record runs from hour1 on its day over the span, past midnight if
    # needed: shifted back by hour1, its samples fall on the date it starts,
    # which day1 and day2 select, within [0, span) of that date
    shifted = df.set_axis(df.index - pd.Timedelta(hours=hour1), axis=0)
    shifted = filter._days(shifted, day1, day2)
    record_days = shifted.index.normalize()
    shifted = shifted[shifted.index - record_days < pd.Timedelta(hours=span)]

    event_times = []
    daily_event_counts = []

    coverage_warning_issued = False

    for day, group in shifted.groupby(shifted.index.normalize()):

        day_event_counts = np.full(NLim, np.nan)
        starts = day + pd.to_timedelta(np.arange(NLim) * step, unit='s')
        lower = group.index.searchsorted(starts)
        upper = group.index.searchsorted(starts + pd.Timedelta(seconds=win))

        for i in range(0, NLim):
            # Window [start, start + win) of this record
            temp = group.iloc[lower[i]:upper[i]]

            arr = temp.iloc[:, column]

            if coverage_check:
                # Samples missing from the data (window cut by the start or
                # end of the data) count as NaN
                missing = np.full(max(window_count - len(arr), 0), np.nan)
                passes_threshold = core.check_coverage(
                    np.concatenate([arr.to_numpy(dtype=float), missing]),
                    coverage_threshold,
                    emit_warning = not coverage_warning_issued
                )
                if not passes_threshold:
                    coverage_warning_issued = True
                    continue

            # A window without any value (all NaN, or no sample) has no count
            if not arr.notna().any():
                if not coverage_warning_issued:
                    warnings.warn("No data in some windows: they return NaN.",
                                core.CoverageWarning, stacklevel=2)
                    coverage_warning_issued = True
                continue

            if background_type == 'Leq':
                threshold = (core.equivalent_level(arr) + exceedance)
            elif background_type == 'L50':
                threshold = np.nanpercentile(arr, 50) + exceedance
            elif background_type == 'L90':
                threshold = np.nanpercentile(arr, 10) + exceedance
            elif isinstance(background_type, (int)):
                threshold = background_type
            else:
                raise ValueError("Invalid background type. Use 'Leq', "
                                "'L50', 'L90', or an int value.")
            day_event_counts[i] = core.noise_events(
                temp, column, threshold, min_gap)
        daily_event_counts.append(day_event_counts)

    daily_event_counts = np.array(daily_event_counts)
    with warnings.catch_warnings():
        # Windows that failed the coverage check on every day stay NaN
        warnings.simplefilter('ignore', category=RuntimeWarning)
        average_event_counts = np.nanmean(daily_event_counts, axis=0)

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
    column: Optional[Union[int, str]] = 0,
    start_at_midnight: bool = False,
    coverage_check: bool = False,
    coverage_threshold: float = 0.5
) -> pd.DataFrame:
    """Sliding average of the entire sound level array, in terms of
    equivalent level (LEQ), and statistical levels (L10, L50 and L90).
    Percentiles are computed by linear interpolation between samples
    (numpy's default method). For statistical levels, data should 
    be sampled at 1 s or finer. Percentiles are computed from the 
    samples as given, so their values depend on the integration time 
    of the data.

    Parameters
    ---------- 
    df: pd.DataFrame
        DataFrame with a datetime index and sound level values.
    win: int, default 3600
        window size (in whole seconds) for the averaging function. For daily,
        weekly or monthly levels, we recommend using
        noisemonitor.summary.periodic instead.
    step: int, default 0
        step size (in whole seconds) to compute a sliding average. If set
        to 0 (default value), the function will compute non-sliding
        averages.
    column: int or str, default 0
        column index (int) or column name (str) to use for calculations. 
        If None, the first column of the DataFrame will be used.
    start_at_midnight: bool, default False
        if set to True, the first window starts at the first sample at or
        after the first midnight of the data (ValueError if fewer than one
        window of samples follow it).
    coverage_check: bool, default False
        if set to True, assess data coverage and automatically filter periods
        with insufficient data coverage and emit warnings. Windows without
        any value (all NaN) are NaN in any case, with a CoverageWarning.
    coverage_threshold: float, default 0.5
        minimum data coverage ratio required (0.0 to 1.0).

    Returns
    ----------
    DataFrame: dataframe containing datetime index and averaged
        Leq, L10, L50 and L90 at the corresponding columns

    """
    column = core._column_to_index(df, column)
    
    interval = core.get_interval(df)

    if interval > 1:
        warnings.warn("Computing the L10, L50, and L90 should be done with "
            "an integration time equal to or below 1s. Results"
            " might not be valid for these indicators.\n",
            core.IntegrationTimeWarning, stacklevel=2)

    # Window and step in whole samples; the tolerance absorbs float errors
    # of sub-second intervals (3600 // 0.1 gives 35999)
    step = int(np.floor(step / interval + 1e-9))
    win = int(np.floor(win / interval + 1e-9))

    N = len(df)

    if N < win:
        raise ValueError(
            f"Insufficient data: need at least {win} samples, got {N}"
        )

    if step == 0:
        step = win

    if start_at_midnight:
        start_time = df.index[0].replace(
            hour=0, minute=0, second=0, microsecond=0)
        if df.index[0] > start_time:
            start_time += pd.Timedelta(days=1)
        # First sample at or after midnight
        start_index = df.index.searchsorted(start_time)
        if N - start_index < win:
            raise ValueError(
                f"Insufficient data after midnight ({start_time}): need at "
                f"least {win} samples, got {N - start_index}"
            )
    else:
        start_index = 0

    NLim = (N - start_index - win) // step + 1

    overallmean = np.full(NLim, np.nan)
    overallL10 = np.full(NLim, np.nan)
    overallL50 = np.full(NLim, np.nan)
    overallL90 = np.full(NLim, np.nan)
    overalltime = []

    coverage_warning_issued = False

    for i in range(0, NLim):
        arr = df.iloc[
            int(start_index + i * step):int(start_index + i * step + win)
        ].iloc[:, column]
        if start_at_midnight:
            overalltime.append(
                start_time
                + pd.Timedelta(seconds=(i * step + win / 2) * interval)
            )
        else:
            overalltime.append(
                df.index[int(start_index + i * step + win / 2)]
            )

        if coverage_check:
            passes_threshold = core.check_coverage(
                arr,
                coverage_threshold,
                emit_warning = not coverage_warning_issued
            )
            if not passes_threshold:
                coverage_warning_issued = True
                continue

        # A window without any value stays NaN
        if not arr.notna().any():
            if not coverage_warning_issued:
                warnings.warn("No data in some windows: they return NaN.",
                            core.CoverageWarning, stacklevel=2)
                coverage_warning_issued = True
            continue

        overallmean[i] = core.equivalent_level(arr)
        overallL10[i] = np.nanpercentile(arr, 90)
        overallL50[i] = np.nanpercentile(arr, 50)
        overallL90[i] = np.nanpercentile(arr, 10)

    meandf = pd.DataFrame(
        index=overalltime,
        data={
            'Leq': overallmean,
            'L10': overallL10,
            'L50': overallL50,
            'L90': overallL90
        })
    return meandf