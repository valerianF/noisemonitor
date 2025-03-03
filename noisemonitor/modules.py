import pandas as pd
import numpy as np
import warnings
import functools

from datetime import time, timedelta
from typing import Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

from .utilities import *

class NoiseMonitor:
    """Compute discrete values and different types of sliding mean averages
    for various kinds of sound level descriptors, including Leq, L10, L50,
    L90, Lden, Number of Noise Events, overall or at daily or weekly rates, 
    from sound level monitor data, weighted or unweighted. 

    Parameters
    ---------- 
    df: DataFrame
        a compatible DataFrame (typically generated with function load_data()),
        with a datetime, time or pd.Timestamp index and corresponding sound 
        level values in the first column.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.interval = (self.df.index[2] - self.df.index[1]).seconds

    # Decorators
    def validate_column(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            column = kwargs.get('column', args[0] if args else None)
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")
            return func(self, *args, **kwargs)
        return wrapper

    def validate_interval(indicator):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                if self.interval > 1:
                    warnings.warn(f"Computing the {indicator} should be done with "
                                "an integration time equal to or below 1s. Results"
                                " might not be valid for this descriptor.\n")
                return func(self, *args, **kwargs)
            return wrapper
        return decorator

    # Class methods
    @validate_interval("L10, L50, and L90")
    def daily(
        self, 
        column: str, 
        hour1: int, 
        hour2: int, 
        *args: Optional[pd.DataFrame], 
        win: int=3600, 
        step: int=0
    ) -> pd.DataFrame:
        """Compute daily, sliding average of the sound level, in terms of
        equivalent level (Leq), and percentiles (L10, L50 and L90).

        Parameters
        ---------- 
        column: str
            column name to use for calculations.
        hour1: int, between 0 and 23
            hour for the starting time of the daily average.
        hour2: int, between 0 and 23
            hour (included) for the ending time of the daily average. 
            If hour2 > hour1 the average will be computed outside of 
            these hours.
        *args: DataFrame
            used as a pipeline for SoundLevel.weekly() function. Can be used 
            to compute the daily average of a custom dataframe.
        win: int, default 3600
            window size for the averaging function, in seconds.
        step: int, default 0
            step size to compute a sliding average. If set to 0 (default 
            value), the function will compute non-sliding averages.

        
        Returns
        ---------- 
        DataFrame: dataframe containing time index and daily averaged
            Leq, L10, L50 and L90 at the corresponding columns

        """
        
        if step == 0:
            step = win
        
        NLim = ((hour2-hour1)%24*3600)//step + 1

        dailymean = np.zeros(NLim)
        dailyL10 = np.zeros(NLim)
        dailyL50 = np.zeros(NLim)
        dailyL90 = np.zeros(NLim)
        dailytime = []

        for i in range(0, NLim):
            t = hour1*3600 + i*step + win//2
            t1 = hour1*3600 + i*step
            t2 = hour1*3600 + i*step + win

            if not args:
                temp = self.df
            else:
                temp = args[0]

            temp = temp.between_time(
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

            dailymean[i] = compute_equivalent_level(temp[column])
            dailyL10[i] = np.nanpercentile(temp[column], 90)
            dailyL50[i] = np.nanpercentile(temp[column], 50)
            dailyL90[i] = np.nanpercentile(temp[column], 10)
            dailytime.append(time(
                hour=(t//3600)%24, 
                minute=(t%3600)//60, 
                second=(t%3600)%60
                ))

        dailymeandf = pd.DataFrame(
            index=dailytime, 
            data={
                'Leq': dailymean, 
                'L10': dailyL10, 
                'L50': dailyL50, 
                'L90': dailyL90
            })

        return dailymeandf
    
    def daily_weekly_harmonica(
            self, 
            column: str, 
            use_chunks: bool = True,
            day1: Optional[str] = None,
            day2: Optional[str] = None
            ) -> pd.DataFrame:
        """Compute the average HARMONICA indicators for each hour of a day.

        Parameters
        ----------
        column: str
            Column name containing the LAeq values.
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
        temp_df = filter_by_days(self.df, day1, day2)
        harmonica_df = compute_harmonica(temp_df, column, use_chunks)

        # Compute the average values for each hour of the day
        daily_avg = harmonica_df.groupby(harmonica_df.index.hour).mean()

        # Create a time index for the result
        time_index = [time(hour=h) for h in range(24)]
        daily_avg.index = time_index

        return daily_avg
    
    @validate_column
    def daily_weekly_indicators(
        self, 
        column: str, 
        freq: str='D', 
        values: bool=False
    ) -> pd.DataFrame:
        """Compute Leq,24h and Lden on a daily or weekly basis.

        Parameters
        ----------
        column: str
            Column name to use for calculations.
        freq: str, default 'D'
            frequency for the computation. 'D' for daily and 'W' for weekly.
        values: bool, default False
            if set to True, the function will return individual day, evening
            and night values in addition to the lden.

        Returns
        ----------
        DataFrame: DataFrame with Leq,24h and Lden values for each day or week.
        """

        results = []

        if freq == 'D':
            resampled = self.df.resample('D')
        elif freq == 'W':
            resampled = self.df.resample('W-MON')
        else:
            raise ValueError("Invalid frequency. Use 'D' for daily or 'W' for" 
                             " weekly.")

        for period, group in resampled:
            if len(group) > 0:
                leq_value = compute_equivalent_level(group[column])
                lden_values = compute_lden(group, column, values=values)
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
    
    @validate_column
    @validate_interval("average Number of Noise Events")
    def daily_weekly_number_of_noise_events(
        self,
        column: str,
        hour1: int, 
        hour2: int, 
        background_type: str = 'leq',
        exceedance: int = 5,
        min_gap: int = 3,
        win: int = 3600,
        step: int = 0,
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
        column: str
            Column name to use for calculations.
        hour1: int, between 0 and 24
            hour for the starting time of the daily average.
        hour2: int, between 0 and 24
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
        day1: Optional[str], default None
            First day of the week to include in the calculation.
        day2: Optional[str], default None
            Last day of the week to include in the calculation.

        Returns
        ----------
        DataFrame: DataFrame with the number of noise events for each sliding 
        window.
        """

        temp_df = filter_by_days(self.df, day1, day2)
        temp_df = filter_by_hours(temp_df, hour1, hour2)

        if step == 0:
            step = win
        
        NLim = ((hour2-hour1)%24*3600)//step - win//step + 1

        event_times = []
        daily_event_counts = []

        # Compute the expected number of values for each day
        freq = (temp_df.index[2] - temp_df.index[1])
        if hour2 > hour1:
            expected_intervals = pd.date_range(
                start=pd.Timestamp('2024-01-01') + pd.Timedelta(hours=hour1),
                end=pd.Timestamp('2024-01-01') + pd.Timedelta(hours=hour2),
                freq=freq
                )
        elif hour1 > hour2:
            expected_intervals = pd.date_range(
                start=pd.Timestamp('2024-01-01') + pd.Timedelta(hours=hour1),
                end=pd.Timestamp('2024-01-02') + pd.Timedelta(hours=hour2),
                freq=freq
                )
        expected_intervals_count = len(expected_intervals)

        for day, group in temp_df.groupby(temp_df.index.date):
            # Check if the day has any NaN values or is missing data
            if group[column].isna().any() or \
                len(group) != expected_intervals_count:
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

                if background_type == 'leq':
                    threshold = compute_equivalent_level(temp[column]) + exceedance
                elif background_type == 'l50':
                    threshold = np.nanpercentile(temp[column], 50) + exceedance
                elif background_type == 'l90':
                    threshold = np.nanpercentile(temp[column], 10) + exceedance
                elif isinstance(background_type, (int)):
                    threshold = background_type
                else:
                    raise ValueError("Invalid background type. Use 'leq', "
                                     "'l50', 'l90', or an int value.")
                day_event_counts[i] = compute_noise_events(
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
    
    @validate_column
    def lden(
        self, 
        column: str, 
        day1: Optional[str] = None, 
        day2: Optional[str] = None, 
        values: bool=True
        ) -> pd.DataFrame:
        """Return the Lden, a descriptor of noise level based on Leq over
        a whole day with a penalty for evening (19h-23h) and night (23h-7h)
        time noise. By default, an average Lden is computed that is 
        representative of all the sound level data. Can return a Lden value
        corresponding to specific days of the week.

        Parameters
        ----------
        column: str
            column name to use for calculations. Should contain LAeq values.
        day1 (optional): str, a day of the week in english, case-insensitive
            first day of the week included in the Lden computation.
        day2 (optional): str, a day of the week in english, case-insensitive
            last (included) day of the week in the Lden computation. If day2 
            happens later in the week than day1 the average will be computed 
            outside of these days.
        values: bool, default False
            If set to True, the function will return individual day, evening
            and night values in addition to the lden.

        Returns
        ---------- 
        dict: daily or weekly lden rounded to two decimals. Associated day,
            evening and night values are returned if values is set to True.
        """

        temp = filter_by_days(self.df, day1, day2)

        return compute_lden(temp, column, values=values)
    
    @validate_column
    def leq(
        self, 
        column: str, 
        hour1: int, 
        hour2: int, 
        day1: Optional[str] = None, 
        day2: Optional[str] = None, 
        stats: bool = True
        ) -> pd.DataFrame:
        """Return the equivalent level (and optionally statistical indicators)
        between two hours of the day. Can return a value corresponding to 
        specific days of the week.

        Parameters
        ----------
        column: str
            column name to use for calculations.
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
        stats: bool, default True
            If set to True, the function will return L10, L50 and L90 
            together with the Leq.

        Returns
        ---------- 
        float: daily or weekly equivalent level rounded to two decimals. 
        Statistical indicators are included if stats is set to True.
        """

        temp = filter_by_days(self.df, day1, day2)
        array = filter_by_hours(temp, hour1, hour2)[column]

        if stats:
            if self.interval > 1:
                warnings.warn("Computing the L10, L50, and L90 should be done with "
                              "an integration time equal to or below 1s. Results"
                              " might not be valid for this descriptor.\n")
            return pd.DataFrame({
                'leq': [np.round(compute_equivalent_level(array), 2)],
                'l10': [np.round(np.nanpercentile(array, 90), 2)],
                'l50': [np.round(np.nanpercentile(array, 50), 2)],
                'l90': [np.round(np.nanpercentile(array, 10), 2)]
            })
        return pd.DataFrame({'leq': [np.round(compute_equivalent_level(array), 2)]})
    
    def nday(
        self, 
        column: str, 
        indicator: str = 'Leq,24h', 
        bins: Optional[List[int]] = None,
        thresholds: List[int] = [55, 60, 65],
        plot: bool = False,
        title: str = None,
        figsize: tuple = (10,8), 
        freq: str = 'D'
    ) -> Union[pd.DataFrame, None]:
        """Compute the number of days in a dataset for which the indicators 
        are between given values of decibels.

        Parameters
        ----------
        column: str
            Column name to use for calculations.
        indicator: str, default 'Leq,24h'
            Indicator to use for the computation. Options are 'Leq,24h', 'Lden', 
            'Lday', 'Levening', and 'Lnight'.
        bins: list of int, optional
            List of decibel values to define the bins. By default: <40 dBA 
            and every 5 dBA until >=80 dBA.
        thresholds: list of int, default [55, 60, 65]
            Thresholds for color coding. 
        plot: bool, default False
            If set to True, the function will plot a histogram with color 
            coding instead of returning a dataframe
        figsize: tuple, default (10,8)
            figure size in inches.
        freq: str, default 'D'
            Frequency for the computation. 'D' for daily and 'W' for weekly.

        Returns
        ----------
        DataFrame or None: If plot is set to False, DataFrame with the number
        of days for each decibel range.
        """
        if bins is None:
            bins = [40, 45, 50, 55, 60, 65, 70, 75, 80]

        # Compute daily or weekly indicators
        indicators_df = self.daily_weekly_indicators(
            column, 
            freq=freq, 
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

        # Plot the histogram if requested
        if plot:
            plt.figure(figsize=figsize)
            plt.rcParams.update({'font.size': 16})

            colors = ['#89dc35' if x.left < thresholds[0] \
                      else '#dcdc35' if x.left < thresholds[1] \
                      else '#dc8935' if x.left < thresholds[2] \
                      else '#dc3535' for x in counts.index]

            
            ax = counts.plot(
                kind='bar', 
                color=colors, 
                width=0.8, 
                zorder=3
                )
            ax.grid(True, linestyle='--', zorder=0)
            plt.xlabel('Decibel Range (dBA)')

            if freq == 'D':
                plt.ylabel('Number of Days')
            elif freq == 'W':
                plt.ylabel('Number of Weeks')            

            # Set custom y-tick labels
            ytick_labels = [f'<{bins[0]}'] + \
                [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)] + \
                [f'>{bins[-1]}']
            ax.set_xticklabels(ytick_labels)

            if title is not None:
                plt.title(title)

            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            return

        return counts_df
    
    @validate_column
    @validate_interval("L10, L50, and L90")
    def sliding_average(
        self, 
        column: str, 
        win: int = 3600, 
        step: int = 0, 
        start_at_midnight: bool = False
        ) -> pd.DataFrame:
        """Sliding average of the entire sound level array, in terms of
        equivalent level (LEQ), and percentiles (L10, L50 and L90).

        Parameters
        ---------- 
        column: str
            column name to use for calculations.
        win: int, default 3600
            window size (in seconds) for the averaging function. For averages
            at a daily or weekly window, we recommend using the
            daily_weekly_indicators function instead.
        step: int, default 0
            step size (in seconds) to compute a sliding average. If set to 0
            (default value), the function will compute non-sliding averages.
        start_at_midnight: bool, default False
            if set to True, the computation will start at midnight.

        Returns
        ---------- 
        DataFrame: dataframe containing datetime index and averaged
            Leq, L10, L50 and L90 at the corresponding columns

        """
        
        step = step // self.interval
        win = win // self.interval

        N = len(self.df)

        if step == 0:
            step = win

        if start_at_midnight:
            start_time = self.df.index[0].replace(
                hour=0, minute=0, second=0, microsecond=0)
            if self.df.index[0] > start_time:
                start_time += pd.Timedelta(days=1)
            start_index = self.df.index.get_indexer(
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
            temp = self.df.iloc[
                int(start_index + i * step):int(start_index + i * step + win)
                ][column]
            overallmean[i] = compute_equivalent_level(temp)
            if np.isnan(temp).all():
                if not nan_warning_issued:
                    warnings.warn(f"All-NaN slice(s) encountered. "
                                  "Check for gaps in the data.")
                    nan_warning_issued = True
                overallL10[i] = np.nan
                overallL50[i] = np.nan
                overallL90[i] = np.nan
            else:
                overallL10[i] = np.nanpercentile(temp, 90)
                overallL50[i] = np.nanpercentile(temp, 50)
                overallL90[i] = np.nanpercentile(temp, 10)
            overalltime.append(self.df.index[int(start_index + i*step + win/2)])

        meandf = pd.DataFrame(
            index=overalltime, 
            data={
                'Leq': overallmean, 
                'L10': overallL10,
                'L50': overallL50,
                'L90': overallL90
            })

        return meandf
    
    
    def weekly(
        self, 
        column: str, 
        hour1: int, 
        hour2: int, 
        day1: str, 
        day2: str, 
        win: int = 3600, 
        step: int = 0
        ) -> pd.DataFrame:
        """Compute weekly, sliding average of the sound level, in terms of
        equivalent level (Leq), and percentiles (L10, L50 and L90). In other
        terms, computes daily averages at specific days of the week.

        Parameters
        ---------- 
        column: str
            column name to use for calculations.
        hour1: int, between 0 and 23
            hour for the starting time of the daily average.
        hour2: int, between 0 and 23
            hour for the ending time of the daily average. Included in the 
            computation. If hour2 > hour1 the average will be computed outside 
            of these hours.
        day1: str, a day of the week in english, case-insensitive
            first day of the week included in the weekly average.
        day2: str, a day of the week in english, case-insensitive
            last (included) day of the week in the weekly average. If day2 
            happens later in the week than day1 the average will be computed 
            outside of these days.
        win: int, default 3600
            window size for the averaging function, in seconds.
        step: int, default 0
            step size to compute a sliding average. If set to 0 
            (default value), the function will compute non-sliding 
            averages.

        Returns
        ---------- 
        DataFrame: dataframe containing time index and weekly averaged
            Leq, L10, L50 and L90 at the corresponding columns.

        """
        temp = filter_by_days(self.df, day1, day2)
            
        return self.daily(column, hour1, hour2, temp, win=win, 
                                  step=step)
    
def compute_equivalent_level(array: np.array) -> float:
    """Compute the equivalent sound level from the input array."""
    if len(array) == 0 or np.isnan(array).all():
        return np.nan
    return 10*np.log10(np.mean(np.power(np.full(len(array), 10), array/10))) 
    
def compute_harmonica(
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

    if use_chunks:
        results = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(
                process_hour_harmonica, 
                hour, 
                group, 
                column, 
                interval) for hour, group in df.resample('H')]

            for future in as_completed(futures):
                results.append(future.result())
    else: 
        results = [process_hour_harmonica(
            hour, 
            group, 
            column, 
            interval) for hour, group in df.resample('H')]

    return pd.DataFrame(results).set_index('hour')
    
def compute_lden(
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
    lday = compute_equivalent_level(df.between_time(
        time(hour=7), time(hour=19))[column])
    levening = compute_equivalent_level(df.between_time(
        time(hour=19), time(hour=23))[column])
    lnight = compute_equivalent_level(df.between_time(
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

def compute_noise_events(
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

def filter_by_days(
    df: pd.DataFrame, 
    day1: Optional[str], 
    day2: Optional[str]
    ) -> pd.DataFrame:
    """Filter the DataFrame based on the specified days of the week.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with a datetime index.
    day1: Optional[str]
        First day of the week to include in the filtering.
    day2: Optional[str]
        Last day of the week to include in the filtering.

    Returns
    ----------
    pd.DataFrame: Filtered DataFrame based on the specified days of the week.
    """
    if day1 and day2:
        week = ['monday', 'tuesday', 'wednesday', 'thursday', 
                'friday', 'saturday', 'sunday']
        if day1.lower() not in week or day2.lower() not in week:
            raise ValueError("Arguments day1 and day2 must be a day of "
                             "the week.")
        
        d1, d2 = week_indexes(day1, day2)

        if d1 <= d2:
            return df.loc[
                (df.index.dayofweek >= d1) & (df.index.dayofweek <= d2)]
        else:
            return df.loc[
                (df.index.dayofweek >= d1) | (df.index.dayofweek <= d2)]
    else:
        return df

def filter_by_hours(
    df: pd.DataFrame, 
    hour1: int, 
    hour2: int
    ) -> pd.DataFrame:
    """Filter the DataFrame based on the specified hours.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with a datetime index.
    hour1: int
        Hour for the starting time of the daily average.
    hour2: int
        Hour for the ending time of the daily average.

    Returns
    ----------
    pd.Series: Filtered Series based on the specified hours.
    """
    
    if not (0 <= hour1 <= 24) or not (0 <= hour2 <= 24):
        raise ValueError("Hours must be between 0 and 24.")  

    if hour1 == 24:
        t1 = time(hour=23, minute=59, second=59)
        t2 = time(hour=hour2)
    elif hour2 == 24:
        t1 = time(hour=hour1)
        t2 = time(hour=23, minute=59, second=59)
    else:
        t1 = time(hour=hour1)
        t2 = time(hour=hour2)

    return df.between_time(t1, t2)

def process_hour_harmonica(hour, group, column, interval):
    """Process a single hour of data to compute HARMONICA indicators."""
    if (len(group) != 3600 // interval) or \
        (group.isna().sum().sum() / len(group) > 0.2):
        return {
            'hour': hour, 
            'EVT': np.nan, 
            'BGN': np.nan, 
            'HARMONICA': np.nan
            }

    # Compute LAeq for the hour
    laeq = compute_equivalent_level(group[column])

    # Compute LA95eq for the hour using a rolling window
    la95 = group[column].rolling(
        window=int(600 // interval),
        step=int(max(1, 1//interval)),
        min_periods=1
    ).apply(lambda x: np.nanpercentile(x, 5), raw=True)
    la95eq = compute_equivalent_level(la95.dropna())

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

def week_indexes(
    day1: str, 
    day2: str
    ) -> tuple:
    """Return datetime compatible weekday indexes from weekday strings."""
    week = [
        'monday', 'tuesday', 'wednesday', 'thursday', 
        'friday', 'saturday', 'sunday'
        ]            
    day1 = day1.lower()
    day2 = day2.lower()
    
    if any(d not in week for d in [day1, day2]):
        raise ValueError("Arguments day1 and day2 must be a day of the week.")
    d1 = week.index(day1)
    d2 = week.index(day2)
    return d1, d2