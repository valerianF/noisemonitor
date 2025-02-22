import pandas as pd
import numpy as np
import warnings

from datetime import time
from typing import Optional

from .utilities import *

class NoiseMonitor:
    """Compute discrete values and different types of sliding mean averages
    for various kinds of sound level descriptors, including LEQ, L10, L50,
    L90, LDEN, overall or at daily or weekly rates, from sound level monitor
    data, weighted or unweighted. 

    Parameters
    ---------- 
    df: DataFrame
        a compatible DataFrame (typically generated with function load_data()),
        with a datetime, time or pd.Timestamp index and corresponding sound 
        level values in the first column.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

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

            dailymean[i] = equivalent_level(temp.iloc[:][column])
            dailyL10[i] = np.nanpercentile(temp.iloc[:][column], 90)
            dailyL50[i] = np.nanpercentile(temp.iloc[:][column], 50)
            dailyL90[i] = np.nanpercentile(temp.iloc[:][column], 10)
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
        DataFrame: DataFrame with Leq and Lden values for each day or week.
        """
        self.validate_column(column)

        results = []

        if freq == 'D':
            resampled = self.df.resample('D')
        elif freq == 'W':
            resampled = self.df.resample('W-MON')
        else:
            raise ValueError("Invalid frequency. Use 'D' for daily or 'W' for weekly.")

        for period, group in resampled:
            if len(group) > 0:
                leq_value = equivalent_level(group[column])
                lden_values = compute_lden(group, column, values=values)
                result = {
                    'Period': period,
                    'Leq': leq_value,
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
        self.validate_column(column)
        if day1 and day2:
            self.validate_days(day1, day2)

        if all(day is not None for day in [day1, day2]):
            d1, d2 = week_indexes(day1, day2)

            if d1 <= d2:
                temp = self.df.loc[(self.df.index.dayofweek >= d1) 
                                   & (self.df.index.dayofweek <= d2)]
            else:
                temp = self.df.loc[(self.df.index.dayofweek >= d1) 
                                   | (self.df.index.dayofweek <= d2)]
        else:
            temp = self.df

        return compute_lden(temp, column, values=values)
    
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
        self.validate_column(column)
        self.validate_hours(hour1, hour2)
        if day1 and day2:
            self.validate_days(day1, day2)

        if all(day is not None for day in [day1, day2]):
            d1, d2 = week_indexes(day1, day2)

            if d1 <= d2:
                temp = self.df.loc[(self.df.index.dayofweek >= d1) 
                                   & (self.df.index.dayofweek <= d2)]
            else:
                temp = self.df.loc[(self.df.index.dayofweek >= d1) 
                                   | (self.df.index.dayofweek <= d2)]
        else:
            temp = self.df

        if hour1 == 24:
            t1 = time(hour=23, minute=59, second=59)
            t2 = time(hour=hour2)
        elif hour2 == 24:
            t1 = time(hour=hour1)
            t2 = time(hour=23, minute=59, second=59)
        else:
            t1 = time(hour=hour1)
            t2 = time(hour=hour2)

        array = temp.between_time(t1, t2).iloc[:][column]

        if stats:
            return pd.DataFrame({
                'leq': [np.round(equivalent_level(array), 2)],
                'l10': [np.round(np.nanpercentile(array, 90), 2)],
                'l50': [np.round(np.nanpercentile(array, 50), 2)],
                'l90': [np.round(np.nanpercentile(array, 10), 2)]
            })
        return pd.DataFrame({'leq': [np.round(equivalent_level(array), 2)]})
    
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
        self.validate_column(column)

        interval = (self.df.index[2] - self.df.index[1]).seconds
        
        step = step // interval
        win = win // interval

        N = len(self.df)

        if step == 0:
            step = win

        if start_at_midnight:
            start_time = self.df.index[0].replace(hour=0, minute=0, second=0, microsecond=0)
            if self.df.index[0] > start_time:
                start_time += pd.Timedelta(days=1)
            start_index = self.df.index.get_indexer([start_time], method='nearest')[0]
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
            overallmean[i] = equivalent_level(temp)
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
            overalltime.append(self.df.index[int(start_index + i * step + win / 2)])

        meandf = pd.DataFrame(
            index=overalltime, 
            data={
                'Leq': overallmean, 
                'L10': overallL10,
                'L50': overallL50,
                'L90': overallL90
            })

        return meandf
    
    def validate_days(self, day1: str, day2: str) -> None:
        week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                'saturday', 'sunday']
        if day1.lower() not in week or day2.lower() not in week:
            raise ValueError("Arguments day1 and day2 must be a day of the week.") 
    
    def validate_column(self, column: str) -> None:
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

    def validate_hours(self, hour1: int, hour2: int) -> None:
        if not (0 <= hour1 <= 24) or not (0 <= hour2 <= 24):
            raise ValueError("Hours must be between 0 and 24.")   
    
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
            computation. If hour2 > hour1 the average will be computed outside of
            these hours.
        day1: str, a day of the week in english, case-insensitive
            first day of the week included in the weekly average.
        day2: str, a day of the week in english, case-insensitive
            last (included) day of the week in the weekly average. If day2 
            happens later in the week than day1 the average will be computed 
            outside of these days.
        win: int, default 3600
            window size for the averaging function, in seconds.
        step: int, default 0
            step size to compute a sliding average. If set to 0 (default value),
            the function will compute non-sliding averages.

        Returns
        ---------- 
        DataFrame: dataframe containing time index and weekly averaged
            Leq, L10, L50 and L90 at the corresponding columns

        """
        d1, d2 = week_indexes(day1, day2)
        if d1 <= d2:
            temp = self.df.loc[(self.df.index.dayofweek >= d1) 
                                & (self.df.index.dayofweek <= d2)]
        else:
            temp = self.df.loc[(self.df.index.dayofweek >= d1) 
                                | (self.df.index.dayofweek <= d2)]

        weeklymeandf = self.daily(column, hour1, hour2, temp, win=win, 
                                  step=step)
            
        return weeklymeandf
    
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
    
def equivalent_level(array: np.array) -> float:
    """Compute the equivalent sound level from the input array."""
    if len(array) == 0 or np.isnan(array).all():
        return np.nan
    return 10*np.log10(np.mean(np.power(np.full(len(array), 10), array/10))) 

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