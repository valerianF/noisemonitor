import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

from datetime import time
from typing import Optional, List

from noisemonitor.utilities.compute import harmonica, lden, equivalent_level
from noisemonitor.utilities.process import filter_by_days, filter_by_hours
from noisemonitor.utilities.decorators import validate_column

class Indicators:
    def __init__(self, noise_monitor):
        self._noise_monitor = noise_monitor
    
    @validate_column
    def weekly_harmonica(
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
        temp_df = filter_by_days(self._noise_monitor.df, day1, day2)
        harmonica_df = harmonica(temp_df, column, use_chunks)

        # Compute the average values for each hour of the day
        daily_avg = harmonica_df.groupby(harmonica_df.index.hour).mean()

        # Create a time index for the result
        time_index = [time(hour=h) for h in range(24)]
        daily_avg.index = time_index

        return daily_avg
    
    @validate_column
    def weekly_levels(
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
            resampled = self._noise_monitor.df.resample('D')
        elif freq == 'W':
            resampled = self._noise_monitor.df.resample('W-MON')
        else:
            raise ValueError("Invalid frequency. Use 'D' for daily or 'W' for" 
                             " weekly.")

        for period, group in resampled:
            if len(group) > 0:
                leq_value = equivalent_level(group[column])
                lden_values = lden(group, column, values=values)
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
    def overall_lden(
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

        temp = filter_by_days(self._noise_monitor.df, day1, day2)

        return lden(temp, column, values=values)
    
    @validate_column
    def overall_leq(
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

        temp = filter_by_days(self._noise_monitor.df, day1, day2)
        array = filter_by_hours(temp, hour1, hour2)[column]

        if stats:
            if self._noise_monitor.interval > 1:
                warnings.warn("Computing the L10, L50, and L90 should be done with "
                              "an integration time equal to or below 1s. Results"
                              " might not be valid for this descriptor.\n")
            return pd.DataFrame({
                'leq': [np.round(equivalent_level(array), 2)],
                'l10': [np.round(np.nanpercentile(array, 90), 2)],
                'l50': [np.round(np.nanpercentile(array, 50), 2)],
                'l90': [np.round(np.nanpercentile(array, 10), 2)]
            })
        return pd.DataFrame({'leq': [np.round(equivalent_level(array), 2)]})
    
    @validate_column
    def nday(
        self, 
        column: str, 
        indicator: str = 'Leq,24h', 
        bins: Optional[List[int]] = None,
        freq: str = 'D'
    ) -> pd.DataFrame:
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
        freq: str, default 'D'
            Frequency for the computation. 'D' for daily and 'W' for weekly.

        Returns
        ----------
        DataFrame : DataFrame with the number of days for each decibel range.
        """
        if bins is None:
            bins = [40, 45, 50, 55, 60, 65, 70, 75, 80]

        # Compute daily or weekly indicators
        indicators_df = self.weekly_levels(
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

        return counts_df, bins   
    
