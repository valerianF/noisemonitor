""" Various functions used across noisemonitor modules to filter and process
dataframes. 
"""

import pandas as pd
import numpy as np

from datetime import datetime, time
from typing import Optional, Union
from . import core

def all_data(
    df: pd.DataFrame, 
    start_datetime: datetime, 
    end_datetime: datetime, 
    between: bool = False
) -> pd.DataFrame:
    """Filter a datetime index DataFrame (typically the output of load_data())
    between two particular dates. By setting between = True, you can filter 
    out the times that are between the two dates.

    Parameters
    ---------- 
    df: DataFrame
        a compatible DataFrame (typically generated with functions load_data(),
        NoiseMonitor.daily() or NoiseMonitor.weekly() ),
        with a datetime, time or pandas.Timestamp index.
    start_datetime: datetime object
        first date from which to filter the data. 
        Must be earlier than end_datetime.
    end_datetime: datetime object
        second date from which to filter the data. 
        Must be later than start_datetime.
    between: bool, default False
        If set to True, will filter out data that is between the two dates.
        Else and by default, will filter data that is outside the two dates.

    Returns
    ---------- 
    DataFrame: input dataframe filtered to the specified dates range.
    """
    try:
        if between:
            return df.loc[
                (df.index < start_datetime) |
                (df.index > end_datetime)
            ]
        return df.loc[
            (df.index >= start_datetime) &
            (df.index <= end_datetime)
        ]
    except TypeError as e:
        raise TypeError(
            "Invalid comparison. Please check whether a timezone "
            "argument should be indicated when loading data."
        ) from e
    
def extreme_values(
    df: pd.DataFrame, 
    column: Optional[Union[int, str]] = None, 
    min_value: int = 30, 
    max_value: int = 100
) -> pd.DataFrame:
    """
    Replace values in the specified column that are below min_value or 
    above max_value with NaN.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the data.
    column: Union[int, str], default None
        The column name or index to use for calculations. If None, the first column
        will be used.
    min_value: int, default 30
        The minimum value threshold.
    max_value: int, default 95
        The maximum value threshold.

    Returns
    ----------
    pd.DataFrame: DataFrame with extreme values replaced by NaN.
    """
    if column is None:
        column = 0
    
    column = core._column_to_index(df, column)
    column_name = df.columns[column]

    initial_count = df[column_name].notna().sum()
    df[column_name] = df[column_name].apply(
        lambda x: x if min_value <= x <= max_value else np.nan
    )

    filtered_count = df[column_name].notna().sum()
    filtered_out = initial_count - filtered_count
    proportion_filtered = (filtered_out / initial_count) * 100

    print(f"Proportion of values filtered out: {proportion_filtered:.2f}%")

    return df

def weather_flags(
    df: pd.DataFrame, 
    column: Union[int, str] = None,  
    filter_wind_flag: bool = True,
    filter_rain_flag: bool = True,
    filter_temp_flag: bool = False,
    filter_rel_hum_flag: bool = False,
    filter_snow_flag: bool = False
) -> pd.DataFrame:
    """
    Replace values in the specified column with NaN based on weather flags.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the data.
    column: Union[int, str], default None
        The column name or index to use for calculations. 
        If None, the first column will be used.
    filter_wind_flag: bool, default False
        Whether to filter data with Wind Speed Flag.
    filter_rain_flag: bool, default True
        Whether to filter data with Rain Flag.
    filter_temp_flag: bool, default False
        Whether to filter data with Temperature Flag.
    filter_rel_hum_flag: bool, default False
        Whether to filter data with Relative Humidity Flag.
    filter_snow_flag: bool, default False
        Whether to filter data with Snow Flag.

    Returns
    ----------
    pd.DataFrame: DataFrame with flagged values replaced by NaN.
    """
    if column is None:
        column = 0
    
    column = core._column_to_index(df, column)
    column_name = df.columns[column]

    df = df.copy()

    flags = {
        'Wind_Spd_Flag': filter_wind_flag,
        'Rain_Flag_Roll': filter_rain_flag,
        'Temp_Flag': filter_temp_flag,
        'Rel_Hum_Flag': filter_rel_hum_flag,
        'Snow_Flag_Roll': filter_snow_flag
    }

    initial_count = df[column_name].notna().sum()

    for flag, include in flags.items():
        if include and flag in df.columns:
            df.loc[df[flag], column_name] = np.nan

    filtered_count = df[column_name].notna().sum()
    filtered_out = initial_count - filtered_count
    proportion_filtered = (filtered_out / initial_count) * 100

    print(f"Proportion of values filtered out: "
          f"{proportion_filtered:.2f}%")

    return df

def _get_week_indexes(
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

def _days(
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

        d1, d2 = _get_week_indexes(day1, day2)

        if d1 <= d2:
            filtered_df = df.loc[
                (df.index.dayofweek >= d1) & (df.index.dayofweek <= d2)]
        else:
            filtered_df = df.loc[
                (df.index.dayofweek >= d1) | (df.index.dayofweek <= d2)]
        
        if filtered_df.empty:
            raise ValueError(f"No data found for the specified day range "
                           f"({day1.lower()} to {day2.lower()}). ")
        
        return filtered_df
    else:
        return df

def _hours(
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




