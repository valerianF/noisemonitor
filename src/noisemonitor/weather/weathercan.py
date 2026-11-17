"""
This chunk of code is derived from the pip package env_canada version 0.15.0.
It allows to retrieve historical weather data from environment canada's 
publicly available weather stations 
(see https://climate.weather.gc.ca/historical_data/). 
"""

import requests
import asyncio
import aiohttp
import lxml.html
import pandas as pd
import warnings
import numpy as np
from typing import Optional, Union

from ..util.core import _column_to_index, get_interval

from dateutil.relativedelta import relativedelta
from datetime import datetime
from io import StringIO

from typing import List, Optional

from noisemonitor.summary import leq, lden

USER_AGENT = "noisemonitor/1.1.0"

STATIONS_URL = (
    "https://climate.weather.gc.ca/historical_data/"
    "search_historic_data_stations_{}.html"
)
WEATHER_URL = (
    "https://climate.weather.gc.ca/climate_data/bulk_data_{}.html"
)

_TODAY = datetime.today().date()
_ONE_YEAR_AGO = _TODAY - relativedelta(years=1, months=1, day=1)
_YEAR = datetime.today().year

def get_historical_stations(
    coordinates: List[float],
    radius: int = 25,
    start_year: int = 1840,
    end_year: int = _YEAR,
    limit: int = 25
) -> pd.DataFrame:
    """
    Get list of all historical stations from Environment Canada.

    Parameters
    ----------
    coordinates: list of float
        List of two floats for latitude and longitude
    radius: int, default 25
        Radius in kilometers to search for stations surrounding the specified
        coordinates (must be between 25 and 100)
    start_year: int, default 1840
        Starting year for the database query
    end_year: int, default present year
        Ending year for the database query
    limit: int, default 25
        limit of weather stations to list

    Returns
    ----------
    DataFrame: includes all available weather stations, their proximity to the
    point in kilometers, id number, and the time range of covered data and 
    associated time granulation (hourly, daily or monthly).
    """
    if radius < 25 or radius > 100:
        raise ValueError(
            "Radius available range is between 25 and 100 kilometers"
        )
    lat, lng = coordinates
    params = {
        "searchType": "stnProx",
        "timeframe": "2",
        "txtRadius": radius,
        "optProxType": "decimal",
        "txtLatDecDeg": lat,
        "txtLongDecDeg": lng,
        "optLimit": "yearRange",
        "StartYear": start_year,
        "EndYear": end_year,
        "Year": start_year,
        "Month": "1",
        "Day": "1",
        "selRowPerPage": limit,
        "selCity": "",
        "selPark": "",
        "txtCentralLatDeg": "",
        "txtCentralLatMin": "",
        "txtCentralLatSec": "",
        "txtCentralLongDeg": "",
        "txtCentralLongMin": "",
        "txtCentralLongSec": "",
    }

    response = requests.get(
        STATIONS_URL.format("e"),
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    result = response.text

    station_html = result
    station_tree = lxml.html.fromstring(station_html)
    station_req_forms = station_tree.xpath(
        "//form[starts-with(@id, 'stnRequest') and '-sm' = substring(@id, "
        "string-length(@id) - string-length('-sm') +1)]"
    )

    stations = {}
    for station_req_form in station_req_forms:
        station = {}
        station_table = station_req_form.xpath(
            './/div[@class="col-md-10 col-sm-8 col-xs-8"]'
        )
        station_name = station_table[0].text
        station["prov"] = station_table[1].text
        station["proximity"] = float(station_table[2].text)
        station["id"] = station_req_form.find(
            "input[@name='climate_id']"
        ).attrib.get("value")
        station["hlyRange"] = station_req_form.find(
            "input[@name='hlyRange']"
        ).attrib.get("value")
        station["dlyRange"] = station_req_form.find(
            "input[@name='dlyRange']"
        ).attrib.get("value")
        station["mlyRange"] = station_req_form.find(
            "input[@name='mlyRange']"
        ).attrib.get("value")
        stations[station_name] = station

    return pd.DataFrame(stations)

async def get_historical_data(
    station_id: Union[int, str],
    daterange=(
        _ONE_YEAR_AGO,
        _TODAY,
    ),
    timeframe="hourly"
) -> pd.DataFrame:
    """
    Get historical weather data from Environment Canada in
    the given range for the given station.

    Parameters
    ----------
    station_id: str
        the climate ID of the station, as given in the 'id' row of
        get_historical_stations (e.g., '702S006').
    daterange: tuple of datetime
        the dates between which the data are retrieved, in either order
    timeframe: str
        selection of granularity : 'hourly' or 'daily'

    Returns
    ----------
    DataFrame: All data in the range
    """
    startdate, stopdate = sorted(pd.to_datetime(daterange))
    months = monthlist(daterange=daterange)
    _tf = {"hourly": 1, "daily": 2}
    timeframe = _tf[timeframe]

    rows = []
    semaphore = asyncio.Semaphore(10)

    async def _fetch_data(session, year, month, day):
        async with semaphore:
            params = {
                "climate_id": station_id,
                "Year": year,
                "Month": month,
                "Day": day,
                "format": "csv",
                "timeframe": timeframe,
                "submit": "Download+Data",
            }

            async with session.get(
                WEATHER_URL.format("e"),
                params=params,
                headers={"User-Agent": USER_AGENT}
                ) as response:
                response.raise_for_status()
                result = await response.text()
                rows.append(pd.read_csv(StringIO(result)))

    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_data(session, year, month, day) for year, month, day in months]
        await asyncio.gather(*tasks)

    df = pd.concat(rows, ignore_index=False) if rows else pd.DataFrame()

    df = df.set_index(
        df.filter(regex="Date/*", axis=1).columns.to_numpy()[0]
    )
    df.index = pd.to_datetime(df.index)

    df = df[startdate <= df.index]
    df = df[stopdate >= df.index]

    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    return df

async def merge_weather(
    df: pd.DataFrame, 
    station_id: Union[int, str],
    wind_speed_flag: int = 18,
    temp_range_flag: tuple = (-10, 30),
    hum_flag: int = 90,
    rolling_window_hours: int = 48,
    timezone: Optional[str] = 'America/Toronto'
) -> pd.DataFrame:
    """
    Async function. Merge weather data with the input DataFrame based on the
    datetime index. Will also include flags for weather conditions.

    Samples at intervals of 1 h or less take the values of the hourly
    weather row nearest to them. Coarser samples (e.g., daily LAeq,24h)
    take the weather rows of their own period [t, t + period), period being
    the frequency of df (e.g., a day, a week or a month) or, if it cannot be
    inferred, its most common interval, and ending at the next sample at the
    latest, with a UserWarning: wind speed, temperature and relative
    humidity are averaged and their flags follow these averages,
    precipitation is summed, 'Weather' lists the conditions met, and the
    rain and snow flags (rolling ones included) are raised if they are
    raised in any hour of the period. Data with fewer than 3 samples, whose
    interval cannot be determined, always take the nearest hourly row.

    The rain and snow flags of an hour are raised when it has precipitation
    and its 'Weather' condition reports rain or snow; precipitation without
    a condition is then not flagged, with a UserWarning giving the number of
    such hours. If the station reports no condition at all, rain and snow
    are inferred from the temperature (precipitation at or above 0 °C is
    rain, below 0 °C snow), with a UserWarning: check them against other
    weather archives.

    Environment Canada gives the times of its rows in Local Standard
    Time (LST) all year: without timezone, they are merged as they are, so
    the times of df must also be in standard time; with timezone, they are
    first converted to the local clock time of that zone, daylight saving
    time included (the time base of data loaded with noisemonitor.load).

    Parameters
    ----------
    df: DataFrame
        Input DataFrame with a datetime index.
    station_id: str
        The climate ID of the weather station, as given in the 'id' row of
        get_historical_stations (e.g., '702S006').
    wind_speed_flag: int, default 18
        Threshold for wind speed flag.
    temp_range_flag: tuple, default (-10, 30)
        Temperature range for temperature flag.
    hum_flag: int, default 90
        Threshold for relative humidity flag.
    rolling_window_hours: int, default 48
        Size of the rolling window for rain and snow flags in hours.
    timezone: str, optional
        Time zone of the times of df (default: 'America/Toronto'), to convert
        the weather times from LST to local time. At the end of daylight
        saving time, the repeated local hour keeps its first weather row.
        If None, weather times are kept in LST.

    Returns
    ----------
    DataFrame: Merged DataFrame with weather data and flags.
    """
    # Interval of df (unknown below 3 samples, then merged as hourly data)
    interval = pd.Timedelta(seconds=get_interval(df) if len(df) > 2 else 0)
    averaged = interval > pd.Timedelta(hours=1)
    # Period [t, t + period) of each sample: the frequency of df (e.g., a
    # month), else its interval
    period = interval
    if averaged:
        period = (pd.DatetimeIndex(df.index, freq='infer').freq
                or df.index.to_series().diff().mode()[0])

    # Import from the start of the rolling window of the first sample, 
    # with a margin for the nearest merge and the shift from LST
    margin = pd.Timedelta(hours=2)
    hourly = await get_historical_data(station_id, daterange=(
        df.index[0] - pd.Timedelta(hours=rolling_window_hours) - margin,
        df.index[-1] + period + margin))

    # Rain and snow: precipitation with the condition reported in 'Weather',
    # or, from a station reporting no condition, at or above / below 0 °C
    precip = hourly['Precip. Amount (mm)'] > 0
    if hourly['Weather'].isna().all():
        if precip.any():
            warnings.warn(
                "No weather condition reported by the station: rain and snow "
                "are inferred from the temperature (precipitation at or above "
                "0 °C is rain, below 0 °C snow). Check them against other "
                "weather archives.", UserWarning, stacklevel=2)
        rain, snow = hourly['Temp (°C)'] >= 0, hourly['Temp (°C)'] < 0
    else:
        unreported = (precip & hourly['Weather'].isna()).sum()
        if unreported:
            warnings.warn(
                f"No weather condition reported for {unreported} hour(s) with "
                "precipitation: not flagged as rain or snow.",
                UserWarning, stacklevel=2)
        text = hourly['Weather'].fillna('')
        rain, snow = text.str.contains('Rain'), text.str.contains('Snow')
    window = pd.Timedelta(hours=rolling_window_hours)
    for flag, condition in (('Rain_Flag', rain), ('Snow_Flag', snow)):
        hourly[flag] = precip & condition
        # Raised in any hour of the last rolling_window_hours hours
        hourly[f'{flag}_Roll'] = hourly[flag].rolling(window).max().astype(bool)

    if timezone is not None:
        # LST is the standard UTC offset of the zone all year: through UTC,
        # weather times become local clock times
        first = hourly.index[0].tz_localize(
            timezone, ambiguous=True, nonexistent='shift_forward')
        lst = first.utcoffset() - first.dst()
        hourly.index = ((hourly.index - lst).tz_localize('UTC')
                        .tz_convert(timezone).tz_localize(None))

    flags = ['Rain_Flag', 'Snow_Flag', 'Rain_Flag_Roll', 'Snow_Flag_Roll']
    hourly = hourly[['Wind Spd (km/h)', 'Precip. Amount (mm)', 'Weather',
                    'Temp (°C)', 'Rel Hum (%)'] + flags]
    if averaged:
        warnings.warn(
            f"The data are at intervals of {interval}, longer than the hourly "
            "weather data: weather values are averaged over each sample "
            "(precipitation summed).", UserWarning, stacklevel=2)
        # Hourly rows of [t, t + period) of each sample t (and before the next)
        start = pd.DatetimeIndex(
            pd.Series(df.index, index=df.index).asof(hourly.index))
        inside = hourly.index < start + period
        weather = hourly[inside].groupby(start[inside]).agg({
            'Wind Spd (km/h)': 'mean',
            'Precip. Amount (mm)': lambda p: p.sum(min_count=1),
            # Distinct conditions in order (e.g., 'Rain,Fog'), NaN if none
            'Weather': lambda w: ','.join(
                dict.fromkeys(','.join(w.dropna()).split(','))) or np.nan,
            'Temp (°C)': 'mean',
            'Rel Hum (%)': 'mean',
            **{flag: 'any' for flag in flags}}).reindex(df.index)
        weather[flags] = weather[flags].eq(True)  # no weather row: not raised
    else:
        # The local hour repeated at the end of daylight saving time keeps
        # its first row, as in noisemonitor.load
        weather = pd.merge_asof(df[[]], hourly[~hourly.index.duplicated()],
                                left_index=True, right_index=True,
                                direction='nearest')

    # Thresholds, on the hourly or averaged values
    weather['Wind_Spd_Flag'] = weather['Wind Spd (km/h)'] >= wind_speed_flag
    weather['Temp_Flag'] = ((weather['Temp (°C)'] < temp_range_flag[0])
                            | (weather['Temp (°C)'] > temp_range_flag[1]))
    weather['Rel_Hum_Flag'] = weather['Rel Hum (%)'] > hum_flag

    merged_df = df.join(weather[[
        'Wind Spd (km/h)', 'Precip. Amount (mm)', 'Weather', 'Temp (°C)',
        'Rel Hum (%)', 'Wind_Spd_Flag', 'Rain_Flag', 'Snow_Flag', 'Temp_Flag',
        'Rel_Hum_Flag', 'Rain_Flag_Roll', 'Snow_Flag_Roll']])
    merged_df.attrs.update(
        wind_speed_flag=wind_speed_flag, temp_range_flag=temp_range_flag,
        hum_flag=hum_flag, rolling_window_hours=rolling_window_hours,
        timezone=timezone)
    return merged_df

def monthlist(daterange):
    """List the months requested from Environment Canada for a date range.

    Parameters
    ----------
    daterange: tuple of datetime
        the dates of the range, in either order.

    Returns
    ----------
    list of tuple: (year, month, 1) of each month from the month of the
        first date to the month of the last date.
    """
    startdate, stopdate = sorted(pd.to_datetime(daterange))

    def total_months(dt):
        return dt.month + 12 * dt.year

    mlist = []
    for tot_m in range(total_months(startdate) - 1, total_months(stopdate)):
        y, m = divmod(tot_m, 12)
        mlist.append((y, m + 1, 1))
    return mlist

def contingency_weather_flags(
    df: pd.DataFrame,
    column: Optional[Union[int, str]] = 0,  
    include_wind_flag: bool = True, 
    include_rain_flag: bool = True, 
    include_temp_flag: bool = False,
    include_rel_hum_flag: bool = False,
    include_snow_flag: bool = False,
    start_day: int = 7,
    start_evening: int = 19,
    start_night: int = 23
) -> pd.DataFrame:
    """
    Create a contingency table for LAeq, Lden, Lday, Levening, and Lnight
    with and without the different weather flags. Lden follows the
    Directive 2002/49/EC (see noisemonitor.summary.lden).

    Parameters
    ----------
    df: DataFrame
        Input DataFrame with a datetime index. Typically, the output of
        merge_weather().
    column: Optional[int or str], default 0
        The column index (int) or column name (str) to use for calculations. 
    include_wind_flag: bool, default True
        Whether to include the Wind Speed Flag in the contingency table.
    include_rain_flag: bool, default True
        Whether to include the Rain Flag in the contingency table.
    include_temp_flag: bool, default False
        Whether to include the Temperature Flag in the contingency table.
    include_rel_hum_flag: bool, default False
        Whether to include the Relative Humidity Flag in the contingency table.
    include_snow_flag: bool, default False
        Whether to include the Snow Flag in the contingency table.
    start_day: int, default 7
        hour (0-23) at which the Lden day period starts.
    start_evening: int, default 19
        hour (0-23) at which the Lden evening period starts.
    start_night: int, default 23
        hour (0-23) at which the Lden night period starts.

    Returns
    ----------
    pd.DataFrame:
        Contingency table with the Leq and Lden of all the data, of the
        rows with and without each included flag, of the rows with any
        included flag raised ('Any Flag') and of those with none ('No
        Flag'), their differences from all the data, and the proportion of
        the data in each row. Rows with fewer than 3 samples are left out.
        The Lden of a row without data in the day, evening or night period
        (e.g., a flag raised only at night) is NaN, and its Leq is kept.
    """
    column = _column_to_index(df, column)

    flags = {
        'Wind_Spd_Flag': include_wind_flag,
        'Rain_Flag_Roll': include_rain_flag,
        'Temp_Flag': include_temp_flag,
        'Rel_Hum_Flag': include_rel_hum_flag,
        'Snow_Flag_Roll': include_snow_flag
    }

    _active_flags = {
        flag: include for flag, include in flags.items() if include}

    subsets = {
        'All Data': df
    }

    for flag in _active_flags:
        subsets[f'No {flag}'] = df[~df[flag]]
        subsets[flag] = df[df[flag]]

    # Rows with at least one active flag raised, and rows with none
    any_flag = df[list(_active_flags.keys())].any(axis=1)
    subsets['Any Flag'] = df[any_flag]
    subsets['No Flag'] = df[~any_flag]

    results = {}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        for key, datafr in subsets.items():
            # Skip subsets with fewer than 3 rows
            if len(datafr) < 3:
                results[key] = {
                    'Leq': np.nan,
                    'Lden': np.nan
                }
                continue
            
            ov_leq = leq(
                df=datafr,
                column=column, 
                hour1=0, 
                hour2=24
                )
            ov_lden = lden(
                df=datafr,
                column=column,
                start_day=start_day,
                start_evening=start_evening,
                start_night=start_night
                )
            results[key] = {
                'Leq': ov_leq['Leq'][0],
                'Lden': ov_lden['Lden'][0]
            }
    
    contingency_table = pd.DataFrame(results).T

    # Calculate the difference in levels compared with the full dataset
    full_data_levels = contingency_table.loc['All Data']
    for level in ['Leq', 'Lden']:
        contingency_table[f'Diff {level}'] = (
            contingency_table[level] - full_data_levels[level]
        )

    total_data_points = len(subsets['All Data'].iloc[:, column])
    contingency_table['Covered data (%)'] = contingency_table.index.map(
        lambda x: np.round(
            (len(subsets[x].iloc[:, column]) / total_data_points) * 100,
            1
        ),
    )

    # Rename columns based on flag thresholds
    wind_speed_flag = df.attrs.get('wind_speed_flag', 18)
    temp_range_flag = df.attrs.get('temp_range_flag', (-10, 30))
    hum_flag = df.attrs.get('hum_flag', 90)
    rolling_window_hours = df.attrs.get('rolling_window_hours', 48)

    row_rename_map = {
        'Wind_Spd_Flag': f'Wind speed >= {wind_speed_flag} km/h',
        'No Wind_Spd_Flag': 'No Flag - Wind',
        'Rain_Flag_Roll': f'Rain in the last {rolling_window_hours}h',
        'No Rain_Flag_Roll': 'No Flag - Rain',
        'Temp_Flag': (
            f'Temperature below {temp_range_flag[0]}°C or above '
            f'{temp_range_flag[1]}°C'
        ),
        'No Temp_Flag': 'No Flag - Temperature',
        'Rel_Hum_Flag': f'Humidity above {hum_flag}%',
        'No Rel_Hum_Flag': 'No Flag - Humidity',
        'Snow_Flag_Roll': f'Snow in the last {rolling_window_hours}h',
        'No Snow_Flag_Roll': 'No Flag - Snow'
    }

    contingency_table = contingency_table.rename(index=row_rename_map)

    # Reorder rows
    row_order = [
        'All Data',
        f'Wind speed >= {wind_speed_flag} km/h',
        f'Rain in the last {rolling_window_hours}h',
        (
            f'Temperature below {temp_range_flag[0]}°C or above '
            f'{temp_range_flag[1]}°C'
        ),
        f'Humidity above {hum_flag}%',
        f'Snow in the last {rolling_window_hours}h',
        'No Flag - Wind',
        'No Flag - Rain',
        'No Flag - Temperature',
        'No Flag - Humidity',
        'No Flag - Snow',
        'Any Flag',
        'No Flag'
    ]

    contingency_table = contingency_table.reindex(row_order)

    # Rows without any level: fewer than 3 samples, or flags not included
    return contingency_table.dropna(subset=['Leq', 'Lden'], how='all')
