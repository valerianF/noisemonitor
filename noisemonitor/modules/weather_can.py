"""
This chunk of code is derived from the pip package env_canada version 0.8.0.
It allows to retrieve historical weather data from environment canada's 
publicly available weather stations. 
"""

import requests
import asyncio
import aiohttp
import lxml.html
import pandas as pd
import warnings
import numpy as np

from dateutil.relativedelta import relativedelta
from datetime import datetime
from io import StringIO

from typing import List

from noisemonitor.modules.noisemonitor import NoiseMonitor

USER_AGENT = "env_canada/0.8.0"

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

def get_historical_stations_can(
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
        raise ValueError("Radius available range is between 25 and 100 kilometers")
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
        station_name = station_req_form.xpath(
            './/div[@class="col-md-10 col-sm-8 col-xs-8"]'
        )[0].text
        station["prov"] = station_req_form.xpath(
            './/div[@class="col-md-10 col-sm-8 col-xs-8"]'
        )[1].text
        station["proximity"] = float(
            station_req_form.xpath(
                './/div[@class="col-md-10 col-sm-8 col-xs-8"]'
            )[2].text
        )
        station["id"] = station_req_form.find(
            "input[@name='StationID']"
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

def flip_daterange(f):
    def wrapper(*args, **kwargs):
        if kwargs.get("daterange") in globals():
            if kwargs.get("daterange")[0] > kwargs.get("daterange")[1]:
                kwargs["daterange"] = (
                    kwargs.get("daterange")[1],
                    kwargs.get("daterange")[0],
                )
        return f(*args, **kwargs)

    return wrapper

@flip_daterange
async def get_historical_data_can(
    station_id: int,
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
    station_id: int
        the ID of the station found with get_historical_stations
    daterange: tuple of datetime    
        the dates between which the data are retrieved
    timeframe: str
        selection of granularity : 'hourly' or 'daily'

    Returns
    ----------
    DataFrame: All data in the range
    """
    df = pd.DataFrame()

    startdate, stopdate = pd.to_datetime(daterange)
    months = monthlist(daterange=daterange)
    _tf = {"hourly": 1, "daily": 2}
    timeframe = _tf[timeframe]

    async def _fetch_data(year, month):
        params = {
            "stationID": station_id,
            "Year": year,
            "Month": month,
            "format": "csv",
            "timeframe": timeframe,
            "submit": "Download+Data",
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                WEATHER_URL.format("e"), 
                params=params, 
                headers={"User-Agent": USER_AGENT}
                ) as response:
                response.raise_for_status()
                result = await response.text()
                f = StringIO(result)
                nonlocal df
                df = pd.concat((df, pd.read_csv(f)))

    tasks = [_fetch_data(year, month) for year, month in months]
    await asyncio.gather(*tasks)

    df = df.set_index(
        df.filter(regex="Date/*", axis=1).columns.to_numpy()[0]
    )
    df.index = pd.to_datetime(df.index)

    df = df[startdate <= df.index]
    df = df[stopdate >= df.index]

    # Ensure the index is unique and sorted
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    return df

async def merge_weather_can(
    df: pd.DataFrame, 
    station_id: int,
    wind_speed_flag: int = 18,
    temp_range_flag: tuple = (-10, 30),
    hum_flag: int = 90,
    rolling_window_hours: int = 48
) -> pd.DataFrame:
    """
    Merge weather data with the input DataFrame based on the datetime index.
    Will also include flags for weather conditions.

    Parameters
    ----------
    df: DataFrame
        Input DataFrame with a datetime index.
    station_id: int
        The ID of the weather station.
    wind_speed_flag: int, default 18
        Threshold for wind speed flag.
    temp_range_flag: tuple, default (-10, 30)
        Temperature range for temperature flag.
    hum_flag: int, default 90
        Threshold for relative humidity flag.
    rolling_window_hours: int, default 48
        Size of the rolling window for rain and snow flags in hours.

    Returns
    ----------
    DataFrame: Merged DataFrame with weather data and flags.
    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Retrieve weather data
    weather_df = await get_historical_data_can(
        station_id=station_id,
        daterange=(df.index.min(), df.index.max()),
        timeframe="hourly"
    )

    # Ensure the index of weather_df is a DatetimeIndex
    if not isinstance(weather_df.index, pd.DatetimeIndex):
        weather_df.index = pd.to_datetime(weather_df.index)

    # Determine the time granularity of the input DataFrame
    time_diff = df.index.to_series().diff().dropna().mode()[0]
    if time_diff >= pd.Timedelta(days=1):
        rolling_window = rolling_window_hours // 24
    else:
        rolling_window = rolling_window_hours

    # Create flags
    weather_df['Wind_Spd_Flag'] = (
        weather_df['Wind Spd (km/h)'] >= wind_speed_flag
    ).astype(bool)
    weather_df['Rain_Flag'] = (
        (weather_df['Precip. Amount (mm)'] > 0) & 
        weather_df['Weather'].str.contains('Rain', na=False)
    ).astype(bool)
    weather_df['Snow_Flag'] = (
        weather_df['Weather'].fillna('').str.contains('Snow')
    ).astype(bool)
    weather_df['Temp_Flag'] = (
        (weather_df['Temp (°C)'] < temp_range_flag[0]) | 
        (weather_df['Temp (°C)'] > temp_range_flag[1])
    ).astype(bool)
    weather_df['Rel_Hum_Flag'] = (
        weather_df['Rel Hum (%)'] > hum_flag
    ).astype(bool)

    # Apply rolling window for the past 48 hours (or days if daily data)
    weather_df['Rain_Flag_Roll'] = (
        weather_df['Rain_Flag'].rolling(window=rolling_window, min_periods=1)
        .max().astype(bool)
    )
    weather_df['Snow_Flag_Roll'] = (
        weather_df['Snow_Flag'].rolling(window=rolling_window, min_periods=1)
        .max().astype(bool)
    )

    # Select relevant columns
    weather_columns = [
        "Wind Spd (km/h)", 
        "Precip. Amount (mm)", 
        "Weather", 
        "Temp (°C)", 
        "Rel Hum (%)",
        "Wind_Spd_Flag", 
        "Rain_Flag", 
        "Snow_Flag", 
        "Temp_Flag", 
        "Rel_Hum_Flag",
        "Rain_Flag_Roll", 
        "Snow_Flag_Roll"
    ]
    weather_df_selected = weather_df[weather_columns]

    # Merge the input DataFrame with the weather data
    merged_df = pd.merge_asof(
        df.sort_index(),
        weather_df_selected.sort_index(),
        left_index=True,
        right_index=True,
        direction='nearest'
    )

    # Store the flag thresholds in the merged DataFrame
    merged_df.attrs['wind_speed_flag'] = wind_speed_flag
    merged_df.attrs['temp_range_flag'] = temp_range_flag
    merged_df.attrs['hum_flag'] = hum_flag
    merged_df.attrs['rolling_window_hours'] = rolling_window_hours

    return merged_df

@flip_daterange
def monthlist(daterange):
    startdate, stopdate = daterange

    def total_months(dt):
        return dt.month + 12 * dt.year

    mlist = []
    for tot_m in range(total_months(startdate) - 1, total_months(stopdate)):
        y, m = divmod(tot_m, 12)
        mlist.append((y, m + 1))
    return mlist

def contingency_weather_flags(
    df: pd.DataFrame,
    column: str, 
    include_wind_flag: bool = True, 
    include_rain_flag: bool = True, 
    include_temp_flag: bool = False,
    include_rel_hum_flag: bool = False, 
    include_snow_flag: bool = False
) -> pd.DataFrame:
    """
    Create a contingency table for LAeq, Lden, Lday, Levening, and Lnight 
    with and without the different weather flags.

    Parameters
    ----------
    df: DataFrame
        Input DataFrame with a datetime index. Typically, the output of
        merge_weather_can().
    column: str
        Column name to use for calculations.
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

    Returns
    ----------
    pd.DataFrame: Contingency table with LAeq,24h, Lden, Lday, Levening, Lnight, 
    and the proportion of data covered from the initial dataset.
    """
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

    subsets['All Flags'] = df[df[list(_active_flags.keys())].any(axis=1)]
    subsets['Neither Flags'] = df[~df[list(_active_flags.keys())].any(axis=1)]

    results = {}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        for key, df in subsets.items():
            nm_instance = NoiseMonitor(df)
            overall_leq = nm_instance.indicators.overall_leq(
                column, 
                hour1=0, 
                hour2=24
                )
            overall_lden = nm_instance.indicators.overall_lden(column)
            results[key] = {
                'Leq': overall_leq['leq'][0],
                'Lden': overall_lden['lden'][0]
            }
    
    contingency_table = pd.DataFrame(results).T

    # Calculate the difference in levels compared with the full dataset
    full_data_levels = contingency_table.loc['All Data']
    for level in ['Leq', 'Lden']:
        contingency_table[f'Diff {level}'] = (
            contingency_table[level] - full_data_levels[level]
        )

    total_data_points = len(subsets['All Data'][column])
    contingency_table['Covered data (%)'] = contingency_table.index.map(
        lambda x: np.round(((len(subsets[x][column]) / total_data_points) * 100), 1), 
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
        'Temp_Flag': f'Temperature below {temp_range_flag[0]}°C or above {temp_range_flag[1]}°C',
        'No Temp_Flag': 'No Flag - Temperature',
        'Rel_Hum_Flag': f'Humidity above {hum_flag}%',
        'No Rel_Hum_Flag': 'No Flag - Humidity',
        'Snow_Flag_Roll': f'Snow in the last {rolling_window_hours}h',
        'No Snow_Flag_Roll': 'No Flag - Snow'
    }

    contingency_table.rename(index=row_rename_map, inplace=True)

    # Reorder rows
    row_order = [
        'All Data',
        f'Wind speed >= {wind_speed_flag} km/h',
        f'Rain in the last {rolling_window_hours}h',
        f'Temperature below {temp_range_flag[0]}°C or above {temp_range_flag[1]}°C',
        f'Humidity above {hum_flag}%',
        f'Snow in the last {rolling_window_hours}h',
        'No Flag - Wind',
        'No Flag - Rain',
        'No Flag - Temperature',
        'No Flag - Humidity',
        'No Flag - Snow',
        'All Flags',
        'Neither Flags'
    ]

    contingency_table = contingency_table.reindex(row_order)

    return contingency_table.dropna()
