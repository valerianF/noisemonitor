"""Unit tests for weather.weathercan module. Without network access (fake
requests and sessions), except TestLive (marked live: Environment Canada service, run
with pytest --live), which also covers filter.weather_flags and the weather
plots on the merged data."""

import asyncio
import re
import warnings
from datetime import date

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from noisemonitor.summary import leq, lden
from noisemonitor.util import display
from noisemonitor.util.core import CoverageWarning, IntegrationTimeWarning
from noisemonitor.util.filter import weather_flags

matplotlib.use('Agg')

# Needs the optional dependencies (pip install noisemonitor[weather])
weathercan = pytest.importorskip("noisemonitor.weather.weathercan")

# Request handling only: no acoustic reference values
pytestmark = pytest.mark.behaviour

# Montréal-Trudeau, the station of the docstrings, which reports precipitation
# amounts but no weather condition, and its coordinates
LIVE_STATION = '702S006'
COORDINATES = [45.47, -73.74]
# Montréal International Airport: weather conditions but no precipitation amount
LIVE_STATION_CONDITIONS = '7025251'

FLAGS = ['Wind_Spd_Flag', 'Rain_Flag_Roll', 'Temp_Flag', 'Rel_Hum_Flag', 'Snow_Flag_Roll']


def _weather_session(rain=(), conditions=None, temp_offset=0.0, no_amount=()):
    """aiohttp.ClientSession stand-in serving hourly weather rows of each
    requested month: 1 mm of precipitation at the given times (LST), none
    otherwise, and no amount (NaN) at the times of no_amount (True: every
    hour), the hour h of the row (LST) plus temp_offset as its
    temperature, 2 h km/h of wind, and the given 'Weather' conditions
    ({time (LST): text}, NaN otherwise)."""
    rain = pd.DatetimeIndex(rain)
    conditions = {pd.Timestamp(t): c for t, c in (conditions or {}).items()}
    no_amount = no_amount if no_amount is True else pd.DatetimeIndex(no_amount)

    class _Response:
        def __init__(self, params):
            self.params = params

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        async def text(self):
            start = pd.Timestamp(self.params['Year'], self.params['Month'], 1)
            hours = pd.date_range(start, start + pd.offsets.MonthBegin(1), freq='h',
                                inclusive='left')
            return pd.DataFrame({
                'Date/Time (LST)': hours.strftime('%Y-%m-%d %H:%M'),
                'Temp (°C)': hours.hour + temp_offset,
                'Rel Hum (%)': 50.0,
                'Wind Spd (km/h)': 2.0 * hours.hour,
                'Precip. Amount (mm)': np.where(
                    hours.isin(hours if no_amount is True else no_amount),
                    np.nan, np.where(hours.isin(rain), 1.0, 0.0)),
                'Weather': [conditions.get(h, np.nan) for h in hours],
            }).to_csv(index=False)

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def get(self, url, params=None, headers=None):
            return _Response(params)

    return _Session


def _merge(monkeypatch, index, session=None, timezone=None, **kwargs):
    """merge_weather of constant levels at the times of index, with the
    weather of session (default: _weather_session())."""
    monkeypatch.setattr(weathercan.aiohttp, 'ClientSession',
                        session or _weather_session())
    df = pd.DataFrame({'LAeq': 50.0}, index=index)
    return asyncio.run(weathercan.merge_weather(df, 1, timezone=timezone, **kwargs))


def _stations_page(stations):
    """Station search results page of Environment Canada, laid out as the
    real one: each station (name, province, proximity in km, climate ID,
    hourly, daily and monthly ranges) in a form for wide screens, then in a
    form for small screens (id ending in '-sm'), whose label/value rows
    list the name, province and proximity (padded), then other fields."""
    forms = []
    for k, (name, prov, proximity, climate_id, *ranges) in enumerate(stations):
        inputs = ''.join(
            f'<input type="hidden" name="{field}" value="{value}"/>'
            for field, value in zip(['hlyRange', 'dlyRange', 'mlyRange', 'climate_id'],
                                    [*ranges, climate_id]))
        value = '<div class="col-md-10 col-sm-8 col-xs-8">{}</div>'
        forms.append(
            f'<form method="post" id="stnRequest{k}">{inputs}'
            f'<div class="col-lg-3 col-md-3 col-sm-3 col-xs-3">{name}</div>'
            f'<div class="col-lg-1 col-md-1 col-sm-1 col-xs-1">{prov}</div></form>'
            f'<form method="post" id="stnRequest{k}-sm">'
            + ''.join(f'<div class="row"><div class="col-md-2 col-sm-4 col-xs-4">'
                    f'{label}</div>{value.format(text)}</div>'
                    for label, text in [('Station', name), ('Prov.', prov),
                                        ('Proximity', f'   {proximity:.2f}'),
                                        ('Year', '<select name="Year"></select>')])
            + f'{inputs}</form>')
    return f'<html><body><main>{"".join(forms)}</main></body></html>'


class _FakeRequests:
    """Stands in for requests.get: records the requests and answers with
    the given page."""

    def __init__(self, page):
        self.page = page
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(dict(url=url, params=params, headers=headers))
        response = type('Response', (), {})()
        response.text = self.page
        response.raise_for_status = lambda: None
        return response


class TestStations:
    """Stations listed by the search results page."""

    @pytest.mark.parametrize("stations", [
        [('MONTREAL INTL A', 'QC', 0.09, '7025251', '2013-02-13|2026-11-17',
        '2013-02-14|2026-11-17', '|'),
        ('OTTAWA CDA', 'ON', 99.5, '6105976', '|', '1889-11-01|2026-11-17',
        '1889-01-01|2006-12-01')],
        [],
    ], ids=['two_stations', 'none'])
    def test_stations_page(self, monkeypatch, stations):
        """One column per station (its name), with its province, proximity,
        climate ID and data ranges, from its small-screen form (the
        wide-screen ones are ignored). No station: an empty frame."""
        monkeypatch.setattr(weathercan.requests, 'get',
                            _FakeRequests(_stations_page(stations)).get)

        df = weathercan.get_historical_stations(COORDINATES)

        assert list(df.columns) == [s[0] for s in stations]
        assert list(df.index) == (['prov', 'proximity', 'id', 'hlyRange', 'dlyRange',
                                'mlyRange'] if stations else [])
        for name, prov, proximity, *values in stations:
            assert list(df[name]) == [prov, proximity, *values]

    @pytest.mark.parametrize("radius, valid", [
        (24, False), (25, True), (100, True), (101, False)])
    def test_radius(self, monkeypatch, radius, valid):
        """A radius of 25 to 100 km is sent with the coordinates, years and
        limit to the station search; outside, a ValueError and no request."""
        fake = _FakeRequests(_stations_page([]))
        monkeypatch.setattr(weathercan.requests, 'get', fake.get)

        if not valid:
            with pytest.raises(ValueError, match="between 25 and 100"):
                weathercan.get_historical_stations(COORDINATES, radius=radius)
            assert fake.calls == []
            return
        weathercan.get_historical_stations(COORDINATES, radius=radius,
                                        start_year=2000, end_year=2020, limit=10)

        [call] = fake.calls
        assert call['url'] == weathercan.STATIONS_URL.format('e')
        assert call['headers'] == {'User-Agent': weathercan.USER_AGENT}
        assert {k: call['params'][k] for k in [
            'txtLatDecDeg', 'txtLongDecDeg', 'txtRadius', 'StartYear', 'EndYear',
            'selRowPerPage']} == {
            'txtLatDecDeg': 45.47, 'txtLongDecDeg': -73.74, 'txtRadius': radius,
            'StartYear': 2000, 'EndYear': 2020, 'selRowPerPage': 10}


class TestDaterange:
    """Months requested for a date range, given in either order."""

    @pytest.mark.parametrize("daterange, expected", [
        ((date(2023, 1, 15), date(2023, 1, 20)), [(2023, 1, 1)]),
        # Timestamps as merge_weather gives them, across the new year
        ((pd.Timestamp('2022-11-30 23:00'), pd.Timestamp('2023-02-01 00:30')),
        [(2022, 11, 1), (2022, 12, 1), (2023, 1, 1), (2023, 2, 1)]),
        ((date(2023, 3, 2), date(2023, 1, 15)),
        [(2023, 1, 1), (2023, 2, 1), (2023, 3, 1)]),
    ], ids=['one_month', 'new_year', 'reversed'])
    def test_monthlist(self, daterange, expected):
        """Each month from the month of the earlier date to the month of the
        later one, as (year, month, 1)."""
        assert weathercan.monthlist(daterange) == expected

    def test_get_historical_data_reversed(self, monkeypatch):
        """Hourly rows from the earlier to the later date, whatever the
        order of the range."""
        monkeypatch.setattr(weathercan.aiohttp, 'ClientSession', _weather_session())
        start, stop = pd.Timestamp('2023-01-15 06:00'), pd.Timestamp('2023-03-02 12:00')

        ordered = asyncio.run(weathercan.get_historical_data(1, daterange=(start, stop)))
        reversed_ = asyncio.run(weathercan.get_historical_data(1, daterange=(stop, start)))

        pd.testing.assert_frame_equal(reversed_, ordered)
        assert list(ordered.index) == list(pd.date_range(start, stop, freq='h'))


def _jan10(times):
    """Times of the day 2023-01-10 given as 'hh:mm'."""
    return pd.DatetimeIndex([f'2023-01-10 {t}' for t in times])


# Conditions of the hours reported without a precipitation amount
NO_AMOUNT_CONDITIONS = {'03:00': 'Rain', '05:00': 'Snow', '08:00': 'Fog', '12:00': 'Rain'}
DRIZZLE = "flagged as rain from a drizzle or ice pellets condition"


class TestMergeWeather:
    """Weather rows merged onto the noise data."""

    @pytest.mark.filterwarnings("error")
    @pytest.mark.parametrize("freq, start, periods, rain", [
        ('h', '2023-06-10', 5 * 24, '2023-06-11 06:00'),   # hourly noise data
        ('20min', '2023-06-10', 5 * 72, '2023-06-11 06:00'),
        ('h', '2023-06-11', 3 * 24, '2023-06-10 20:00'),   # rain before the data
    ], ids=['hourly', '20_min', 'rain_before_data'])
    def test_rain_flag_window(self, monkeypatch, freq, start, periods, rain):
        """A sample at t is flagged for rain recorded in (t - 48 h, t] (from
        the hourly row nearest to t), even when the rain falls before the
        first sample, without warning."""
        index = pd.date_range(start, periods=periods, freq=freq)

        merged = _merge(monkeypatch, index, _weather_session([rain], {rain: 'Rain'}))

        # Nearest hourly row (half hours go to the next one)
        since_rain = index.round('h') - pd.Timestamp(rain)
        expected = (since_rain >= pd.Timedelta(0)) & (since_rain < pd.Timedelta(hours=48))
        assert expected.any()
        assert list(merged['Rain_Flag_Roll']) == list(expected)

    @pytest.mark.parametrize("rain, conditions, session, rain_at, snow_at, warning", [
        # 1 mm at 03:00: rain, drizzle and ice pellets are rain, snow is snow
        (['03:00'], {'03:00': 'Rain'}, {}, ['03:00'], [], None),
        (['03:00'], {'03:00': 'Freezing Rain'}, {}, ['03:00'], [], None),
        (['03:00'], {'03:00': 'Rain,Drizzle'}, {}, ['03:00'], [], None),
        (['03:00'], {'03:00': 'Drizzle'}, {}, ['03:00'], [], DRIZZLE),
        (['03:00'], {'03:00': 'Freezing Drizzle'}, {}, ['03:00'], [], DRIZZLE),
        (['03:00'], {'03:00': 'Ice Pellets'}, {}, ['03:00'], [], DRIZZLE),
        (['03:00'], {'03:00': 'Snow,Ice Pellets'}, {}, ['03:00'], ['03:00'], DRIZZLE),
        (['03:00'], {'03:00': 'Snow'}, {}, [], ['03:00'], None),
        (['03:00'], {'03:00': 'Blowing Snow,Drifting Snow'}, {}, [], ['03:00'], None),
        (['03:00'], {'03:00': 'Fog'}, {}, [], [], None),
        # Precipitation at 03:00 without a condition: neither
        (['03:00', '05:00'], {'05:00': 'Snow'}, {}, [], ['05:00'],
         r"No weather condition reported for 1 hour\(s\) with precipitation"),
        # No condition at all: from the temperature (-9 °C at 03:00, 3 °C at 15:00)
        (['03:00', '15:00'], {}, {'temp_offset': -12}, ['15:00'], ['03:00'],
         "inferred from the temperature"),
        # Conditions without an amount: from the condition alone (fog: neither;
        # 12:00 has 1 mm, or no amount from a station without amounts)
        (['12:00'], NO_AMOUNT_CONDITIONS, {'no_amount': True}, ['03:00', '12:00'],
         ['05:00'], r"No precipitation amount reported for 3 hour\(s\)"),
        (['12:00'], NO_AMOUNT_CONDITIONS, {'no_amount': _jan10(['03:00', '05:00', '08:00'])},
         ['03:00', '12:00'], ['05:00'], r"No precipitation amount reported for 2 hour\(s\)"),
    ], ids=['rain', 'freezing_rain', 'rain_drizzle', 'drizzle', 'freezing_drizzle',
            'ice_pellets', 'snow_ice_pellets', 'snow', 'blowing_snow', 'fog',
            'precipitation_without_condition', 'no_condition_reported',
            'no_amount_station', 'no_amount_hours'])
    def test_rain_snow_flags(self, monkeypatch, rain, conditions, session, rain_at,
                             snow_at, warning):
        """Hours with precipitation are rain when their condition names rain,
        drizzle or ice pellets, and snow when it names snow; a condition
        without an amount is flagged alone; a station reporting no condition
        gets rain (>= 0 °C) or snow from the temperature. One UserWarning
        when drizzle or ice pellets alone flag rain, or when a condition or
        an amount is missing."""
        index = pd.date_range('2023-01-10', periods=24, freq='h')
        conditions = dict(zip(_jan10(conditions), conditions.values()))

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            merged = _merge(monkeypatch, index,
                            _weather_session(_jan10(rain), conditions, **session))

        assert len(record) == (warning is not None)
        if warning:
            assert issubclass(record[0].category, UserWarning)
            assert re.search(warning, str(record[0].message))
        assert list(merged['Rain_Flag']) == list(index.isin(_jan10(rain_at)))
        assert list(merged['Snow_Flag']) == list(index.isin(_jan10(snow_at)))

    @pytest.mark.parametrize("index, period, timezone", [
        (pd.date_range('2023-06-10', periods=6, freq='D'), pd.DateOffset(days=1), None),
        (pd.date_range('2023-06-05', periods=3, freq='W-MON'), pd.DateOffset(weeks=1),
         None),
        # Without freq (as summary.periodic gives): months inferred
        (pd.DatetimeIndex(pd.date_range('2023-04-01', periods=3, freq='MS').tolist()),
         pd.DateOffset(months=1), None),
        # A missing day: each sample still covers its own day only
        (pd.DatetimeIndex(['2023-06-10', '2023-06-11', '2023-06-13', '2023-06-14']),
         pd.DateOffset(days=1), None),
        # Local days, one hour ahead of LST in June
        (pd.date_range('2023-06-10', periods=6, freq='D'), pd.DateOffset(days=1),
         'America/Toronto'),
    ], ids=['daily', 'weekly', 'monthly', 'daily_gap', 'daily_local'])
    def test_averaged(self, monkeypatch, index, period, timezone):
        """Samples coarser than 1 h take the hourly rows of [t, t + period),
        with a UserWarning: mean temperature and wind, wind flag from the
        mean, summed precipitation, distinct conditions in order, and rain
        flags raised if raised in any hour."""
        rain = pd.DatetimeIndex(['2023-05-20 10:00', '2023-06-11 06:00',
                                '2023-06-11 23:00', '2023-06-12 15:00'])  # LST
        conditions = {'2023-05-20 10:00': 'Rain', '2023-06-11 06:00': 'Rain,Fog',
                    '2023-06-11 08:00': 'Fog', '2023-06-11 23:00': 'Rain',
                    '2023-06-12 15:00': 'Rain'}

        with pytest.warns(UserWarning, match="averaged"):
            merged = _merge(monkeypatch, index, _weather_session(rain, conditions),
                            timezone=timezone, wind_speed_flag=30)

        # Hourly rows (LST) and their local times
        lst = pd.date_range('2023-03-01', '2023-08-01', freq='h', inclusive='left')
        local = lst + pd.Timedelta(hours=1 if timezone else 0)
        texts = pd.Series(conditions).set_axis(pd.DatetimeIndex(list(conditions)))
        for t, row in merged.iterrows():
            hours = lst[(local >= t) & (local < t + period)]
            assert row['Temp (°C)'] == hours.hour.to_numpy().mean()
            assert row['Wind Spd (km/h)'] == 2 * hours.hour.to_numpy().mean()
            # The mean (23 km/h) stays below 30 km/h, reached in some hours
            assert not row['Wind_Spd_Flag']
            assert row['Precip. Amount (mm)'] == hours.isin(rain).sum()
            assert row['Rain_Flag'] == hours.isin(rain).any()
            since_rain = hours.to_numpy()[:, None] - rain.to_numpy()[None, :]
            assert row['Rain_Flag_Roll'] == bool(
                ((since_rain >= np.timedelta64(0, 'h'))
                 & (since_rain < np.timedelta64(48, 'h'))).any())
            found = [c for text in texts[texts.index.isin(hours)]
                    for c in text.split(',')]
            if found:
                assert row['Weather'] == ','.join(dict.fromkeys(found))
            else:
                assert pd.isna(row['Weather'])
        assert merged['Rain_Flag_Roll'].any() and not merged['Rain_Flag_Roll'].all()

    @pytest.mark.parametrize("timezone, start, expected", [
        (None, '2023-07-10 12:00', [12, 13, 14]),
        ('America/Toronto', '2023-07-10 12:00', [11, 12, 13]),  # daylight time
        ('America/Toronto', '2023-01-10 12:00', [12, 13, 14]),  # standard time
        # End of daylight time, 5 Nov 2023: 01:00 local twice (first kept)
        ('America/Toronto', '2023-11-05 00:00', [23, 0, 2]),
    ], ids=['lst', 'summer', 'winter', 'end_of_dst'])
    def test_timezone(self, monkeypatch, timezone, start, expected):
        """Weather times are Local Standard Time: with a timezone, the row of
        hour h (LST) is merged at h + 1 local time during daylight saving
        time."""
        merged = _merge(monkeypatch, pd.date_range(start, periods=3, freq='h'),
                        timezone=timezone)

        assert list(merged['Temp (°C)']) == expected
        assert merged.attrs['timezone'] == timezone


def _flagged_levels(**flags):
    """Random levels at 1 min over two days with the weather flags of
    contingency_weather_flags: the given ones (functions of the hour of
    each sample), False otherwise."""
    index = pd.date_range('2023-01-01', periods=2 * 1440, freq='min')
    rng = np.random.default_rng(0)
    df = pd.DataFrame({'LAeq': rng.uniform(40, 70, len(index))}, index=index)
    for flag in FLAGS:
        df[flag] = flags[flag](index.hour) if flag in flags else False
    return df


class TestContingency:
    """Rows of the contingency table."""

    def test_any_flag_no_flag(self):
        """'Any Flag' holds the rows with at least one included flag raised,
        'No Flag' the rows with none: their Leq and Lden are those of
        these rows."""
        # Flags raised in every Lden period, so that each subset has an Lden;
        # Temp_Flag is not included by default
        df = _flagged_levels(Wind_Spd_Flag=lambda h: h % 2 == 0,
                             Rain_Flag_Roll=lambda h: h % 3 == 0,
                             Temp_Flag=lambda h: h % 5 == 0)
        any_flag = df['Wind_Spd_Flag'] | df['Rain_Flag_Roll']

        table = weathercan.contingency_weather_flags(df)

        assert 'All Flags' not in table.index and 'Neither Flags' not in table.index
        for label, rows in (('Any Flag', any_flag), ('No Flag', ~any_flag)):
            subset = df[rows]
            assert table.loc[label, 'Leq'] == leq(subset, 0, 24, stats=False)['Leq'][0]
            assert table.loc[label, 'Lden'] == lden(subset)['Lden'][0]
            assert table.loc[label, 'Covered data (%)'] == round(100 * rows.mean(), 1)

    def test_row_without_lden(self):
        """A flag raised only at night: its row has no Lden (no day or
        evening data) but keeps its Leq; rows of flags not included are
        left out."""
        df = _flagged_levels(Rain_Flag_Roll=lambda h: (h >= 23) | (h < 7))
        night = df['Rain_Flag_Roll']

        table = weathercan.contingency_weather_flags(df, include_wind_flag=False)

        rain = 'Rain in the last 48h'
        assert list(table.index) == ['All Data', rain, 'No Flag - Rain', 'Any Flag',
                                    'No Flag']
        assert table.loc[rain, 'Leq'] == leq(df[night], 0, 24, stats=False)['Leq'][0]
        assert np.isnan(table.loc[rain, 'Lden'])
        assert table.loc['No Flag - Rain', 'Leq'] == leq(df[~night], 0, 24,
                                                        stats=False)['Leq'][0]


@pytest.fixture(scope="module")
def live_merged():
    """Random levels at 20 min over a week of July 2023 (daylight saving
    time), merged with the live weather of LIVE_STATION, and the warnings
    raised by the merge."""
    index = pd.date_range('2023-07-03', '2023-07-10', freq='20min', inclusive='left')
    rng = np.random.default_rng(0)
    df = pd.DataFrame({'LAeq': rng.uniform(40, 70, len(index))}, index=index)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        merged = asyncio.run(weathercan.merge_weather(df, LIVE_STATION))
    return df, merged, caught


@pytest.mark.live
@pytest.mark.slow
class TestLive:
    """Environment Canada service: format of its pages and files, and the
    functions on its live data (ranges and consistency, not exact values)."""

    def test_stations(self):
        """Stations within 25 km, nearest first, LIVE_STATION among them
        with hourly data over the dates of the other live tests."""
        df = weathercan.get_historical_stations(COORDINATES, radius=25)

        assert list(df.index) == ['prov', 'proximity', 'id', 'hlyRange', 'dlyRange',
                                'mlyRange']
        assert 0 < df.shape[1] <= 25
        proximity = df.loc['proximity'].astype(float)
        assert proximity.is_monotonic_increasing and proximity.between(0, 25).all()
        ranges = df.loc[['hlyRange', 'dlyRange', 'mlyRange']].stack()
        assert ranges.str.fullmatch(r'(\d{4}-\d\d-\d\d)?\|(\d{4}-\d\d-\d\d)?').all()
        [hourly] = df.loc['hlyRange', df.loc['id'] == LIVE_STATION]
        first, last = hourly.split('|')
        assert first <= '2022-12-01' and last >= '2023-07-31'

    @pytest.mark.parametrize("timeframe, start, stop, freq, index_name, ranges", [
        ('hourly', '2023-07-01 00:00', '2023-07-14 23:00', 'h', 'Date/Time (LST)',
        {'Temp (°C)': (-50, 50), 'Rel Hum (%)': (0, 100), 'Wind Spd (km/h)': (0, 200),
        'Precip. Amount (mm)': (0, 200)}),
        # A daily file holds a whole year: the two months of each year
        # requested return the same rows (duplicates dropped)
        ('daily', '2022-11-15', '2023-02-15', 'D', 'Date/Time',
        {'Mean Temp (°C)': (-50, 50), 'Total Precip (mm)': (0, 500)}),
    ], ids=['hourly', 'daily'])
    def test_historical_data(self, timeframe, start, stop, freq, index_name, ranges):
        """One row per hour or day of the range, both ends included, indexed
        by the time column of the files (Local Standard Time for hourly
        data); values mostly present and within physical ranges."""
        start, stop = pd.Timestamp(start), pd.Timestamp(stop)

        df = asyncio.run(weathercan.get_historical_data(
            LIVE_STATION, daterange=(start, stop), timeframe=timeframe))

        assert list(df.index) == list(pd.date_range(start, stop, freq=freq))
        assert df.index.name == index_name
        assert (df['Climate ID'].astype(str) == LIVE_STATION).all()
        for column, (low, high) in ranges.items():
            assert df[column].notna().mean() >= 0.9
            assert df[column].dropna().between(low, high).all()

    def test_merge_weather(self, live_merged):
        """Each sample takes the hourly row nearest to it in local time (LST
        + 1 h in July). The station reports no condition: rain and snow come
        from the temperature, with a UserWarning; it rained this week."""
        df, merged, caught = live_merged
        lst = asyncio.run(weathercan.get_historical_data(LIVE_STATION, daterange=(
            df.index[0] - pd.Timedelta(hours=2), df.index[-1] + pd.Timedelta(hours=2))))
        local = (lst.set_axis(lst.index + pd.Timedelta(hours=1))
                .reindex(df.index, method='nearest'))

        assert [w.category for w in caught] == [UserWarning]
        assert "inferred from the temperature" in str(caught[0].message)
        pd.testing.assert_frame_equal(merged[df.columns], df)
        for column in ['Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)',
                    'Precip. Amount (mm)']:
            pd.testing.assert_series_equal(merged[column], local[column],
                                        check_names=False)
        flags = ['Wind_Spd_Flag', 'Rain_Flag', 'Snow_Flag', 'Temp_Flag', 'Rel_Hum_Flag',
                'Rain_Flag_Roll', 'Snow_Flag_Roll']
        assert (merged[flags].dtypes == bool).all()
        rain = (merged['Precip. Amount (mm)'] > 0) & (merged['Temp (°C)'] >= 0)
        assert rain.any() and merged['Rain_Flag'].equals(rain)
        assert (merged['Rain_Flag_Roll'] >= merged['Rain_Flag']).all()
        assert not merged[['Snow_Flag', 'Snow_Flag_Roll']].any().any()

    def test_merge_weather_conditions_only(self):
        """A station reporting conditions but no precipitation amount: the
        hours reporting rain (or drizzle) are flagged from the condition
        alone, with a UserWarning for each; it rained these two weeks,
        and drizzled without rain in some hours."""
        index = pd.date_range('2023-06-25', '2023-07-09', freq='h', inclusive='left')
        df = pd.DataFrame({'LAeq': 50.0}, index=index)

        with pytest.warns(UserWarning) as record:
            merged = asyncio.run(weathercan.merge_weather(df, LIVE_STATION_CONDITIONS))

        messages = [str(w.message) for w in record]
        assert len(messages) == 2
        assert messages[0].startswith("No precipitation amount reported")
        assert DRIZZLE in messages[1]
        assert merged['Precip. Amount (mm)'].isna().all()
        rain = merged['Weather'].fillna('').str.contains('Rain|Drizzle|Ice Pellets')
        assert rain.any() and merged['Rain_Flag'].equals(rain)

    def test_contingency(self, live_merged):
        """One row per default flag raised and not raised, any and none,
        each holding its share of the samples, with an Leq."""
        _, merged, _ = live_merged
        wind, rain = merged['Wind_Spd_Flag'], merged['Rain_Flag_Roll']
        rows = {'All Data': wind | ~wind, 'Wind speed >= 18 km/h': wind,
                'Rain in the last 48h': rain, 'No Flag - Wind': ~wind,
                'No Flag - Rain': ~rain, 'Any Flag': wind | rain, 'No Flag': ~(wind | rain)}

        table = weathercan.contingency_weather_flags(merged)

        assert list(table.index) == list(rows)
        assert table['Leq'].notna().all()
        for label, mask in rows.items():
            assert table.loc[label, 'Covered data (%)'] == round(100 * mask.mean(), 1)

    def test_filter_weather_flags(self, live_merged):
        """filter.weather_flags removes the levels of the samples with wind
        or rain flags (the default) and keeps the others."""
        _, merged, _ = live_merged
        flagged = merged['Wind_Spd_Flag'] | merged['Rain_Flag_Roll']

        filtered = weather_flags(merged)

        assert filtered['LAeq'].isna().equals(flagged)
        pd.testing.assert_series_equal(filtered['LAeq'][~flagged],
                                    merged['LAeq'][~flagged])

    @pytest.mark.parametrize("plot", ['line_weather', 'compare_weather_daily'])
    def test_plots(self, live_merged, plot):
        """The weather plots draw the merged data."""
        _, merged, _ = live_merged

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # compare_weather_daily: profiles of 20 min data, and of flag
            # subsets with windows holding no sample
            warnings.simplefilter("ignore", IntegrationTimeWarning)
            warnings.simplefilter("ignore", CoverageWarning)
            ax = getattr(display, plot)(merged, show=False)

        assert isinstance(ax, matplotlib.axes.Axes) and ax.has_data()
        plt.close('all')
