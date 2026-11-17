"""Unit tests for weather.weathercan module (no network access)."""

import asyncio
from datetime import date

import numpy as np
import pandas as pd
import pytest

# Needs the optional dependencies (pip install noisemonitor[weather])
weathercan = pytest.importorskip("noisemonitor.weather.weathercan")

# Request handling only: no acoustic reference values
pytestmark = pytest.mark.behaviour


class _FakeResponse:
    """Hourly rows of the requested month, as a bulk data CSV."""

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
        return pd.DataFrame({'Date/Time (LST)': hours.strftime('%Y-%m-%d %H:%M'),
                            'Temp (°C)': 0.0}).to_csv(index=False)


class _FakeSession:
    """Stands in for aiohttp.ClientSession."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def get(self, url, params=None, headers=None):
        return _FakeResponse(params)


def _weather_session(rain=(), conditions=None, temp_offset=0.0):
    """aiohttp.ClientSession stand-in serving hourly weather rows of each
    requested month: 1 mm of precipitation at the given times (LST), none
    otherwise, the hour h of the row (LST) plus temp_offset as its
    temperature, 2 h km/h of wind, and the given 'Weather' conditions
    ({time (LST): text}, NaN otherwise)."""
    rain = pd.DatetimeIndex(rain)
    conditions = {pd.Timestamp(t): c for t, c in (conditions or {}).items()}

    class _Response(_FakeResponse):
        async def text(self):
            start = pd.Timestamp(self.params['Year'], self.params['Month'], 1)
            hours = pd.date_range(start, start + pd.offsets.MonthBegin(1), freq='h',
                                inclusive='left')
            return pd.DataFrame({
                'Date/Time (LST)': hours.strftime('%Y-%m-%d %H:%M'),
                'Temp (°C)': hours.hour + temp_offset,
                'Rel Hum (%)': 50.0,
                'Wind Spd (km/h)': 2.0 * hours.hour,
                'Precip. Amount (mm)': np.where(hours.isin(rain), 1.0, 0.0),
                'Weather': [conditions.get(h, np.nan) for h in hours],
            }).to_csv(index=False)

    class _Session(_FakeSession):
        def get(self, url, params=None, headers=None):
            return _Response(params)

    return _Session


class TestDaterange:
    """A daterange may be given in either order."""

    def test_monthlist_reversed(self):
        """The months covered by the range, whatever its order."""
        ordered = (date(2023, 1, 15), date(2023, 3, 2))

        assert (weathercan.monthlist(ordered[::-1]) == weathercan.monthlist(ordered)
                == [(2023, 1, 1), (2023, 2, 1), (2023, 3, 1)])

    def test_get_historical_data_reversed(self, monkeypatch):
        """Hourly rows from the earlier to the later date, whatever the
        order of the range."""
        monkeypatch.setattr(weathercan.aiohttp, 'ClientSession', _FakeSession)
        start, stop = pd.Timestamp('2023-01-15 06:00'), pd.Timestamp('2023-03-02 12:00')

        ordered = asyncio.run(weathercan.get_historical_data(1, daterange=(start, stop)))
        reversed_ = asyncio.run(weathercan.get_historical_data(1, daterange=(stop, start)))

        pd.testing.assert_frame_equal(reversed_, ordered)
        assert list(ordered.index) == list(pd.date_range(start, stop, freq='h'))


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
        monkeypatch.setattr(weathercan.aiohttp, 'ClientSession',
                            _weather_session([rain], {rain: 'Rain'}))
        index = pd.date_range(start, periods=periods, freq=freq)
        df = pd.DataFrame({'LAeq': 50.0}, index=index)

        merged = asyncio.run(weathercan.merge_weather(df, 1, timezone=None))

        # Nearest hourly row (half hours go to the next one)
        since_rain = index.round('h') - pd.Timestamp(rain)
        expected = (since_rain >= pd.Timedelta(0)) & (since_rain < pd.Timedelta(hours=48))
        assert expected.any()
        assert list(merged['Rain_Flag_Roll']) == list(expected)

    def test_precipitation_without_condition(self, monkeypatch):
        """Precipitation is flagged as rain or snow only when the condition is
        reported: at 03:00 (3 °C) without any condition, neither flag, with a
        UserWarning giving one hour; at 05:00 with 'Snow', the snow flag."""
        monkeypatch.setattr(weathercan.aiohttp, 'ClientSession', _weather_session(
            ['2023-01-10 03:00', '2023-01-10 05:00'], {'2023-01-10 05:00': 'Snow'}))
        index = pd.date_range('2023-01-10', periods=24, freq='h')
        df = pd.DataFrame({'LAeq': 50.0}, index=index)

        with pytest.warns(UserWarning, match=r"for 1 hour\(s\) with precipitation"):
            merged = asyncio.run(weathercan.merge_weather(df, 1, timezone=None))

        assert not merged['Rain_Flag'].any() and not merged['Rain_Flag_Roll'].any()
        assert list(merged['Snow_Flag']) == list(index == '2023-01-10 05:00')
        assert list(merged['Snow_Flag_Roll']) == list(index >= '2023-01-10 05:00')

    def test_no_condition_reported(self, monkeypatch):
        """A station reporting no condition: precipitation at 03:00 (-9 °C) is
        snow, at 15:00 (3 °C) rain, with a UserWarning to check them."""
        monkeypatch.setattr(weathercan.aiohttp, 'ClientSession', _weather_session(
            ['2023-01-10 03:00', '2023-01-10 15:00'], temp_offset=-12))
        index = pd.date_range('2023-01-10', periods=24, freq='h')
        df = pd.DataFrame({'LAeq': 50.0}, index=index)

        with pytest.warns(UserWarning, match="inferred from the temperature"):
            merged = asyncio.run(weathercan.merge_weather(df, 1, timezone=None))

        assert list(merged['Snow_Flag']) == list(index == '2023-01-10 03:00')
        assert list(merged['Rain_Flag']) == list(index == '2023-01-10 15:00')

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
        monkeypatch.setattr(weathercan.aiohttp, 'ClientSession',
                            _weather_session(rain, conditions))
        df = pd.DataFrame({'LAeq': 50.0}, index=index)

        with pytest.warns(UserWarning, match="averaged"):
            merged = asyncio.run(weathercan.merge_weather(
                df, 1, wind_speed_flag=30, timezone=timezone))

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
        monkeypatch.setattr(weathercan.aiohttp, 'ClientSession', _weather_session())
        index = pd.date_range(start, periods=3, freq='h')
        df = pd.DataFrame({'LAeq': 50.0}, index=index)

        merged = asyncio.run(weathercan.merge_weather(df, 1, timezone=timezone))

        assert list(merged['Temp (°C)']) == expected
        assert merged.attrs['timezone'] == timezone


class TestContingency:
    """Rows of the contingency table."""

    def test_any_flag_no_flag(self):
        """'Any Flag' holds the rows with at least one included flag raised,
        'No Flag' the rows with none: their Leq and Lden are those of
        these rows."""
        from noisemonitor.summary import leq, lden
        index = pd.date_range('2023-01-01', periods=2 * 1440, freq='min')
        rng = np.random.default_rng(0)
        df = pd.DataFrame({'LAeq': rng.uniform(40, 70, len(index))}, index=index)
        # Flags raised in every Lden period, so that each subset has an Lden
        hour = index.hour
        df['Wind_Spd_Flag'] = hour % 2 == 0
        df['Rain_Flag_Roll'] = hour % 3 == 0
        df['Temp_Flag'] = hour % 5 == 0  # not included by default
        df['Rel_Hum_Flag'] = False
        df['Snow_Flag_Roll'] = False
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
        from noisemonitor.summary import leq
        index = pd.date_range('2023-01-01', periods=2 * 1440, freq='min')
        rng = np.random.default_rng(0)
        df = pd.DataFrame({'LAeq': rng.uniform(40, 70, len(index))}, index=index)
        night = (index.hour >= 23) | (index.hour < 7)
        df['Wind_Spd_Flag'] = False
        df['Rain_Flag_Roll'] = night
        df['Temp_Flag'] = False
        df['Rel_Hum_Flag'] = False
        df['Snow_Flag_Roll'] = False

        table = weathercan.contingency_weather_flags(df, include_wind_flag=False)

        rain = 'Rain in the last 48h'
        assert list(table.index) == ['All Data', rain, 'No Flag - Rain', 'Any Flag',
                                    'No Flag']
        assert table.loc[rain, 'Leq'] == leq(df[night], 0, 24, stats=False)['Leq'][0]
        assert np.isnan(table.loc[rain, 'Lden'])
        assert table.loc['No Flag - Rain', 'Leq'] == leq(df[~night], 0, 24,
                                                        stats=False)['Leq'][0]
