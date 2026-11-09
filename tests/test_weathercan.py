"""Unit tests for weather.weathercan module (no network access)."""

import asyncio
from datetime import date

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
