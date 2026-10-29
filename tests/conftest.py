"""Shared by the test modules: fixtures of the internal test data
(data/internal), and fixtures, constants and helpers of the Accra external
data (data/external). Constants and helpers are imported with
`from tests.conftest import ...`.

Module scope: each test module gets its own copy of the data, so a function
that modifies its input cannot affect other modules."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import noisemonitor as nm


# Internal test data

@pytest.fixture(scope="module")
def test_data_paths():
    """Provide paths to the test data files."""
    base_path = Path(__file__).parent / "data" / "internal"
    return {
        "laeq1m": base_path / "test_data_laeq1m.csv",
        "laeq1s": base_path / "test_data_laeq1s.csv"
    }


@pytest.fixture(scope="module")
def laeq1m_data(test_data_paths):
    """Load LAeq,1min test data (one week of data)."""
    df = nm.load(
        str(test_data_paths["laeq1m"]),
        datetimeindex=0,
        valueindexes=1,
        header=0,
        sep=','
    )
    return df


@pytest.fixture(scope="module")
def laeq1s_data(test_data_paths):
    """Load LAeq,1s test data (one day of data)."""
    df = nm.load(
        str(test_data_paths["laeq1s"]),
        datetimeindex=0,
        valueindexes=1,
        header=0,
        sep=','
    )
    return df


@pytest.fixture(scope="module")
def data_with_gaps(laeq1m_data):
    """Create data with gaps: 3 days with 60% of day 2 set to NaN."""
    # Take 3 days of data
    test_data = laeq1m_data[:3 * 1440].copy()

    # Find day 2 indices (1440 to 2880)
    day2_start = 1440
    day2_end = 2880
    day2_indices = test_data.index[day2_start:day2_end]

    # Randomly set 60% of day 2 data to NaN
    np.random.seed(42)
    nan_indices = np.random.choice(
        day2_indices,
        size=int(len(day2_indices) * 0.6),
        replace=False
    )
    test_data.loc[nan_indices] = np.nan
    return test_data


@pytest.fixture(scope="module")
def data_with_time_gaps(laeq1s_data):
    """Create data with time-specific gaps: 70% of evening period set to NaN."""
    # Take 1 day of data
    test_data = laeq1s_data[:86400].copy()

    # Find evening period (19:00-23:00)
    evening_mask = (test_data.index.hour >= 19) & (test_data.index.hour < 23)
    evening_indices = test_data.index[evening_mask]

    # Randomly set 70% of evening data to NaN
    np.random.seed(42)
    nan_indices = np.random.choice(
        evening_indices,
        size=int(len(evening_indices) * 0.7),
        replace=False
    )
    test_data.loc[nan_indices] = np.nan
    return test_data


# External data: Clark et al. (2021), Accra (data/external/README.md)

EXT_PATH = Path(__file__).parent / "data" / "external"
ACCRA_PERIODS = dict(start_day=6, start_evening=19, start_night=22)
ACCRA_COLUMNS = {'EL': 'EL', 'JT': 1}  # `column` given as str and as int
# The source's coverage rule: a value is empty only when its period has no data
SOURCE_COVERAGE = dict(coverage_check=True, coverage_threshold=1e-9)
TOL = 1e-4  # dB
TOL_ROUNDED = 0.005 + TOL  # summary.lden and summary.leq round to 0.01 dB


def assert_close(got, expected, tol=TOL):
    """Same NaNs, and values within tol."""
    np.testing.assert_allclose(got, expected, rtol=0, atol=tol)


def lday_06_22(lday, levening, days, sites):
    """LDay (06-22) of the source from Lday (06-19) and Levening (19-22),
    each weighted by its number of hours with data."""
    n_day = np.array([day[site].between_time('06:00', '19:00', inclusive='left')
                      .count() for day, site in zip(days, sites)])
    n_evening = np.array([day[site].between_time('19:00', '22:00', inclusive='left')
                          .count() for day, site in zip(days, sites)])
    energy = np.nansum([n_day * 10 ** (np.asarray(lday) / 10),
                        n_evening * 10 ** (np.asarray(levening) / 10)], axis=0)
    with np.errstate(invalid='ignore'):  # no data in 06-22: NaN
        return 10 * np.log10(energy / (n_day + n_evening))


@pytest.fixture(scope="module")
def accra_hourly():
    """Hourly LAeq at two Accra sites (EL, JT), one column per site."""
    return nm.load(str(EXT_PATH / "ext_data_accra_h.csv"),
                   datetimeindex=0, valueindexes=[1, 2])


@pytest.fixture(scope="module")
def accra_daily():
    """Published daily values: one row per site and reproducible day."""
    return pd.read_csv(EXT_PATH / "ext_data_accra_d.csv", parse_dates=['date'])


@pytest.fixture(scope="module")
def accra_days(accra_hourly, accra_daily):
    """Hourly LAeq (both sites) of the calendar day of each published row."""
    return [accra_hourly.loc[d:d + pd.Timedelta(hours=23)] for d in accra_daily['date']]


@pytest.fixture(scope="module")
def accra_spans(accra_hourly, accra_daily):
    """(hourly data, day1 = day2) for each published row: its calendar day
    ('day', no day filter), and the 7 days starting on it, filtered back to
    it by its weekday ('week')."""
    def span(n_days, weekday):
        end = pd.Timedelta(days=n_days) - pd.Timedelta(hours=1)
        return [(accra_hourly.loc[d:d + end], weekday(d)) for d in accra_daily['date']]
    return {'day': span(1, lambda d: None), 'week': span(7, lambda d: d.day_name())}
