"""Shared by the test modules: fixtures of the internal test data
(data/internal), and fixtures, constants and helpers of the external data
(data/external). Constants and helpers are imported with
`from tests.conftest import ...`.

Module scope: each test module gets its own copy of the data, so a function
that modifies its input cannot affect other modules."""

import lzma
import shutil
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


# External data: Bruitparif (data/external/README.md)

BRUITPARIF_TZ = 'Europe/Paris'
# Rows of the LAeq,1s file for 23 and 24 October and the 10 min before them;
# the file goes on with 25 October 00:00-05:00, which spans the end of
# daylight saving time (used by test_load only)
BRUITPARIF_ROWS = 600 + 2 * 86400
# noisemonitor output column: Bruitparif hourly column
BRUITPARIF_LEVELS = {'Leq': 'LAeq_1h', 'L10': 'L10_1h', 'L50': 'L50_1h',
                    'L90': 'L90_1h'}
HARMONICA_INDICATORS = ['HARMONICA', 'BGN', 'EVT']
# Hours whose span (the hour and the 10 min before it) has a missing second:
# Bruitparif computed HARMONICA with values for seconds that are missing from
# its LAeq,1s export. Other hours: TOL.
TOL_HARMONICA = 0.005


@pytest.fixture(scope="session")
def bruitparif_1s_path(tmp_path_factory):
    """The LAeq,1s file, stored xz-compressed, unpacked once per session."""
    path = tmp_path_factory.mktemp("bruitparif") / "ext_data_bruitparif_s.csv"
    with lzma.open(EXT_PATH / "ext_data_bruitparif_s.csv.xz") as src:
        with open(path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return path


@pytest.fixture(scope="module")
def bruitparif_1s(bruitparif_1s_path):
    """LAeq,1s over two days and the 10 min before them (the first
    BRUITPARIF_ROWS rows of the file); timestamps with a UTC offset in the
    file, converted to naive local time by load()."""
    return nm.load(str(bruitparif_1s_path), datetimeindex=0, valueindexes=1,
                timezone=BRUITPARIF_TZ, nrows=BRUITPARIF_ROWS)


@pytest.fixture(scope="module")
def bruitparif_hourly():
    """Hourly indicators computed by Bruitparif for 23 and 24 October,
    indexed by the start of their hour in naive local time."""
    df = pd.read_csv(EXT_PATH / "ext_data_bruitparif_h.csv")
    df = df[df['ts'] < '2026-10-25'].reset_index(drop=True)
    df.index = (pd.DatetimeIndex(pd.to_datetime(df.pop('ts')))
                .tz_convert(BRUITPARIF_TZ).tz_localize(None))
    return df


# Two 0.5 s levels whose energetic mean is the LAeq,1s L: L + 1 dB and
# L + 10 lg(2 - 10^0.1) dB (about L - 1.30 dB)
HALF_SECOND_OFFSETS = (1.0, 10 * np.log10(2 - 10 ** 0.1))
# Last second of the span used at 0.5 s: the 10 min before 23 October and
# its first three hours
HALF_SECOND_END = '2026-10-23 02:59:59'


@pytest.fixture(scope="module")
def bruitparif_halfsecond(bruitparif_1s, tmp_path_factory):
    """Sub-second data with known LAeq,1s: each LAeq,1s of Bruitparif over
    HALF_SECOND_END and the 10 min before it, split into two 0.5 s levels
    (HALF_SECOND_OFFSETS) whose energetic mean is the LAeq,1s (missing
    seconds give two missing levels). Written with millisecond timestamps
    and read by load()."""
    levels = bruitparif_1s.iloc[:, 0].loc[:HALF_SECOND_END]
    halves = pd.concat([
        pd.Series(levels.to_numpy() + offset,
                index=levels.index + pd.Timedelta(milliseconds=500 * k))
        for k, offset in enumerate(HALF_SECOND_OFFSETS)]).sort_index()
    path = tmp_path_factory.mktemp("halfsecond") / "halfsecond.csv"
    halves.rename('LAeq').to_csv(path, index_label='datetime',
                                date_format='%Y-%m-%d %H:%M:%S.%f')
    return nm.load(str(path), datetimeindex=0, valueindexes=1,
                use_chunks=False)


# Band column names in mixed formats and in no particular order, and the
# frequency (Hz) that core._parse_freq_columns gives them
BANDS = {'1 kHz': 1000.0, '125 Hz': 125.0, '63': 63.0, '500Hz': 500.0}


@pytest.fixture(scope="module")
def bruitparif_bands(bruitparif_1s):
    """Four bands built from the LAeq,1s, one per BANDS column: the series
    rolled by a different number of hours and offset by a different level,
    so that each band gives different values in every window."""
    levels = bruitparif_1s.iloc[:, 0].to_numpy()
    return pd.DataFrame({name: np.roll(levels, 3600 * (5 * k + 1)) - 2.0 * k
                        for k, name in enumerate(BANDS)},
                        index=bruitparif_1s.index)


@pytest.fixture(scope="module")
def bruitparif_span_complete(bruitparif_1s, bruitparif_hourly):
    """For each hour of bruitparif_hourly, whether its HARMONICA span (the
    hour and the 10 min before it, the window of its first LA95) has no
    missing second."""
    levels = bruitparif_1s.iloc[:, 0]
    end = pd.Timedelta(hours=1) - pd.Timedelta(seconds=1)
    return pd.Series(
        [levels.loc[h - pd.Timedelta(minutes=10):h + end].notna().sum() == 4200
        for h in bruitparif_hourly.index],
        index=bruitparif_hourly.index)
