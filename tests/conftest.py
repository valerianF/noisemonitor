"""Shared by the test modules: fixtures of the internal test data
(data/internal), and fixtures, constants and helpers of the external data
(data/external). Constants and helpers are imported with
`from tests.conftest import ...`.

Module scope: each test module gets its own copy of the data, so a function
that modifies its input cannot affect other modules."""

import lzma
import re
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
        "laeq1s": base_path / "test_data_laeq1s.csv.xz",
        "third_octave": base_path / "test_data_third_octave.csv.xz"
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


# External data: Tartu (data/external/README.md)

TARTU_RECEIVERS = (10428, 13914, 5080)
# Day of the traffic simulated by the source, used as the date of the series
TARTU_DATE = '2021-10-16'
# Indoor level (Baclet and Rumpler, 2024, Eq. 3): energetic sum of the
# outdoor level minus the facade attenuation (window open or closed) and of
# 35 dB of indoor noise. Seconds with no vehicle (empty) are 0 dB outdoors.
TARTU_ATTENUATION = {'OP': 5.0, 'CL': 25.0}
TARTU_INDOOR = 35.0
TARTU_SILENCE = 0.0
# Source indicator names: <threshold>E<exceedance>G<gap><window>
TARTU_INDICATOR = re.compile(r'(T\d\d|L50|L90|LEQ)E(\d\d)G(\d\d)(OP|CL)')


def tartu_parameters(indicator):
    """nne arguments (background_type, exceedance, min_gap) and window of a
    source indicator, e.g., 'L90E10G30CL' -> ('L90', 10, 30), 'CL'; a fixed
    threshold 'T45' gives background_type=45."""
    threshold, exceedance, gap, window = TARTU_INDICATOR.fullmatch(indicator).groups()
    background = {'LEQ': 'Leq', 'L50': 'L50', 'L90': 'L90'}.get(threshold)
    return dict(background_type=background or int(threshold[1:]),
                exceedance=int(exceedance), min_gap=int(gap)), window


@pytest.fixture(scope="module")
def tartu_indoor():
    """Indoor levels at 1 s over one day, for each (receiver, window), from
    the simulated outdoor LAeq,1s."""
    df = pd.read_csv(EXT_PATH / "ext_data_tartu_s.csv.xz")
    index = pd.DatetimeIndex(pd.Timestamp(TARTU_DATE)
                            + pd.to_timedelta(df['second_of_day'], unit='s'))
    indoor = {}
    for pk in TARTU_RECEIVERS:
        outdoor = df[f'R{pk}'].fillna(TARTU_SILENCE).to_numpy()
        for window, attenuation in TARTU_ATTENUATION.items():
            level = 10 * np.log10(10 ** ((outdoor - attenuation) / 10)
                                + 10 ** (TARTU_INDOOR / 10))
            indoor[pk, window] = pd.DataFrame({'LAeq': level}, index=index)
    return indoor


@pytest.fixture(scope="module")
def tartu_counts():
    """Hourly numbers of noise events computed by the source: one row per
    (receiver, hour), one column per indicator."""
    return pd.read_csv(EXT_PATH / "ext_data_tartu_h.csv", index_col=['pk', 'hour'])


# External data: published tables for the band labels and the low-frequency
# criteria (data/external/README.md, "Published tables")

# ISO 266:1997, Table 1: preferred frequencies and calculated frequencies
# (exact base-10 frequencies to five significant figures), in Hz or kHz
ISO_266_PREFERRED = (
    1.00, 1.25, 1.60, 2.00, 2.50, 3.15, 4.00, 5.00, 6.30, 8.00, 10.0, 12.5,
    16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100, 125, 160, 200, 250,
    315, 400, 500, 630, 800)
ISO_266_CALCULATED = (
    1.0000, 1.2589, 1.5849, 1.9953, 2.5119, 3.1623, 3.9811, 5.0119, 6.3096,
    7.9433, 10.000, 12.589, 15.849, 19.953, 25.119, 31.623, 39.811, 50.119,
    63.096, 79.433, 100.00, 125.89, 158.49, 199.53, 251.19, 316.23, 398.11,
    501.19, 630.96, 794.33)

# Montano & Gushiken (2023), Table 2: Z-weighted levels (dB) of the bands from
# 16 Hz to 200 Hz. Their A- and C-weighted energetic sums are the levels the
# paper gives for Fig. 3-c, p. 8 (CLF 68.1 dB, ALF 51.4 dB, CLF - ALF
# 16.7 dB); the 74.1 and 56.0 dB printed under the table are those of
# Fig. 3-a.
MONTANO_TABLE2 = {
    'bands': (16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0,
            160.0, 200.0),
    'Z': (48.0, 50.8, 53.0, 55.5, 55.7, 57.3, 58.8, 59.7, 60.1, 62.7, 59.9,
        55.3),
}
MONTANO_FIG3C = {'LA': 51.4, 'LC': 68.1, 'C-A': 16.7}

# Generalitat de Catalunya (2012), guide "10. Motor que genera soroll a baixa
# freqüència" (A6), Tables 3 and 5: Z-weighted levels (dB) of the bands from
# 20 Hz to 160 Hz from two sets of measurements, the guide's LAeq,20-160Hz,
# LCeq,20-160Hz and their difference Lf.
GENCAT_A6 = {
    'T3': {'Z': (56.3, 58.8, 60.7, 61.5, 63.7, 69.1, 65.4, 59.2, 59.5, 55.6),
        'LA': 49.6, 'LC': 71.9, 'C-A': 22.3},
    'T5': {'Z': (57.0, 60.3, 64.8, 61.8, 64.0, 70.6, 66.6, 61.0, 59.3, 55.7),
        'LA': 50.4, 'LC': 73.2, 'C-A': 22.8},
}
GENCAT_A6_BANDS = (20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0,
                160.0)

TOL_PUBLISHED = 0.05 + TOL  # values published to 0.1 dB

# One-third-octave bands used by the low-frequency criteria (16 Hz to 20 kHz)
LF_BANDS = (16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0,
            160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0,
            1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0,
            8000.0, 10000.0, 12500.0, 16000.0, 20000.0)


# Band range (Hz, both included) of each low-frequency method
LF_METHOD_BANDS = {'botsford': (20.0, 20000.0), 'montano': (16.0, 200.0),
                'barcelona': (20.0, 160.0)}


def weighted_band_sum(df, method, weighting):
    """Energetic sum, for each row of df (columns named f'{band:g}'), of the
    band levels plus their weighting (dict band: dB) over the bands of the
    method, as a one-column DataFrame."""
    f_min, f_max = LF_METHOD_BANDS[method]
    energy = sum(np.power(10, (df[f'{band:g}'].to_numpy() + weighting[band]) / 10)
                for band in LF_BANDS if f_min <= band <= f_max)
    return pd.DataFrame({'L': 10 * np.log10(energy)}, index=df.index)


def band_frame(levels, bands, index):
    """DataFrame with one column per band, named by its frequency (e.g.
    '31.5'), holding the given level of the band (a constant or an array) at
    every timestamp of index."""
    return pd.DataFrame(
        {f'{band:g}': np.broadcast_to(level, len(index)).astype(float)
        for band, level in zip(bands, levels)}, index=index)


@pytest.fixture(scope="module")
def lf_bands():
    """Three days (from a Monday) at 1 min of one-third-octave band levels
    from 16 Hz to 20 kHz: a spectrum falling by 1 dB per band, an offset that
    changes every hour (the same for all bands) and random fluctuations of
    each band and sample."""
    index = pd.date_range('2023-01-02', periods=3 * 1440, freq='min')
    rng = np.random.default_rng(42)
    hourly = np.repeat(rng.normal(0, 5, 3 * 24), 60)
    levels = [70.0 - k + hourly + rng.normal(0, 3, len(index))
            for k in range(len(LF_BANDS))]
    return band_frame(levels, LF_BANDS, index)


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
