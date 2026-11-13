"""Unit tests for core module functions."""

import pytest
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, time, timedelta
from unittest.mock import patch

import noisemonitor as nm
from noisemonitor.util.core import (
    equivalent_level,
    get_interval,
    harmonica,
    hourly_harmonica,
    lden,
    noise_events,
    _column_to_index,
    _laeq_1s,
    check_coverage,
    CoverageWarning,
    _A_WEIGHTING,
    _C_WEIGHTING,
    _lf_apply,
    _lf_levels,
    _parse_freq_columns
)
from tests.conftest import (
    ACCRA_COLUMNS, ACCRA_PERIODS, BANDS, HARMONICA_INDICATORS,
    ISO_266_CALCULATED, ISO_266_PREFERRED, LF_BANDS, SOURCE_COVERAGE,
    TOL_HARMONICA, assert_close, band_frame, lday_06_22
)



@pytest.mark.external
@pytest.mark.filterwarnings("ignore::noisemonitor.util.core.CoverageWarning")
class TestExternalAccra:
    """Published daily values of two Accra sites (Clark et al., 2021),
    reproducible days only."""

    @pytest.mark.parametrize("indicator, hours", [
        ('LAeq24hr', range(24)),
        ('LDay', range(6, 22)),
        ('LNight', [22, 23, 0, 1, 2, 3, 4, 5]),
    ], ids=['LAeq24hr', 'LDay', 'LNight'])
    def test_equivalent_level(self, accra_daily, accra_days, indicator, hours):
        """Leq of the hourly values of a period, each calendar day."""
        got = [equivalent_level(day.loc[day.index.hour.isin(hours), site].to_numpy())
            for day, site in zip(accra_days, accra_daily['ID'])]
        assert_close(got, accra_daily[indicator])

    @pytest.mark.slow
    @pytest.mark.parametrize("coverage", [{}, SOURCE_COVERAGE],
                            ids=['no_check', 'coverage_check'])
    def test_lden(self, accra_daily, accra_days, coverage):
        """Lden, Lnight, and Lday + Levening (as the source's LDay), each
        calendar day with the Accra periods."""
        sites = accra_daily['ID']
        result = pd.concat([
            lden(day, ACCRA_COLUMNS[site], values=True, **ACCRA_PERIODS, **coverage)
            for day, site in zip(accra_days, sites)], ignore_index=True)

        assert_close(result['Lden'], accra_daily['LDEN'])
        assert_close(result['Lnight'], accra_daily['LNight'])
        assert_close(lday_06_22(result['Lday'], result['Levening'], accra_days, sites),
                    accra_daily['LDay'])


@pytest.mark.external
class TestExternalBruitparif:
    """Hourly indicators computed by Bruitparif from the LAeq,1s of one
    station, over two days."""

    def test_equivalent_level(self, bruitparif_1s, bruitparif_hourly):
        """Leq of the LAeq,1s of each hour: LAeq_1h."""
        hour = pd.Timedelta(hours=1) - pd.Timedelta(seconds=1)
        got = [equivalent_level(bruitparif_1s.loc[h:h + hour].iloc[:, 0].to_numpy())
            for h in bruitparif_hourly.index]
        assert_close(got, bruitparif_hourly['LAeq_1h'])

    @pytest.mark.slow
    @pytest.mark.parametrize("use_chunks", [True, False], ids=['chunks', 'no_chunks'])
    def test_harmonica(self, bruitparif_1s, bruitparif_hourly,
                    bruitparif_span_complete, use_chunks):
        """HARMONICA, BGN and EVT of each hour: within TOL when its span has
        no missing second, within TOL_HARMONICA otherwise."""
        result = harmonica(bruitparif_1s, 0, use_chunks=use_chunks)
        result = result.loc[bruitparif_hourly.index]
        complete = bruitparif_span_complete.to_numpy()
        for indicator in HARMONICA_INDICATORS:
            assert_close(result.loc[complete, indicator],
                        bruitparif_hourly.loc[complete, indicator])
            assert_close(result.loc[~complete, indicator],
                        bruitparif_hourly.loc[~complete, indicator], TOL_HARMONICA)

    def test_harmonica_subsecond(self, bruitparif_halfsecond, bruitparif_hourly,
                                bruitparif_span_complete):
        """Sub-second data (0.5 s levels averaging exactly to the LAeq,1s):
        the hours covered give Bruitparif's HARMONICA, BGN and EVT, as from
        the LAeq,1s."""
        result = harmonica(bruitparif_halfsecond, 0, use_chunks=False)
        hours = bruitparif_hourly.index.intersection(result.index)
        assert len(hours) == 3
        complete = bruitparif_span_complete.loc[hours].to_numpy()
        for indicator in HARMONICA_INDICATORS:
            got = result.loc[hours, indicator].to_numpy()
            expected = bruitparif_hourly.loc[hours, indicator].to_numpy()
            assert_close(got[complete], expected[complete])
            assert_close(got[~complete], expected[~complete], TOL_HARMONICA)


@pytest.mark.external
class TestExternalBands:
    """External validation of the band labels against ISO 266:1997."""

    @pytest.mark.parametrize("unit, factor", [('', 1), ('k', 1000)],
                             ids=['Hz', 'kHz'])
    def test_parse_freq_columns_iso_266(self, unit, factor):
        """The calculated frequencies of ISO 266, Table 1, read in Hz and in
        kHz, are labelled with their preferred frequencies."""
        names = [f'{f}{unit}' for f in ISO_266_CALCULATED]
        result = _parse_freq_columns(pd.DataFrame(columns=names))
        expected = [round(f * factor, 10) for f in ISO_266_PREFERRED]
        assert result.columns.tolist() == expected


@pytest.mark.internal
class TestCalculations:
    """Internal validation: closed-form results on synthetic signals, or
    consistency with another noisemonitor function."""

    def test_lden_default_periods(self, accra_daily, accra_days):
        """Default periods of the EU Directive (07h-19h-23h), on each Accra
        calendar day: Lday, Levening and Lnight are the Leq of hours 7-18,
        19-22 and 23, 0-6, and Lden combines them with 12, 4 and 8 hours
        and the 5 and 10 dB penalties."""
        sites = accra_daily['ID']
        result = pd.concat([lden(day, ACCRA_COLUMNS[site], values=True)
                            for day, site in zip(accra_days, sites)], ignore_index=True)

        def period_leq(hours):
            return np.array([
                equivalent_level(day.loc[day.index.hour.isin(hours), site].to_numpy())
                for day, site in zip(accra_days, sites)])

        lday = period_leq(range(7, 19))
        levening = period_leq(range(19, 23))
        lnight = period_leq([23, *range(7)])
        expected = 10 * np.log10((12 * 10 ** (lday / 10)
                                + 4 * 10 ** ((levening + 5) / 10)
                                + 8 * 10 ** ((lnight + 10) / 10)) / 24)

        assert_close(result['Lday'], lday)
        assert_close(result['Levening'], levening)
        assert_close(result['Lnight'], lnight)
        assert_close(result['Lden'], expected)

    @pytest.mark.parametrize("missing", [False, True], ids=['complete', 'missing_second'])
    def test_harmonica_la95_window(self, missing):
        """40 dB for one hour, then 60 dB. The LA95 of second t of the second
        hour covers [t - 600 s, t) and is the 31st lowest of its 600 values:
        40 dB while at least 31 of them are 40 dB (t < 570 s), 60 dB after.
        A missing second (00:50:01) is ignored: it leaves these LA95 unchanged."""
        dates = pd.date_range('2023-01-01', periods=2 * 3600, freq='s')
        levels = np.where(dates.hour == 0, 40.0, 60.0)
        if missing:
            levels[3001] = np.nan
        df = pd.DataFrame({'LAeq': levels}, index=dates)
        result = harmonica(df, 0, use_chunks=False).loc[dates[3600]]

        la95eq = 10 * np.log10((570 * 10 ** 4 + 3030 * 10 ** 6) / 3600)
        assert np.isclose(result['BGN'], 0.2 * (la95eq - 30), rtol=0, atol=1e-9)
        assert np.isclose(result['EVT'], 0.25 * (60 - la95eq), rtol=0, atol=1e-9)

    def test_harmonica_la95_window_coverage(self):
        """One hour without data before it: 90 dB for 10 s, then 50 dB. The
        windows of t < 300 s hold fewer than half of their 600 values and
        give no LA95; the later ones give 50 dB."""
        dates = pd.date_range('2023-01-01', periods=3600, freq='s')
        df = pd.DataFrame({'LAeq': np.where(np.arange(3600) < 10, 90.0, 50.0)},
                        index=dates)
        result = harmonica(df, 0, use_chunks=False).iloc[0]

        laeq = 10 * np.log10((10 * 10 ** 9 + 3590 * 10 ** 5) / 3600)
        assert np.isclose(result['BGN'], 0.2 * (50 - 30), rtol=0, atol=1e-9)
        assert np.isclose(result['EVT'], 0.25 * (laeq - 50), rtol=0, atol=1e-9)

    def test_laeq_1s(self):
        """0.1 s levels starting mid-second, with missing values and one
        empty second: each LAeq,1s is equivalent_level of the samples in
        [s, s + 1 s), NaN ignored, and NaN for the empty second."""
        rng = np.random.default_rng(0)
        index = pd.date_range('2023-01-01 00:00:00.500', periods=100, freq='100ms')
        values = 55 + 6 * rng.standard_normal(100)
        values[[7, 23, 24, 61]] = np.nan
        values[45:55] = np.nan  # the whole second 00:00:05
        levels = pd.Series(values, index=index)

        result = _laeq_1s(levels)

        seconds = pd.date_range('2023-01-01 00:00:00', '2023-01-01 00:00:10', freq='s')
        expected = [equivalent_level(levels[(index >= s)
                                            & (index < s + pd.Timedelta(seconds=1))]
                                    .to_numpy())
                    for s in seconds]
        assert list(result.index) == list(seconds)
        assert np.isnan(result.loc['2023-01-01 00:00:05'])
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=0, atol=1e-9)


    @pytest.mark.parametrize("method, band", [
        ('montano', 16.0), ('montano', 63.0), ('montano', 200.0),
        ('barcelona', 20.0), ('barcelona', 160.0),
        ('botsford', 31.5), ('botsford', 1000.0), ('botsford', 20000.0)])
    def test_lf_single_band(self, method, band):
        """With the energy in one band (60 dB, the others at -100 dB), C-A is
        the difference of the C and A weightings of that band."""
        index = pd.date_range('2024-01-01', periods=3, freq='s')
        levels = [60.0 if b == band else -100.0 for b in LF_BANDS]
        result = _lf_levels(band_frame(levels, LF_BANDS, index), method)
        difference = result[('LC', method)] - result[('LA', method)]
        assert_close(difference.to_numpy(),
                    _C_WEIGHTING[band] - _A_WEIGHTING[band], 1e-6)

    @pytest.mark.parametrize("method, f_min, f_max", [
        ('montano', 16.0, 200.0), ('barcelona', 20.0, 160.0),
        ('botsford', 20.0, 20000.0)])
    def test_lf_flat_spectrum(self, method, f_min, f_max):
        """For the same level in every band, LA and LC are that level plus the
        energetic sum of the weightings over the method's bands."""
        index = pd.date_range('2024-01-01', periods=3, freq='s')
        flat = band_frame([60.0] * len(LF_BANDS), LF_BANDS, index)
        result = _lf_levels(flat, method)
        bands = [b for b in LF_BANDS if f_min <= b <= f_max]
        for indicator, weighting in (('LA', _A_WEIGHTING),
                                    ('LC', _C_WEIGHTING)):
            expected = 60.0 + 10 * np.log10(
                sum(10 ** (weighting[b] / 10) for b in bands))
            assert_close(result[(indicator, method)].to_numpy(), expected,
                        1e-9)

    @pytest.mark.parametrize("method, la, lc, expected", [
        ('botsford', 50.0, 60.0, False), ('botsford', 50.0, 60.5, True),
        ('montano', 50.0, 65.0, False), ('montano', 50.0, 65.5, True),
        ('barcelona', 50.0, 69.5, False), ('barcelona', 50.0, 70.0, True)])
    def test_lf_flag_threshold(self, method, la, lc, expected):
        """Botsford (> 10 dB) and Montano (> 15 dB) flag a difference above
        their threshold; Barcelona flags 20 dB and above."""
        levels = pd.DataFrame({('LA', method): [la], ('LC', method): [lc]})
        levels.columns.names = ['Indicator', 'Method']
        result = _lf_apply(
            lambda df, column: pd.DataFrame({'Leq': df.iloc[:, column]}),
            levels, flag=True)
        assert result[('C-A', method)].iloc[0] == lc - la
        assert result[('Lowfreq_Flag', method)].iloc[0] == expected


@pytest.mark.behaviour
class TestCheckCoverage:
    """Test the check_coverage function."""

    def test_check_coverage_full_coverage(self):
        """Test with 100% data coverage."""
        array = np.array([60.0, 65.0, 70.0, 75.0])
        passes = check_coverage(array, threshold=0.5)
        assert passes is True

    def test_check_coverage_above_threshold(self):
        """Test with coverage above threshold."""
        array = np.array([60.0, 65.0, np.nan, 70.0])
        passes = check_coverage(array, threshold=0.5)
        assert passes is True

    def test_check_coverage_at_threshold(self):
        """Test with coverage exactly at threshold."""
        array = np.array([60.0, np.nan, 70.0, np.nan])
        passes = check_coverage(array, threshold=0.5)
        assert passes is True

    def test_check_coverage_below_threshold(self):
        """Test with coverage below threshold."""
        array = np.array([60.0, np.nan, np.nan, np.nan])
        passes = check_coverage(array, threshold=0.5)
        assert passes is False

    def test_check_coverage_empty_array(self):
        """Test with empty array."""
        array = np.array([])
        passes = check_coverage(array, threshold=0.5)
        assert passes is False

    def test_check_coverage_all_nan(self):
        """Test with all NaN values."""
        array = np.array([np.nan, np.nan, np.nan])
        passes = check_coverage(array, threshold=0.5)
        assert passes is False

    def test_check_coverage_with_warning(self):
        """Test that warning is emitted when emit_warning=True."""
        array = np.array([60.0, np.nan, np.nan, np.nan])
        with pytest.warns(CoverageWarning, match="Insufficient data coverage"):
            passes = check_coverage(
                array, 
                threshold=0.5, 
                emit_warning=True
            )
        assert passes is False

    def test_check_coverage_no_warning_when_disabled(self):
        """Test that no warning is emitted when emit_warning=False."""
        array = np.array([60.0, np.nan, np.nan, np.nan])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            passes = check_coverage(
                array, 
                threshold=0.5, 
                emit_warning=False
            )
        assert passes is False

    def test_check_coverage_different_thresholds(self):
        """Test with various threshold values."""
        array = np.array([60.0, 65.0, np.nan, np.nan, 70.0])
        
        # 60% coverage should pass with 0.5 threshold
        passes = check_coverage(array, threshold=0.5)
        assert passes is True
        
        # 60% coverage should fail with 0.7 threshold
        passes = check_coverage(array, threshold=0.7)
        assert passes is False


@pytest.mark.behaviour
class TestEquivalentLevel:
    """Test the equivalent_level function without data."""

    def test_equivalent_level_empty_array(self):
        """Test with empty array."""
        array = np.array([])
        result = equivalent_level(array)
        assert np.isnan(result)

    def test_equivalent_level_all_nan(self):
        """Test with all NaN values."""
        array = np.array([np.nan, np.nan, np.nan])
        result = equivalent_level(array)
        assert np.isnan(result)


@pytest.mark.behaviour
class TestGetInterval:
    """Test the get_interval function."""

    def test_get_interval_basic(self):
        """Test basic interval calculation."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=5, freq='1s')
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]}, index=dates)
        result = get_interval(df)
        assert result == 1

    def test_get_interval_minutes(self):
        """Test with minute intervals."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=5, freq='1min')
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]}, index=dates)
        result = get_interval(df)
        assert result == 60

    def test_get_interval_hours(self):
        """Test with hour intervals."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=5, freq='1h')
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]}, index=dates)
        result = get_interval(df)
        assert result == 3600

    def test_get_interval_insufficient_data(self):
        """Test with insufficient data (less than 3 rows)."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=2, freq='1s')
        df = pd.DataFrame({'value': [1, 2]}, index=dates)
        with pytest.raises(ValueError, match="DataFrame index must have at least three entries"):
            get_interval(df)

    @pytest.mark.parametrize("freq, seconds", [
        ('100ms', 0.1), ('500ms', 0.5), ('1500ms', 1.5), ('1D', 86400.0),
    ])
    def test_get_interval_not_whole_seconds(self, freq, seconds):
        """Intervals below one second, with a fraction of a second, or of a
        day or more: the whole interval, in seconds."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=5, freq=freq)
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]}, index=dates)
        assert get_interval(df) == seconds


@pytest.mark.behaviour
class TestHarmonicaValidation:
    """Test the harmonica function input validation."""

    def test_harmonica_invalid_interval(self):
        """Test that harmonica raises ValueError for intervals > 1s."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=100, freq='5s')
        df = pd.DataFrame({'sound_level': np.random.uniform(40, 80, 100)}, index=dates)
        
        with pytest.raises(ValueError, match="Computing the HARMONICA indicator requires "
                        "an integration time equal to or below 1s"):
            harmonica(df, 0)

    @pytest.mark.slow
    def test_harmonica_valid_interval_subsecond(self):
        """Test that harmonica accepts sub-second intervals (one hour at 0.1 s)."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=36000, freq='100ms')
        df = pd.DataFrame({'sound_level': np.random.uniform(50, 70, 36000)}, index=dates)

        result = harmonica(df, 0)
        assert len(result) == 1
        assert list(result.columns) == ['EVT', 'BGN', 'HARMONICA']
        assert not result.isna().any().any()

    def test_harmonica_empty_dataframe(self):
        """Test harmonica with empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(IndexError):
            harmonica(df, 0)


@pytest.mark.behaviour
class TestHourlyHarmonica:
    """Test the hourly_harmonica helper function."""

    def test_hourly_harmonica_complete_hour(self):
        """Test hourly harmonica with complete hour of data."""
        hour = pd.Timestamp('2023-01-01 12:00:00')
        dates = pd.date_range('2023-01-01 12:00:00', periods=3600, freq='1s')
        group = pd.DataFrame({'sound_level': np.random.uniform(50, 70, 3600)}, index=dates)
        previous_data = pd.DataFrame()
        
        result = hourly_harmonica(hour, group, 0, 1, previous_data)
        
        assert isinstance(result, dict)
        assert 'hour' in result
        assert 'EVT' in result
        assert 'BGN' in result
        assert 'HARMONICA' in result
        assert result['hour'] == hour
        assert not np.isnan(result['EVT'])
        assert not np.isnan(result['BGN'])
        assert not np.isnan(result['HARMONICA'])

    def test_hourly_harmonica_too_many_nans(self):
        """Test hourly harmonica with too many NaN values (>20%)."""
        hour = pd.Timestamp('2023-01-01 12:00:00')
        dates = pd.date_range('2023-01-01 12:00:00', periods=3600, freq='1s')
        data = np.random.uniform(50, 70, 3600)
        data[:1000] = np.nan  # 27.8% NaN values
        group = pd.DataFrame({'sound_level': data}, index=dates)
        previous_data = pd.DataFrame()
        
        result = hourly_harmonica(hour, group, 0, 1, previous_data)
        
        assert isinstance(result, dict)
        assert result['hour'] == hour
        assert np.isnan(result['EVT'])
        assert np.isnan(result['BGN'])
        assert np.isnan(result['HARMONICA'])

    def test_hourly_harmonica_with_previous_data(self):
        """Test hourly harmonica with previous data for rolling window."""
        hour = pd.Timestamp('2023-01-01 12:00:00')
        
        prev_dates = pd.date_range('2023-01-01 11:45:00', periods=900, freq='1s')
        previous_data = pd.DataFrame({'sound_level': np.random.uniform(45, 65, 900)}, index=prev_dates)
        
        dates = pd.date_range('2023-01-01 12:00:00', periods=3600, freq='1s')
        group = pd.DataFrame({'sound_level': np.random.uniform(50, 70, 3600)}, index=dates)
        
        result = hourly_harmonica(hour, group, 0, 1, previous_data)
        
        assert isinstance(result, dict)
        assert result['hour'] == hour
        assert not np.isnan(result['EVT'])
        assert not np.isnan(result['BGN'])
        assert not np.isnan(result['HARMONICA'])

    def test_hourly_harmonica_missing_rows_under_threshold(self):
        """Hour with a few missing rows but >=80% coverage should NOT return NaN."""
        # 3600 expected samples at 1s; drop 100 rows (97.2% coverage)
        dates = pd.date_range('2023-01-01 00:00:00', periods=3600, freq='1s')
        data = np.random.uniform(40, 70, 3600)
        df = pd.DataFrame({'LAeq': data}, index=dates).iloc[:-100]  # 3500 rows
        result = hourly_harmonica(
            hour=dates[0], group=df, column=0,
            interval=1, previous_data=pd.DataFrame()
        )
        assert not np.isnan(result['HARMONICA'])


@pytest.mark.behaviour
class TestLden:
    """Test the lden function."""

    @pytest.mark.parametrize("starts", [
        {'start_day': 24},
        {'start_evening': -1},
        {'start_day': 7.0},
        {'start_night': True},
        {'start_day': 7, 'start_evening': 23, 'start_night': 19},
        {'start_day': 7, 'start_evening': 7, 'start_night': 23},
    ])
    def test_lden_invalid_start_hours(self, starts):
        """Invalid or misordered start hours raise a ValueError."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=24*60, freq='1min')
        df = pd.DataFrame({'sound_level': 60.0}, index=dates)

        with pytest.raises(ValueError):
            lden(df, 0, **starts)


@pytest.mark.internal
class TestNoiseEvents:
    """Test the noise_events function."""

    def test_noise_events_basic(self):
        """Test basic noise event counting."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=100, freq='1s')
        
        data = [50] * 100
        data[10:15] = [75] * 5 
        data[30:35] = [80] * 5
        data[50:55] = [70] * 5
        
        df = pd.DataFrame({'sound_level': data}, index=dates)
        result = noise_events(df, 0, threshold=65.0, min_gap=5)
        
        assert result == 3

    def test_noise_events_no_events(self):
        """Test with no events above threshold."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=100, freq='1s')
        data = [50] * 100  # All below threshold
        df = pd.DataFrame({'sound_level': data}, index=dates)
        
        result = noise_events(df, 0, threshold=65.0, min_gap=5)
        assert result == 0

    def test_noise_events_continuous_event(self):
        """Test with one continuous event starting after the first timestamp."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=100, freq='1s')

        data = [50] + [75] * 99
        df = pd.DataFrame({'sound_level': data}, index=dates)
        
        result = noise_events(df, 0, threshold=65.0, min_gap=1)
        assert result == 1

    def test_noise_events_insufficient_gap(self):
        """Test events with insufficient gap between them."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=100, freq='1s')
        
        data = [50] * 100
        data[10:15] = [75] * 5
        data[17:22] = [80] * 5
        
        df = pd.DataFrame({'sound_level': data}, index=dates)
        result = noise_events(df, 0, threshold=65.0, min_gap=5)
        
        assert result == 1

    def test_noise_events_boundary_conditions(self):
        """Exceedances at both ends of the slice: the slice starts within the
        first one, which is an event, and the second one ends with the slice
        and follows 5 s at or below the threshold (>= 3 s): 2 events."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=10, freq='1s')

        data = [75, 75, 50, 50, 50, 50, 50, 75, 75, 75]
        df = pd.DataFrame({'sound_level': data}, index=dates)

        result = noise_events(df, 0, threshold=65.0, min_gap=3)
        assert result == 2

    @pytest.mark.parametrize("min_gap", [1, 5, 30])
    def test_noise_events_first_crossing(self, min_gap):
        """The first exceedance is an event whatever its distance to the start
        of the slice (here 2 s, shorter than min_gap = 5 or 30 s); the second
        one, 3 s after the end of the first, is a new event only if
        min_gap <= 3 s."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=60, freq='1s')

        data = [50] * 60
        data[2:4] = [75] * 2
        data[7:9] = [75] * 2
        df = pd.DataFrame({'sound_level': data}, index=dates)

        result = noise_events(df, 0, threshold=65.0, min_gap=min_gap)
        assert result == (2 if min_gap <= 3 else 1)

    def test_noise_events_single_point_events(self):
        """Test single-point events."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=20, freq='1s')
        
        data = [50] * 20
        data[5] = 75
        data[10] = 80
        data[15] = 70
        
        df = pd.DataFrame({'sound_level': data}, index=dates)
        result = noise_events(df, 0, threshold=65.0, min_gap=2)
        
        assert result == 3


@pytest.mark.behaviour
class TestParseFreqColumns:
    """Band labels of core._parse_freq_columns."""

    def test_nominal_names(self):
        """Names of nominal bands, in any format, give their frequency."""
        result = _parse_freq_columns(pd.DataFrame(columns=list(BANDS)))
        assert result.columns.tolist() == list(BANDS.values())

    @pytest.mark.parametrize("name, band", [
        ('31.62', 31.5), ('31.6', 31.5), ('31.25', 31.5), (19.69, 20.0),
        ('12.6', 12.5), ('1.6k', 1600.0), ('3.15k', 3150.0),
        (np.float64(125.89), 125.0), (np.int64(63), 63.0), ('6.3', 6.3)])
    def test_exact_and_rounded_names(self, name, band):
        """Exact (base 10 or base 2) and rounded mid-band frequencies are
        labelled with the nominal frequency of their band."""
        result = _parse_freq_columns(pd.DataFrame(columns=[name]))
        assert result.columns[0] == band

    @pytest.mark.parametrize("name, value", [
        ('22', 22.0), ('1059.3', 1059.3), ('0', 0.0), ('15.625', 15.625)])
    def test_other_frequencies_unchanged(self, name, value):
        """Frequencies more than 1.59 % from a preferred frequency (e.g.
        one-twelfth-octave bands, or 15.625 Hz, base-2 frequency of the 16 Hz
        band) keep their parsed value."""
        result = _parse_freq_columns(pd.DataFrame(columns=[name]))
        assert result.columns[0] == value

    def test_same_band_raises(self):
        """Two columns of the same band raise a ValueError naming them."""
        with pytest.raises(ValueError,
                        match=r"'31.5', '31.62' \(31.5 Hz\)"):
            _parse_freq_columns(pd.DataFrame(columns=['31.5', '31.62', '63']))

    def test_unparseable_name(self):
        """A name that is not a frequency leaves the columns unchanged, with
        a warning."""
        df = pd.DataFrame(columns=['63', 'LAeq'])
        with pytest.warns(UserWarning,
                        match="Could not parse column name: LAeq"):
            result = _parse_freq_columns(df)
        assert result.columns.tolist() == ['63', 'LAeq']


@pytest.mark.behaviour
class TestLfLevels:
    """Band selection, errors and flag of core._lf_levels and
    core._lf_apply."""

    @pytest.fixture
    def bands(self):
        index = pd.date_range('2024-01-01', periods=4, freq='s')
        return band_frame([60.0] * len(LF_BANDS), LF_BANDS, index)

    @pytest.mark.parametrize("method", ['din', ['montano', 'din'], []])
    def test_unknown_method(self, bands, method):
        with pytest.raises(ValueError, match="Unknown low-frequency method"):
            _lf_levels(bands, method)

    def test_columns(self, bands):
        """One LA and one LC column per method, in the order given; a method
        given twice is computed once."""
        result = _lf_levels(bands, ['barcelona', 'montano', 'barcelona'])
        assert result.columns.names == ['Indicator', 'Method']
        assert result.columns.tolist() == [
            ('LA', 'barcelona'), ('LC', 'barcelona'),
            ('LA', 'montano'), ('LC', 'montano')]
        assert result.index.equals(bands.index)

    def test_missing_band(self, bands):
        """A missing band of the method raises a ValueError naming it."""
        with pytest.raises(ValueError,
                        match=r"'montano'.*missing: 16, 160 Hz"):
            _lf_levels(bands.drop(columns=['16', '160']), 'montano')

    def test_only_method_bands_needed(self, bands):
        """Montano and Barcelona need no band above 200 Hz; Botsford does."""
        low = bands[[f'{b:g}' for b in LF_BANDS if b <= 200]]
        result = _lf_levels(low, ['montano', 'barcelona'])
        assert not result.isna().any().any()
        with pytest.raises(ValueError, match="'botsford'.*missing: 250, "):
            _lf_levels(low, 'botsford')

    def test_exact_band_names(self, bands):
        """Exact mid-band frequencies as names give the same levels."""
        def exact(name):
            x = round(10 * np.log10(float(name) / 1000))
            return f'{1000 * 10 ** (x / 10):.5g}'

        renamed = bands.rename(columns=exact)
        assert renamed.columns[3] == '31.623'
        pd.testing.assert_frame_equal(_lf_levels(renamed, 'montano'),
                                    _lf_levels(bands, 'montano'))

    def test_column_not_a_band(self, bands):
        """A column that is not a band raises a ValueError naming it, without
        the warning of _parse_freq_columns."""
        df = bands.assign(LAeq=50.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with pytest.raises(ValueError,
                            match=r"could not read \['LAeq'\]"):
                _lf_levels(df, 'montano')

    def test_missing_level(self, bands):
        """A sample with a missing band is NaN for the methods using that
        band only."""
        df = bands.copy()
        df.iloc[1, df.columns.get_loc('16')] = np.nan
        result = _lf_levels(df, ['montano', 'barcelona'])
        for indicator in ('LA', 'LC'):
            assert result[(indicator, 'montano')].isna().tolist() == [
                False, True, False, False]
            assert not result[(indicator, 'barcelona')].isna().any()

    def test_flag_missing_difference(self):
        """Lowfreq_Flag is a nullable boolean, <NA> where C-A is NaN."""
        levels = pd.DataFrame({('LA', 'montano'): [50.0, np.nan],
                            ('LC', 'montano'): [70.0, 70.0]})
        levels.columns.names = ['Indicator', 'Method']
        result = _lf_apply(
            lambda df, column: pd.DataFrame({'Leq': df.iloc[:, column]}),
            levels, flag=True)
        flag = result[('Lowfreq_Flag', 'montano')]
        assert flag.dtype == 'boolean'
        assert flag.iloc[0]
        assert flag.isna().tolist() == [False, True]


if __name__ == '__main__':
    pytest.main([__file__])