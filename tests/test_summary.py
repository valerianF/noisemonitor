"""Unit tests for summary module."""

import pytest
import pandas as pd
import numpy as np
import warnings
from datetime import time, datetime

import noisemonitor as nm
from noisemonitor.summary import (
    harmonica_periodic, periodic, freq_periodic, lden, leq, 
    freq_indicators, nday
)
from noisemonitor.util.core import CoverageWarning, IntegrationTimeWarning
from tests.conftest import (
    ACCRA_COLUMNS, ACCRA_PERIODS, BANDS, BRUITPARIF_LEVELS, HALF_SECOND_END,
    HARMONICA_INDICATORS, SOURCE_COVERAGE, TOL_HARMONICA, TOL_ROUNDED, assert_close,
    lday_06_22
)

LDEN_COLUMNS = ['Leq', 'Lden', 'Lday', 'Levening', 'Lnight']


@pytest.fixture(scope="module")
def sample_octave_data():
    """Create sample octave band data for frequency analysis tests."""
    dates = pd.date_range('2023-01-01', periods=1440, freq='min') 

    octave_bands = ['63', '125', '250', '500', '1000', '2000', '4000', '8000']
    data = {}
    
    np.random.seed(42)  # For reproducible tests
    for i, band in enumerate(octave_bands):
        base_level = 50 + i * 2
        variation = np.random.normal(0, 5, len(dates))
        data[band] = base_level + variation
    
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def one_loud_night_data():
    """Three days at 1 min, 40 dB everywhere except the night from
    2023-01-01 23:00 to 2023-01-02 07:00 at 50 dB."""
    dates = pd.date_range('2023-01-01', periods=3*1440, freq='min')
    loud = ((dates >= '2023-01-01 23:00') & (dates < '2023-01-02 07:00'))
    return pd.DataFrame({'LAeq': np.where(loud, 50.0, 40.0)}, index=dates)


@pytest.mark.external
@pytest.mark.filterwarnings("ignore::noisemonitor.util.core.CoverageWarning")
class TestExternalAccra:
    """Published daily values of two Accra sites (Clark et al., 2021),
    reproducible days only."""

    @pytest.mark.slow
    @pytest.mark.parametrize("coverage", [{}, SOURCE_COVERAGE],
                            ids=['no_check', 'coverage_check'])
    def test_periodic(self, accra_hourly, accra_daily, accra_spans, coverage):
        """Daily Leq, Lden, Lnight, and Lday + Levening (as the source's
        LDay), calendar days, on the whole series."""
        per_site = {site: periodic(accra_hourly, freq='D', column=column,
                                values=True, start_at_day=False,
                                   **ACCRA_PERIODS, **coverage)
                    for site, column in ACCRA_COLUMNS.items()}
        result = pd.DataFrame([per_site[site].loc[date] for site, date
                            in zip(accra_daily['ID'], accra_daily['date'])])
        days = [day for day, _ in accra_spans['day']]

        assert_close(result['Leq'], accra_daily['LAeq24hr'])
        assert_close(result['Lden'], accra_daily['LDEN'])
        assert_close(result['Lnight'], accra_daily['LNight'])
        assert_close(lday_06_22(result['Lday'], result['Levening'], days,
                                accra_daily['ID']), accra_daily['LDay'])

    @pytest.mark.slow
    @pytest.mark.parametrize("span", ['day', 'week'])
    @pytest.mark.parametrize("coverage", [{}, SOURCE_COVERAGE],
                            ids=['no_check', 'coverage_check'])
    def test_lden(self, accra_daily, accra_spans, span, coverage):
        """Lden, Lnight, and Lday + Levening (as the source's LDay), rounded:
        each calendar day, or the 7 days starting on it with day1 = day2."""
        result = pd.concat([
            lden(data, weekday, weekday, column=ACCRA_COLUMNS[site],
                 values=True, **ACCRA_PERIODS, **coverage)
            for (data, weekday), site in zip(accra_spans[span], accra_daily['ID'])],
            ignore_index=True)
        days = [day for day, _ in accra_spans['day']]

        assert_close(result['Lden'], accra_daily['LDEN'], TOL_ROUNDED)
        assert_close(result['Lnight'], accra_daily['LNight'], TOL_ROUNDED)
        assert_close(lday_06_22(result['Lday'], result['Levening'], days,
                                accra_daily['ID']), accra_daily['LDay'], TOL_ROUNDED)

    @pytest.mark.parametrize("indicator, hour1, hour2", [
        ('LAeq24hr', 0, 24),
        ('LDay', 6, 22),
        ('LNight', 22, 6),
    ], ids=['LAeq24hr', 'LDay', 'LNight'])
    @pytest.mark.parametrize("span", ['day', 'week'])
    @pytest.mark.parametrize("coverage", [{}, SOURCE_COVERAGE],
                            ids=['no_check', 'coverage_check'])
    def test_leq(self, accra_daily, accra_spans, indicator, hour1, hour2, span,
                coverage):
        """Leq of [hour1, hour2), rounded: each calendar day, or the 7 days
        starting on it with day1 = day2."""
        got = [leq(data, hour1, hour2, weekday, weekday,
                   column=ACCRA_COLUMNS[site], stats=False, **coverage)['Leq'][0]
            for (data, weekday), site in zip(accra_spans[span], accra_daily['ID'])]
        assert_close(got, accra_daily[indicator], TOL_ROUNDED)


@pytest.mark.external
class TestExternalBruitparif:
    """Hourly indicators computed by Bruitparif from the LAeq,1s of one
    station, over two days."""

    def test_leq(self, bruitparif_1s, bruitparif_hourly):
        """Leq, L10, L50 and L90 of [h, h + 1) on each day (day1 = day2),
        rounded."""
        result = pd.concat([
            leq(bruitparif_1s, h.hour, h.hour + 1, h.day_name(), h.day_name(),
                column=0, stats=True)
            for h in bruitparif_hourly.index], ignore_index=True)
        for got, ref in BRUITPARIF_LEVELS.items():
            assert_close(result[got], bruitparif_hourly[ref], TOL_ROUNDED)

    def test_leq_subsecond(self, bruitparif_halfsecond, bruitparif_hourly):
        """Sub-second data (0.5 s levels averaging exactly to the LAeq,1s):
        Leq of [h, h + 1) for the three hours covered, rounded."""
        expected = bruitparif_hourly.loc[:HALF_SECOND_END, 'LAeq_1h']
        result = pd.concat([leq(bruitparif_halfsecond, h.hour, h.hour + 1,
                                column=0, stats=False)
                            for h in expected.index], ignore_index=True)
        assert len(result) == 3
        assert_close(result['Leq'], expected, TOL_ROUNDED)

    # the 10 min before the first hour form a partial hour (NaN)
    @pytest.mark.filterwarnings("ignore::noisemonitor.util.core.CoverageWarning")
    def test_harmonica_periodic_subsecond(self, bruitparif_halfsecond,
                                        bruitparif_hourly, bruitparif_span_complete):
        """Sub-second data (0.5 s levels averaging exactly to the LAeq,1s),
        one day: hours 00 to 02 give Bruitparif's values, as from the
        LAeq,1s (see test_core)."""
        result = harmonica_periodic(bruitparif_halfsecond, column=0, use_chunks=False)
        hours = bruitparif_hourly.loc[:HALF_SECOND_END].index
        complete = bruitparif_span_complete.loc[hours].to_numpy()
        for indicator in HARMONICA_INDICATORS:
            got = result.loc[[time(h.hour) for h in hours], indicator].to_numpy()
            expected = bruitparif_hourly.loc[hours, indicator].to_numpy()
            assert_close(got[complete], expected[complete])
            assert_close(got[~complete], expected[~complete], TOL_HARMONICA)

    @pytest.mark.slow
    # the 10 min before the first day form a partial hour (NaN)
    @pytest.mark.filterwarnings("ignore::noisemonitor.util.core.CoverageWarning")
    def test_harmonica_periodic(self, bruitparif_1s, bruitparif_hourly,
                                bruitparif_span_complete):
        """HARMONICA, BGN and EVT of each hour of the day, averaged over the
        two days: within TOL when the span of that hour has no missing
        second on both days, within TOL_HARMONICA otherwise (see test_core)."""
        result = harmonica_periodic(bruitparif_1s, column=0)
        ref = bruitparif_hourly[HARMONICA_INDICATORS]
        expected = ref.groupby(ref.index.hour).mean()
        complete = (bruitparif_span_complete
                    .groupby(bruitparif_span_complete.index.hour).all().to_numpy())

        assert list(result.index) == [time(h) for h in range(24)]
        for indicator in HARMONICA_INDICATORS:
            assert_close(result[indicator].to_numpy()[complete],
                        expected[indicator].to_numpy()[complete])
            assert_close(result[indicator].to_numpy()[~complete],
                        expected[indicator].to_numpy()[~complete], TOL_HARMONICA)

    def test_harmonica_periodic_sub_day(self, bruitparif_1s, bruitparif_hourly,
                                        bruitparif_span_complete):
        """A few hours of one day, from 07:50 to 14:00: hours 08 to 13 give
        Bruitparif's values (the 10 min before 08:00 hold the first LA95
        window), as in test_harmonica_periodic. The 07:00 hour (10 min of
        data) is NaN with a CoverageWarning, and hours without data are NaN."""
        day = bruitparif_hourly.index[0].normalize()
        hours = pd.date_range(day + pd.Timedelta(hours=8), periods=6, freq='h')
        data = bruitparif_1s.loc[day + pd.Timedelta(hours=7, minutes=50):
                                 day + pd.Timedelta(hours=14) - pd.Timedelta(seconds=1)]
        with pytest.warns(CoverageWarning):
            result = harmonica_periodic(data, column=0, use_chunks=False)
        expected = bruitparif_hourly.loc[hours, HARMONICA_INDICATORS]
        complete = bruitparif_span_complete.loc[hours].to_numpy()
        included = [time(h.hour) for h in hours]

        assert list(result.index) == [time(h) for h in range(24)]
        for indicator in HARMONICA_INDICATORS:
            got = result.loc[included, indicator].to_numpy()
            assert_close(got[complete], expected[indicator].to_numpy()[complete])
            assert_close(got[~complete], expected[indicator].to_numpy()[~complete],
                         TOL_HARMONICA)
        assert result.drop(index=included).isna().all().all()


@pytest.mark.internal
class TestCalculations:
    """Internal validation: closed-form results on synthetic signals, or
    consistency with another noisemonitor function."""

    def test_periodic_custom_periods(self, one_loud_night_data):
        """Start hours are passed to core.lden and set the day start."""
        starts = dict(start_day=6, start_evening=18, start_night=22)
        result = periodic(one_loud_night_data, freq='D', values=True,
                          start_at_day=True, **starts)

        assert (result.index.hour == 6).all()
        day = pd.Timestamp('2023-01-01 06:00')
        group = one_loud_night_data.loc[day:day + pd.Timedelta(hours=24)
                                        - pd.Timedelta(minutes=1)]
        expected = nm.util.core.lden(group, 0, values=True, **starts)
        for col in ['Lden', 'Lday', 'Levening', 'Lnight']:
            assert result.loc[day, col] == expected[col][0]

    def test_freq_lden_custom_periods(self, sample_octave_data):
        """Start hours are passed through to Lden in both freq functions."""
        starts = dict(start_day=6, start_evening=18, start_night=22)
        indicators = freq_indicators(sample_octave_data, **starts)
        periodic_bands = freq_periodic(sample_octave_data, values=True,
                                       start_at_day=True, **starts)

        assert (periodic_bands.index.hour == 6).all()
        for i, band in enumerate(sample_octave_data.columns):
            expected = lden(sample_octave_data, column=i, **starts)
            for col in ['Lden', 'Lday', 'Levening', 'Lnight']:
                assert indicators.loc[col, float(band)] == expected[col][0]

    def test_nday_start_at_day(self, one_loud_night_data):
        """nday passes start hours and start_at_day to periodic."""
        result, _ = nday(one_loud_night_data, indicator='Lnight',
                        bins=[45], column=0, start_at_day=True)
        counts = result.set_index('Decibel Range')['Number of Days']

        # Days from 07:00: partial day (00-07 of 2023-01-01), the loud night,
        # 40 dB night, partial last day (23-24 of 2023-01-03)
        assert counts.iloc[1] == 1
        assert counts.sum() == 4

    @pytest.mark.slow
    @pytest.mark.parametrize("start_at_day", [False, True],
                            ids=['calendar', 'start_at_day'])
    @pytest.mark.parametrize("freq", ['D', 'W', 'MS'])
    def test_periodic_periods(self, accra_hourly, freq, start_at_day):
        """Leq and Lden (default periods) of each day, week (Monday to
        Sunday) or calendar month, starting at midnight or at start_day
        (07:00), on the whole Accra series: core.equivalent_level and
        core.lden of the hours of each period. The period of each hour is
        found from its date, without resampling."""
        shift = pd.Timedelta(hours=7 if start_at_day else 0)
        shifted = accra_hourly.index - shift
        if freq == 'D':
            starts = shifted.normalize()
        elif freq == 'W':
            starts = shifted.normalize() - pd.to_timedelta(shifted.day_of_week, unit='D')
        else:
            starts = shifted.to_period('M').start_time
        starts = starts + shift

        for site, column in ACCRA_COLUMNS.items():
            result = periodic(accra_hourly, freq=freq, column=column, values=True,
                            start_at_day=start_at_day)
            expected = pd.DataFrame([
                {'Leq': nm.util.core.equivalent_level(group[site].to_numpy()),
                 **nm.util.core.lden(group, site, values=True).iloc[0]}
                for _, group in accra_hourly.groupby(starts)])

            assert list(result.index) == list(starts.unique())
            for col in LDEN_COLUMNS:
                assert_close(result[col], expected[col])

    @pytest.mark.slow
    @pytest.mark.parametrize("freq", ['D', 'W'])
    def test_nday_counts(self, accra_hourly, freq):
        """Number of days or weeks per bin, for each indicator: the values of
        summary.periodic counted in [lower, upper) bins, NaN left out."""
        bins = [50, 55, 60, 65, 70, 75]
        edges = [-np.inf] + bins + [np.inf]
        for site, column in ACCRA_COLUMNS.items():
            indicators = periodic(accra_hourly, freq=freq, column=column, values=True)
            for indicator in LDEN_COLUMNS:
                result, returned_bins = nday(accra_hourly, indicator=indicator,
                                             bins=bins, freq=freq, column=column)
                values = indicators[indicator]
                expected = [int(((values >= lower) & (values < upper)).sum())
                            for lower, upper in zip(edges[:-1], edges[1:])]

                assert returned_bins == bins
                assert list(result['Number of Days']) == expected
                assert sum(expected) == values.notna().sum()
                ranges = list(result['Decibel Range'])
                assert [r.left for r in ranges] == edges[:-1]
                assert [r.right for r in ranges] == edges[1:]
                assert all(r.closed == 'left' for r in ranges)

    def test_nday_bin_edge(self):
        """A value equal to a bin edge is counted in the bin above it: three
        days at a constant 60 dB (Leq exactly 60.0)."""
        dates = pd.date_range('2023-01-01', periods=3 * 24, freq='h')
        df = pd.DataFrame({'LAeq': 60.0}, index=dates)
        result, _ = nday(df, bins=[55, 60, 65])

        assert list(result['Number of Days']) == [0, 0, 3, 0]

    def test_freq_periodic_bands(self, bruitparif_bands):
        """Each band (column names in mixed formats, unsorted) gives the
        values of summary.periodic on its own column, under its frequency;
        bands in ascending order."""
        result = freq_periodic(bruitparif_bands, freq='D', values=True)

        assert result.columns.names == ['Indicator', 'Frequency Band']
        assert (list(result.columns.get_level_values('Frequency Band').unique())
                == sorted(BANDS.values()))
        for name, band in BANDS.items():
            expected = periodic(bruitparif_bands, freq='D', column=name, values=True)
            pd.testing.assert_frame_equal(
                result.xs(band, axis=1, level='Frequency Band'), expected,
                check_names=False, check_exact=True)

    def test_freq_indicators_bands(self, bruitparif_bands):
        """Each band gives the values of summary.leq (with L10, L50, L90) and
        summary.lden on its own column, under its frequency; bands in
        ascending order."""
        result = freq_indicators(bruitparif_bands, hour1=22, hour2=6, stats=True,
                                values=True)

        assert list(result.columns) == sorted(BANDS.values())
        for name, band in BANDS.items():
            levels = leq(bruitparif_bands, 22, 6, column=name, stats=True).iloc[0]
            periods = lden(bruitparif_bands, column=name, values=True).iloc[0]
            expected = pd.Series({
                'Leq': levels['Leq'], 'Lden': periods['Lden'],
                'L10': levels['L10'], 'L50': levels['L50'], 'L90': levels['L90'],
                'Lday': periods['Lday'], 'Levening': periods['Levening'],
                'Lnight': periods['Lnight']})
            pd.testing.assert_series_equal(result[band], expected, check_names=False,
                                        check_exact=True)


@pytest.mark.behaviour
class TestHarmonicaPeriodic:
    """Test cases for the harmonica_periodic function."""

    def test_harmonica_1m(self, laeq1m_data):
        """Test harmonica periodic raises error for intervals > 1s."""
        # Test that it raises ValueError for insufficient data
        with pytest.raises(
            ValueError,
            match="Computing the HARMONICA indicator requires"
        ):
            harmonica_periodic(
                laeq1m_data, 
                column=0
            )

    @pytest.mark.slow
    def test_harmonica_periodic_no_chunks(self, laeq1s_data):
        """Test harmonica periodic without chunks."""
        with pytest.warns(CoverageWarning, 
                        match="Insufficient data coverage or gaps detected"):
            result = harmonica_periodic(laeq1s_data, column=0, use_chunks=False)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 24


@pytest.mark.behaviour
class TestPeriodic:
    """Test cases for the periodic function."""

    def test_periodic_invalid_frequency(self, laeq1m_data):
        """Test periodic with invalid frequency."""
        with pytest.raises(ValueError, match="Invalid frequency"):
            periodic(laeq1m_data, freq='X', column=0)


@pytest.mark.behaviour
class TestFreq:
    """Test cases for the frequency bands function."""

    def test_freq_indicators(self, sample_octave_data):
        """Test basic frequency indicators computation."""
        with pytest.warns(
            IntegrationTimeWarning,
            match="Computing the L10, L50, and L90"
            ):
            result = freq_indicators(
                sample_octave_data, 
                stats=True,
                hour1=6,
                hour2=22
            )
        
        assert isinstance(result, pd.DataFrame)

        expected_bands = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 
                        4000.0, 8000.0]
        for band in expected_bands:
            assert band in result.columns

        expected_indicators = ['Leq', 'Lden', 'L10', 'L50', 'L90', 'Lday',
                            'Levening', 'Lnight']
        for indicator in expected_indicators:
            assert indicator in result.index


@pytest.mark.behaviour
class TestNday:
    """Test cases for the nday function."""

    def test_nday_invalid_indicator(self, laeq1m_data):
        """Test nday with invalid indicator."""
        with pytest.raises(ValueError, match="Invalid indicator"):
            nday(laeq1m_data, indicator='InvalidIndicator', column=0)


@pytest.mark.behaviour
class TestSummaryEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe(self):
        """Test summary functions with empty DataFrame."""
        empty_df = pd.DataFrame(
            {'Leq': []}, 
            index=pd.DatetimeIndex([])
        )

        with pytest.raises((ValueError, IndexError, KeyError)):
            periodic(empty_df, freq='D')

        result = lden(empty_df, column=0)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_invalid_column_index(self, laeq1m_data):
        """Test with invalid column index."""
        with pytest.raises((IndexError, KeyError)):
            periodic(laeq1m_data, column=99)
        
        with pytest.raises((IndexError, KeyError)):
            lden(laeq1m_data, column=99)

    def test_single_day_data(self, laeq1s_data):
        """Test functions with single day of data."""
        single_day_data = laeq1s_data.iloc[:-1]
        
        result = lden(single_day_data, column=0)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        
        daily_result = periodic(single_day_data, freq='D', column=0)
        assert isinstance(daily_result, pd.DataFrame)
        print(daily_result)
        assert len(daily_result) == 1


@pytest.mark.behaviour
class TestCoverageCheck:
    """Tests for coverage_check functionality across summary functions."""

    def test_periodic_coverage_check(self, data_with_gaps):
        """Test periodic function with coverage_check enabled."""
        # With coverage_check, day 2 should be filtered (40% < 50% threshold)
        with pytest.warns(CoverageWarning, match="Insufficient data coverage detected"):
            result = periodic(
                data_with_gaps, 
                freq='D', 
                column=0, 
                coverage_check=True
            )
        
        assert len(result) == 3
        # Day 2 (index 1) should be NaN due to insufficient coverage
        assert pd.isna(result.iloc[1]['Leq'])
        assert pd.isna(result.iloc[1]['Lden'])
        # Days 1 and 3 should have values
        assert not pd.isna(result.iloc[0]['Leq'])
        assert not pd.isna(result.iloc[2]['Leq'])

    def test_lden_coverage_check(self, data_with_time_gaps):
        """Test lden function with coverage_check enabled."""
        # With coverage_check, evening period should be filtered
        with pytest.warns(CoverageWarning, match="Insufficient data coverage detected"):
            result = lden(
                data_with_time_gaps,
                column=0,
                coverage_check=True,
                values=True
            )
        
        # Should have daily result
        assert len(result) == 1
        
        # Evening value should be NaN (70% removed > 50% threshold)
        assert pd.isna(result['Levening'].iloc[0])
        
        # Day and night should still have values (sufficient coverage)
        assert not pd.isna(result['Lday'].iloc[0])
        assert not pd.isna(result['Lnight'].iloc[0])

    def test_leq_coverage_check(self, laeq1m_data):
        """Test leq function with coverage_check enabled."""
        # Create data with evening period having only 40% coverage
        test_data = laeq1m_data[:1440].copy()  # 1 day
        
        # Set 60% of evening data (19:00-23:00) to NaN
        evening_mask = (test_data.index.hour >= 19) & (test_data.index.hour < 23)
        evening_indices = test_data.index[evening_mask]
        np.random.seed(42)
        nan_indices = np.random.choice(
            evening_indices,
            size=int(len(evening_indices) * 0.6),
            replace=False
        )
        test_data.loc[nan_indices] = np.nan
        
        with pytest.warns(CoverageWarning, match="Insufficient data coverage detected"):
            result = leq(
                test_data,
                hour1=19,
                hour2=23,
                column=0,
                coverage_check=True
            )
        
        # Result should be NaN for evening period with insufficient coverage
        assert pd.isna(result.iloc[0, 0])

    def test_freq_indicators_coverage_check(self, laeq1m_data):
        """Test freq_indicators function with coverage_check enabled."""
        # Create data with gaps in nighttime period
        test_data = laeq1m_data[:1440].copy()  # 1 day
        
        # Set 60% of night data (23:00-7:00) to NaN
        night_mask = (test_data.index.hour >= 23) | (test_data.index.hour < 7)
        night_indices = test_data.index[night_mask]
        np.random.seed(42)
        nan_indices = np.random.choice(
            night_indices,
            size=int(len(night_indices) * 0.6),
            replace=False
        )
        test_data.loc[nan_indices] = np.nan
        
        with pytest.warns(UserWarning, match="Could not parse column name"):
            with pytest.warns(CoverageWarning, match="Insufficient data coverage detected"):
                result = freq_indicators(
                    test_data,
                    hour1=23,
                    hour2=7,
                    coverage_check=True
                )
        
        # Should not have result with NaN values for insufficient coverage
        # Check that at least one frequency band has NaN
        has_nan = result.isna().any().any()
        assert has_nan

    def test_nday_coverage_check(self, data_with_gaps):
        """Test nday function with coverage_check enabled."""
        with pytest.warns(CoverageWarning, match="Insufficient data coverage detected"):
            result, bins = nday(
                data_with_gaps,
                column=0,
                coverage_check=True
            )
        
        # Check that the sum of 'Number of Days' equals 2 (days 1 and 3)
        total_days = result['Number of Days'].sum()
        assert total_days == 2

    @pytest.mark.slow
    def test_harmonica_periodic_coverage_check(self, laeq1s_data):
        """Test harmonica_periodic function emits warning when coverage is below 80%."""
        test_data = laeq1s_data[:86400].copy()  # 1 full day
        
        hour_mask = (test_data.index.hour == 10)
        hour_indices = test_data.index[hour_mask]
        np.random.seed(42)
        nan_indices = np.random.choice(
            hour_indices,
            size=int(len(hour_indices) * 0.25),
            replace=False
        )
        test_data.loc[nan_indices] = np.nan
        
        with pytest.warns(CoverageWarning, match="Insufficient data coverage or gaps detected"):
            result = harmonica_periodic(
                test_data,
                column=0,
                use_chunks=True
            )
        
        assert len(result) == 24
        assert 'HARMONICA' in result.columns
        assert 'BGN' in result.columns
        assert 'EVT' in result.columns
        
        assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__])