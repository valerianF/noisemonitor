"""Unit tests for profile module."""

import pytest
import pandas as pd
import numpy as np
import warnings
import os
from datetime import time, datetime

import noisemonitor as nm
from noisemonitor.profile import (
    periodic, series, nne, freq_periodic, freq_series, lf_periodic, lf_series
)
from noisemonitor.util.core import (
    CoverageWarning, IntegrationTimeWarning, _A_WEIGHTING, _C_WEIGHTING
)
from tests.conftest import (
    ACCRA_COLUMNS, BANDS, BRUITPARIF_LEVELS, GENCAT_A6, GENCAT_A6_BANDS,
    HALF_SECOND_END, LF_BANDS, LF_METHOD_BANDS, MONTANO_FIG3C, MONTANO_TABLE2,
    SOURCE_COVERAGE, TOL_PUBLISHED, assert_close, band_frame, weighted_band_sum
)


def window_levels(levels):
    """Leq, L10, L50 and L90 of the samples of one window (percentiles by
    linear interpolation, as checked against Bruitparif's hourly values)."""
    return {'Leq': nm.util.core.equivalent_level(levels.to_numpy()),
            'L10': np.nanpercentile(levels, 90),
            'L50': np.nanpercentile(levels, 50),
            'L90': np.nanpercentile(levels, 10)}


@pytest.fixture(scope="module")
def sample_octave_data():
    """Create sample octave band data for frequency analysis tests."""
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    index = pd.date_range(start=start_time, periods=360, freq='1min')  # 6 hours of data
    
    np.random.seed(42) 
    octave_bands = ['63', '125', '250', '500', '1000', '2000', '4000', '8000']
    
    data = {}
    for i, band in enumerate(octave_bands):
        base_level = 45 + i * 2 
        variation = np.random.normal(0, 3, len(index))
        data[band] = base_level + variation
    
    return pd.DataFrame(data, index=index)


@pytest.fixture(scope="module")
def octave_data_with_gaps(sample_octave_data):
    """Create octave band data with gaps: 70% of hour 2 set to NaN."""
    test_data = sample_octave_data.copy()
    
    # Find hour 2 (02:00-03:00)
    hour_mask = (test_data.index.hour == 2)
    hour_indices = test_data.index[hour_mask]
    
    # Randomly set 70% of that hour's data to NaN for all bands
    np.random.seed(42)
    nan_indices = np.random.choice(
        hour_indices,
        size=int(len(hour_indices) * 0.7),
        replace=False
    )
    test_data.loc[nan_indices] = np.nan
    return test_data


@pytest.fixture
def hour_start_peaks_data():
    """One day at 1 s, 40 dB except 100 dB at every hh:00:00."""
    dates = pd.date_range('2023-01-01', periods=86400, freq='s')
    peaks = (dates.minute == 0) & (dates.second == 0)
    return pd.DataFrame({'LAeq': np.where(peaks, 100.0, 40.0)}, index=dates)


@pytest.mark.external
@pytest.mark.filterwarnings("ignore::noisemonitor.util.core.CoverageWarning")
@pytest.mark.filterwarnings("ignore::noisemonitor.util.core.IntegrationTimeWarning")  # hourly data
@pytest.mark.filterwarnings("ignore:All-NaN slice:UserWarning")  # days not kept
class TestExternalAccra:
    """Published daily values of two Accra sites (Clark et al., 2021),
    reproducible days only."""

    @pytest.mark.parametrize("indicator, hour, win", [
        ('LDay', 6, 16 * 3600),  # window [06, 22)
        ('LNight', 22, 8 * 3600),  # window [22, 06), past midnight
    ], ids=['LDay', 'LNight'])
    @pytest.mark.parametrize("span", ['day', 'week'])
    @pytest.mark.parametrize("coverage", [{}, SOURCE_COVERAGE],
                            ids=['no_check', 'coverage_check'])
    def test_periodic(self, accra_daily, accra_spans, indicator, hour, win, span,
                    coverage):
        """Leq of one window matching a published period (hour1 = hour2), on
        one day so that no averaging across days happens: each calendar day,
        or the 7 days starting on it with day1 = day2."""
        got = [periodic(data, hour, hour, weekday, weekday,
                        column=ACCRA_COLUMNS[site], win=win, **coverage)['Leq'].iloc[0]
            for (data, weekday), site in zip(accra_spans[span], accra_daily['ID'])]
        assert_close(got, accra_daily[indicator])

    @pytest.mark.parametrize("coverage", [{}, SOURCE_COVERAGE],
                            ids=['no_check', 'coverage_check'])
    def test_series(self, accra_hourly, accra_daily, coverage):
        """24 h windows starting at midnight, labelled by their centre
        (12:00), on the whole series: LAeq24hr."""
        per_site = {site: series(accra_hourly, win=86400, column=column,
                                 start_at_midnight=True, **coverage)
                    for site, column in ACCRA_COLUMNS.items()}
        got = [per_site[site].loc[date + pd.Timedelta(hours=12), 'Leq'] for site, date
            in zip(accra_daily['ID'], accra_daily['date'])]
        assert_close(got, accra_daily['LAeq24hr'])


@pytest.mark.external
class TestExternalBruitparif:
    """Hourly indicators computed by Bruitparif from the LAeq,1s of one
    station, over two days."""

    @pytest.mark.parametrize("hour1, hour2", [(0, 23), (22, 6)],
                            ids=['whole_day', 'past_midnight'])
    def test_periodic(self, bruitparif_1s, bruitparif_hourly, hour1, hour2):
        """1 h windows from hour1 to hour2 on one day (day1 = day2), so that
        no averaging across days happens, labelled by their centre (hh:30):
        Leq, L10, L50 and L90 of each hour."""
        hours = [(hour1 + i) % 24 for i in range((hour2 - hour1) % 24 + 1)]
        days = bruitparif_hourly.index.normalize().unique()
        result = pd.concat([periodic(bruitparif_1s, hour1, hour2, d.day_name(),
                                    d.day_name(), column=0, win=3600) for d in days])
        expected = bruitparif_hourly.loc[[d + pd.Timedelta(hours=h)
                                        for d in days for h in hours]]

        assert list(result.index) == [time(h, 30) for h in hours] * len(days)
        for got, ref in BRUITPARIF_LEVELS.items():
            assert_close(result[got], expected[ref])

    @pytest.mark.parametrize("stat", [[10, 50, 90], 50], ids=['list', 'int'])
    def test_periodic_stat(self, bruitparif_1s, bruitparif_hourly, stat):
        """stat as a list or an int: each statistical level L_s is the
        (100 - s)th percentile, and these columns replace the default ones.
        1 h windows on one day (day1 = day2): Bruitparif's Ln of each hour."""
        days = bruitparif_hourly.index.normalize().unique()
        result = pd.concat([periodic(bruitparif_1s, 0, 23, d.day_name(), d.day_name(),
                                    column=0, win=3600, stat=stat)
                            for d in days])
        expected = bruitparif_hourly.loc[[d + pd.Timedelta(hours=h)
                                        for d in days for h in range(24)]]

        for s in stat if isinstance(stat, list) else [stat]:
            assert_close(result[f'L{s}'], expected[BRUITPARIF_LEVELS[f'L{s}']])

    def test_series(self, bruitparif_1s, bruitparif_hourly):
        """1 h windows starting at midnight, labelled by their centre (hh:30),
        on the whole series: Leq, L10, L50 and L90 of each hour."""
        result = series(bruitparif_1s, win=3600, column=0, start_at_midnight=True)

        assert list(result.index) == list(bruitparif_hourly.index + pd.Timedelta(minutes=30))
        for got, ref in BRUITPARIF_LEVELS.items():
            assert_close(result[got], bruitparif_hourly[ref])

    def test_subsecond(self, bruitparif_halfsecond, bruitparif_hourly):
        """Sub-second data (0.5 s levels averaging exactly to the LAeq,1s)
        over three hours: the Leq of each 1 h window is Bruitparif's LAeq of
        the hour, with periodic and with series starting at midnight."""
        expected = bruitparif_hourly.loc[:HALF_SECOND_END, 'LAeq_1h']
        assert len(expected) == 3

        result = periodic(bruitparif_halfsecond, 0, 2, column=0, win=3600)
        assert list(result.index) == [time(h, 30) for h in range(3)]
        assert_close(result['Leq'], expected)

        result = series(bruitparif_halfsecond, win=3600, column=0,
                        start_at_midnight=True)
        assert list(result.index) == list(expected.index + pd.Timedelta(minutes=30))
        assert_close(result['Leq'], expected)


@pytest.mark.external
class TestExternalLowFrequency:
    """External validation of lf_series and lf_periodic against published
    worked examples (constant band levels over one day at 1 min)."""

    @pytest.mark.parametrize("function, args", [
        (lf_series, {'win': 3600}), (lf_periodic, {'hour1': 0, 'hour2': 23})],
        ids=['lf_series', 'lf_periodic'])
    def test_montano_table2(self, function, args):
        """Montano & Gushiken (2023), Table 2: CLF, ALF and their difference
        as given for Fig. 3-c, in every window."""
        index = pd.date_range('2024-01-01', periods=1440, freq='min')
        df = band_frame(MONTANO_TABLE2['Z'], MONTANO_TABLE2['bands'], index)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = function(df, method='montano', **args)
        assert len(result) == 24
        for indicator, value in MONTANO_FIG3C.items():
            assert_close(result[(indicator, 'montano')].to_numpy(), value,
                        TOL_PUBLISHED)

    @pytest.mark.parametrize("function, args", [
        (lf_series, {'win': 3600}), (lf_periodic, {'hour1': 0, 'hour2': 23})],
        ids=['lf_series', 'lf_periodic'])
    @pytest.mark.parametrize("case", ['T3', 'T5'])
    def test_generalitat_a6(self, case, function, args):
        """Generalitat de Catalunya (2012), guide A6: LAeq and LCeq of the
        bands from 20 Hz to 160 Hz and their difference Lf, in every
        window."""
        index = pd.date_range('2024-01-01', periods=1440, freq='min')
        df = band_frame(GENCAT_A6[case]['Z'], GENCAT_A6_BANDS, index)
        result = function(df, method='barcelona', **args)
        assert len(result) == 24
        for indicator in ('LA', 'LC', 'C-A'):
            assert_close(result[(indicator, 'barcelona')].to_numpy(),
                        GENCAT_A6[case][indicator], TOL_PUBLISHED)


@pytest.mark.internal
class TestCalculations:
    """Internal validation: closed-form results on synthetic signals, or
    consistency with another noisemonitor function."""

    @pytest.mark.parametrize("hour2", [23, 24], ids=['to_23h', 'full_day'])
    def test_nne_windows_half_open(self, hour_start_peaks_data, hour2):
        """Each window covers [start, start + win): the hh:00:00 event is
        counted in one window only, not also in the previous one. The last
        window ends at hour2, so (0, 24) gives the 24 hours of the day."""
        result = nne(hour_start_peaks_data, hour1=0, hour2=hour2,
                    background_type=70, min_gap=0, win=3600, column=0)

        assert list(result.index) == [time(h, 30) for h in range(hour2)]
        assert (result.iloc[:, 0] == 1).all()

    def test_periodic_tni(self, bruitparif_1s, bruitparif_hourly):
        """TNI = 4 (L10 - L90) + L90 - 30 (Griffiths and Langdon, 1968) of
        each hour, against the formula applied to Bruitparif's hourly L10 and
        L90 (the percentiles are checked in TestExternalBruitparif): 1 h
        windows on one day (day1 = day2)."""
        days = bruitparif_hourly.index.normalize().unique()
        result = pd.concat([periodic(bruitparif_1s, 0, 23, d.day_name(), d.day_name(),
                                    column=0, win=3600, traffic_noise_indicators=True)
                            for d in days])
        l10 = bruitparif_hourly[BRUITPARIF_LEVELS['L10']]
        l90 = bruitparif_hourly[BRUITPARIF_LEVELS['L90']]

        assert_close(result['TNI'], 4 * (l10 - l90) + l90 - 30)

    @pytest.mark.parametrize("step, hour2", [(0, 23), (900, 22)],
                            ids=['hourly', 'sliding'])
    @pytest.mark.parametrize("one_day", [False, True], ids=['pooled', 'one_day'])
    def test_periodic_windows(self, bruitparif_1s, bruitparif_hourly, step, hour2,
                            one_day):
        """1 h windows starting every hour (step 0) or every 15 min, from
        00:00 to hour2, labelled by their centre: each window holds the
        samples whose time of day lies in [start, start + 1 h), on one day
        (day1 = day2) or pooled over all days without a day filter (the 10 min
        before the first day included). Leq and percentiles of the pooled
        samples, not means of daily values."""
        levels = bruitparif_1s.iloc[:, 0]
        index = levels.index
        seconds = index.hour * 3600 + index.minute * 60 + index.second
        starts = range(0, hour2 * 3600 + 1, step or 3600)
        days = ([d.day_name() for d in bruitparif_hourly.index.normalize().unique()]
                if one_day else [None])

        for day in days:
            result = periodic(bruitparif_1s, 0, hour2, day, day, column=0,
                            win=3600, step=step)
            on_day = index.day_name() == day if day else np.ones(len(index), bool)
            expected = pd.DataFrame([
                window_levels(levels[on_day & (seconds >= s) & (seconds < s + 3600)])
                for s in starts])

            assert list(result.index) == [
                (pd.Timestamp(0) + pd.Timedelta(seconds=s + 1800)).time() for s in starts]
            for col in expected.columns:
                assert_close(result[col], expected[col])

    def test_periodic_roughness_gaps(self):
        """Roughness from the differences between samples one interval apart
        only: |dL| = 1 dB between every such pair, while the jumps across a
        missing value (50/51 to 70/71 dB) and between the two days pooled in
        the window (70/71 to 30/31 dB) are left out, so dLav = dLmax,1 =
        dLmin,90 = 1. A window holding a single sample has no difference: NaN."""
        start = pd.Timestamp('2023-01-02')
        hour = pd.timedelta_range(0, periods=3600, freq='s')
        alternating = 50.0 + np.arange(3600) % 2
        day1 = np.where(np.arange(3600) < 1800, alternating, alternating + 20)
        day1[1800] = np.nan
        index = ((start + hour)
                .append(pd.DatetimeIndex([start + pd.Timedelta(hours=1)]))
                .append(start + pd.Timedelta(days=1) + hour))
        df = pd.DataFrame({'LAeq': np.concatenate([day1, [60.0], alternating - 20])},
                        index=index)
        result = periodic(df, 0, 1, column=0, win=3600, roughness_indicators=True)

        roughness = ['dLav', 'dLmax,1', 'dLmin,90']
        assert list(result.loc[time(0, 30), roughness]) == [1.0, 1.0, 1.0]
        assert result.loc[time(1, 30), roughness].isna().all()

    def test_periodic_roughness_subsecond(self, bruitparif_halfsecond, bruitparif_1s):
        """Roughness is defined on LAeq,1s: 0.5 s levels averaging exactly to
        Bruitparif's LAeq,1s give the roughness of the LAeq,1s (differences
        between the 0.5 s levels themselves would differ by several dB)."""
        roughness = ['dLav', 'dLmax,1', 'dLmin,90']
        kwargs = dict(column=0, win=3600, roughness_indicators=True)
        result = periodic(bruitparif_halfsecond, 0, 2, **kwargs)
        expected = periodic(bruitparif_1s.loc[:HALF_SECOND_END], 0, 2, **kwargs)

        for col in roughness:
            assert_close(result[col], expected[col])

    @pytest.mark.parametrize("min_gap, events", [(3, 3), (0, 4)])
    def test_nne_subsecond(self, min_gap, events):
        """0.1 s levels over one hour, 50 dB except 70 dB over [10 s, 12 s),
        [12.5 s, 13 s), [20 s, 20.3 s) and [40 s, 45 s), threshold 60 dB:
        4 upward crossings, the second one 0.5 s after the end of the first,
        so 3 events with a minimum gap of 3 s."""
        index = pd.date_range('2023-01-02', periods=36000, freq='100ms')
        sample = np.arange(36000)
        loud = np.zeros(36000, bool)
        for start, stop in [(100, 120), (125, 130), (200, 203), (400, 450)]:
            loud |= (sample >= start) & (sample < stop)
        df = pd.DataFrame({'LAeq': np.where(loud, 70.0, 50.0)}, index=index)

        result = nne(df, 0, 1, background_type=60, min_gap=min_gap, win=3600,
                    column=0)

        assert list(result.index) == [time(0, 30)]
        assert result.iloc[0, 0] == events

    @pytest.mark.parametrize("start_at_midnight", [False, True],
                            ids=['first_sample', 'midnight'])
    def test_series_sliding(self, bruitparif_1s, start_at_midnight):
        """1 h windows every 10 min, from the first sample or from the first
        midnight, labelled by their centre: Leq and percentiles of the
        samples in [start, start + 1 h), for every window that ends within
        the data."""
        levels = bruitparif_1s.iloc[:, 0]
        win, step = pd.Timedelta(hours=1), pd.Timedelta(minutes=10)
        first = levels.index[0]
        if start_at_midnight:
            first = first.ceil('D')
        end = levels.index[-1] + pd.Timedelta(seconds=1)
        starts = pd.date_range(first, end - win, freq=step)
        result = series(bruitparif_1s, win=3600, step=600, column=0,
                        start_at_midnight=start_at_midnight)
        expected = pd.DataFrame([
            window_levels(levels[(levels.index >= s) & (levels.index < s + win)])
            for s in starts])

        assert list(result.index) == list(starts + win / 2)
        for col in expected.columns:
            assert_close(result[col], expected[col])

    @pytest.mark.parametrize("chunks", [False, pytest.param(True, marks=pytest.mark.slow)],
                            ids=['no_chunks', 'chunks'])
    def test_freq_periodic_bands(self, bruitparif_bands, chunks):
        """Each band (column names in mixed formats, unsorted) gives the
        values of profile.periodic on its own column, under its frequency;
        bands in ascending order."""
        params = dict(win=3600, step=1800, stat=[5, 95])
        result = freq_periodic(bruitparif_bands, 0, 23, chunks=chunks, **params)

        assert result.columns.names == ['Indicator', 'Frequency Band']
        assert (list(result.columns.get_level_values('Frequency Band').unique())
                == sorted(BANDS.values()))
        for name, band in BANDS.items():
            expected = periodic(bruitparif_bands, 0, 23, column=name, **params)
            pd.testing.assert_frame_equal(
                result.xs(band, axis=1, level='Frequency Band'), expected,
                check_names=False, check_exact=True, check_like=True)

    @pytest.mark.parametrize("chunks", [False, pytest.param(True, marks=pytest.mark.slow)],
                            ids=['no_chunks', 'chunks'])
    def test_freq_series_bands(self, bruitparif_bands, chunks):
        """Each band gives the values of profile.series on its own column,
        under its frequency; bands in ascending order."""
        params = dict(win=3600, step=1800)
        result = freq_series(bruitparif_bands, chunks=chunks, **params)

        assert result.columns.names == ['Indicator', 'Frequency Band']
        assert (list(result.columns.get_level_values('Frequency Band').unique())
                == sorted(BANDS.values()))
        for name, band in BANDS.items():
            expected = series(bruitparif_bands, column=name, **params)
            pd.testing.assert_frame_equal(
                result.xs(band, axis=1, level='Frequency Band'), expected,
                check_names=False, check_exact=True, check_like=True)


    @pytest.mark.filterwarnings(
        "ignore::noisemonitor.util.core.IntegrationTimeWarning")  # series
    def test_lf_series_levels(self, lf_bands):
        """LA and LC are the Leq of profile.series applied to the weighted
        band sums of each sample, and C-A their difference."""
        params = dict(win=3600, step=1800, start_at_midnight=True)
        result = lf_series(lf_bands, method=list(LF_METHOD_BANDS), **params)
        for method in LF_METHOD_BANDS:
            for indicator, weighting in (('LA', _A_WEIGHTING),
                                        ('LC', _C_WEIGHTING)):
                expected = series(weighted_band_sum(lf_bands, method, weighting),
                                  **params)['Leq']
                assert result.index.equals(expected.index)
                assert_close(result[(indicator, method)].to_numpy(),
                            expected.to_numpy(), 1e-9)
            assert_close(result[('C-A', method)].to_numpy(),
                        (result[('LC', method)] - result[('LA', method)])
                        .to_numpy(), 0)

    @pytest.mark.filterwarnings(
        "ignore::noisemonitor.util.core.IntegrationTimeWarning")  # periodic
    def test_lf_periodic_levels(self, lf_bands):
        """Windows pooled over the days: LA and LC are the Leq of
        profile.periodic applied to the weighted band sums of each sample."""
        params = dict(hour1=22, hour2=5, day1='Tuesday', day2='Wednesday',
                    win=1800, step=900)
        result = lf_periodic(lf_bands, method=['montano', 'barcelona'], **params)
        for method in ('montano', 'barcelona'):
            for indicator, weighting in (('LA', _A_WEIGHTING),
                                        ('LC', _C_WEIGHTING)):
                expected = periodic(weighted_band_sum(lf_bands, method, weighting),
                                    **params)['Leq']
                assert result.index.equals(expected.index)
                assert_close(result[(indicator, method)].to_numpy(),
                            expected.to_numpy(), 1e-9)

    def test_lf_series_freq_series(self, lf_bands):
        """LA and LC are the weighted sums of the band Leq of freq_series
        (energetic averaging commutes with weighting and band summation)."""
        bands = [b for b in LF_BANDS if 16 <= b <= 200]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', IntegrationTimeWarning)
            leq = freq_series(lf_bands[[f'{b:g}' for b in bands]],
                            chunks=False)['Leq']
        result = lf_series(lf_bands)
        for indicator, weighting in (('LA', _A_WEIGHTING), ('LC', _C_WEIGHTING)):
            expected = 10 * np.log10(sum(np.power(10, (leq[b] + weighting[b]) / 10)
                                        for b in bands))
            assert_close(result[(indicator, 'montano')].to_numpy(),
                        expected.to_numpy(), 1e-9)


@pytest.mark.behaviour
class TestPeriodic:
    """Test cases for the periodic function."""

    def test_periodic_traffic_indicators(self, laeq1s_data):
        """Test periodic with traffic noise indicators using 1s data."""
        result = periodic(
                laeq1s_data,
                hour1=7,
                hour2=19,
                column=0,
                win=1800, 
                traffic_noise_indicators=True
            )
        
        expected_columns = ['Leq', 'L10', 'L50', 'L90', 'TNI', 'NPL']
        for col in expected_columns:
            assert col in result.columns
        
        if len(result) > 0:
            # Allow some tolerance
            assert result['TNI'].between(0, 150).all()
            assert (result['NPL'] >= result['Leq']).all()

    def test_periodic_roughness_indicators(self, laeq1s_data):
        """Test periodic with roughness indicators using 1s data."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = periodic(
                laeq1s_data,
                hour1=8,
                hour2=18,
                column=0, 
                win=1800, 
                roughness_indicators=True
            )
        
        roughness_columns = ['dLav', 'dLmax,1', 'dLmin,90']
        for col in roughness_columns:
            assert col in result.columns
        
        for col in roughness_columns:
            assert (result[col] >= 0).all()

    @pytest.mark.parametrize("func", [periodic, nne], ids=['periodic', 'nne'])
    def test_warns_with_large_interval(self, laeq1m_data, func):
        """Percentiles and noise events need data at 1 s or finer: 1 min data
        raises an IntegrationTimeWarning."""
        with pytest.warns(IntegrationTimeWarning, match="integration time"):
            func(laeq1m_data, hour1=10, hour2=16, column=0, win=3600)


@pytest.mark.behaviour
class TestSeries:
    """Test cases for the series function."""

    def test_series_warns_with_large_interval(self, laeq1m_data):
        """Test that series warns when using large time intervals."""
        with pytest.warns(IntegrationTimeWarning, match="Computing the L10, L50"):
            series(
                laeq1m_data,
                win=1800,
                column=0
            )


@pytest.mark.behaviour
class TestNne:
    """Test cases for the nne (Number of Noise Events) function."""

    @pytest.mark.slow
    def test_nne_basic(self, laeq1s_data):
        """Test NNE with different background level types."""
        background_types = ['Leq', 'L50', 'L90']
                
        for bg_type in background_types:
            result = nne(
                laeq1s_data,
                hour1=0,
                hour2=23,
                background_type=bg_type,
                exceedance=5,
                min_gap=3,
                win=1800,
                column=0
            )
            
            assert isinstance(result, pd.DataFrame)
            assert 'Average NNEs' in result.columns
            assert len(result) > 0
            assert (result['Average NNEs'] >= 0).all()

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_nne_insufficient_data(self, laeq1s_data):
        """Windows without data on any day are NaN, without a numpy
        warning about an empty mean."""
        subset_data = laeq1s_data[:7200]  # 2 hours of data
        
        result = nne(
            subset_data,
            hour1=0,
            hour2=23,
            background_type='Leq',
            exceedance=5,
            min_gap=3,
            win=1800,
            column=0
        )

        assert isinstance(result, pd.DataFrame)
        assert result.iloc[:4].notna().all().all()  # data from 00:00 to 02:00
        assert result.iloc[4:].isna().all().all()

    def test_nne_constant_threshold(self, laeq1s_data):
        """Test NNE with constant threshold value."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = nne(
                laeq1s_data,
                hour1=0,
                hour2=23,
                background_type=60, 
                min_gap=3,
                win=1800,
                column=0
            )
        
        assert isinstance(result, pd.DataFrame)
        assert 'Average NNEs' in result.columns
        assert len(result) > 0
        assert (result['Average NNEs'] >= 0).all()

    def test_nne_invalid_background_type(self, laeq1s_data):
        """Test NNE with invalid background type."""
        with pytest.raises(ValueError, match="Invalid background type"):
            nne(
                laeq1s_data,
                hour1=0,
                hour2=23,
                background_type='invalid',
                win=900,
                column=0
            )


@pytest.mark.behaviour
class TestProfileEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe(self):
        """Test profile functions with empty DataFrame."""
        empty_df = pd.DataFrame(
            {'Leq': []}, 
            index=pd.DatetimeIndex([])
        )
        
        with pytest.raises((ValueError, IndexError)):
            series(empty_df, win=300)

    def test_insufficient_data(self, laeq1m_data):
        """Test with insufficient data for window size."""
        small_subset = laeq1m_data.iloc[:5]
        
        with pytest.warns(
            IntegrationTimeWarning, match="Computing the L10, L50, and L90"
        ):
            with pytest.raises(
                ValueError, match="Insufficient data: need at least"
            ):
                series(small_subset, win=600, column=0)

    @pytest.mark.parametrize("func, hour1, hour2, kwargs", [
        (periodic, 0, 24, {}),
        (periodic, -1, 5, {}),
        (periodic, 7.5, 9, {}),
        (nne, 0, 25, {}),
        (nne, 5, 5, {}),
        (nne, 0, 1, {'win': 7200}),
    ], ids=['periodic_24', 'periodic_negative', 'periodic_float', 'nne_25',
            'nne_empty_range', 'nne_win_too_long'])
    def test_invalid_hours(self, laeq1s_data, func, hour1, hour2, kwargs):
        """Hours outside 0-23 for periodic (hour2 starts the last window) or
        0-24 for nne (hour2 ends the last window), non-integer hours, and
        ranges that hold no window raise a ValueError."""
        with pytest.raises(ValueError, match="hour"):
            func(laeq1s_data, hour1, hour2, column=0, **kwargs)

    def test_invalid_column_index(self, laeq1m_data):
        """Test with invalid column index."""
        with pytest.raises((IndexError, KeyError)):
            with pytest.warns(IntegrationTimeWarning, match="Computing the L10, L50"):
                series(laeq1m_data, win=300, column=99)


@pytest.mark.behaviour
class TestCoverageCheck:
    """Tests for coverage_check functionality across profile functions."""

    def test_periodic_coverage_check(self, data_with_time_gaps):
        """Test periodic function with coverage_check enabled."""
        with pytest.warns(CoverageWarning, match="Insufficient data coverage detected"):
            result = periodic(
                data_with_time_gaps,
                hour1=18,
                hour2=23,
                column=0,
                win=3600,  # 1-hour windows
                coverage_check=True
            )
        
        assert len(result) == 6
        
        assert not pd.isna(result.iloc[0]['Leq'])
        assert not pd.isna(result.iloc[0]['L10'])
        assert not pd.isna(result.iloc[0]['L50'])
        assert not pd.isna(result.iloc[0]['L90'])
        
        for i in range(1, 5):
            assert pd.isna(result.iloc[i]['Leq']), f"Expected NaN at index {i} (hour {19+i-1})"
            assert pd.isna(result.iloc[i]['L10']), f"Expected NaN at index {i}"
            assert pd.isna(result.iloc[i]['L50']), f"Expected NaN at index {i}"
            assert pd.isna(result.iloc[i]['L90']), f"Expected NaN at index {i}"
        
        assert len(result) == 6

    @pytest.mark.slow
    def test_freq_periodic_coverage_check(self, octave_data_with_gaps):
        """Test freq_periodic function with coverage_check enabled."""
        with pytest.warns(CoverageWarning, match="Insufficient data coverage detected"):
            result = freq_periodic(
                octave_data_with_gaps,
                hour1=0,
                hour2=5,
                win=3600,
                chunks=True,
                coverage_check=True,
                coverage_threshold=0.5
            )

        assert isinstance(result.columns, pd.MultiIndex)
        assert result.columns.names == ["Indicator", "Frequency Band"]
        assert len(result) > 0
        
        target_time = time(2, 30, 0)
        matching_indices = [i for i, t in enumerate(result.index) if t == target_time]
        
        if matching_indices:
            idx = matching_indices[0]
            nan_count = 0
            for band in ['63', '125', '250', '500', '1000', '2000', '4000', '8000']:
                if ('Leq', float(band)) in result.columns:
                    if pd.isna(result.iloc[idx][('Leq', float(band))]):
                        nan_count += 1
            assert nan_count > 0, "Expected some NaN values for bands at hour 2 due to insufficient coverage"

    @pytest.mark.slow
    def test_freq_series_coverage_check(self, octave_data_with_gaps):
        """Test freq_series function with coverage_check enabled."""
        with pytest.warns(CoverageWarning, match="Insufficient data coverage detected"):
            result = freq_series(
                octave_data_with_gaps,
                win=3600,
                step=1800,
                chunks=True,
                coverage_check=True,
                coverage_threshold=0.5
            )
        
        assert isinstance(result.columns, pd.MultiIndex)
        assert result.columns.names == ["Indicator", "Frequency Band"]
        assert len(result) > 0
        
        has_nans = False
        for band in ['63', '125', '250', '500', '1000', '2000', '4000', '8000']:
            if ('Leq', float(band)) in result.columns:
                if result[('Leq', float(band))].isna().any():
                    has_nans = True
                    break
        assert has_nans, "Expected some NaN values due to insufficient coverage in gaps"

    def test_series_coverage_check(self, data_with_time_gaps):
        """Test series function with coverage_check enabled."""
        with pytest.warns(CoverageWarning, match="Insufficient data coverage detected"):
            result = series(
                data_with_time_gaps,
                win=3600,
                step=1800,
                column=0,
                coverage_check=True,
                coverage_threshold=0.5
            )
        
        assert 'Leq' in result.columns
        assert 'L10' in result.columns
        assert 'L50' in result.columns
        assert 'L90' in result.columns
        assert len(result) > 0
        
        evening_results = result[(result.index.hour >= 19) & (result.index.hour < 23)]
        
        if len(evening_results) > 0:
            assert evening_results['Leq'].isna().any(), \
                "Expected some NaN values in evening hours due to insufficient coverage"

@pytest.mark.behaviour
class TestLowFrequency:
    """Test cases for the lf_series and lf_periodic functions."""

    def test_columns(self, lf_bands):
        """Columns (Indicator, Method), methods in the order given, no flag."""
        for result in (lf_series(lf_bands, method=['barcelona', 'montano']),
                    lf_periodic(lf_bands, 0, 23,
                                method=['barcelona', 'montano'])):
            assert result.columns.names == ['Indicator', 'Method']
            assert result.columns.tolist() == [
                (indicator, method) for indicator in ('LA', 'LC', 'C-A')
                for method in ('barcelona', 'montano')]
        assert len(lf_periodic(lf_bands, 0, 23)) == 24
        assert len(lf_series(lf_bands)) == 72

    def test_no_integration_time_warning(self, lf_bands):
        """Only Leq is computed: no warning on 1 min data."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            lf_series(lf_bands, method='botsford')
            lf_periodic(lf_bands, 0, 23)
        assert caught == []

    def test_coverage_check(self, lf_bands):
        """Windows below the coverage threshold are NaN, with a single
        CoverageWarning for each call."""
        df = lf_bands.copy()
        df.loc[df.index.hour == 2] = np.nan
        for function, args in ((lf_series, ()), (lf_periodic, (0, 23))):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result = function(df, *args, method=['montano', 'barcelona'],
                                coverage_check=True)
            assert [w.category for w in caught] == [CoverageWarning]
            assert result.isna().any(axis=1).sum() == (3 if function is lf_series
                                                    else 1)

    def test_invalid_hours(self, lf_bands):
        with pytest.raises(ValueError, match="hour2 must be an integer"):
            lf_periodic(lf_bands, 0, 24)



if __name__ == "__main__":
    pytest.main([__file__])