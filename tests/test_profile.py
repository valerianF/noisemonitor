"""Unit tests for profile module."""

import pytest
import pandas as pd
import numpy as np
import warnings
from datetime import time

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
    SOURCE_COVERAGE, TARTU_RECEIVERS, TOL_PUBLISHED, assert_close, band_frame,
    tartu_parameters, weighted_band_sum
)


def window_levels(levels):
    """Leq, L10, L50 and L90 of the samples of one window (percentiles by
    linear interpolation, as checked against Bruitparif's hourly values)."""
    return {'Leq': nm.util.core.equivalent_level(levels.to_numpy()),
            'L10': np.nanpercentile(levels, 90),
            'L50': np.nanpercentile(levels, 50),
            'L90': np.nanpercentile(levels, 10)}


@pytest.fixture(scope="module")
def nan_hour_data():
    """Three hours at 1 s from 00:00, the second one (01:00) all NaN; no
    data from 03:00."""
    index = pd.date_range('2024-01-01', periods=3 * 3600, freq='s')
    rng = np.random.default_rng(0)
    levels = rng.uniform(40, 60, len(index))
    levels[3600:7200] = np.nan
    return pd.DataFrame({'LAeq': levels}, index=index)


@pytest.fixture
def hour_start_peaks_data():
    """One day at 1 s, 40 dB except 100 dB at every hh:00:00."""
    dates = pd.date_range('2023-01-01', periods=86400, freq='s')
    peaks = (dates.minute == 0) & (dates.second == 0)
    return pd.DataFrame({'LAeq': np.where(peaks, 100.0, 40.0)}, index=dates)


@pytest.mark.external
@pytest.mark.filterwarnings("ignore::noisemonitor.util.core.CoverageWarning")
@pytest.mark.filterwarnings("ignore::noisemonitor.util.core.IntegrationTimeWarning")  # hourly data
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


@pytest.mark.external
class TestExternalTartu:
    """Hourly numbers of noise events computed by Baclet and Rumpler's
    implementation from simulated indoor levels at 1 s, at three receivers
    over one day."""

    @pytest.mark.parametrize("receiver", [
        pytest.param(pk, marks=pytest.mark.slow) for pk in TARTU_RECEIVERS])
    def test_nne(self, tartu_indoor, tartu_counts, receiver):
        """1 h windows over the whole day, (0, 24), one day: the source's
        count of each hour, exactly, for its 128 indicators (fixed, Leq, L50
        and L90 thresholds, exceedances of 3 to 15 dB, minimum gaps of 1 to
        30 s, window open or closed)."""
        expected = tartu_counts.loc[receiver]
        assert list(expected.index) == list(range(24))

        mismatches = []
        for indicator in expected.columns:
            kwargs, window = tartu_parameters(indicator)
            result = nne(tartu_indoor[receiver, window], 0, 24, win=3600,
                        column=0, **kwargs)
            assert list(result.index) == [time(h, 30) for h in range(24)]
            mismatches += [(indicator, hour, got, ref) for hour, (got, ref)
                        in enumerate(zip(result.iloc[:, 0], expected[indicator]))
                        if got != ref]
        assert not mismatches


@pytest.mark.internal
class TestCalculations:
    """Internal validation: closed-form results on synthetic signals, or
    consistency with another noisemonitor function."""

    @pytest.mark.parametrize("min_gap", [0, 3])
    @pytest.mark.parametrize("hour2", [23, 24], ids=['to_23h', 'full_day'])
    def test_nne_windows_half_open(self, hour_start_peaks_data, hour2, min_gap):
        """Each window covers [start, start + win): the hh:00:00 event is
        counted in one window only, not also in the previous one, and is
        counted whatever min_gap, as the first exceedance of its window. The
        last window ends at hour2, so (0, 24) gives the 24 hours of the day."""
        result = nne(hour_start_peaks_data, hour1=0, hour2=hour2,
                    background_type=70, min_gap=min_gap, win=3600, column=0)

        assert list(result.index) == [time(h, 30) for h in range(hour2)]
        assert (result.iloc[:, 0] == 1).all()

    @staticmethod
    def _bursts(start, end, bursts):
        """1 s data at 40 dB from start to end (excluded), with 10 s at 80 dB
        from each burst time."""
        index = pd.date_range(start, end, freq='s', inclusive='left')
        levels = np.full(len(index), 40.0)
        for burst in bursts:
            levels[index.get_loc(pd.Timestamp(burst)) + np.arange(10)] = 80.0
        return pd.DataFrame({'LAeq': levels}, index=index)

    def test_nne_past_midnight(self):
        """Windows from 23:00 to 01:00 (1 h, step 30 min) are counted over
        each night, contiguous across midnight. Three nights, bursts at
        23:40 on the 1st, 00:15 on the 2nd and 00:45 on the 3rd: the
        [23:30, 00:30) window of the first night holds two events. Events
        per night: [23:00, 00:00) 1, 0, 0; [23:30, 00:30) 2, 0, 0;
        [00:00, 01:00) 1, 1, 0."""
        df = self._bursts('2024-01-01 23:00', '2024-01-04 01:00',
                        ['2024-01-01 23:40', '2024-01-02 00:15', '2024-01-03 00:45'])
        result = nne(df, 23, 1, background_type=60, min_gap=1, win=3600, step=1800)

        assert list(result.index) == [time(23, 30), time(0), time(0, 30)]
        assert np.allclose(result['Average NNEs'], [1 / 3, 2 / 3, 2 / 3],
                        rtol=0, atol=1e-12)

    def test_nne_day_of_night(self):
        """day1 = day2 = 'friday' with windows from 22:00 to 06:00 keeps the
        night starting on Friday (5 Jan 2024), early Saturday included, and
        not the end of Thursday's night, early Friday. Two bursts at 01:00
        and 01:20 on Saturday, one at 01:00 on Friday: two events at 01:00,
        none elsewhere."""
        df = self._bursts('2024-01-05', '2024-01-07', ['2024-01-05 01:00',
                                                    '2024-01-06 01:00',
                                                    '2024-01-06 01:20'])
        result = nne(df, 22, 6, background_type=60, min_gap=1, win=3600,
                    day1='friday', day2='friday')

        expected = [2.0 if h == 1 else 0.0 for h in [22, 23, 0, 1, 2, 3, 4, 5]]
        assert list(result['Average NNEs']) == expected

    def test_nne_window_without_data(self):
        """Without coverage check, a window without any value has no count:
        one burst at 01:10 on the first day, the 01:00 hour of the second
        day all NaN. The mean of the 01:00 window is 1 (first day only), not
        1/2."""
        df = self._bursts('2024-01-01', '2024-01-03', ['2024-01-01 01:10'])
        df.loc['2024-01-02 01:00':'2024-01-02 01:59:59', 'LAeq'] = np.nan

        with pytest.warns(CoverageWarning):
            result = nne(df, 0, 24, background_type=60, min_gap=1, win=3600,
                        coverage_check=False)

        assert result.loc[time(1, 30), 'Average NNEs'] == 1.0
        assert (result.drop(index=time(1, 30))['Average NNEs'] == 0.0).all()

    def test_nne_windows_cut_by_data_range(self):
        """Data from 00:30 on the first day to 23:30 on the second: the
        00:00 window of the first day and the 23:00 window of the second
        hold 50% of their samples, below 80%, and are left out, although
        all their present samples are valid. One burst in each: the means
        of these windows come from the other day only (0, not 1/2)."""
        df = self._bursts('2024-01-01 00:30', '2024-01-02 23:30',
                        ['2024-01-01 00:40', '2024-01-02 23:10'])

        with pytest.warns(CoverageWarning):
            result = nne(df, 0, 24, background_type=60, min_gap=1, win=3600)

        assert (result['Average NNEs'] == 0.0).all()

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

    def test_periodic_npl(self):
        """NPL = Leq + 2.56 σ (Robinson, 1971), σ the standard deviation of
        the levels of the window (sample, n - 1): 1 h windows of 1 s levels
        alternating 40 and 60 dB give Leq = 10 lg((10^4 + 10^6) / 2) and
        σ = 10 √(3600 / 3599)."""
        index = pd.date_range('2023-01-01', periods=2 * 3600, freq='s')
        df = pd.DataFrame({'LAeq': np.where(np.arange(len(index)) % 2, 60.0, 40.0)},
                        index=index)
        result = periodic(df, 0, 1, column=0, win=3600, traffic_noise_indicators=True)

        leq = 10 * np.log10((10 ** 4 + 10 ** 6) / 2)
        assert len(result) == 2
        assert_close(result['NPL'], leq + 25.6 * np.sqrt(3600 / 3599), 1e-9)

    def test_periodic_roughness_definitions(self):
        """Roughness (Defrance et al., 2010) of one hour of 1 s levels whose
        3599 differences are, in absolute value, 360 of 0 dB, 3202 of 2 dB
        and 37 of 20 dB: dLav is their mean, dLmax,1 the mean of the largest
        1 % (the 20 dB ones) and dLmin,90 the mean of the lowest 90 % (the
        0 and 2 dB ones), 2 × 3202 / 3562 dB."""
        steps = np.repeat([0.0, 2.0, 20.0], [360, 3202, 37])
        signs = np.where(np.arange(steps.size) % 2, -1.0, 1.0)
        levels = 50.0 + np.concatenate([[0.0], np.cumsum(steps * signs)])
        df = pd.DataFrame({'LAeq': levels},
                        index=pd.date_range('2023-01-02', periods=3600, freq='s'))

        result = periodic(df, 0, 0, column=0, win=3600, roughness_indicators=True)

        assert len(result) == 1
        assert_close(result['dLav'], (2 * 3202 + 20 * 37) / 3599, 1e-12)
        assert_close(result['dLmax,1'], 20.0, 1e-12)
        assert_close(result['dLmin,90'], 2 * 3202 / 3562, 1e-12)

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

    @pytest.mark.parametrize("use_chunks", [False, pytest.param(True, marks=pytest.mark.slow)],
                            ids=['no_chunks', 'chunks'])
    def test_freq_periodic_bands(self, bruitparif_bands, use_chunks):
        """Each band (column names in mixed formats, unsorted) gives the
        values of profile.periodic on its own column, under its frequency;
        bands in ascending order."""
        params = dict(win=3600, step=1800, stat=[5, 95])
        result = freq_periodic(bruitparif_bands, 0, 23, use_chunks=use_chunks,
                            **params)

        assert result.columns.names == ['Indicator', 'Frequency Band']
        assert (list(result.columns.get_level_values('Frequency Band').unique())
                == sorted(BANDS.values()))
        for name, band in BANDS.items():
            expected = periodic(bruitparif_bands, 0, 23, column=name, **params)
            pd.testing.assert_frame_equal(
                result.xs(band, axis=1, level='Frequency Band'), expected,
                check_names=False, check_exact=True, check_like=True)

    @pytest.mark.parametrize("use_chunks", [False, pytest.param(True, marks=pytest.mark.slow)],
                            ids=['no_chunks', 'chunks'])
    def test_freq_series_bands(self, bruitparif_bands, use_chunks):
        """Each band gives the values of profile.series on its own column,
        under its frequency; bands in ascending order."""
        params = dict(win=3600, step=1800)
        result = freq_series(bruitparif_bands, use_chunks=use_chunks, **params)

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
                            use_chunks=False)['Leq']
        result = lf_series(lf_bands)
        for indicator, weighting in (('LA', _A_WEIGHTING), ('LC', _C_WEIGHTING)):
            expected = 10 * np.log10(sum(np.power(10, (leq[b] + weighting[b]) / 10)
                                        for b in bands))
            assert_close(result[(indicator, 'montano')].to_numpy(),
                        expected.to_numpy(), 1e-9)


@pytest.mark.behaviour
class TestPeriodic:
    """Test cases for the periodic function."""

    @pytest.mark.parametrize("func, kwargs", [
        (periodic, dict(hour1=10, hour2=16)), (nne, dict(hour1=10, hour2=16)),
        (series, dict())], ids=['periodic', 'nne', 'series'])
    # nne: the last day ends at 10:29, so its later windows are empty
    @pytest.mark.filterwarnings("ignore::noisemonitor.util.core.CoverageWarning")
    def test_warns_with_large_interval(self, laeq1m_data, func, kwargs):
        """Percentiles and noise events need data at 1 s or finer: 1 min data
        raises an IntegrationTimeWarning."""
        with pytest.warns(IntegrationTimeWarning, match="integration time"):
            func(laeq1m_data, column=0, win=3600, **kwargs)


@pytest.mark.behaviour
class TestNne:
    """Test cases for the nne (Number of Noise Events) function."""

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

    def test_series_insufficient_data_after_midnight(self):
        """Starting at midnight, 30 min of data are left for 1 h windows:
        ValueError, not a window of half an hour."""
        index = pd.date_range('2024-01-01 20:00', '2024-01-02 00:30', freq='s',
                            inclusive='left')
        df = pd.DataFrame({'LAeq': 50.0}, index=index)

        with pytest.raises(ValueError, match="Insufficient data after midnight"):
            series(df, win=3600, column=0, start_at_midnight=True)

    def test_series_start_at_midnight_first_sample_after(self):
        """Samples at hh:mm:40 (1 min): the windows start at the first sample
        at or after midnight (00:00:40), not at the nearest one (23:59:40)."""
        index = pd.date_range('2024-01-01 22:00:40', periods=240, freq='min')
        df = pd.DataFrame({'LAeq': np.where(index < '2024-01-02', 70.0, 50.0)},
                        index=index)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', IntegrationTimeWarning)
            result = series(df, win=3600, column=0, start_at_midnight=True)

        assert (result['Leq'] == 50.0).all()

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

    @pytest.mark.parametrize("func, args", [(freq_periodic, (0, 23)), (freq_series, ())],
                            ids=['freq_periodic', 'freq_series'])
    def test_freq_broken_process_pool(self, monkeypatch, bruitparif_bands, func, args):
        """When parallel processing fails, the bands are processed one after
        the other, with a RuntimeWarning, and give the same values."""
        from concurrent.futures import Future
        from concurrent.futures.process import BrokenProcessPool

        class _BrokenPool:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def submit(self, *args, **kwargs):
                future = Future()
                future.set_exception(BrokenProcessPool('worker died'))
                return future

        monkeypatch.setattr(nm.util.core, 'ProcessPoolExecutor', _BrokenPool)
        with pytest.warns(RuntimeWarning, match="Parallel processing unavailable"):
            result = func(bruitparif_bands, *args, win=3600, use_chunks=True)

        pd.testing.assert_frame_equal(
            result, func(bruitparif_bands, *args, win=3600, use_chunks=False))

    @pytest.mark.parametrize("chunks", [pytest.param(True, marks=pytest.mark.slow), False])
    @pytest.mark.parametrize("func, args", [(freq_periodic, (0, 23)), (freq_series, ())],
                            ids=['freq_periodic', 'freq_series'])
    def test_freq_chunks_deprecated(self, bruitparif_bands, func, args, chunks):
        """The former keyword chunks still sets use_chunks, with a
        DeprecationWarning."""
        from unittest.mock import patch

        with patch.object(nm.util.core, '_map_bands',
                          wraps=nm.util.core._map_bands) as spy:
            with pytest.warns(DeprecationWarning, match="use 'use_chunks'"):
                func(bruitparif_bands[bruitparif_bands.columns[:1]], *args,
                     win=3600, use_chunks=not chunks, chunks=chunks)

        assert spy.call_args.args[2] is chunks


@pytest.mark.behaviour
class TestCoverageCheck:
    """Tests for coverage_check functionality across profile functions."""

    @pytest.mark.parametrize("coverage_check", [False, True], ids=['no_check', 'check'])
    @pytest.mark.parametrize("func, args, empty", [
        (periodic, dict(hour1=0, hour2=3), [1, 3]),
        (series, dict(), [1]),
        (nne, dict(hour1=0, hour2=4, background_type='L50'), [1, 3]),
    ], ids=['periodic', 'series', 'nne'])
    def test_windows_without_data(self, nan_hour_data, func, args, empty,
                                coverage_check):
        """Windows without any value (all NaN, or no sample) are NaN, with
        a single CoverageWarning and no numpy warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = func(nan_hour_data, win=3600, column=0,
                        coverage_check=coverage_check, **args)

        assert [w.category for w in caught] == [CoverageWarning]
        assert result.iloc[empty].isna().all().all()
        assert result.drop(index=result.index[empty]).notna().all().all()

    @pytest.mark.parametrize("use_chunks", [
        False, pytest.param(True, marks=pytest.mark.slow)], ids=['sequential', 'chunks'])
    @pytest.mark.parametrize("coverage_check", [False, True], ids=['no_check', 'check'])
    @pytest.mark.parametrize("func, args", [
        (freq_periodic, dict(hour1=0, hour2=2)), (freq_series, dict())],
        ids=['freq_periodic', 'freq_series'])
    def test_freq_windows_without_data(self, nan_hour_data, func, args,
                                    coverage_check, use_chunks):
        """Two bands, each with a window without any value (01:00): a single
        CoverageWarning per call, with or without parallel processing."""
        bands = pd.concat({'125': nan_hour_data['LAeq'],
                        '250': nan_hour_data['LAeq'] + 3}, axis=1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = func(bands, win=3600, use_chunks=use_chunks,
                        coverage_check=coverage_check, **args)

        assert [w.category for w in caught] == [CoverageWarning]
        assert result.iloc[1].isna().all()
        assert result.drop(index=result.index[1]).notna().all().all()

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