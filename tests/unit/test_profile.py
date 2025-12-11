"""Unit tests for profile module."""

import pytest
import pandas as pd
import numpy as np
import warnings
import os
from datetime import time, datetime
from pathlib import Path

import noisemonitor as nm
from noisemonitor.profile import (
    periodic, series, nne, freq_periodic, freq_series
)
from noisemonitor.util.core import CoverageWarning


@pytest.fixture(scope="module")
def test_data_paths():
    """Provide paths to the test data files."""
    base_path = Path(__file__).parent.parent / "data"
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


class TestPeriodic:
    """Test cases for the periodic function."""
    
    def test_periodic_exact_values(self, laeq1s_data):
        """Test periodic with exact expected values from dataset."""
        # Test full day profile (0-23h) with 1-hour windows
        result = periodic(
            laeq1s_data,
            hour1=0, 
            hour2=23,
            column=0,
            win=3600 
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 24  # 24 hourly values
        
        expected_columns = ['Leq', 'L10', 'L50', 'L90']
        for col in expected_columns:
            assert col in result.columns
        
        # Test exact values for first three hours from dataset
        # Hour 0 (00:30:00)
        assert abs(result.iloc[0]['Leq'] - 45.164192) < 1e-2
        assert abs(result.iloc[0]['L10'] - 46.485907) < 1e-2
        assert abs(result.iloc[0]['L50'] - 44.285907) < 1e-2
        assert abs(result.iloc[0]['L90'] - 43.185907) < 1e-2
        
        # Hour 1 (01:30:00)
        assert abs(result.iloc[1]['Leq'] - 43.236100) < 1e-2
        assert abs(result.iloc[1]['L10'] - 44.685907) < 1e-2
        assert abs(result.iloc[1]['L50'] - 42.685907) < 1e-2
        assert abs(result.iloc[1]['L90'] - 41.685907) < 1e-2
        
        # Hour 2 (02:30:00)
        assert abs(result.iloc[2]['Leq'] - 42.633263) < 1e-2
        assert abs(result.iloc[2]['L10'] - 44.185907) < 1e-2
        assert abs(result.iloc[2]['L50'] - 41.985907) < 1e-2
        assert abs(result.iloc[2]['L90'] - 41.085907) < 1e-2
        
        # Verify ordering relationships hold
        assert (result['L10'] >= result['L50']).all()
        assert (result['L50'] >= result['L90']).all()
    
    def test_periodic_basic_daytime(self, laeq1m_data):
        """Test basic daytime periodic averaging with 1-minute data."""

        with pytest.warns(UserWarning) as record:
            result = periodic(
                laeq1m_data,
                hour1=6, 
                hour2=22,
                column=0,
                win=3600 
            )
        
        # Check for L10/L50 warning
        warning_messages = [str(w.message) for w in record]
        assert any("Computing the L10, L50" in msg for msg in warning_messages)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        expected_columns = ['Leq', 'L10', 'L50', 'L90']
        for col in expected_columns:
            assert col in result.columns
        
        assert all(isinstance(idx, time) for idx in result.index)
        
        assert (result['L10'] >= result['L50']).all()
        assert (result['L50'] >= result['L90']).all()
        assert result['Leq'].between(30, 90).all()
    
    def test_periodic_nighttime_exact_values(self, laeq1s_data):
        """Test nighttime periodic with exact expected values from dataset."""

        result = periodic(
            laeq1s_data,
            hour1=22, 
            hour2=6,
            column=0,
            win=3600  
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 9  # Hours 22, 23, 0, 1, 2, 3, 4, 5, 6
        
        expected_columns = ['Leq', 'L10', 'L50', 'L90']
        for col in expected_columns:
            assert col in result.columns
        
        # Test exact values for first two night hours from dataset
        # Hour 22 (22:30:00) - first entry in night profile
        assert abs(result.iloc[0]['Leq'] - 52.327007) < 1e-2
        assert abs(result.iloc[0]['L10'] - 55.185907) < 1e-2
        assert abs(result.iloc[0]['L50'] - 49.385907) < 1e-2
        assert abs(result.iloc[0]['L90'] - 46.585907) < 1e-2
        
        # Hour 23 (23:30:00) - second entry in night profile
        assert abs(result.iloc[1]['Leq'] - 51.256863) < 1e-2
        assert abs(result.iloc[1]['L10'] - 53.885907) < 1e-2
        assert abs(result.iloc[1]['L50'] - 47.185907) < 1e-2
        assert abs(result.iloc[1]['L90'] - 44.085907) < 1e-2
        
        # Verify ordering relationships hold
        assert (result['L10'] >= result['L50']).all()
        assert (result['L50'] >= result['L90']).all()
    
    def test_periodic_nighttime(self, laeq1m_data):
        """Test nighttime periodic averaging (hour2 < hour1)."""
        with pytest.warns(UserWarning) as record:
            result = periodic(
                laeq1m_data,
                hour1=22,
                hour2=6,
                column=0,
                win=3600
            )
        
        # Check for L10/L50 warning
        warning_messages = [str(w.message) for w in record]
        assert any("Computing the L10, L50" in msg for msg in warning_messages)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        if len(result) > 0:
            night_hours = [22, 23, 0, 1, 2, 3, 4, 5, 6]
            result_hours = [idx.hour for idx in result.index]
            assert any(hour in night_hours for hour in result_hours)
    
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
    
    def test_periodic_sliding_windows(self, laeq1m_data):
        """Test periodic with sliding windows (step < win)."""
        with pytest.warns(UserWarning) as record:
            result = periodic(
                laeq1m_data,
                hour1=9,
                hour2=17,
                column=0,
                win=3600,
                step=1800
            )
        
        # Check for L10/L50 warning
        warning_messages = [str(w.message) for w in record]
        assert any("Computing the L10, L50" in msg for msg in warning_messages)
        
        with pytest.warns(UserWarning) as record2:
            result_no_overlap = periodic(
                laeq1m_data,
                hour1=9,
                hour2=17, 
                column=0,
                win=3600
            )
        
        # Check for warnings in second call too
        warning_messages2 = [str(w.message) for w in record2]
        assert any("Computing the L10, L50" in msg for msg in warning_messages2)
        
        assert len(result) >= len(result_no_overlap)
    
    def test_periodic_warns_with_large_interval(self, laeq1m_data):
        """Test that periodic warns when using large time intervals for percentiles."""
        with pytest.warns(UserWarning) as record:
            periodic(
                laeq1m_data,
                hour1=10,
                hour2=16,
                column=0,
                win=3600
            )
        
        # Check for L10/L50/L90 warning
        warning_messages = [str(w.message) for w in record]
        assert any("Computing the L10, L50, L90" in msg for msg in warning_messages)
    
    def test_periodic_stat_single_value(self, laeq1m_data):
        """Test periodic with single stat parameter value."""
        with pytest.warns(UserWarning, match="Computing the L10, L50"):
            result = periodic(
                laeq1m_data,
                hour1=7,
                hour2=19,
                column=0,
                win=3600,
                stat=25
            )
        
        assert isinstance(result, pd.DataFrame)
        assert 'L25' in result.columns
        
        # Check that default columns are still present
        expected_columns = ['Leq', 'L10', 'L50', 'L90', 'L25']
        for col in expected_columns:
            assert col in result.columns
        
        # Test exact value for first hour (07:30:00)
        assert abs(result.iloc[0]['L25'] - 53.961804) < 1e-5
        
        # Verify ordering relationships
        assert (result['L10'] >= result['L25']).all()
        assert (result['L25'] >= result['L50']).all()
    
    def test_periodic_stat_multiple_values(self, laeq1m_data):
        """Test periodic with multiple stat parameter values."""
        with pytest.warns(UserWarning, match="Computing the L10, L50"):
            result = periodic(
                laeq1m_data,
                hour1=7,
                hour2=19,
                column=0,
                win=3600,
                stat=[1, 5, 95, 99]
            )
        
        assert isinstance(result, pd.DataFrame)
        
        # Check that custom stat columns are present
        custom_columns = ['L1', 'L5', 'L95', 'L99']
        for col in custom_columns:
            assert col in result.columns
        
        # Check that default columns are still present
        default_columns = ['Leq', 'L10', 'L50', 'L90']
        for col in default_columns:
            assert col in result.columns
        
        # Test exact values for first hour (07:30:00)
        assert abs(result.iloc[0]['L1'] - 61.067000) < 1e-5
        assert abs(result.iloc[0]['L5'] - 57.027961) < 1e-5
        assert abs(result.iloc[0]['L95'] - 44.373262) < 1e-5
        assert abs(result.iloc[0]['L99'] - 41.830812) < 1e-5
        
        # Verify ordering relationships
        assert (result['L1'] >= result['L5']).all()
        assert (result['L5'] >= result['L10']).all()
        assert (result['L50'] >= result['L90']).all()
        assert (result['L90'] >= result['L95']).all()
        assert (result['L95'] >= result['L99']).all()

class TestSeries:
    """Test cases for the series function."""
    
    def test_series_basic(self, laeq1m_data):
        """Test basic time series calculation."""
        with pytest.warns(UserWarning, match="Computing the L10, L50"):
            result = series(
                laeq1m_data,
                win=1800,
                column=0
            )
        

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        expected_columns = ['Leq', 'L10', 'L50', 'L90']
        for col in expected_columns:
            assert col in result.columns
        
        assert isinstance(result.index, pd.DatetimeIndex)
        assert (result['L10'] >= result['L50']).all()
        assert (result['L50'] >= result['L90']).all()
        assert len(result) < len(laeq1m_data)
    
    def test_series_sliding_windows(self, laeq1m_data):
        """Test series with sliding windows."""
        with pytest.warns(UserWarning, match="Computing the L10, L50"):
            result = series(
                laeq1m_data,
                win=1800,
                step=600,
                column=0
            )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        with pytest.warns(UserWarning, match="Computing the L10, L50"):
            result_no_overlap = series(
                laeq1m_data,
                win=1800,
                column=0
            )
        
        assert len(result) >= len(result_no_overlap)
    
    def test_series_start_at_midnight(self, laeq1m_data):
        """Test series starting at midnight."""
        win_seconds = 3600  # 1 hour window
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = series(
                laeq1m_data,
                win=win_seconds,
                start_at_midnight=True,
                column=0
            )
        
        if len(result) > 0:
            first_time = result.index[0]
            
            data_start = laeq1m_data.index[0]
            expected_midnight = data_start.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            
            if data_start > expected_midnight:
                expected_midnight += pd.Timedelta(days=1)
            
            expected_first_time = expected_midnight + pd.Timedelta(seconds=win_seconds/2)
            
            assert first_time == expected_first_time
    
    def test_series_warns_with_large_interval(self, laeq1m_data):
        """Test that series warns when using large time intervals."""
        with pytest.warns(UserWarning, match="Computing the L10, L50"):
            series(
                laeq1m_data,
                win=1800,
                column=0
            )
    
    def test_series_handles_nan_data(self, laeq1m_data):
        """Test series handling of NaN data."""
        test_data = laeq1m_data.copy()
        test_data.iloc[100:110, 0] = np.nan
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = series(
                test_data,
                win=900,
                column=0
            )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestNne:
    """Test cases for the nne (Number of Noise Events) function."""
    
    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
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
    
    def test_nne_insufficient_data(self, laeq1s_data):
        """Test that NNE handles insufficient daily data gracefully."""
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


class TestFreqPeriodic:
    """Test cases for the freq_periodic function."""
    
    def test_freq_periodic_basic(self, sample_octave_data):
        """Test basic frequency-domain periodic analysis."""
        with pytest.warns(
            UserWarning, match="Computing the L10, L50, L90"
        ):
            with pytest.warns(
                RuntimeWarning, match="Mean of empty slice"
            ):
                result = freq_periodic(
                    sample_octave_data,
                    hour1=8,
                    hour2=18,
                    win=1800,
                    chunks=False
                )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert isinstance(result.columns, pd.MultiIndex)
        
        octave_bands = [
            '63', '125', '250', '500', '1000', '2000', '4000', '8000'
        ]
        for band in octave_bands:
            assert float(band) in result.columns.levels[1]
    
    def test_freq_periodic_with_chunks(self, sample_octave_data):
        """Test frequency periodic with parallel processing."""
        result = freq_periodic(
            sample_octave_data,
            hour1=6,
            hour2=20,
            win=1800,
            chunks=True
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert isinstance(result.columns, pd.MultiIndex)


class TestFreqSeries:
    """Test cases for the freq_series function."""
    
    def test_freq_series_basic(self, sample_octave_data):
        """Test basic frequency-domain time series."""
        with pytest.warns(
            UserWarning, match="Computing the L10, L50"
        ):
            result = freq_series(
                sample_octave_data,
                win=1800,
                chunks=False
            )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert isinstance(result.columns, pd.MultiIndex)
        assert isinstance(result.index, pd.DatetimeIndex)
        
        octave_bands = ['63', '125', '250', '500', '1000', '2000', '4000', '8000']
        for band in octave_bands:
            assert float(band) in result.columns.levels[1]
    
    def test_freq_series_sliding_windows(self, sample_octave_data):
        """Test frequency series with sliding windows."""
        with pytest.warns(
            UserWarning, match="Computing the L10, L50"
        ):
            result = freq_series(
                sample_octave_data,
                win=1200,
                step=600, 
                chunks=False
            )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        with pytest.warns(
            UserWarning, match="Computing the L10, L50"
        ):
            result_no_overlap = freq_series(
                sample_octave_data,
                win=1200,
                chunks=False
            )
        
        assert len(result) >= len(result_no_overlap)

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
            UserWarning, match="Computing the L10, L50, and L90"
        ):
            with pytest.raises(
                ValueError, match="Insufficient data: need at least"
            ):
                series(small_subset, win=600, column=0)
    
    def test_invalid_column_index(self, laeq1m_data):
        """Test with invalid column index."""
        with pytest.raises((IndexError, KeyError)):
            with pytest.warns(UserWarning, match="Computing the L10, L50"):
                series(laeq1m_data, win=300, column=99)

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

if __name__ == "__main__":
    pytest.main([__file__])