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


@pytest.fixture
def sample_octave_data():
    """Create sample octave band data for frequency analysis tests."""
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    index = pd.date_range(start=start_time, periods=100, freq='1min')
    
    np.random.seed(42) 
    octave_bands = ['63', '125', '250', '500', '1000', '2000', '4000', '8000']
    
    data = {}
    for i, band in enumerate(octave_bands):
        base_level = 45 + i * 2 
        variation = np.random.normal(0, 3, len(index))
        data[band] = base_level + variation
    
    return pd.DataFrame(data, index=index)


class TestPeriodic:
    """Test cases for the periodic function."""
    
    def test_periodic_exact_values(self, laeq1s_data):
        """Test periodic with exact expected values from dataset."""
        # Test full day profile (0-23h) with 1-hour windows
        # Expect coverage warnings for incomplete hourly windows
        with pytest.warns(UserWarning, match="Coverage filter"):
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
        
        # Check for both the L10/L50 warning and coverage warning
        warning_messages = [str(w.message) for w in record]
        assert any("Computing the L10, L50" in msg for msg in warning_messages)
        assert any("Coverage filter" in msg for msg in warning_messages)
        
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

        with pytest.warns(UserWarning, match="Coverage filter"):
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
        
        # Check for both the L10/L50 warning and coverage warning
        warning_messages = [str(w.message) for w in record]
        assert any("Computing the L10, L50" in msg for msg in warning_messages)
        assert any("Coverage filter" in msg for msg in warning_messages)
        
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
        
        # Check for both the L10/L50 warning and coverage warning
        warning_messages = [str(w.message) for w in record]
        assert any("Computing the L10, L50" in msg for msg in warning_messages)
        assert any("Coverage filter" in msg for msg in warning_messages)
        
        with pytest.warns(UserWarning) as record2:
            result_no_overlap = periodic(
                laeq1m_data,
                hour1=9,
                hour2=17, 
                column=0,
                win=3600
            )
        
        # Check for both warnings in second call too
        warning_messages2 = [str(w.message) for w in record2]
        assert any("Computing the L10, L50" in msg for msg in warning_messages2)
        assert any("Coverage filter" in msg for msg in warning_messages2)
        
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
        
        # Check for both the L10/L50/L90 warning and coverage warning
        warning_messages = [str(w.message) for w in record]
        assert any("Computing the L10, L50, L90" in msg for msg in warning_messages)
        assert any("Coverage filter" in msg for msg in warning_messages)


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
    def test_nne_basic(self, laeq1s_data):
        """Test NNE with different background level types."""
        background_types = ['leq', 'l50', 'l90']
                
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
        """Test that NNE raises error with insufficient daily data."""
        subset_data = laeq1s_data[:7200]  # 2 hours of data
        
        with pytest.raises(
            ValueError, match="No complete days found in the dataset"
        ):
            nne(
                subset_data,
                hour1=0,
                hour2=23,
                background_type='leq',
                exceedance=5,
                min_gap=3,
                win=1800,
                column=0
            )
    
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
        result = freq_series(
            sample_octave_data,
            win=1200,
            step=600, 
            chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
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


if __name__ == "__main__":
    pytest.main([__file__])