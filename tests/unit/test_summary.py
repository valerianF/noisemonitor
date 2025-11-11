"""Unit tests for summary module."""

import pytest
import pandas as pd
import numpy as np
import warnings
from datetime import time, datetime
from pathlib import Path

import noisemonitor as nm
from noisemonitor.summary import (
    harmonica_periodic, periodic, freq_periodic, lden, leq, 
    freq_descriptors, nday
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


class TestHarmonicaPeriodic:
    """Test cases for the harmonica_periodic function."""
    
    def test_harmonica_periodic(self, laeq1s_data):
        """Test basic harmonica periodic computation with exact expected values."""
        result = harmonica_periodic(
            laeq1s_data, 
            column=0,
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 24 
        assert all(isinstance(idx, time) for idx in result.index)
        
        expected_columns = ['BGN', 'EVT', 'HARMONICA']
        for col in expected_columns:
            assert col in result.columns
        
        # Test exact values for first three hours from dataset
        assert abs(result.iloc[0]['BGN'] - 0.496360) < 1e-5
        assert abs(result.iloc[0]['EVT'] - 2.635803) < 1e-5  
        assert abs(result.iloc[0]['HARMONICA'] - 3.132164) < 1e-5
        
        assert abs(result.iloc[1]['BGN'] - 0.339076) < 1e-5
        assert abs(result.iloc[1]['EVT'] - 2.375998) < 1e-5
        assert abs(result.iloc[1]['HARMONICA'] - 2.715074) < 1e-5
        
        assert abs(result.iloc[2]['BGN'] - 0.326900) < 1e-5
        assert abs(result.iloc[2]['EVT'] - 2.265118) < 1e-5
        assert abs(result.iloc[2]['HARMONICA'] - 2.592018) < 1e-5
    
    def test_harmonica_1m(self, laeq1m_data):
        """Test harmonica periodic raises error for intervals > 1s."""
        with pytest.raises(ValueError, match="Computing the HARMONICA indicator requires"):
            harmonica_periodic(
                laeq1m_data, 
                column=0
            )
    
    def test_harmonica_periodic_no_chunks(self, laeq1s_data):
        """Test harmonica periodic without chunks."""
        result = harmonica_periodic(laeq1s_data, column=0, use_chunks=False)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 24


class TestPeriodic:
    """Test cases for the periodic function."""
    
    def test_periodic_daily_exact_values(self, laeq1s_data):
        """Test daily periodic with exact expected values from dataset."""
        result = periodic(laeq1s_data, freq='D', column=0, values=True)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1  # At least one day of data
        expected_columns = ['Leq,24h', 'Lden', 'Lday', 'Levening', 'Lnight']
        for col in expected_columns:
            assert col in result.columns
        
        # Test exact values from dataset (first day)
        first_day = result.iloc[0] 
        assert abs(first_day['Leq,24h'] - 49.736828) < 1e-5
        assert abs(first_day['Lden'] - 54.72) < 0.01
        assert abs(first_day['Lday'] - 49.66) < 0.01
        assert abs(first_day['Levening'] - 53.02) < 0.01
        assert abs(first_day['Lnight'] - 46.38) < 0.01
    
    def test_periodic_daily(self, laeq1m_data):
        """Test daily periodic with individual day/evening/night values."""
        result = periodic(laeq1m_data, freq='D', column=0, values=True)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        expected_columns = ['Leq,24h', 'Lden', 'Lday', 'Levening', 'Lnight']
        for col in expected_columns:
            assert col in result.columns
        
        valid_mask = result['Leq,24h'].notna() & result['Lden'].notna()
        if valid_mask.any():
            assert (result.loc[valid_mask, 'Leq,24h'] <= \
                    result.loc[valid_mask, 'Lden']).all()
    
    def test_periodic_weekly(self, laeq1m_data):
        """Test weekly periodic computation."""
        result = periodic(laeq1m_data, freq='W', column=0)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert isinstance(result.index, pd.DatetimeIndex)
        
        expected_columns = ['Leq,24h', 'Lden']
        for col in expected_columns:
            assert col in result.columns
    
    def test_periodic_monthly(self, laeq1m_data):
        """Test monthly periodic computation."""
        result = periodic(laeq1m_data, freq='M', column=0)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert isinstance(result.index, pd.DatetimeIndex)
    
    def test_periodic_invalid_frequency(self, laeq1m_data):
        """Test periodic with invalid frequency."""
        with pytest.raises(ValueError, match="Invalid frequency"):
            periodic(laeq1m_data, freq='X', column=0)


class TestFreq:
    """Test cases for the frequency bands function."""
    
    def test_freq_periodic(self, sample_octave_data):
        """Test frequency periodic with individual day/evening/night values."""
        result = freq_periodic(sample_octave_data, freq='D', values=True)
        
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.columns, pd.MultiIndex)
        
        indicators = result.columns.get_level_values(0)
        expected_indicators = ['Leq,24h', 'Lden', 'Lday', 'Levening', 'Lnight']
        for indicator in expected_indicators:
            assert indicator in indicators
            
        frequency_bands = result.columns.get_level_values(1)
        octave_bands = ['63', '125', '250', '500', '1000', '2000', '4000', '8000']
        for band in octave_bands:
            assert band in frequency_bands
            
        band = '63'

        valid_mask = result[('Leq,24h', band)].notna() & result[('Lden', band)].notna()
        assert (result.loc[valid_mask, ('Leq,24h', band)] <= 
            result.loc[valid_mask, ('Lden', band)]).all()
        
    def test_freq_descriptors(self, sample_octave_data):
        """Test basic frequency descriptors computation."""
        with pytest.warns(
            UserWarning, 
            match="Computing the L10, L50, and L90"
            ):
            result = freq_descriptors(
                sample_octave_data, 
                stats=True,
                hour1=6,
                hour2=22
            )
        
        assert isinstance(result, pd.DataFrame)

        expected_bands = ['63', '125', '250', '500', '1000', '2000', 
                        '4000', '8000']
        for band in expected_bands:
            assert band in result.columns

        expected_indicators = ['Leq', 'Lden', 'L10', 'L50', 'L90', 'Lday', 
                            'Levening', 'Lnight']
        for indicator in expected_indicators:
            assert indicator in result.index

class TestLevels:
    """Test cases for the levels (Lden and Leq) functions."""
    
    def test_lden(self, laeq1s_data):
        """Test basic Lden computation with exact expected values."""
        # Use 1s data for exact value testing
        result = lden(laeq1s_data, column=0)
        
        assert isinstance(result, pd.DataFrame)
        assert 'lden' in result
        assert 'lday' in result
        assert 'levening' in result
        assert 'lnight' in result
        
        # Test exact values from dataset
        assert abs(result['lden'][0] - 54.72) < 0.01
        assert abs(result['lday'][0] - 49.66) < 0.01  
        assert abs(result['levening'][0] - 53.02) < 0.01
        assert abs(result['lnight'][0] - 46.38) < 0.01
    
    def test_lden_1m_data(self, laeq1m_data):
        """Test Lden computation with 1m data (original generic test).""" 
        result = lden(
            laeq1m_data, 
            column=0, 
            day1='Monday', 
            day2='Friday'
            )
        
        assert isinstance(result, pd.DataFrame)
        assert 'lden' in result
        assert 'lday' in result
        assert 'levening' in result
        assert 'lnight' in result
        
        lden_value = result['lden'][0]
        assert 30 <= lden_value <= 90
    
    def test_leq(self, laeq1s_data):
        """Test basic Leq computation with exact expected values."""
        # Test day period (7-22h) with exact values
        result_day = leq(laeq1s_data, hour1=7, hour2=22, column=0)
            
        assert isinstance(result_day, pd.DataFrame)
        
        expected_columns = ['leq', 'l10', 'l50', 'l90']
        for col in expected_columns:
            assert col in result_day.columns
        
        # Test exact values for day period from dataset
        assert abs(result_day['leq'][0] - 50.65) < 0.01
        assert abs(result_day['l10'][0] - 52.89) < 0.01
        assert abs(result_day['l50'][0] - 48.59) < 0.01 
        assert abs(result_day['l90'][0] - 44.89) < 0.01
        
        # Test night period (22-7h) 
        result_night = leq(laeq1s_data, hour1=22, hour2=7, column=0)
            
        # Test exact values for night period from dataset
        assert abs(result_night['leq'][0] - 47.6) < 0.01
        assert abs(result_night['l10'][0] - 49.29) < 0.01
        assert abs(result_night['l50'][0] - 44.89) < 0.01
        assert abs(result_night['l90'][0] - 41.79) < 0.01
        
        # Verify ordering relationships
        assert (result_day['l10'] >= result_day['l50']).all()
        assert (result_day['l50'] >= result_day['l90']).all()

class TestNday:
    """Test cases for the nday function."""
    
    def test_nday(self, laeq1m_data):
        """Test basic nday computation."""
        custom_bins = [45, 55, 65, 75]
        result, bins = nday(laeq1m_data, bins=custom_bins, column=0)
        
        assert isinstance(result, pd.DataFrame)
        assert bins == custom_bins
        assert len(result) == len(custom_bins) + 1 
        
        expected_columns = ['Decibel Range', 'Number of Days']
        for col in expected_columns:
            assert col in result.columns
        
        assert (result['Number of Days'] >= 0).all()
        assert result['Number of Days'].dtype in ['int64', 'int32']
    
    def test_nday_invalid_indicator(self, laeq1m_data):
        """Test nday with invalid indicator."""
        with pytest.raises(ValueError, match="Invalid indicator"):
            nday(laeq1m_data, indicator='InvalidIndicator', column=0)

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

if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__])