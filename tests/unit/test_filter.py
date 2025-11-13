"""Unit tests for filter module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time
from pathlib import Path

import noisemonitor as nm
from noisemonitor.util.filter import (
    all_data, extreme_values, weather_flags, 
    _get_week_indexes, _days, _hours
)


@pytest.fixture(scope="module")
def test_data_paths():
    """Provide paths to the test data files."""
    base_path = Path(__file__).parent.parent / "data"
    return {
        "laeq1m": base_path / "test_data_laeq1m.csv"
    }


@pytest.fixture(scope="module") 
def laeq1m_data(test_data_paths):
    """Load LEQ dB -A,1min test data (one week of data)."""
    df = nm.load(
        str(test_data_paths["laeq1m"]),
        datetimeindex=0,
        valueindexes=1,
        header=0,
        sep=','
    )
    return df

@pytest.fixture(scope="module")
def sample_weather_data():
    """Sample weather data."""
    start_date = datetime(2025, 3, 22, 0, 0, 0)
    index = pd.date_range(start=start_date, periods=100, freq='h')
    
    np.random.seed(123)
    df = pd.DataFrame({
        'LEQ dB -A': 50 + np.random.normal(0, 5, len(index)),
        'Wind_Spd_Flag': [True] * 10 + [False] * 90,
        'Rain_Flag_Roll': [True] * 5 + [False] * 95,
        'Temp_Flag': [True] * 3 + [False] * 97,
        'Rel_Hum_Flag': [True] * 2 + [False] * 98,
        'Snow_Flag_Roll': [True] * 1 + [False] * 99
    }, index=index)
    
    return df


class TestAllData:
    """Test cases for the all_data function."""
    
    def test_filter_within_range(self, laeq1m_data):
        """Test filtering data within a specific datetime range."""
        start_datetime = datetime(2025, 3, 21, 6, 0, 0)
        end_datetime = datetime(2025, 3, 22, 18, 0, 0)  
        
        result = all_data(laeq1m_data, start_datetime, end_datetime)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert result.index.min() >= start_datetime
        assert result.index.max() <= end_datetime
    
    def test_filter_outside_range(self, laeq1m_data):
        """Test filtering data outside a specific datetime range."""
        start_datetime = datetime(2025, 3, 22, 6, 0, 0)
        end_datetime = datetime(2025, 3, 23, 18, 0, 0)
        
        original_len = len(laeq1m_data)
        result = all_data(laeq1m_data, start_datetime, end_datetime, between=True)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) < original_len
        # Verify all returned timestamps are outside the range
        outside_range = (
            (result.index < start_datetime) |
            (result.index > end_datetime)
        )
        assert outside_range.all()
    
    def test_empty_result(self, laeq1m_data):
        """Test when filtering results in empty DataFrame."""
        # Filter for a time range outside the data
        start_datetime = datetime(2025, 1, 1, 0, 0, 0)
        end_datetime = datetime(2025, 1, 2, 0, 0, 0)
        
        result = all_data(laeq1m_data, start_datetime, end_datetime)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_invalid_datetime_comparison(self):
        """Test error handling for timezone-related issues."""
        index = pd.date_range('2023-01-01', periods=10, freq='h', tz='UTC')
        df = pd.DataFrame({'LEQ dB -A': range(10)}, index=index)
        
        start_datetime = datetime(2025, 3, 23, 2, 0, 0)
        end_datetime = datetime(2025, 3, 25, 8, 0, 0)
        
        with pytest.raises(TypeError) as exc_info:
            all_data(df, start_datetime, end_datetime)
        
        assert "Invalid comparison" in str(exc_info.value)
        assert "timezone" in str(exc_info.value)


class TestExtremeValues:
    """Test cases for the extreme_values function."""
    
    def test_filter_extreme_values(self, laeq1m_data, capsys):
        """Test filtering extreme values with default parameters."""
        result = extreme_values(laeq1m_data.copy(), 
                                min_value=40, 
                                max_value=90)
        
        assert result['LEQ dB -A'].min() >= 40
        assert result['LEQ dB -A'].max() <= 90
        
        original_count = laeq1m_data['LEQ dB -A'].notna().sum()
        filtered_count = result['LEQ dB -A'].notna().sum()
        assert filtered_count < original_count
        
        captured = capsys.readouterr()
        assert "Proportion of values filtered out:" in captured.out
    
    
    def test_filter_by_column_index(self, laeq1m_data):
        """Test filtering using column index."""
        result = extreme_values(laeq1m_data.copy(), 
                                column=0,
                                min_value=40, 
                                max_value=90)

        assert result['LEQ dB -A'].min() >= 40
        assert result['LEQ dB -A'].max() <= 90
    
    def test_no_extreme_values(self):
        """Test when no values are filtered out."""
        # Create data within normal range
        index = pd.date_range('2023-01-01', periods=10, freq='h')
        df = pd.DataFrame({'LEQ dB -A': [45, 50, 55, 60, 65] * 2}, index=index)

        result = extreme_values(df.copy(), min_value=40, max_value=90)

        # No values should be filtered
        assert result['LEQ dB -A'].notna().sum() == df['LEQ dB -A'].notna().sum()


class TestWeatherFlags:
    """Test cases for the weather_flags function."""
    
    def test_filter_flags(self, sample_weather_data, capsys):
        """Test filtering with multiple weather flags."""
        original_count = sample_weather_data['LEQ dB -A'].notna().sum()
        result = weather_flags(
            sample_weather_data.copy(),
            filter_wind_flag=True,
            filter_rain_flag=True,
            filter_temp_flag=True
        )
        
        # Check that all flagged values are NaN
        flagged_mask = (
            sample_weather_data['Wind_Spd_Flag'] |
            sample_weather_data['Rain_Flag_Roll'] |
            sample_weather_data['Temp_Flag']
        )
        flagged_indices = sample_weather_data[flagged_mask].index
        filtered_count = result['LEQ dB -A'].notna().sum()

        assert result.loc[flagged_indices, 'LEQ dB -A'].isna().all()
        assert filtered_count < original_count

        captured = capsys.readouterr()
        assert "Proportion of values filtered out:" in captured.out
    
    def test_filter_no_flags(self, sample_weather_data):
        """Test when no flags are enabled."""
        result = weather_flags(
            sample_weather_data.copy(),
            filter_wind_flag=False,
            filter_rain_flag=False,
            filter_temp_flag=False
        )
        
        # No values should be filtered
        original_count = sample_weather_data['LEQ dB -A'].notna().sum()
        filtered_count = result['LEQ dB -A'].notna().sum()
        assert filtered_count == original_count
    
    def test_missing_flag_columns(self, laeq1m_data):
        """Test behavior when flag columns don't exist."""
        result = weather_flags(laeq1m_data.copy())
        
        assert isinstance(result, pd.DataFrame)

class TestDays:
    """Test cases for the _days function."""
    
    def test_filter_weekdays(self, laeq1m_data):
        """Test filtering for weekdays only."""
        # Sample data starts on Monday (2023-03-20)
        result = _days(laeq1m_data, 'monday', 'friday')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        weekdays = result.index.dayofweek
        assert weekdays.min() >= 0  # Monday
        assert weekdays.max() <= 4  # Friday
    
    def test_filter_weekend(self, laeq1m_data):
        """Test filtering for weekend only."""
        result = _days(laeq1m_data, 'saturday', 'sunday')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        weekdays = result.index.dayofweek
        assert ((weekdays == 5) | (weekdays == 6)).all()
    
    def test_wrap_around_week(self, laeq1m_data):
        """Test filtering across week boundary (e.g., Friday to Tuesday)."""
        result = _days(laeq1m_data, 'friday', 'tuesday')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        weekdays = result.index.dayofweek
        valid_days = (weekdays >= 4) | (weekdays <= 1)
        assert valid_days.all()
    
    def test_no_day_filtering(self, laeq1m_data):
        """Test when no day filtering is applied."""
        result = _days(laeq1m_data, None, None)
        
        assert result.equals(laeq1m_data)
    
    def test_invalid_day_names(self, laeq1m_data):
        """Test error handling for invalid day names."""
        with pytest.raises(ValueError) as exc_info:
            _days(laeq1m_data, 'notaday', 'tuesday')
        
        assert "must be a day of the week" in str(exc_info.value)
    
    def test_no_data_for_specified_days(self, laeq1m_data):
        """Test when no data exists for specified days."""
        single_day = laeq1m_data.iloc[:5]
        
        with pytest.raises(ValueError) as exc_info:
            _days(single_day, 'saturday', 'sunday')
        
        assert "No data found for the specified day range" in str(exc_info.value)


class TestHours:
    """Test cases for the _hours function."""
    
    def test_filter_daytime_hours(self, laeq1m_data):
        """Test filtering for daytime hours."""
        result = _hours(laeq1m_data, 9, 17)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        hours = result.index.hour
        assert hours.min() >= 9
        assert hours.max() <= 17
    
    def test_filter_nighttime_hours(self, laeq1m_data):
        """Test filtering for nighttime hours."""
        result = _hours(laeq1m_data, 22, 6)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        hours = result.index.hour
        valid_hours = (hours >= 22) | (hours <= 6)
        assert valid_hours.all()
    
    def test_single_hour(self, laeq1m_data):
        """Test filtering for a single hour."""
        result = _hours(laeq1m_data, 12, 12)
        
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert (result.index.hour == 12).all()
    
    def test_invalid_hour_range(self, laeq1m_data):
        """Test error handling for invalid hour values."""
        with pytest.raises(ValueError) as exc_info:
            _hours(laeq1m_data, -1, 12)
        
        assert "Hours must be between 0 and 24" in str(exc_info.value)
        
        with pytest.raises(ValueError) as exc_info:
            _hours(laeq1m_data, 12, 25)
        
        assert "Hours must be between 0 and 24" in str(exc_info.value)


class TestFilterIntegration:
    """Integration tests using actual test data."""
    
    def test_combined_filtering(self, laeq1m_data):
        """Test combining multiple filters on real data."""
        filtered_data = extreme_values(laeq1m_data.copy(), min_value=35, max_value=85)

        if len(filtered_data) > 24:
            mid_date = filtered_data.index[len(filtered_data)//2]
            start_date = mid_date - pd.Timedelta(hours=12)
            end_date = mid_date + pd.Timedelta(hours=12)
            
            filtered_data = all_data(filtered_data, start_date, end_date)
        
        assert isinstance(filtered_data, pd.DataFrame)
        valid_values = filtered_data.iloc[:, 0].dropna()
        if len(valid_values) > 0:
            assert valid_values.min() >= 35
            assert valid_values.max() <= 85
    
    def test_filter_with_missing_data(self, laeq1m_data):
        """Test filters handle missing data correctly."""
        if laeq1m_data is None:
            pytest.skip("No test data available")
            
        test_data = laeq1m_data.copy()
        test_data.iloc[10:20, 0] = np.nan
        
        # Apply filters
        result = extreme_values(test_data, min_value=40, max_value=80)
        
        assert isinstance(result, pd.DataFrame)
        assert result.iloc[10:20, 0].isna().all()