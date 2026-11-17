"""Unit tests for filter module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from noisemonitor.util.filter import (
    all_data, extreme_values, weather_flags, _days, _hours
)

# Filtering only: no acoustic reference values
pytestmark = pytest.mark.behaviour


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
    
    @pytest.mark.parametrize("column", [None, 0, 'LEQ dB -A'])
    def test_filter_extreme_values(self, laeq1m_data, capsys, column):
        """Values outside [min_value, max_value] become NaN in a copy: the
        input DataFrame is left unchanged. The column is the first one by
        default, or given by index or name."""
        before = laeq1m_data.copy()
        result = extreme_values(laeq1m_data, column=column, min_value=40,
                                max_value=90)

        pd.testing.assert_frame_equal(laeq1m_data, before)
        assert result['LEQ dB -A'].min() >= 40
        assert result['LEQ dB -A'].max() <= 90

        original_count = laeq1m_data['LEQ dB -A'].notna().sum()
        filtered_count = result['LEQ dB -A'].notna().sum()
        assert filtered_count < original_count

        captured = capsys.readouterr()
        assert "Proportion of values filtered out:" in captured.out

    def test_no_extreme_values(self):
        """Test when no values are filtered out."""
        # Create data within normal range
        index = pd.date_range('2023-01-01', periods=10, freq='h')
        df = pd.DataFrame({'LEQ dB -A': [45, 50, 55, 60, 65] * 2}, index=index)

        result = extreme_values(df.copy(), min_value=40, max_value=90)

        # No values should be filtered
        assert result['LEQ dB -A'].notna().sum() == df['LEQ dB -A'].notna().sum()

    @pytest.mark.filterwarnings("error")
    def test_all_nan_column(self, capsys):
        """A column without any value: nothing to filter, 0% reported,
        without a division by zero."""
        index = pd.date_range('2023-01-01', periods=10, freq='h')
        df = pd.DataFrame({'LEQ dB -A': np.nan}, index=index)

        result = extreme_values(df, min_value=40, max_value=90)

        assert result['LEQ dB -A'].isna().all()
        assert "Proportion of values filtered out: 0.00%" in capsys.readouterr().out


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

    @pytest.mark.filterwarnings("error")
    def test_all_nan_column(self, sample_weather_data, capsys):
        """A column without any value: nothing to filter, 0% reported,
        without a division by zero."""
        df = sample_weather_data.assign(**{'LEQ dB -A': np.nan})

        result = weather_flags(df)

        assert result['LEQ dB -A'].isna().all()
        assert "Proportion of values filtered out: 0.00%" in capsys.readouterr().out

@pytest.mark.filterwarnings("error::DeprecationWarning")  # e.g. pandas deprecations
class TestDays:
    """Test cases for the _days function."""
    
    @pytest.mark.parametrize("day1, day2, kept", [
        ('monday', 'friday', {0, 1, 2, 3, 4}),
        ('saturday', 'sunday', {5, 6}),
        ('friday', 'tuesday', {4, 5, 6, 0, 1}),  # across the end of the week
    ], ids=['weekdays', 'weekend', 'wrap_around'])
    def test_filter_days(self, laeq1m_data, day1, day2, kept):
        """Keeps exactly the days from day1 to day2 (one week of data)."""
        result = _days(laeq1m_data, day1, day2)
        assert set(result.index.day_of_week) == kept

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
    
    @pytest.mark.parametrize("hour1, hour2, first, last, count", [
        (7, 19, '07:00', '18:59', 720),
        (23, 7, '00:00', '23:59', 480),
        (22, 24, '22:00', '23:59', 120),
        (24, 7, '00:00', '06:59', 420),
        (0, 24, '00:00', '23:59', 1440),
        (12, 12, None, None, 0),
    ])
    def test_hours_half_open(self, hour1, hour2, first, last, count):
        """_hours keeps [hour1, hour2): the sample at hour2 is excluded,
        24 maps to midnight, (0, 24) keeps the full day and hour1 = hour2
        (other than 0 and 24) keeps nothing."""
        dates = pd.date_range('2023-01-01', periods=1440, freq='min')
        df = pd.DataFrame({'LAeq': 50.0}, index=dates)

        result = _hours(df, hour1, hour2)
        times = result.index.strftime('%H:%M')

        assert len(result) == count
        if count:
            assert times.min() == first
            assert times.max() == last
        if hour1 != 0 or hour2 != 24:
            assert f'{hour2 % 24:02d}:00' not in times

    def test_invalid_hour_range(self, laeq1m_data):
        """Test error handling for invalid hour values."""
        with pytest.raises(ValueError) as exc_info:
            _hours(laeq1m_data, -1, 12)
        
        assert "Hours must be between 0 and 24" in str(exc_info.value)
        
        with pytest.raises(ValueError) as exc_info:
            _hours(laeq1m_data, 12, 25)
        
        assert "Hours must be between 0 and 24" in str(exc_info.value)