"""Unit tests for core module functions."""

import pytest
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, time, timedelta
from unittest.mock import patch
from pathlib import Path

import noisemonitor as nm
from noisemonitor.util.core import (
    equivalent_level,
    get_interval,
    harmonica,
    hourly_harmonica,
    lden,
    noise_events,
    _column_to_index
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
    """Load LAeq,1s test data (24h dataset - 86,401 samples)."""
    df = nm.load(
        str(test_data_paths["laeq1s"]),
        datetimeindex=0,
        valueindexes=1,
        header=0,
        sep=','
    )
    return df


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
    """Load LAeq,1s test data (24 hours of data)."""
    df = nm.load(
        str(test_data_paths["laeq1s"]),
        datetimeindex=0,
        valueindexes=1,
        header=0,
        sep=','
    )
    return df


class TestEquivalentLevel:
    """Test the equivalent_level function."""

    def test_equivalent_level_basic(self):
        """Test basic equivalent level calculation."""
        array = np.array([60.0, 65.0, 70.0])
        result = equivalent_level(array)
        expected = 10 * np.log10(np.mean(np.power(10, array / 10)))
        assert np.isclose(result, expected, rtol=1e-10)

    def test_equivalent_level_single_value(self):
        """Test with single value."""
        array = np.array([50.0])
        result = equivalent_level(array)
        assert np.isclose(result, 50.0, rtol=1e-10)

    def test_equivalent_level_empty_array(self):
        """Test with empty array."""
        array = np.array([])
        result = equivalent_level(array)
        assert np.isnan(result)

    def test_equivalent_level_all_nan(self):
        """Test with all NaN values."""
        array = np.array([np.nan, np.nan, np.nan])
        with pytest.warns(UserWarning, match="Coverage check"):
            result = equivalent_level(array)
        assert np.isnan(result)

    def test_equivalent_level_with_nan(self):
        """Test with some NaN values."""
        array = np.array([60.0, np.nan, 70.0])
        result = equivalent_level(array)
        expected = 10 * np.log10(np.mean(np.power(10, array / 10)))
        assert np.isnan(result) == np.isnan(expected)


class TestCoverage:
    """Test coverage assessment and warnings."""
    
    def test_coverage_warning_with_sparse_data(self, laeq1m_data):
        """Test that coverage warning is raised with exact expected values for sparse data."""
        # Create sparse data by removing specific days to create known gaps
        # Remove data for 2 complete days out of ~12 days total
        sparse_data = laeq1m_data.copy()
        
        # Remove March 22 and March 23 (2 full days)
        mask = (sparse_data.index.date != pd.Timestamp('2025-03-22').date()) & \
               (sparse_data.index.date != pd.Timestamp('2025-03-23').date())
        sparse_data = sparse_data[mask]
        
        with pytest.warns(UserWarning) as record:
            result = lden(
                sparse_data, 
                column=0,
                coverage_check=True,
                coverage_threshold=0.5
            )
        
        # Verify warning was raised
        assert len(record) > 0
        warning_message = str(record[0].message)
        
        # Check that warning contains coverage filter information
        assert "Coverage filter" in warning_message
        assert "below 50.0% threshold" in warning_message
        
        # Extract the actual values from warning
        # Format: "Coverage filter: X/Y periods (Z.Z%) below 50.0% threshold"
        import re
        # Extract the numbers from the warning message
        match = re.search(
            r'(\d+)/(\d+) periods \((\d+\.\d+)%\)', warning_message
        )
        assert match is not None
        
        filtered_count = int(match.group(1))
        total_count = int(match.group(2))
        percentage = float(match.group(3))
        
        # Verify the values - we removed 2 complete days,
        # so expect 2 filtered periods
        assert (
            filtered_count == 2
        ), f"Expected 2 filtered periods, got {filtered_count}"
        assert (
            total_count >= 10
        ), f"Expected at least 10 total periods, got {total_count}"
        assert abs(percentage - (filtered_count / total_count * 100)) < 0.1
        
        # Result should still be valid despite filtering
        assert isinstance(result, pd.DataFrame)
        assert 'lden' in result.columns
    
    def test_coverage_no_warning_with_dense_data(self, laeq1m_data):
        """Test that no coverage warning is raised with complete dataset."""
        # laeq1m_data has good coverage (most days have full data)
        # Should NOT raise warnings when data meets threshold
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = lden(
                laeq1m_data, 
                column=0,
                coverage_check=True,
                coverage_threshold=0.5
            )
            
            # Check that no coverage warnings were raised
            coverage_warnings = [warning for warning in w 
                                   if "Coverage filter" in str(warning.message)]
            assert len(coverage_warnings) == 0, \
                f"Expected no coverage warnings, but got {len(coverage_warnings)}"
        
        # Result should be valid
        assert isinstance(result, pd.DataFrame)
        assert 'lden' in result.columns
    
    def test_coverage_lden_requires_all_periods(self, laeq1m_data):
        """Test that Lden coverage requires ALL three periods (day/evening/night) to meet threshold."""
        # Remove only the evening period (19:00-23:00) from March 22
        # Even though day and night periods are complete, the day should be filtered
        partial_data = laeq1m_data.copy()
        
        # Remove evening period from March 22
        march_22_evening = partial_data[
            (partial_data.index.date == pd.Timestamp('2025-03-22').date()) &
            (partial_data.index.hour >= 19) &
            (partial_data.index.hour < 23)
        ]
        partial_data = partial_data.drop(march_22_evening.index)
        
        # With 50% threshold, this should trigger a warning since evening period has 0% data
        with pytest.warns(UserWarning) as record:
            result = lden(
                partial_data, 
                column=0,
                coverage_check=True,
                coverage_threshold=0.5
            )
        
        # Verify warning was raised
        assert len(record) > 0
        warning_message = str(record[0].message)
        assert "Coverage filter" in warning_message
        
        # Extract filtered count - should be at least 1 (March 22)
        import re
        match = re.search(r'(\d+)/(\d+) periods', warning_message)
        assert match is not None
        
        filtered_count = int(match.group(1))
        
        assert (
            filtered_count >= 1
        ), (
            f"Expected at least 1 filtered period "
            f"(March 22 missing evening), got {filtered_count}"
        )        # Result should be valid
        assert isinstance(result, pd.DataFrame)
        assert 'lden' in result.columns
    
    def test_coverage_with_custom_threshold(self, laeq1m_data):
        """Test coverage warnings with different custom thresholds."""
        # Create data with complete gap - remove 2 full days
        sparse_data = laeq1m_data.copy()
        mask = (sparse_data.index.date != pd.Timestamp('2025-03-22').date()) & \
               (sparse_data.index.date != pd.Timestamp('2025-03-23').date())
        sparse_data = sparse_data[mask]
        
        # Test with very permissive threshold (0%) - should still warn about missing days
        with pytest.warns(UserWarning) as record_low:
            result_low = lden(
                sparse_data, 
                column=0,
                coverage_check=True,
                coverage_threshold=0.0
            )
        
        # Test with stricter threshold (0.5) - should also warn
        with pytest.warns(UserWarning) as record_high:
            result_high = lden(
                sparse_data, 
                column=0,
                coverage_check=True,
                coverage_threshold=0.5
            )
        
        # Both should produce warnings about the 2 missing days
        assert len(record_low) > 0
        assert len(record_high) > 0
        
        low_warning = str(record_low[0].message)
        high_warning = str(record_high[0].message)
        
        assert "Coverage filter" in low_warning
        assert "Coverage filter" in high_warning
        
        # Both should mention 2 filtered periods
        import re
        match_low = re.search(r'(\d+)/(\d+) periods', low_warning)
        match_high = re.search(r'(\d+)/(\d+) periods', high_warning)
        
        assert match_low is not None
        assert match_high is not None
        
        filtered_low = int(match_low.group(1))
        filtered_high = int(match_high.group(1))
        
        # Both thresholds should filter the same 2 complete missing days
        assert filtered_low == 2, f"Expected 2 filtered periods with 0% threshold, got {filtered_low}"
        assert filtered_high == 2, f"Expected 2 filtered periods with 50% threshold, got {filtered_high}"
        # Should filter at least 1 period (the one with partial data)
        assert filtered_high >= 1, f"Expected at least 1 filtered period, got {filtered_high}"
        
        # Both results should be valid
        assert isinstance(result_low, pd.DataFrame)
        assert isinstance(result_high, pd.DataFrame)


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


class TestHarmonicaValidation:
    """Test the harmonica function input validation."""

    def test_harmonica_invalid_interval(self):
        """Test that harmonica raises ValueError for intervals > 1s."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=100, freq='5s')
        df = pd.DataFrame({'sound_level': np.random.uniform(40, 80, 100)}, index=dates)
        
        with pytest.raises(ValueError, match="Computing the HARMONICA indicator requires "
                           "an integration time equal to or below 1s"):
            harmonica(df, 0)

    def test_harmonica_valid_interval_1s(self):
        """Test that harmonica accepts 1-second intervals."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=7200, freq='1s')
        df = pd.DataFrame({'sound_level': np.random.uniform(40, 80, 7200)}, index=dates)
        
        result = harmonica(df, 0)
        assert isinstance(result, pd.DataFrame)
        assert 'EVT' in result.columns
        assert 'BGN' in result.columns
        assert 'HARMONICA' in result.columns

    def test_harmonica_valid_interval_subsecond(self):
        """Test that harmonica accepts sub-second intervals."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=36000, freq='0.1s')
        df = pd.DataFrame({'sound_level': np.random.uniform(40, 80, 36000)}, index=dates)
        
        result = harmonica(df, 0)
        assert isinstance(result, pd.DataFrame)

    def test_harmonica_empty_dataframe(self):
        """Test harmonica with empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(IndexError):
            harmonica(df, 0)


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


class TestLden:
    """Test the lden function."""

    def test_lden_basic(self):
        """Test basic lden calculation."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=24*3600, freq='1s')
        
        # Create different levels for different periods
        data = []
        for date in dates:
            if 7 <= date.hour < 19:
                data.append(65.0)
            elif 19 <= date.hour < 23:
                data.append(60.0)
            else:
                data.append(55.0)
        
        df = pd.DataFrame({'sound_level': data}, index=dates)
        result = lden(df, 0)
        
        assert isinstance(result, pd.DataFrame)
        assert 'lden' in result.columns
        assert len(result) == 1
        assert not np.isnan(result['lden'].iloc[0])

    def test_lden_with_values(self):
        """Test lden calculation with individual period values."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=24*3600, freq='1s')
        
        data = []
        for date in dates:
            if 7 <= date.hour < 19:
                data.append(65.0)
            elif 19 <= date.hour < 23:
                data.append(60.0)
            else:
                data.append(55.0)
        
        df = pd.DataFrame({'sound_level': data}, index=dates)
        result = lden(df, 0, values=True)
        
        assert isinstance(result, pd.DataFrame)
        assert 'lden' in result.columns
        assert 'lday' in result.columns
        assert 'levening' in result.columns
        assert 'lnight' in result.columns
        assert len(result) == 1
        
        assert np.isclose(result['lday'].iloc[0], 65.0, rtol=0.01)
        assert np.isclose(result['levening'].iloc[0], 60.0, rtol=0.01)
        assert np.isclose(result['lnight'].iloc[0], 55.0, rtol=0.01)

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
        """Test events at data boundaries."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=10, freq='1s')
        
        data = [75, 75, 50, 50, 50, 50, 50, 75, 75, 75]
        df = pd.DataFrame({'sound_level': data}, index=dates)
        
        result = noise_events(df, 0, threshold=65.0, min_gap=3)
        assert result == 1

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


class TestHarmonicaIntegration:
    """Integration tests for harmonica function with different scenarios."""

    def test_harmonica_single_hour(self):
        """Test harmonica with exactly one hour of data."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=3600, freq='1s')
        data = np.random.uniform(50, 70, 3600)
        df = pd.DataFrame({'sound_level': data}, index=dates)
        
        result = harmonica(df, 0)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert 'EVT' in result.columns
        assert 'BGN' in result.columns
        assert 'HARMONICA' in result.columns

    def test_harmonica_multiple_hours(self):
        """Test harmonica with multiple hours of data."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=3*3600, freq='1s')  # 3 hours
        data = np.random.uniform(50, 70, 3*3600)
        df = pd.DataFrame({'sound_level': data}, index=dates)
        
        result = harmonica(df, 0)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'EVT' in result.columns
        assert 'BGN' in result.columns
        assert 'HARMONICA' in result.columns

    def test_harmonica_no_chunks(self):
        """Test harmonica with chunks disabled."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=7200, freq='1s')  # 2 hours
        data = np.random.uniform(50, 70, 7200)
        df = pd.DataFrame({'sound_level': data}, index=dates)
        
        result = harmonica(df, 0, use_chunks=False)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 'EVT' in result.columns
        assert 'BGN' in result.columns
        assert 'HARMONICA' in result.columns

    def test_harmonica_subsecond_interval(self):
        """Test harmonica with sub-second intervals."""
        # Create data with 0.1 second intervals
        dates = pd.date_range('2023-01-01 00:00:00', periods=36000, freq='0.1s')  # 1 hour at 0.1s
        data = np.random.uniform(50, 70, 36000)
        df = pd.DataFrame({'sound_level': data}, index=dates)
        
        result = harmonica(df, 0)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert 'EVT' in result.columns
        assert 'BGN' in result.columns
        assert 'HARMONICA' in result.columns


class TestLdenCalculations:
    """Test lden calculation accuracy."""

    def test_lden_calculation_accuracy(self):
        """Test lden calculation with known values."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=24*60, freq='1min')
        
        data = []
        for date in dates:
            if 7 <= date.hour < 19:
                data.append(65.0)
            elif 19 <= date.hour < 23:
                data.append(60.0)
            else:
                data.append(55.0)
        
        df = pd.DataFrame({'sound_level': data}, index=dates)
        result = lden(df, 0, values=True)
        
        expected_lden = 10 * np.log10((
            12 * np.power(10, 65/10) + 
            4 * np.power(10, 65/10) + 
            8 * np.power(10, 65/10)
        ) / 24)
        
        assert np.isclose(result['lday'].iloc[0], 65.0, atol=0.1)
        assert np.isclose(result['levening'].iloc[0], 60.0, atol=0.1)
        assert np.isclose(result['lnight'].iloc[0], 55.0, atol=0.1)
        assert np.isclose(result['lden'].iloc[0], expected_lden, atol=0.1)


if __name__ == '__main__':
    pytest.main([__file__])