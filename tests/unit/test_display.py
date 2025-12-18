"""
Test suite for noisemonitor.util.display module.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

# Use non-interactive backend for testing
matplotlib.use('Agg')

import noisemonitor as nm
from noisemonitor.util.display import (
    compare, freq_line, freq_map, harmonica, line, 
    line_weather, compare_weather_daily
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
    dates = pd.date_range('2023-01-01', periods=1440, freq='min') 

    octave_bands = ['63', '125', '250', '500', '1000', '2000', '4000', '8000']
    data = {}
    
    np.random.seed(42) 
    for i, band in enumerate(octave_bands):
        base_level = 60 - (i * 5)
        data[band] = np.random.normal(base_level, 3, len(dates))
    
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_weather_data(laeq1m_data):
    """Generate sample data with weather flags for testing, based on real data."""
    df = laeq1m_data.copy()
    
    np.random.seed(42)
    data_length = len(df)
    df['Wind_Spd_Flag'] = np.random.choice([True, False], data_length, p=[0.2, 0.8])
    df['Rain_Flag_Roll'] = np.random.choice([True, False], data_length, p=[0.1, 0.9])
    df['Temp_Flag'] = np.random.choice([True, False], data_length, p=[0.05, 0.95])
    df['Rel_Hum_Flag'] = np.random.choice([True, False], data_length, p=[0.05, 0.95])
    df['Snow_Flag_Roll'] = np.random.choice([True, False], data_length, p=[0.02, 0.98])
    
    return df


class TestBasicPlotFunctions:
    """Test basic plotting functions that don't require special data."""
    
    @patch('matplotlib.pyplot.show')
    def test_line_basic(self, mock_show, laeq1m_data):
        """Test basic line plotting function."""
        column_name = laeq1m_data.columns[0]
        
        ax = line(laeq1m_data, column_name)
        assert ax is not None
        assert hasattr(ax, 'lines')
        assert len(ax.lines) == 1 
        ax = line(laeq1m_data, column_name)
        assert ax is not None
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_line_multiple_columns(self, mock_show, laeq1m_data):
        """Test line plotting with multiple columns."""
        df_subset = laeq1m_data.iloc[:100].copy() 
        df_subset['LAmax'] = df_subset.iloc[:, 0] + 10
        
        column1 = df_subset.columns[0]
        column2 = df_subset.columns[1]
        
        ax = line(df_subset, column1, column2)
        assert ax is not None
        assert len(ax.lines) >= 2
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_line_with_options(self, mock_show, laeq1s_data):
        """Test line plotting with various options."""
        # Use 1-second data for more detailed plotting
        column_name = laeq1s_data.columns[0]
        
        ax = line(
            laeq1s_data.iloc[:200],
            column_name,
            step=True,
            show_points=True,
            ylabel="Custom Label",
            figsize=(12, 6)
        )
        assert ax is not None
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_compare_basic(self, mock_show, laeq1m_data):
        """Test compare function with multiple DataFrames."""
        df1 = laeq1m_data.iloc[:100].copy()
        df2 = laeq1m_data.iloc[100:200].copy()

        column_name = df1.columns[0]
        assert column_name in df2.columns, f"Column {column_name} not in second dataset"
        
        dfs = [df1, df2]
        labels = ['Dataset 1', 'Dataset 2']
        
        ax = compare(dfs, labels, column_name)
        assert ax is not None
        assert hasattr(ax, 'legend_')
        mock_show.assert_called()
        plt.close('all')


class TestAdvancedPlotFunctions:
    """Test frequency-specific plotting functions."""
    
    @patch('matplotlib.pyplot.show')
    def test_freq_line(self, mock_show, sample_octave_data):
        """Test frequency line plotting with options."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = freq_line(
                sample_octave_data,
                figsize=(14, 8)
            )
        assert mock_show.call_count >= 1
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_harmonica_basic(self, mock_show):
        """Test harmonica (bar chart) plotting."""
        times = pd.date_range('2023-01-01', periods=24, freq='h').time
        
        np.random.seed(42)
        harmonica_values = np.random.randint(1, 10, 24)
        bgn_values = np.random.uniform(2, 8, 24)
        harmonica_data = pd.DataFrame(
            {
                'HARMONICA': harmonica_values,
                'BGN': bgn_values
            }, 
            index=times
        )
        
        result = harmonica(harmonica_data)
        assert mock_show.call_count >= 1
        plt.close('all')


class TestWeatherPlotFunctions:
    """Test weather-related plotting functions."""
    
    @patch('matplotlib.pyplot.show')
    def test_line_weather(self, mock_show, sample_weather_data):
        """Test weather line plotting with different flag combinations."""
        column_name = sample_weather_data.columns[0]
        result = line_weather(
            sample_weather_data, 
            column=column_name,
            include_wind_flag=True,
            include_rain_flag=True,
            include_temp_flag=False,
            include_rel_hum_flag=False,
            include_snow_flag=False
        )
        assert mock_show.call_count >= 1
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_line_weather_with_window(self, mock_show, sample_weather_data):
        """Test weather line plotting with rolling window."""
        column_name = sample_weather_data.columns[0]
        
        with pytest.warns(
            UserWarning, match="Computing the L10, L50, and L90"
        ):
            result = line_weather(
                sample_weather_data, 
                column=column_name,
                win=3600
            )
            
        assert mock_show.call_count >= 1
        plt.close('all')

class TestKwargsSupport:
    """Test kwargs support for all display functions."""
    
    @patch('matplotlib.pyplot.show')
    def test_line_with_kwargs(self, mock_show, laeq1m_data):
        """Test line function with various matplotlib kwargs."""
        column_name = laeq1m_data.columns[0]
        
        # Test with color, linewidth, and alpha
        ax = line(
            laeq1m_data.iloc[:100],
            column_name,
            color='red',
            linewidth=3,
            alpha=0.7,
            linestyle='--'
        )
        assert ax is not None
        assert len(ax.lines) == 1
        line_obj = ax.lines[0]
        assert line_obj.get_color() == 'red'
        assert line_obj.get_linewidth() == 3
        assert line_obj.get_alpha() == 0.7
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_line_with_ylim_xlim(self, mock_show, laeq1m_data):
        """Test line function with ylim and xlim kwargs."""
        column_name = laeq1m_data.columns[0]
        
        ax = line(
            laeq1m_data.iloc[:100],
            column_name,
            ylim=(40, 80),
            title='Custom Title'
        )
        assert ax is not None
        assert ax.get_ylim() == (40, 80)
        assert ax.get_title() == 'Custom Title'
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_line_step_with_kwargs(self, mock_show, laeq1m_data):
        """Test line function with step=True and kwargs."""
        column_name = laeq1m_data.columns[0]
        
        ax = line(
            laeq1m_data.iloc[:100],
            column_name,
            step=True,
            color='blue',
            linewidth=2,
            marker='o'
        )
        assert ax is not None
        assert len(ax.lines) == 1
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_compare_with_kwargs(self, mock_show, laeq1m_data):
        """Test compare function with kwargs."""
        df1 = laeq1m_data.iloc[:100].copy()
        df2 = laeq1m_data.iloc[100:200].copy()
        column_name = df1.columns[0]
        
        ax = compare(
            [df1, df2],
            ['Dataset 1', 'Dataset 2'],
            column_name,
            linewidth=2.5,
            alpha=0.8,
            ylim=(30, 90)
        )
        assert ax is not None
        assert ax.get_ylim() == (30, 90)
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_freq_line_with_kwargs(self, mock_show, sample_octave_data):
        """Test freq_line function with kwargs."""
        # Create overall levels DataFrame
        overall_df = pd.DataFrame({
            'Leq': sample_octave_data.mean(axis=0)
        }).T
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = freq_line(
                overall_df,
                color='green',
                linewidth=2,
                marker='s',
                markersize=8
            )
        assert mock_show.call_count >= 1
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_freq_map_with_kwargs(self, mock_show, sample_octave_data):
        """Test freq_map function with kwargs."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = freq_map(
                sample_octave_data.iloc[:100],
                cmap='viridis',
                vmin=40,
                vmax=80
            )
        assert mock_show.call_count >= 1
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_harmonica_with_kwargs(self, mock_show):
        """Test harmonica function with kwargs."""
        times = pd.date_range('2023-01-01', periods=24, freq='h').time
        np.random.seed(42)
        harmonica_data = pd.DataFrame(
            {
                'HARMONICA': np.random.randint(1, 10, 24),
                'BGN': np.random.uniform(2, 8, 24)
            },
            index=times
        )
        
        result = harmonica(
            harmonica_data,
            figsize=(15, 6),
            ylim=(0, 12)
        )
        assert mock_show.call_count >= 1
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_line_weather_with_kwargs(self, mock_show, sample_weather_data):
        """Test line_weather function with kwargs."""
        column_name = sample_weather_data.columns[0]
        
        result = line_weather(
            sample_weather_data.iloc[:200],
            column=column_name,
            title='Custom Weather Plot',
            figsize=(14, 8),
            color='purple',
            linewidth=2
        )
        assert mock_show.call_count >= 1
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_compare_weather_daily_with_kwargs(self, mock_show, sample_weather_data):
        """Test compare_weather_daily function with kwargs."""
        column_name = sample_weather_data.columns[0]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = compare_weather_daily(
                sample_weather_data,
                column=column_name,
                include_wind_flag=True,
                include_rain_flag=True,
                linewidth=1.5,
                alpha=0.9
            )
        assert mock_show.call_count >= 1
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_line_with_marker_kwargs(self, mock_show, laeq1m_data):
        """Test line function with marker-related kwargs."""
        column_name = laeq1m_data.columns[0]
        
        ax = line(
            laeq1m_data.iloc[:50],
            column_name,
            marker='D',
            markersize=6,
            markerfacecolor='red',
            markeredgecolor='black',
            markeredgewidth=1.5
        )
        assert ax is not None
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_multiple_functions_with_consistent_kwargs(self, mock_show, laeq1m_data):
        """Test that kwargs work consistently across different functions."""
        column_name = laeq1m_data.columns[0]
        df_subset = laeq1m_data.iloc[:100]
        
        # Define consistent styling kwargs
        style_kwargs = {
            'color': 'steelblue',
            'linewidth': 2,
            'alpha': 0.75
        }
        
        # Test line function
        ax1 = line(df_subset, column_name, **style_kwargs)
        assert ax1 is not None
        
        # Test compare function
        ax2 = compare(
            [df_subset, df_subset],
            ['Data 1', 'Data 2'],
            column_name,
            **style_kwargs
        )
        assert ax2 is not None
        
        plt.close('all')


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_dataframe(self):
        """Test plotting functions with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with patch('matplotlib.pyplot.show'):
            with pytest.raises((IndexError, ValueError, KeyError)):
                line(empty_df, 'nonexistent_column')
            plt.close('all')
    
    def test_invalid_column_name(self, laeq1m_data):
        """Test plotting functions with invalid column names."""
        with patch('matplotlib.pyplot.show'):
            with pytest.raises((KeyError, IndexError)):
                line(laeq1m_data, 'nonexistent_column')
            plt.close('all')
    
    def test_invalid_column_index(self, laeq1m_data):
        """Test plotting functions with invalid column indices."""
        with patch('matplotlib.pyplot.show'):
            with pytest.raises((IndexError, KeyError)):
                line(laeq1m_data, 999)
            plt.close('all')


if __name__ == "__main__":
    pytest.main([__file__])