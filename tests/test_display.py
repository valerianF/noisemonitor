"""
Test suite for noisemonitor.util.display module.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from unittest.mock import patch

# Use non-interactive backend for testing
matplotlib.use('Agg')

import noisemonitor as nm
from noisemonitor.profile import periodic
from noisemonitor.util.display import (
    compare, freq_line, freq_map, harmonica, line,
    line_weather, compare_weather_daily
)

# Plots only: no acoustic reference values
pytestmark = pytest.mark.behaviour


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


@pytest.fixture
def harmonica_profile():
    """24 hourly HARMONICA and BGN values."""
    np.random.seed(42)
    return pd.DataFrame(
        {'HARMONICA': np.random.randint(1, 10, 24),
         'BGN': np.random.uniform(2, 8, 24)},
        index=pd.date_range('2023-01-01', periods=24, freq='h').time)


class TestBasicPlotFunctions:
    """Test basic plotting functions that don't require special data."""

    @patch('matplotlib.pyplot.show')
    def test_line_kwargs(self, mock_show, laeq1m_data):
        """Matplotlib kwargs reach the line; ylim and title set the axes."""
        ax = line(laeq1m_data.iloc[:100], laeq1m_data.columns[0], color='red',
                  linewidth=3, alpha=0.7, linestyle='--', ylim=(40, 80),
                  title='Custom Title')

        assert len(ax.lines) == 1
        line_obj = ax.lines[0]
        assert line_obj.get_color() == 'red'
        assert line_obj.get_linewidth() == 3
        assert line_obj.get_alpha() == 0.7
        assert line_obj.get_linestyle() == '--'
        assert ax.get_ylim() == (40, 80)
        assert ax.get_title() == 'Custom Title'
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_line_step_points(self, mock_show, laeq1s_data):
        """Step plot with points, custom label and figure size."""
        ax = line(laeq1s_data.iloc[:200], laeq1s_data.columns[0], step=True,
                  show_points=True, ylabel="Custom Label", figsize=(12, 6))
        assert ax.get_ylabel() == "Custom Label"
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_line_multiple_columns(self, mock_show, laeq1m_data):
        """Test line plotting with multiple columns."""
        df_subset = laeq1m_data.iloc[:100].copy()
        df_subset['LAmax'] = df_subset.iloc[:, 0] + 10

        ax = line(df_subset, df_subset.columns[0], df_subset.columns[1])
        assert len(ax.lines) >= 2
        plt.close('all')

    def test_line_object_datetime_index(self, laeq1m_data):
        """An index of datetime objects (object dtype) is plotted as the
        same DatetimeIndex."""
        df = laeq1m_data.iloc[:100]
        as_objects = df.set_axis(pd.Index(df.index.to_pydatetime(), dtype=object))
        assert as_objects.index.dtype == object

        expected = line(df, df.columns[0]).lines[0].get_xdata()
        got = line(as_objects, df.columns[0]).lines[0].get_xdata()

        assert list(got) == list(expected)
        plt.close('all')

    @pytest.mark.parametrize("freq, periods, fmt", [
        ('h', 30 * 24, '%Y-%m-%d'),  # hourly values over 30 days
        ('MS', 12, '%b %Y'),         # monthly values
    ], ids=['hourly', 'monthly'])
    def test_time_axis_format(self, freq, periods, fmt):
        """Dates are labelled by day, or by month for data at intervals of
        15 days or more."""
        index = pd.date_range('2023-01-01', periods=periods, freq=freq)
        ax = line(pd.DataFrame({'Leq': 50.0}, index=index), 'Leq')

        assert ax.xaxis.get_major_formatter().fmt == fmt
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_compare(self, mock_show, laeq1m_data):
        """Two DataFrames on the same axes, with a legend and kwargs."""
        df1 = laeq1m_data.iloc[:100]
        df2 = laeq1m_data.iloc[100:200]

        ax = compare([df1, df2], ['Dataset 1', 'Dataset 2'], df1.columns[0],
                     linewidth=2.5, alpha=0.8, ylim=(30, 90))

        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            'Dataset 1 - LEQ dB -A', 'Dataset 2 - LEQ dB -A']
        assert ax.get_ylim() == (30, 90)
        mock_show.assert_called()
        plt.close('all')

    @pytest.mark.parametrize("column", ['nonexistent_column', 999])
    def test_invalid_column(self, laeq1m_data, column):
        """An unknown column name or index raises an error."""
        with patch('matplotlib.pyplot.show'):
            with pytest.raises((KeyError, IndexError)):
                line(laeq1m_data, column)
        plt.close('all')


class TestAdvancedPlotFunctions:
    """Test frequency-specific plotting functions."""

    @pytest.mark.slow
    @patch('matplotlib.pyplot.show')
    def test_freq_line(self, mock_show, sample_octave_data):
        """Test frequency line plotting with options."""
        with pytest.warns(RuntimeWarning, match="Frequency bands are not nu"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                freq_line(sample_octave_data, figsize=(14, 8))
        assert mock_show.call_count >= 1
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_freq_line_with_kwargs(self, mock_show, sample_octave_data):
        """Test freq_line function with kwargs."""
        overall_df = pd.DataFrame({'Leq': sample_octave_data.mean(axis=0)}).T

        with pytest.warns(RuntimeWarning, match="Frequency bands are not nu"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                freq_line(overall_df, color='green', linewidth=2, marker='s',
                          markersize=8)
        assert mock_show.call_count >= 1
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_freq_map_with_kwargs(self, mock_show, sample_octave_data):
        """Test freq_map function with kwargs."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            freq_map(sample_octave_data.iloc[:100], cmap='viridis', vmin=40, vmax=80)
        assert mock_show.call_count >= 1
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_harmonica(self, mock_show, harmonica_profile):
        """Bar chart of the HARMONICA profile, with kwargs."""
        harmonica(harmonica_profile, figsize=(15, 6), ylim=(0, 12))
        assert mock_show.call_count >= 1
        plt.close('all')

    @pytest.mark.parametrize("freq, ylabel", [
        ('D', 'Number of Days'), ('W', 'Number of Weeks'), ('MS', 'Number of Months')])
    def test_nday_ylabel(self, freq, ylabel):
        """The y-axis is labelled with the period counted."""
        counts = pd.DataFrame({'Decibel Range': ['<50', '50-55', '>55'],
                            'Count': [1, 2, 3]})

        nm.display.nday(counts, bins=[50, 55], freq=freq, show=False)

        assert plt.gca().get_ylabel() == ylabel
        plt.close('all')


class TestWeatherPlotFunctions:
    """Test weather-related plotting functions."""

    @patch('matplotlib.pyplot.show')
    def test_line_weather(self, mock_show, sample_weather_data):
        """Weather line plot with some flags and kwargs."""
        line_weather(
            sample_weather_data.iloc[:200],
            column=sample_weather_data.columns[0],
            include_wind_flag=True,
            include_rain_flag=True,
            include_temp_flag=False,
            include_rel_hum_flag=False,
            include_snow_flag=False,
            title='Custom Weather Plot',
            figsize=(14, 8),
            color='purple',
            linewidth=2
        )
        assert mock_show.call_count >= 1
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_line_weather_with_window(self, mock_show, sample_weather_data):
        """Test weather line plotting with rolling window."""
        with pytest.warns(UserWarning, match="Computing the L10, L50, and L90"):
            line_weather(sample_weather_data, column=sample_weather_data.columns[0],
                         win=3600)
        assert mock_show.call_count >= 1
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_compare_weather_daily_full_day(self, mock_show, sample_weather_data):
        """Each daily profile starts at 00:00: periodic is called with hours
        0 to 23 (hour2 is the start of the last window)."""
        with patch('noisemonitor.util.display.periodic', wraps=periodic) as spy:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                compare_weather_daily(sample_weather_data, column=0)

        assert spy.call_count > 0
        assert all(call.args[1:3] == (0, 23) for call in spy.call_args_list)
        plt.close('all')

    def test_compare_weather_daily_any_flag(self, sample_weather_data):
        """The profiles of the rows with any flag raised and with none are
        labelled 'Any Flag' and 'No Flag'; kwargs reach the plot."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            compare_weather_daily(sample_weather_data, column=0, show=False,
                                  include_wind_flag=True, include_rain_flag=True,
                                  linewidth=1.5, alpha=0.9)

        labels = [t.get_text() for t in plt.gca().get_legend().get_texts()]
        assert 'Any Flag - Leq' in labels and 'No Flag - Leq' in labels
        assert not any(label.startswith(('All Flags', 'Neither Flags'))
                    for label in labels)
        plt.close('all')
