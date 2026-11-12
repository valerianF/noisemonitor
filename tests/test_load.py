"""Unit tests for load module functions."""

import bz2
import gzip
import lzma
import shutil
import zipfile

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import warnings
import openpyxl
from pathlib import Path
from datetime import datetime, date, time
from unittest.mock import patch, mock_open

import noisemonitor as nm
from noisemonitor.util.core import equivalent_level
from noisemonitor.util.load import load, _parse_data
from tests.conftest import (
    BRUITPARIF_ROWS, BRUITPARIF_TZ, EXT_PATH, assert_close
)

# The 10 min of LAeq,1s before 25 October, and the first row at UTC+1 after
# them (second occurrence of 02:00:00): 00:00 to 03:00 local time at UTC+2
DST_LEAD = 600
DST_CHANGE = DST_LEAD + 3 * 3600


@pytest.mark.behaviour
class TestParseData:
    """Test the _parse_data helper function."""

    def test_parse_data_datetime_index(self):
        """Test _parse_data with datetime in single column."""
        data = {
            'datetime_col': ['2023-01-01 10:00:00', '2023-01-01 10:01:00', '2023-01-01 10:02:00'],
            'sound_level': [65.5, 67.2, 64.8]
        }
        chunk = pd.DataFrame(data)
        
        result = _parse_data(
            chunk=chunk,
            datetimeindex='datetime_col',
            timeindex=None,
            dateindex=None,
            timezone=None
        )
        
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 3
        assert len(result.columns) == 1
        assert all(result.iloc[:, 0] == [65.5, 67.2, 64.8])

    def test_parse_data_separate_date_time(self):
        """Test _parse_data with separate date and time columns."""
        data = {
            'date_col': ['2023-01-01', '2023-01-01', '2023-01-01'],
            'time_col': ['10:00:00', '10:01:00', '10:02:00'],
            'sound_level': [65.5, 67.2, 64.8]
        }
        chunk = pd.DataFrame(data)
        
        result = _parse_data(
            chunk=chunk,
            datetimeindex=None,
            timeindex='time_col',
            dateindex='date_col',
            timezone=None 
        )
        
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 3
        assert len(result.columns) == 1

    def test_parse_data_multiple_value_indexes(self):
        """Test _parse_data with multiple sound level columns."""
        data = {
            'datetime_col': ['2023-01-01 10:00:00', '2023-01-01 10:01:00'],
            'LAeq': [65.5, 67.2],
            'LCeq': [68.1, 69.5],
            'LAmax': [72.3, 74.1]
        }
        chunk = pd.DataFrame(data)
        
        result = _parse_data(
            chunk=chunk,
            datetimeindex='datetime_col',
            timeindex=None,
            dateindex=None,
            timezone=None
        )
        
        assert len(result.columns) == 3
        assert list(result.iloc[0, :]) == [65.5, 68.1, 72.3]

    def test_parse_data_timezone_conversion(self):
        """Test _parse_data with timezone conversion."""
        data = {
            'datetime_col': ['2023-01-01 10:00:00+00:00', '2023-01-01 11:00:00+00:00'],
            'sound_level': [65.5, 67.2]
        }
        chunk = pd.DataFrame(data)
        chunk['datetime_col'] = pd.to_datetime(chunk['datetime_col'])
        
        result = _parse_data(
            chunk=chunk,
            datetimeindex='datetime_col',
            timeindex=None,
            dateindex=None,
            timezone='America/New_York'
        )

        # Local time with timezone information (load() removes it)
        assert str(result.index.tz) == 'America/New_York'
        assert list(result.index.tz_localize(None)) == [
            pd.Timestamp('2023-01-01 05:00'), pd.Timestamp('2023-01-01 06:00')]
        assert len(result) == 2
        assert len(result.columns) == 1

    def test_parse_data_missing_datetime_index(self):
        """Test _parse_data raises exception when datetime index is missing."""
        data = {
            'datetime_col': ['2023-01-01 10:00:00'],
            'sound_level': [65.5]
        }
        chunk = pd.DataFrame(data)
        
        with pytest.raises(Exception, match="You must provide either a datetime"):
            _parse_data(
                chunk=chunk,
                datetimeindex=None,
                timeindex=None,
                dateindex=None,
                timezone=None
            )

    def test_parse_data_incomplete_date_time_indexes(self):
        """Test _parse_data raises exception when only one of date/time index provided."""
        data = {
            'date_col': ['2023-01-01'],
            'time_col': ['10:00:00'],
            'sound_level': [65.5]
        }
        chunk = pd.DataFrame(data)
        
        with pytest.raises(Exception, match="You must provide either a datetime"):
            _parse_data(
                chunk=chunk,
                datetimeindex=None,
                timeindex='time_col',
                dateindex=None,
                timezone=None
            )


@pytest.mark.behaviour
class TestLoad:
    """Test the main load function."""

    def setup_method(self):
        """Set up test files for each test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test CSV file
        self.csv_file = os.path.join(self.temp_dir, 'test.csv')
        csv_content = """datetime,LAeq,LCeq
                        2023-01-01 10:00:00,65.5,68.1
                        2023-01-01 10:01:00,67.2,69.5
                        2023-01-01 10:02:00,64.8,67.9
                        2023-01-01 10:03:00,66.1,68.7
                    """
        with open(self.csv_file, 'w') as f:
            f.write(csv_content)
        
        self.txt_file = os.path.join(self.temp_dir, 'test.txt')
        txt_content = """date\ttime\tLAeq
                        2023-01-01\t10:00:00\t65.5
                        2023-01-01\t10:01:00\t67.2
                        2023-01-01\t10:02:00\t64.8
                    """
        with open(self.txt_file, 'w') as f:
            f.write(txt_content)

    def teardown_method(self):
        """Clean up test files after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_single_csv_file(self):
        """Test loading a single CSV file."""
        result = load(
            path=self.csv_file,
            datetimeindex=0,
            valueindexes=[1, 2],
            sep=',',
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 4
        assert len(result.columns) == 2
        assert result.index.is_monotonic_increasing

    def test_load_single_txt_file_separate_datetime(self):
        """Test loading a TXT file with separate date and time columns."""
        result = load(
            path=self.txt_file,
            dateindex=0,
            timeindex=1,
            valueindexes=[2],
            sep='\t',
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 3
        assert len(result.columns) == 1

    def test_load_multiple_files(self):
        """Test loading multiple files."""
        csv_file2 = os.path.join(self.temp_dir, 'test2.csv')
        csv_content2 = """datetime,LAeq
                        2023-01-01 11:00:00,63.2
                        2023-01-01 11:01:00,64.5
                        """
        with open(csv_file2, 'w') as f:
            f.write(csv_content2)
        
        result = load(
            path=[self.csv_file, csv_file2],
            datetimeindex=0,
            valueindexes=[1],  # Column 1 is LAeq (0-based indexing)
            sep=',',
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 4
        assert isinstance(result.index, pd.DatetimeIndex)

    @pytest.mark.slow
    def test_load_with_chunks(self):
        """Test loading with chunk processing enabled."""
        result = load(
            path=self.csv_file,
            datetimeindex=0,
            valueindexes=[1],
            sep=',',
            use_chunks=True,
            chunksize=2
        )
        
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 4

    def test_load_xls_text_file_read_as_csv(self):
        """Text file with an .xls extension is read as CSV, without
        calling an Excel engine."""
        fake_xls = os.path.join(self.temp_dir, 'fake.xls')
        with open(fake_xls, 'w') as f:
            f.write("datetime,LAeq\n2023-01-01 10:00:00,65.5\n")

        with patch('pandas.read_excel') as read_excel:
            result = load(
                path=fake_xls,
                datetimeindex=0,
                valueindexes=[1],
                sep=',',
                use_chunks=False
            )

        read_excel.assert_not_called()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    @pytest.mark.filterwarnings("error::DeprecationWarning")
    @pytest.mark.parametrize("use_chunks", [False, pytest.param(True, marks=pytest.mark.slow)])
    def test_load_binary_xls_file(self, use_chunks):
        """Binary Excel .xls workbook (title row, blank row, header on row
        2) read with the calamine engine, without deprecation warnings."""
        result = load(
            path=str(Path(__file__).parent / 'data' / 'internal'
                     / 'test_data_binary.xls'),
            datetimeindex=0,
            valueindexes=[1, 2],
            header=2,
            use_chunks=use_chunks,
            chunksize=4
        )

        assert list(result.columns) == ['LAeq', 'LCeq']
        assert len(result) == 10
        assert result.index[0] == pd.Timestamp('2023-01-01 10:00:00')
        assert result.index[-1] == pd.Timestamp('2023-01-01 10:00:09')
        np.testing.assert_allclose(result['LAeq'].values, 50 + 0.5 * np.arange(10))
        np.testing.assert_allclose(result['LCeq'].values, 60 + 0.25 * np.arange(10))

    def test_load_xlsx_saved_as_xls(self):
        """An .xlsx workbook renamed .xls is detected and read as Excel."""
        xlsx_file = os.path.join(self.temp_dir, 'test.xlsx')
        data = pd.DataFrame({
            'datetime': ['2023-01-01 10:00:00', '2023-01-01 10:01:00'],
            'LAeq': [65.5, 67.2]
        })
        data.to_excel(xlsx_file, index=False, engine='openpyxl')
        xls_file = os.path.join(self.temp_dir, 'test.xls')
        os.rename(xlsx_file, xls_file)

        result = load(
            path=xls_file,
            datetimeindex=0,
            valueindexes=[1],
            use_chunks=False
        )

        assert list(result.columns) == ['LAeq']
        assert result.index[0] == pd.Timestamp('2023-01-01 10:00:00')
        np.testing.assert_allclose(result['LAeq'].values, [65.5, 67.2])

    def test_load_xlsx_file(self):
        """Test loading XLSX file."""
        xlsx_file = os.path.join(self.temp_dir, 'test.xlsx')
        
        data = pd.DataFrame({
            'datetime': ['2023-01-01 10:00:00', '2023-01-01 10:01:00'],
            'LAeq': [65.5, 67.2]
        })
        data.to_excel(xlsx_file, index=False, engine='openpyxl')
        
        result = load(
            path=xlsx_file,
            datetimeindex=0,
            valueindexes=[1],  # Column 1 is LAeq (0-based indexing)
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_load_unsupported_extension(self):
        """Test loading file with unsupported extension."""
        bad_file = os.path.join(self.temp_dir, 'test.bad')
        with open(bad_file, 'w') as f:
            f.write("test content")
        
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load(path=bad_file, datetimeindex=0, valueindexes=[1], use_chunks=False)

    def test_load_with_header_parameter(self):
        """Test loading with different header parameter."""
        header_file = os.path.join(self.temp_dir, 'header.csv')
        with open(header_file, 'w') as f:
            f.write("File metadata\ndatetime,LAeq\n2023-01-01 10:00:00,65.5\n")
        
        result = load(
            path=header_file,
            datetimeindex=0,
            valueindexes=[1],
            header=1,
            sep=',',
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_load_no_header(self):
        """Test loading file with no header."""
        no_header_file = os.path.join(self.temp_dir, 'no_header.csv')
        with open(no_header_file, 'w') as f:
            f.write("2023-01-01 10:00:00,65.5\n2023-01-01 10:01:00,67.2\n")
        
        result = load(
            path=no_header_file,
            datetimeindex=0,
            valueindexes=[1],  # Column 1 is the value column (0-based indexing)
            header=None,
            sep=',',
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_load_duplicate_timestamps(self):
        """Test loading data with duplicate timestamps."""
        dup_file = os.path.join(self.temp_dir, 'duplicates.csv')
        with open(dup_file, 'w') as f:
            f.write("datetime,LAeq\n")
            f.write("2023-01-01 10:00:00,65.5\n")
            f.write("2023-01-01 10:00:00,66.0\n")
            f.write("2023-01-01 10:01:00,67.2\n")
        
        result = load(
            path=dup_file,
            datetimeindex=0,
            valueindexes=[1],  
            sep=',',
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result.index.is_unique
        assert len(result) == 2

    def test_load_resampling(self):
        """Test that load function applies resampling to fill gaps."""
        gap_file = os.path.join(self.temp_dir, 'gaps.csv')
        with open(gap_file, 'w') as f:
            f.write("datetime,LAeq\n")
            f.write("2023-01-01 10:00:00,65.5\n")
            f.write("2023-01-01 10:00:01,66.0\n")
            f.write("2023-01-01 10:00:02,66.5\n")
            f.write("2023-01-01 10:00:05,67.2\n") 
        
        result = load(
            path=gap_file,
            datetimeindex=0,
            valueindexes=[1],
            sep=',',
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 4

    def test_load_with_timezone(self):
        """Test loading with timezone parameter."""
        tz_file = os.path.join(self.temp_dir, 'timezone.csv')
        with open(tz_file, 'w') as f:
            f.write("datetime,LAeq\n")
            f.write("2023-01-01 10:00:00+00:00,65.5\n")
        
        result = load(
            path=tz_file,
            datetimeindex=0,
            valueindexes=[1],
            timezone='America/New_York',
            sep=',',
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result.index.tz is None

    def test_load_decimal_comma_detected(self):
        """Values with a decimal comma are converted to floats."""
        ns_file = os.path.join(self.temp_dir, 'noisesentry.csv')
        with open(ns_file, 'w') as f:
            f.write("datetime;LAeq\n")
            f.write("2023-01-01 10:00:00;65,5\n")
            f.write("2023-01-01 10:01:00;67,2\n")

        result = load(
            path=ns_file,
            datetimeindex=0,
            valueindexes=[1],
            sep=';',
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert np.isclose(result.iloc[0, 0], 65.5, rtol=1e-10)
        assert np.isclose(result.iloc[1, 0], 67.2, rtol=1e-10)

    def test_load_single_value_index(self):
        """Test load function with single value index (not as list)."""
        result = load(
            path=self.csv_file,
            datetimeindex=0,
            valueindexes=1,
            sep=',',
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 1

    def test_load_path_conversion(self):
        """Test load function converts single path to list."""
        result = load(
            path=self.csv_file,
            datetimeindex=0,
            valueindexes=[1],
            sep=',',
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_load_with_string_valueindexes(self):
        """Test loading using string column names for valueindexes."""
        result = load(
            path=self.csv_file,
            datetimeindex='datetime',
            valueindexes=['LAeq'],
            sep=',',
            use_chunks=False
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 1
        assert np.isclose(result.iloc[0, 0], 65.5, rtol=1e-10)

    def test_conflicting_index_types(self):
        """Mixing integer and string column references should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot mix column names"):
            load(
                path=self.csv_file,
                datetimeindex=0,
                valueindexes=['LAeq'],
                sep=',',
                use_chunks=False
            )

    def test_conflicting_kwargs_usecols(self):
        """Passing 'usecols' in kwargs must raise a ValueError."""
        with pytest.raises(ValueError, match="The parameter 'usecols' conflicts"):
            load(
                path=self.csv_file,
                datetimeindex=0,
                valueindexes=[1],
                usecols=[0],
                use_chunks=False
            )

    def test_conflicting_kwargs_index_col(self):
        """Passing 'index_col' in kwargs (except False) must raise a ValueError."""
        with pytest.raises(ValueError, match="The parameter 'index_col' conflicts"):
            load(
                path=self.csv_file,
                datetimeindex=0,
                valueindexes=[1],
                index_col=1,
                use_chunks=False
            )

    def test_index_col_false_allowed(self):
        """Passing index_col=False in kwargs should be accepted."""
        result = load(
            path=self.csv_file,
            datetimeindex=0,
            valueindexes=[1],
            index_col=False,
            sep=',',
            use_chunks=False
        )

        assert isinstance(result, pd.DataFrame)

    def test_load_empty_dataframe_handling(self):
        """Test load function handles case with minimal data."""
        minimal_file = os.path.join(self.temp_dir, 'minimal.csv')
        with open(minimal_file, 'w') as f:
            f.write("datetime,LAeq\n")
            f.write("2023-01-01 10:00:00,65.5\n")
        
        result = load(
            path=minimal_file,
            datetimeindex=0,
            valueindexes=[1],
            sep=',',
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    @pytest.mark.slow
    def test_load_concurrent_processing(self):
        """Test load function's concurrent processing with ProcessPoolExecutor."""
        large_file = os.path.join(self.temp_dir, 'large.csv')
        with open(large_file, 'w') as f:
            f.write("datetime,LAeq\n")
            for i in range(100):
                minute = i // 60
                second = i % 60
                f.write(f"2023-01-01 10:{minute:02d}:{second:02d},65.{i%10}\n")
        
        result = load(
            path=large_file,
            datetimeindex=0,
            valueindexes=[1],
            sep=',',
            use_chunks=True,
            chunksize=20
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 100


@pytest.mark.behaviour
class TestLoadColumnReference:
    """Test the new column reference functionality (string vs integer indices)."""

    def setup_method(self):
        """Set up test files for each test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test CSV file with named columns
        self.csv_file = os.path.join(self.temp_dir, 'test_named.csv')
        csv_content = """datetime,LAeq,LCeq,LAmax
2023-01-01 10:00:00,65.5,68.1,72.3
2023-01-01 10:01:00,67.2,69.5,74.1
2023-01-01 10:02:00,64.8,67.9,71.8
"""
        with open(self.csv_file, 'w') as f:
            f.write(csv_content)
        
        # Create file with separate date/time columns
        self.datetime_sep_file = os.path.join(self.temp_dir, 'test_sep.csv')
        sep_content = """date,time,LAeq,LCeq
2023-01-01,10:00:00,65.5,68.1
2023-01-01,10:01:00,67.2,69.5
2023-01-01,10:02:00,64.8,67.9
"""
        with open(self.datetime_sep_file, 'w') as f:
            f.write(sep_content)

    def teardown_method(self):
        """Clean up test files after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_with_string_column_names(self):
        """Test loading with string column names instead of integer indices."""
        result = load(
            path=self.csv_file,
            datetimeindex='datetime',
            valueindexes=['LAeq', 'LCeq'],
            sep=',',
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 2
        assert 'LAeq' in result.columns
        assert 'LCeq' in result.columns
        assert len(result) == 3

    def test_load_with_string_single_value(self):
        """Test loading with single string column name (not as list)."""
        result = load(
            path=self.csv_file,
            datetimeindex='datetime',
            valueindexes='LAeq',
            sep=',',
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 1
        assert 'LAeq' in result.columns

    def test_load_with_string_datetime_separate(self):
        """Test loading with string names for separate date and time columns."""
        result = load(
            path=self.datetime_sep_file,
            dateindex='date',
            timeindex='time',
            valueindexes=['LAeq', 'LCeq'],
            sep=',',
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 2
        assert 'LAeq' in result.columns
        assert 'LCeq' in result.columns
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_load_mixed_string_int_raises_error(self):
        """Test that mixing string and integer column references raises error."""
        with pytest.raises(ValueError, match="Cannot mix column names"):
            load(
                path=self.csv_file,
                datetimeindex='datetime',
                valueindexes=[1, 2],  # Integer indices with string datetime
                sep=',',
                use_chunks=False
            )

    def test_load_valueindexes_none_loads_all(self):
        """Test that valueindexes=None loads all columns."""
        result = load(
            path=self.csv_file,
            datetimeindex='datetime',
            valueindexes=None,
            sep=',',
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 3  # LAeq, LCeq, LAmax
        assert all(col in result.columns for col in ['LAeq', 'LCeq', 'LAmax'])

    def test_load_usecols_in_kwargs_raises_error(self):
        """Test that passing usecols in kwargs raises error."""
        with pytest.raises(ValueError, match="'usecols' conflicts with load"):
            load(
                path=self.csv_file,
                datetimeindex='datetime',
                valueindexes=['LAeq'],
                sep=',',
                use_chunks=False,
                usecols=['datetime', 'LAeq']
            )

    def test_load_index_col_in_kwargs_raises_error(self):
        """Test that passing index_col (except False) in kwargs raises error."""
        with pytest.raises(ValueError, match="'index_col' conflicts with load"):
            load(
                path=self.csv_file,
                datetimeindex='datetime',
                valueindexes=['LAeq'],
                sep=',',
                use_chunks=False,
                index_col=0
            )

    def test_load_index_col_false_allowed(self):
        """Test that index_col=False is allowed in kwargs."""
        result = load(
            path=self.csv_file,
            datetimeindex='datetime',
            valueindexes=['LAeq'],
            sep=',',
            use_chunks=False,
            index_col=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 1

    def test_load_with_string_names_uses_usecols(self):
        """Test that using string names properly filters columns via usecols."""
        # This test ensures only specified columns are loaded, not all
        result = load(
            path=self.csv_file,
            datetimeindex='datetime',
            valueindexes=['LAeq'],
            sep=',',
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 1
        assert 'LAeq' in result.columns
        assert 'LCeq' not in result.columns
        assert 'LAmax' not in result.columns

    def test_load_integer_indices_still_work(self):
        """Test that traditional integer indices still work correctly."""
        result = load(
            path=self.csv_file,
            datetimeindex=0,
            valueindexes=[1, 2],
            sep=',',
            use_chunks=False
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 2
        assert isinstance(result.index, pd.DatetimeIndex)

    @pytest.mark.slow
    @pytest.mark.parametrize("datetimeindex, valueindexes, expected", [
        # Integer indices, datetime column on the right
        (4, 1, ['LCeq']),
        (4, [1, 3], ['LCeq', 'LAmax']),          # ordered list
        (4, [3, 0], ['LAeq', 'LAmax']),          # non-ordered list
        (4, [3, 0, 1], ['LAeq', 'LCeq', 'LAmax']),
        # Integer indices, datetime column in the middle
        (2, [3, 0], ['LAeq', 'LAmax']),
        (2, [0, 3], ['LAeq', 'LAmax']),
        # String names (positions resolved by name, must behave the same)
        ('datetime', 'LCeq', ['LCeq']),
        ('datetime', ['LCeq', 'LAmax'], ['LCeq', 'LAmax']),
        ('datetime', ['LAmax', 'LAeq'], ['LAeq', 'LAmax']),
    ])
    def test_load_datetime_column_not_leftmost(
        self, datetimeindex, valueindexes, expected
    ):
        """Datetime column located after value columns or not selected
        columns: integer file positions must be remapped after selection."""
        values = {'LAeq': [65.5, 67.2, 64.8], 'LCeq': [68.1, 69.5, 67.9],
                  'extra': [1, 2, 3], 'LAmax': [72.3, 74.1, 71.8]}
        stamps = ['2023-01-01 10:00:00', '2023-01-01 10:01:00',
                  '2023-01-01 10:02:00']
        # Columns: LAeq(0), LCeq(1), [extra or datetime](2), LAmax(3), ...(4)
        if datetimeindex in (4, 'datetime'):
            order = ['LAeq', 'LCeq', 'extra', 'LAmax', 'datetime']
        else:
            order = ['LAeq', 'LCeq', 'datetime', 'LAmax', 'extra']
        df_file = pd.DataFrame({**values, 'datetime': stamps})[order]
        path = os.path.join(self.temp_dir, 'test_datetime_right.csv')
        df_file.to_csv(path, index=False)

        for use_chunks in (False, True):
            result = load(
                path=path,
                datetimeindex=datetimeindex,
                valueindexes=valueindexes,
                sep=',',
                use_chunks=use_chunks
            )

            assert isinstance(result.index, pd.DatetimeIndex)
            assert result.index[0] == pd.Timestamp('2023-01-01 10:00:00')
            assert len(result) == 3
            assert sorted(result.columns) == sorted(expected)
            for col in expected:
                np.testing.assert_allclose(result[col].values, values[col])

    @pytest.mark.parametrize("dateindex, timeindex, valueindexes, expected", [
        (3, 4, [0, 2], ['LAeq', 'LAmax']),           # ordered
        (4, 3, [2, 0], ['LAeq', 'LAmax']),           # swapped date/time order
        (3, 4, 1, ['LCeq']),
        ('date', 'time', ['LAmax', 'LAeq'], ['LAeq', 'LAmax']),
    ])
    def test_load_separate_date_time_not_leftmost(
        self, dateindex, timeindex, valueindexes, expected
    ):
        """Separate date and time columns on the right of the value columns."""
        # Columns: LAeq(0), LCeq(1), LAmax(2), date/time(3), time/date(4)
        if dateindex in (3, 'date'):
            header = 'LAeq,LCeq,LAmax,date,time\n'
            rows = ['65.5,68.1,72.3,2023-01-01,10:00:00',
                    '67.2,69.5,74.1,2023-01-01,10:01:00',
                    '64.8,67.9,71.8,2023-01-01,10:02:00']
        else:
            header = 'LAeq,LCeq,LAmax,time,date\n'
            rows = ['65.5,68.1,72.3,10:00:00,2023-01-01',
                    '67.2,69.5,74.1,10:01:00,2023-01-01',
                    '64.8,67.9,71.8,10:02:00,2023-01-01']
        path = os.path.join(self.temp_dir, 'test_sep_right.csv')
        with open(path, 'w') as f:
            f.write(header + '\n'.join(rows) + '\n')

        result = load(
            path=path,
            dateindex=dateindex,
            timeindex=timeindex,
            valueindexes=valueindexes,
            sep=',',
            use_chunks=False
        )

        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index[0] == pd.Timestamp('2023-01-01 10:00:00')
        assert sorted(result.columns) == sorted(expected)
        first = {'LAeq': 65.5, 'LCeq': 68.1, 'LAmax': 72.3}
        for col in expected:
            assert result[col].iloc[0] == first[col]

    @pytest.mark.parametrize("dateindex, timeindex, valueindexes", [
        (1, 0, 2),
        ('date', 'time', 'LAeq'),
    ])
    def test_load_time_column_before_date_column(
        self, dateindex, timeindex, valueindexes
    ):
        """Time column placed before the date column: dropping the time
        column must not shift which column becomes the datetime index."""
        path = os.path.join(self.temp_dir, 'test_time_first.csv')
        with open(path, 'w') as f:
            f.write('time,date,LAeq\n'
                    '10:00:00,2023-01-01,65.5\n'
                    '10:01:00,2023-01-01,67.2\n'
                    '10:02:00,2023-01-01,64.8\n')

        result = load(
            path=path,
            dateindex=dateindex,
            timeindex=timeindex,
            valueindexes=valueindexes,
            sep=',',
            use_chunks=False
        )

        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index[0] == pd.Timestamp('2023-01-01 10:00:00')
        assert list(result.columns) == ['LAeq']
        np.testing.assert_allclose(result['LAeq'].values, [65.5, 67.2, 64.8])


@pytest.mark.behaviour
class TestLoadUnnamedColumnsAndDecimalComma:
    """Columns with an empty header ('Unnamed: N') and decimal commas,
    including NoiseSentry exports (tab-separated text, also with an .xls
    extension, ending each line with a tab)."""

    NS_COLUMNS = [' L-Max dB -A ', ' LEQ dB -A ', ' L-Min dB -A ']

    def setup_method(self):
        """Set up test files for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(__file__).parent / "data" / "internal"
        idx = pd.date_range('2023-01-01 10:00', periods=5, freq='1min')
        # DatetimeIndex without a name, as after some pandas operations
        self.df = pd.DataFrame(
            {'LAeq': [60., 61., 62., 63., 64.],
            'LCeq': [70., 71., 72., 73., 74.]}, index=idx)

    def teardown_method(self):
        """Clean up test files after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, name, content):
        path = os.path.join(self.temp_dir, name)
        with open(path, 'w') as f:
            f.write(content)
        return path

    @pytest.mark.parametrize("valueindexes", [[1, 2, 3], None])
    @pytest.mark.parametrize("use_chunks", [False, pytest.param(True, marks=pytest.mark.slow)])
    def test_load_noisesentry_decimal_comma(self, valueindexes, use_chunks):
        """NoiseSentry text export with an .xls extension and decimal commas
        (values and milliseconds): values are converted, the empty trailing
        column is dropped silently."""
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            result = load(
                path=str(self.data_dir / 'test_data_decimal_comma.xls'),
                datetimeindex=0,
                valueindexes=valueindexes,
                header=1,
                sep='\t',
                use_chunks=use_chunks,
                chunksize=20
            )

        assert not any('empty header' in str(w.message) for w in record)
        assert not any(issubclass(w.category, DeprecationWarning)
                       for w in record)
        assert list(result.columns) == self.NS_COLUMNS
        assert all(pd.api.types.is_float_dtype(t) for t in result.dtypes)
        assert len(result) == 60
        assert result.index[0] == pd.Timestamp('2024-01-03 12:48:31.625')
        np.testing.assert_allclose(
            result.iloc[0].values, [60.920141, 61.020290, 60.920141])

    def test_load_noisesentry_dot(self):
        """NoiseSentry export with decimal dots: all columns loaded, empty
        trailing column dropped."""
        result = load(
            path=str(self.data_dir / 'test_data_decimal_dot.csv'),
            datetimeindex=0,
            valueindexes=None,
            header=1,
            sep='\t',
            use_chunks=False
        )

        assert list(result.columns) == self.NS_COLUMNS + [' L-PeakdB -A ']
        assert len(result) == 60
        np.testing.assert_allclose(
            result.iloc[0].values, [34.827647, 32.527647, 33.627647, 44.427647])

    def test_load_thousands_separator_not_converted(self):
        """Numbers with thousands separators (e.g. a NoiseSentry export
        resaved by Excel) are not read as decimal commas."""
        path = self._write(
            'resaved.csv',
            'datetime\tLAeq\tLCeq\n'
            '2022/09/12 15:00:00,000\t59,120932\t59,120,932\n'
            '2022/09/12 15:00:01,000\t60,020932\t60,020,932\n')

        result = load(path=path, datetimeindex=0, valueindexes=None,
                    sep='\t', use_chunks=False)

        np.testing.assert_allclose(result['LAeq'].values, [59.120932, 60.020932])
        assert not pd.api.types.is_numeric_dtype(result['LCeq'])

    def test_load_decimal_comma_conversion_error(self):
        """Detection relies on the first 100 rows; a later value that cannot
        be converted raises an error naming the column."""
        rows = ''.join(f'2023-01-01 10:{i // 60:02d}:{i % 60:02d};65,5\n'
                    for i in range(120))
        path = self._write('late_error.csv', 'datetime;LAeq\n' + rows
                        + '2023-01-01 10:02:00;ERR\n')

        with pytest.raises(ValueError, match="'LAeq'.*decimal comma"):
            load(path=path, datetimeindex=0, valueindexes=[1], sep=';',
                use_chunks=False)

    @pytest.mark.slow
    @pytest.mark.parametrize("ext", ['csv', 'xlsx'])
    @pytest.mark.parametrize("datetimeindex, valueindexes", [
        (0, [1, 2]),
        (0, None),
        ('Unnamed: 0', ['LAeq', 'LCeq']),
    ])
    def test_load_unnamed_datetime_column(self, ext, datetimeindex,
                                        valueindexes):
        """Datetime column with an empty header (DataFrame saved with an
        unnamed index) is used as the index, not dropped."""
        path = os.path.join(self.temp_dir, f'saved.{ext}')
        if ext == 'csv':
            self.df.to_csv(path)
        else:
            self.df.to_excel(path, engine='openpyxl')

        for use_chunks in (False, True):
            result = load(path=path, datetimeindex=datetimeindex,
                        valueindexes=valueindexes, sep=',',
                        use_chunks=use_chunks)

            assert list(result.index) == list(self.df.index)
            assert list(result.columns) == ['LAeq', 'LCeq']
            np.testing.assert_allclose(result.values, self.df.values)

    @pytest.mark.parametrize("valueindexes", [[1, 2], None])
    def test_load_unnamed_column_with_data_kept(self, valueindexes):
        """A value column with an empty header is kept, with a warning."""
        path = self._write(
            'unnamed_values.csv',
            'datetime,,LAeq\n'
            '2023-01-01 10:00:00,65.0,60.0\n'
            '2023-01-01 10:01:00,66.0,61.0\n')

        with pytest.warns(UserWarning, match="'Unnamed: 1' has an empty header"):
            result = load(path=path, datetimeindex=0,
                        valueindexes=valueindexes, use_chunks=False)

        assert list(result.columns) == ['Unnamed: 1', 'LAeq']
        np.testing.assert_allclose(result['Unnamed: 1'].values, [65.0, 66.0])

    @pytest.mark.parametrize("use_chunks", [False, pytest.param(True, marks=pytest.mark.slow)])
    def test_load_empty_unnamed_column_before_datetime(self, use_chunks):
        """Dropping an empty unnamed column located before the datetime
        column does not shift the datetime column position."""
        path = self._write(
            'empty_first.csv',
            ',datetime,LAeq\n'
            ',2023-01-01 10:00:00,60.0\n'
            ',2023-01-01 10:01:00,61.0\n')

        result = load(path=path, datetimeindex=1, valueindexes=None,
                    use_chunks=use_chunks)

        assert list(result.columns) == ['LAeq']
        assert result.index[0] == pd.Timestamp('2023-01-01 10:00:00')
        np.testing.assert_allclose(result['LAeq'].values, [60.0, 61.0])

    def test_load_empty_header_row_raises(self):
        """NoiseSentry export converted to Excel: the blank line is counted
        as a row, so header=1 points to it and raises a clear error;
        header=2 loads the file."""
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(['SITE_C_2021_05_17__18h30m00s'])
        ws.append([])
        ws.append(['Time (Date hh:mm:ss.ms)'] + self.NS_COLUMNS)
        ws.append(['2021/05/17 18:30:00.000', 44.13857, 53.93857, 5.83857])
        ws.append(['2021/05/17 18:30:05.000', 54.73857, 54.13857, 44.73857])
        path = os.path.join(self.temp_dir, 'converted.xlsx')
        wb.save(path)

        with pytest.raises(ValueError, match="empty header"):
            load(path=path, datetimeindex=0, valueindexes=[2], header=1,
                use_chunks=False)

        result = load(path=path, datetimeindex=0, valueindexes=[2], header=2,
                    use_chunks=False)
        np.testing.assert_allclose(result.iloc[:, 0].values,
                                [53.93857, 54.13857])


@pytest.mark.behaviour
class TestLoadIntegration:
    """Integration tests for load function.
    
    With various file formats and edge cases.
    """

    def test_load_comprehensive_csv(self):
        """Test load function with comprehensive CSV data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("timestamp,LAeq,LCeq,LAmax,LAmin\n")
            f.write("2023-01-01 08:00:00,55.2,58.1,68.3,45.2\n")
            f.write("2023-01-01 08:00:01,56.1,59.0,69.1,46.0\n")
            f.write("2023-01-01 08:00:02,54.8,57.9,67.8,44.9\n")
            f.write("2023-01-01 08:00:03,57.3,60.2,70.5,47.1\n")
            temp_file = f.name
        
        try:
            result = load(
                path=temp_file,
                datetimeindex=0,
                valueindexes=[1, 2, 3, 4],
                sep=',',
                use_chunks=False
            )
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 4
            assert len(result.columns) == 4
            assert result.index.is_monotonic_increasing
            assert isinstance(result.index, pd.DatetimeIndex)
            
            assert np.isclose(result.iloc[0, 0], 55.2, rtol=1e-10)  # LAeq
            assert np.isclose(result.iloc[0, 1], 58.1, rtol=1e-10)  # LCeq
            assert np.isclose(result.iloc[0, 2], 68.3, rtol=1e-10)  # LAmax
            assert np.isclose(result.iloc[0, 3], 45.2, rtol=1e-10)  # LAmin
            
        finally:
            os.unlink(temp_file)

    def test_load_error_handling(self):
        """Test load function error handling for invalid parameters."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("datetime,LAeq\n")
            f.write("2023-01-01 10:00:00,65.5\n")
            temp_file = f.name
        
        try:
            with pytest.raises(Exception):
                load(
                    path=temp_file,
                    valueindexes=[1],
                    sep=',',
                    use_chunks=False
                )
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    pytest.main([__file__])


@pytest.mark.behaviour
class TestLoadTimestamps:
    """Sub-second timestamps, Excel date and time cells, numeric timestamps,
    and timestamps with a UTC offset."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_csv(self, name, index, values):
        path = os.path.join(self.temp_dir, name)
        pd.Series(values, index=index, name='LAeq').to_csv(
            path, index_label='datetime')
        return path

    @pytest.mark.parametrize("freq", ['100ms', '125ms', '500ms'])
    def test_load_subsecond(self, freq):
        """Sub-second data with millisecond timestamps and one missing
        sample: a regular grid at the same interval, with one NaN row."""
        index = pd.date_range('2023-01-02 10:00:00.250', periods=20, freq=freq)
        values = 50.0 + np.arange(20)
        path = os.path.join(self.temp_dir, 'subsecond.csv')
        pd.Series(values, index=index, name='LAeq').drop(index[12]).to_csv(
            path, index_label='datetime', date_format='%Y-%m-%d %H:%M:%S.%f')

        result = load(path, datetimeindex=0, valueindexes=1, use_chunks=False)

        assert list(result.index) == list(index)
        assert result['LAeq'].isna().sum() == 1
        assert np.isnan(result.loc[index[12], 'LAeq'])
        np.testing.assert_array_equal(result['LAeq'].drop(index[12]),
                                    np.delete(values, 12))

    @pytest.mark.parametrize("dateindex, timeindex, datetimeindex", [
        ('date', 'time', None),
        ('date', 'time_text', None),
        (None, None, 'datetime'),
    ], ids=['date_time_cells', 'date_cell_time_text', 'datetime_cells'])
    def test_load_excel_date_time_cells(self, dateindex, timeindex, datetimeindex):
        """Excel date, time and datetime cells (read as Timestamp and
        datetime.time), also combined with times written as text."""
        index = pd.date_range('2023-01-02 23:58', periods=4, freq='1min')
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(['date', 'time', 'time_text', 'datetime', 'LAeq'])
        for k, t in enumerate(index):
            t = t.to_pydatetime()
            ws.append([t.date(), t.time(), t.strftime('%H:%M:%S'), t, 50.0 + k])
        path = os.path.join(self.temp_dir, 'cells.xlsx')
        wb.save(path)

        result = load(path, datetimeindex=datetimeindex, dateindex=dateindex,
                    timeindex=timeindex, valueindexes='LAeq', use_chunks=False)

        assert list(result.index) == list(index)
        np.testing.assert_array_equal(result['LAeq'], 50.0 + np.arange(4))

    @pytest.mark.parametrize("value", [1672653600, 44928.5],
                            ids=['epoch', 'excel_serial'])
    def test_load_numeric_timestamps_raise(self, value):
        """Numeric timestamps (Unix epoch, Excel serial numbers) are not
        read as dates: ValueError."""
        path = self._write_csv('numeric.csv', [value, value + 1, value + 2],
                            [50.0, 51.0, 52.0])
        with pytest.raises(ValueError, match="numeric timestamps"):
            load(path, datetimeindex=0, valueindexes=1, use_chunks=False)

    def test_load_start_of_dst(self):
        """Start of daylight saving time in Paris (31 March 2024, 02:00
        UTC+1 -> 03:00 UTC+2), 1 min data with UTC offsets: local times
        without timezone; the hour skipped by the clocks gives 60 NaN rows,
        without warning."""
        utc = pd.date_range('2024-03-30 23:00', '2024-03-31 02:59', freq='1min',
                            tz='UTC')
        local = utc.tz_convert('Europe/Paris')
        path = self._write_csv('spring.csv', [t.isoformat() for t in local],
                            np.full(len(utc), 50.0))

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = load(path, datetimeindex=0, valueindexes=1,
                        timezone='Europe/Paris', use_chunks=False)

        assert result.index.tz is None
        assert list(result.index) == list(pd.date_range(
            '2024-03-31 00:00', '2024-03-31 04:59', freq='1min'))
        skipped = ((result.index >= '2024-03-31 02:00')
                & (result.index < '2024-03-31 03:00'))
        assert result['LAeq'].isna().to_numpy().tolist() == skipped.tolist()

    def test_load_timezone_naive_raises(self):
        """Timestamps without UTC offset are already local times: a
        timezone raises a ValueError."""
        index = pd.date_range('2023-01-02 10:00', periods=5, freq='1min')
        path = self._write_csv('naive.csv', index, np.full(5, 50.0))
        with pytest.raises(ValueError, match="already local times"):
            load(path, datetimeindex=0, valueindexes=1, timezone='Europe/Paris',
                use_chunks=False)


CHUNKS = [
    (False, 10000),
    pytest.param(True, 10000, marks=pytest.mark.slow),  # change in a chunk
    pytest.param(True, DST_CHANGE, marks=pytest.mark.slow),  # between chunks
]
CHUNK_IDS = ['no_chunks', 'change_in_chunk', 'change_between_chunks']


@pytest.mark.external
class TestExternalBruitparif:
    """End of daylight saving time in Paris (25 October 2026, 03:00 UTC+2 ->
    02:00 UTC+1) in Bruitparif's LAeq,1s, whose timestamps carry their UTC
    offset, from 24 October 23:50 to 25 October 04:59:59 (local time)."""

    def _load(self, path, **kwargs):
        return load(str(path), datetimeindex=0, valueindexes=1,
                    skiprows=range(1, BRUITPARIF_ROWS - DST_LEAD + 1), **kwargs)

    @pytest.mark.parametrize("use_chunks, chunksize", CHUNKS, ids=CHUNK_IDS)
    def test_end_of_dst(self, bruitparif_1s_path, use_chunks, chunksize):
        """Converted to local time, 02:00-02:59 occurs twice: its first
        occurrence (UTC+2) is kept, with one UserWarning. The LAeq of each
        local hour from 00:00 to 04:00 is Bruitparif's LAeq of that hour
        (the UTC+2 one for 02:00)."""
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            result = self._load(bruitparif_1s_path, timezone=BRUITPARIF_TZ,
                                use_chunks=use_chunks, chunksize=chunksize)

        assert len([w for w in record if issubclass(w.category, UserWarning)]) == 1
        assert result.index.tz is None
        assert list(result.index) == list(pd.date_range(
            '2026-10-24 23:50:00', '2026-10-25 04:59:59', freq='1s'))

        hourly = pd.read_csv(EXT_PATH / "ext_data_bruitparif_h.csv")
        hourly = hourly[hourly['ts'].str.startswith('2026-10-25')
                        & (hourly['ts'] != '2026-10-25 02:00:00+01:00')]
        starts = pd.to_datetime(hourly['ts'].str[:19])
        end = pd.Timedelta(hours=1) - pd.Timedelta(seconds=1)
        got = [equivalent_level(result.loc[h:h + end].iloc[:, 0].to_numpy())
            for h in starts]
        assert len(got) == 5
        assert_close(got, hourly['LAeq_1h'])

    @pytest.mark.parametrize("use_chunks, chunksize", CHUNKS, ids=CHUNK_IDS)
    def test_end_of_dst_without_timezone(self, bruitparif_1s_path, use_chunks,
                                        chunksize):
        """Timestamps with two UTC offsets and no timezone: ValueError,
        whether the change falls inside a chunk or between two chunks."""
        with pytest.raises(ValueError, match="several UTC offsets"):
            self._load(bruitparif_1s_path, use_chunks=use_chunks,
                    chunksize=chunksize)


@pytest.mark.behaviour
class TestLoadCompressed:
    """Compressed text files, decompressed by pandas.read_csv."""

    SOURCE = Path(__file__).parent / 'data' / 'internal' / 'test_data_decimal_dot.csv'
    ARGS = dict(datetimeindex=0, valueindexes=None, header=1, sep='\t',
                use_chunks=False)

    @pytest.mark.parametrize("suffix", ['.xz', '.gz', '.bz2', '.zip'])
    def test_compressed_csv(self, tmp_path, suffix):
        """A compressed copy of a .csv file loads as the file itself."""
        path = tmp_path / f'{self.SOURCE.name}{suffix}'
        if suffix == '.zip':
            with zipfile.ZipFile(path, 'w') as archive:
                archive.write(self.SOURCE, arcname=self.SOURCE.name)
        else:
            opener = {'.xz': lzma.open, '.gz': gzip.open, '.bz2': bz2.open}[suffix]
            with open(self.SOURCE, 'rb') as src, opener(path, 'wb') as dst:
                shutil.copyfileobj(src, dst)

        pd.testing.assert_frame_equal(load(str(path), **self.ARGS),
                                      load(str(self.SOURCE), **self.ARGS))

    def test_compressed_excel(self, tmp_path):
        """Only .csv and .txt files can be compressed."""
        path = tmp_path / 'data.xlsx.gz'
        path.write_bytes(b'')
        with pytest.raises(ValueError, match=r"Unsupported file extension: "
                                            r"\.xlsx\.gz"):
            load(str(path), datetimeindex=0)

    @pytest.mark.slow
    def test_third_octave_data(self, test_data_paths):
        """The internal one-third-octave file (.csv.xz): 36 bands from 6.3 Hz
        to 20 kHz, usable by the low-frequency criteria."""
        df = load(str(test_data_paths['third_octave']), datetimeindex=0,
                valueindexes=list(range(1, 37)))
        assert df.shape[1] == 36
        assert (df.columns[0], df.columns[-1]) == ('6.3Hz', '20KHz')
        result = nm.summary.lf_periodic(
            df, method=['botsford', 'montano', 'barcelona'])
        assert len(result) == 7
        assert not result.isna().any().any()
