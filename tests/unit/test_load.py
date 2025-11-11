"""Unit tests for load module functions."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, date, time
from unittest.mock import patch, mock_open
from xlrd import XLRDError

from noisemonitor.util.load import load, _parse_data


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
            datetimeindex=0,
            timeindex=None,
            dateindex=None,
            valueindexes=[1],  # Column 1 is 'sound_level' (1-based from original columns)
            slm_type=None,
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
            timeindex=1,
            dateindex=0,
            valueindexes=[2],  # Column 2 is 'sound_level' (0-based indexing)
            slm_type=None,
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
            datetimeindex=0,
            timeindex=None,
            dateindex=None,
            valueindexes=[1, 2, 3],  # Columns 1, 2, 3 are LAeq, LCeq, LAmax (0-based indexing)
            slm_type=None,
            timezone=None
        )
        
        assert len(result.columns) == 3
        assert list(result.iloc[0, :]) == [65.5, 68.1, 72.3]

    def test_parse_data_noisesentry_format(self):
        """Test _parse_data with NoiseSentry SLM type (comma decimal separator)."""
        data = {
            'datetime_col': ['2023-01-01 10:00:00', '2023-01-01 10:01:00'],
            'sound_level': ['65,5', '67,2']  # European decimal format
        }
        chunk = pd.DataFrame(data)
        
        result = _parse_data(
            chunk=chunk,
            datetimeindex=0,
            timeindex=None,
            dateindex=None,
            valueindexes=[1],
            slm_type='NoiseSentry',
            timezone=None
        )
        
        assert np.isclose(result.iloc[0, 0], 65.5, rtol=1e-10)
        assert np.isclose(result.iloc[1, 0], 67.2, rtol=1e-10)

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
            datetimeindex=0,
            timeindex=None,
            dateindex=None,
            valueindexes=[1],
            slm_type=None,
            timezone='America/New_York'
        )
        
        assert result.index.tz is None
        assert len(result) == 2

    def test_parse_data_unnamed_columns(self):
        """Test _parse_data removes unnamed columns."""
        data = {
            'datetime_col': ['2023-01-01 10:00:00', '2023-01-01 10:01:00'],
            'sound_level': [65.5, 67.2],
            'Unnamed: 2': [np.nan, np.nan]
        }
        chunk = pd.DataFrame(data)
        
        result = _parse_data(
            chunk=chunk,
            datetimeindex=0,
            timeindex=None,
            dateindex=None,
            valueindexes=[1],
            slm_type=None,
            timezone=None
        )
        
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
                valueindexes=[1],
                slm_type=None,
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
                timeindex=1,
                dateindex=None,
                valueindexes=[2],
                slm_type=None,
                timezone=None
            )


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

    def test_load_excel_fallback_to_csv(self):
        """Test Excel file loading that falls back to CSV reading."""
        fake_xls = os.path.join(self.temp_dir, 'fake.xls')
        with open(fake_xls, 'w') as f:
            f.write("datetime,LAeq\n2023-01-01 10:00:00,65.5\n")
        
        with patch('pandas.read_excel', side_effect=XLRDError("Test error")):
            result = load(
                path=fake_xls,
                datetimeindex=0,
                valueindexes=[1],
                sep=',',
                use_chunks=False
            )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

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

    def test_load_noisesentry_format(self):
        """Test loading NoiseSentry format data."""
        ns_file = os.path.join(self.temp_dir, 'noisesentry.csv')
        with open(ns_file, 'w') as f:
            f.write("datetime;LAeq\n")
            f.write("2023-01-01 10:00:00;65,5\n")
            f.write("2023-01-01 10:01:00;67,2\n")
        
        result = load(
            path=ns_file,
            datetimeindex=0,
            valueindexes=[1],
            slm_type='NoiseSentry',
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


class TestLoadIntegration:
    """Integration tests for load function with various file formats and edge cases."""

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