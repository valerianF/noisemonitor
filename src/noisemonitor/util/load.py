""" Functions used to load and parse data into a dataframe suitable
for analysis.
"""

import os
import re
import pandas as pd
import numpy as np
import sys
import warnings
import multiprocessing

from datetime import datetime
from dateutil import parser
from typing import Optional, List, Union
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

# Number of rows inspected to detect empty unnamed columns and decimal commas.
_SAMPLE_ROWS = 100
# First bytes of Excel workbooks: OLE2 (.xls) and ZIP (.xlsx).
_EXCEL_SIGNATURES = (b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1', b'PK\x03\x04')

def load(
    path: Union[str, List[str]], 
    datetimeindex: Optional[Union[int, str]] = None, 
    timeindex: Optional[Union[int, str]] = None, 
    dateindex: Optional[Union[int, str]] = None, 
    valueindexes: Optional[Union[int, str, List[Union[int, str]]]] = 1, 
    header: Optional[int] = 0, 
    sep: str = ',',
    timezone: Optional[str] = None,
    use_chunks: bool = True,
    chunksize: int = 10000,
    **kwargs
) -> pd.DataFrame:
    """Take one or several datasheets with date and time indicators
    combined in one column or across two columns, and sound level measured 
    with a sound level monitor as input (e.g., equivalent levels and/or
    third-octave levels) and return a pandas DataFrame suitable for 
    use in the noisemonitor package. Date and time are automatically parsed 
    with dateutil package.

    Columns with an empty header (e.g., trailing column created by a 
    separator at the end of each line) are dropped when their first 100 rows 
    are empty; otherwise they are kept and a UserWarning is issued. 
    Value columns read as text are converted to floats.

    Parameters
    ----------
    path: str or list of str
        absolute or relative pathname (or list of pathnames) to convert as a
        sound level DataFrame. File(s) format can either be .csv, .xls, .xlsx,
        or .txt. Files with an .xls extension are read as Excel workbooks
        or as text files (e.g. NoiseSentry exports), depending on their
        content. When several files are given, they must share the same
        format (typically, files exported by the same instrument).
    datetimeindex: int or str
        column index or column name for date and time if combined in a single 
        column. Not to be indicated if date and time are in different columns.
    timeindex: int or str
        column index or column name for time. Not to be indicated if date and 
        time are combined in a single column. Must be entered conjointly 
        with dateindex.
    dateindex: int or str
        column index or column name for date. Not to be indicated if date and 
        time are combined in a single column. Must be entered conjointly 
        with timeindex.
    valueindexes: int, str, list of int/str, or None, default 1.
        column index/name or list of column indices/names for sound level 
        values to which averages are to be computed. The columns should contain 
        sound levels values, either weighted or unweighted and integrated over 
        a period corresponding to the refresh rate of the sound level meter 
        (typically between 1 second and several minutes, though the module will 
        work with smaller or higher refresh rates). If None, all columns are loaded.
        Example of relevant indices: LAeq, LCeq, LZeq, LAmax, LAmin, LCpeak, etc. 
    header: int, None, default 0
        row index for datasheet header. If None, the datasheet has
        no header. Blank lines are not counted in text files (.csv, .txt,
        and text files with an .xls extension) but are counted in Excel
        files.
    sep: str, default ','
        separator if reading .csv file(s).
    timezone: str, default None
        when indicated, will convert the datetime index from the specified 
        timezone to a timezone unaware format.
    use_chunks: bool, default True
        whether to process the data in chunks for large datasets.
    chunksize: int, default 10000
        number of rows to read at a time for large datasets.
    **kwargs: dict
        additional keyword arguments to pass to pandas read functions 
        (read_csv, read_excel). Note: 'usecols' and 'index_col' parameters 
        are not allowed as they conflict with 'valueindexes' and 
        'datetimeindex'/'timeindex'/'dateindex' respectively.

    Returns
    ---------- 
    DataFrame: dataframe formatted for sound level analysis when associated
    with a LevelMonitor class. Contains a datetime (or pandas Timestamp) array
    as index and corresponding equivalent sound level as first column.
    """

    # Check for conflicting parameters
    if 'usecols' in kwargs:
        raise ValueError(
            "The parameter 'usecols' conflicts with load() arguments and "
            "cannot be passed in kwargs. Use 'valueindexes' instead."
        )
    
    # Allow index_col=False to prevent pandas auto-detection, but not other values
    if 'index_col' in kwargs and kwargs['index_col'] is not False:
        raise ValueError(
            "The parameter 'index_col' conflicts with load() arguments and "
            "cannot be passed in kwargs (except index_col=False). "
            "Use 'datetimeindex'/'timeindex'/'dateindex' to specify the index column."
        )

    df = pd.DataFrame()

    if type(path) is not list:
        path = [path]
    if valueindexes is not None and type(valueindexes) is not list:
        valueindexes = [valueindexes]

    all_indices = [datetimeindex, timeindex, dateindex]
    if valueindexes is not None:
        all_indices += valueindexes
    all_indices = [idx for idx in all_indices if idx is not None]
    
    uses_strings = any(isinstance(idx, str) for idx in all_indices)
    uses_ints = any(isinstance(idx, int) for idx in all_indices)
    
    if uses_strings and uses_ints:
        raise ValueError(
            "Cannot mix column names (strings) and column indices (integers). "
            "Use either all strings or all integers for column references."
        )

    # Using column names - build usecols from strings
    usecols = []
    
    # Add datetime column(s)
    if datetimeindex is not None:
        usecols.append(datetimeindex)
    elif all(ind is not None for ind in [dateindex, timeindex]):
        usecols.append(dateindex)
        usecols.append(timeindex)
    
    # Add value columns
    if valueindexes is not None:
        usecols.extend(valueindexes)
    else:
        usecols = None

    for fp in path:
        ext = os.path.splitext(fp)[-1].lower()

        # Reading datasheet with pandas
        if ext == '.xls':
            if _is_excel_file(fp):
                temp = pd.read_excel(
                    fp,
                    engine='calamine',
                    header=header,
                    usecols=usecols,
                    **kwargs
                    )
            else:
                temp = pd.read_csv(
                    fp, 
                    sep=sep, 
                    header=header,
                    usecols=usecols,
                    **kwargs
                )
        elif ext == '.xlsx':
            temp = pd.read_excel(
                fp, 
                engine='openpyxl', 
                header=header,
                usecols=usecols,
                **kwargs
                )
        elif ext in ['.csv', '.txt']:
            temp = pd.read_csv(
                fp, 
                sep=sep, 
                header=header,
                engine='c',
                usecols=usecols,
                **kwargs
                )
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        if not temp.empty:
            df = pd.concat([df, temp])

    # Refer to the datetime, date and time columns by label from now on.
    if uses_ints:
        positions = (
            sorted(set(usecols)) if usecols is not None
            else range(len(df.columns))
        )
        labels = dict(zip(positions, df.columns))
        datetimeindex, dateindex, timeindex = (
            labels.get(idx) for idx in (datetimeindex, dateindex, timeindex)
        )
    index_labels = [
        idx for idx in (datetimeindex, dateindex, timeindex) if idx is not None
    ]
    df = _drop_unnamed_columns(df, index_labels, header)
    df = _convert_decimal_comma(df, index_labels)

    if use_chunks:
        try:
            with ProcessPoolExecutor() as executor:
                num_chunks = max(1, (len(df) + chunksize - 1) // chunksize)
                slices = []
                for i in range(num_chunks):
                    start = i * chunksize
                    stop = min((i + 1) * chunksize, len(df))
                    slices.append((start, stop))
                
                futures = [
                    executor.submit(
                        _parse_data, 
                        df.iloc[start:stop],
                        datetimeindex, 
                        timeindex, 
                        dateindex,
                        timezone
                    ) for start, stop in slices
                ]
                df = pd.concat([future.result() for future in futures])
        except (BrokenProcessPool, RuntimeError, OSError) as e:
            # If multiprocessing fails fall back to single-threaded processing
            warnings.warn(
                f"Parallel processing unavailable ({type(e).__name__}), "
                "using single-threaded processing instead.",
                RuntimeWarning
            )
            df = _parse_data(
                df,
                datetimeindex,
                timeindex,
                dateindex,
                timezone)
    else:
        df = _parse_data(
            df,
            datetimeindex,
            timeindex,
            dateindex,
            timezone)

    # Ensure the index is unique and sorted
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    # Resample the data to fill gaps based on the interval between rows.
    # The interval is taken between index[1] and index[2] because some
    # devices report a wrong interval for the first two data points.
    if len(df) > 2:
        interval = df.index[2] - df.index[1]
        resample_freq = f'{int(interval.total_seconds())}s'
        df = df.resample(resample_freq, origin=df.index[2]).asfreq()
        
    return df

def _parse_data(chunk, datetimeindex, timeindex, dateindex, timezone):
    """Process a chunk of data to convert it into a DataFrame suitable for
    sound level analysis. Datetime, date and time columns are given by
    column label."""
    chunk = chunk.copy()

    if datetimeindex is not None:
        datetime_col = datetimeindex
        if not isinstance(chunk[datetime_col].iloc[0], pd.Timestamp):
            chunk[datetime_col] = chunk[datetime_col].map(
                lambda a: parser.parse(a))
    elif all(ind is not None for ind in [dateindex, timeindex]):
        datetime_col = dateindex
        dates = chunk[dateindex].map(lambda a: parser.parse(a).date())
        times = chunk[timeindex].map(lambda a: parser.parse(a).time())
        chunk[datetime_col] = [
            datetime.combine(d, t) for d, t in zip(dates, times)]
        chunk = chunk.drop(columns=timeindex)
    else:
        raise Exception("You must provide either a datetime "
                        "index or time and date indexes.")

    chunk = chunk.rename(columns={datetime_col: 'datetime'})

    # Convert datetime column to avoid dtype inference
    chunk['datetime'] = pd.to_datetime(chunk['datetime'])
    chunk = chunk.set_index('datetime')

    if timezone is not None:
        chunk.index = chunk.index.tz_convert(
            timezone).tz_localize(None)

    return chunk

def _is_excel_file(path):
    """Tell Excel workbooks from text files by their first bytes. Some
    instruments (e.g. NoiseSentry) export tab-separated text files with an
    .xls extension; an .xlsx workbook renamed .xls is also detected."""
    with open(path, 'rb') as f:
        return f.read(8).startswith(_EXCEL_SIGNATURES)

def _drop_unnamed_columns(df, index_labels, header):
    """Drop columns with an empty header (named 'Unnamed: N' by pandas)
    whose first rows are empty, e.g. the trailing column created by a
    separator at the end of each line. Only the first rows are inspected to
    keep the check inexpensive. Unnamed columns holding data are kept with
    a warning; datetime, date and time columns are always kept."""
    unnamed = [
        col for col in df.columns
        if isinstance(col, str) and re.fullmatch(r'Unnamed: \d+', col)
    ]
    if not unnamed:
        return df

    if header is not None and len(unnamed) == len(df.columns):
        raise ValueError(
            f"All loaded columns have an empty header: row {header} may not "
            "be the header row. Note that blank lines are not counted in "
            "text files (.csv, .txt) but are counted in Excel files."
        )

    head = df.iloc[:_SAMPLE_ROWS]
    empty = []
    for col in unnamed:
        if col in index_labels:
            continue
        if head[col].isna().all():
            empty.append(col)
        else:
            warnings.warn(
                f"Column '{col}' has an empty header but contains data; "
                "it is kept as a value column.",
                UserWarning
            )
    return df.drop(columns=empty)

def _convert_decimal_comma(df, index_labels):
    """Convert value columns read as text that hold numbers with a decimal
    comma (e.g. '63,723268') to floats. A column is converted when all
    non-empty values of its first rows are numbers with at most one comma,
    so that different formats (e.g. '59,120,932') are never read as
    decimal commas."""
    for col in df.columns:
        if col in index_labels or pd.api.types.is_numeric_dtype(df[col]):
            continue
        head = df[col].iloc[:_SAMPLE_ROWS].dropna()
        head = head[head.map(lambda a: isinstance(a, str))].astype(str)
        if (head.empty
                or not head.str.contains(',', regex=False).any()
                or not head.str.fullmatch(r'\s*[+-]?\d+(,\d+)?\s*').all()):
            continue
        try:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '.', regex=False))
        except ValueError as e:
            raise ValueError(
                f"Column '{col}' holds numbers with a decimal comma but some "
                f"values could not be converted: {e}"
            ) from e
    return df




