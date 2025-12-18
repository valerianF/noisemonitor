""" Functions used to load and parse data into a dataframe suitable
for analysis.
"""

import os
import locale
import pandas as pd
import numpy as np
import sys
import warnings
import multiprocessing

from datetime import datetime
from dateutil import parser
from xlrd import XLRDError
from typing import Optional, List, Union
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

def load(
    path: Union[str, List[str]], 
    datetimeindex: Optional[Union[int, str]] = None, 
    timeindex: Optional[Union[int, str]] = None, 
    dateindex: Optional[Union[int, str]] = None, 
    valueindexes: Optional[Union[int, str, List[Union[int, str]]]] = 1, 
    header: Optional[int] = 0, 
    sep: str = ',', 
    slm_type: Optional[str] = None, 
    timezone: Optional[str] = None,
    use_chunks: bool = True,
    chunksize: int = 10000,
    **kwargs
) -> pd.DataFrame:
    """Take one or several datasheets with date and time indicators
    in combined in one column or across two columns, and sound level measured 
    with a sound level monitor as input and return a DataFrame suitable for 
    sound level averaging and descriptors computation provided in
    the LevelMonitor class. Date and time are automatically parsed with 
    dateutil package. 

    Parameters
    ---------- 
    path: str or list of str
        absolute or relative pathname (or list of pathnames) to convert as a 
        sound level DataFrame. File(s) format can either be .csv, .xls, .xlsx, 
        or .txt
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
        no header.
    sep: str, default ','
        separator if reading .csv file(s).
    slm_type: str, default None
        performs specific parsing operation for known sound level monitor
        manufacturers. For now will only respond to 'NoiseSentry' as input, 
        replacing the commas by dots in the sound level data to convert sound
        levels to floats.
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
    if type(valueindexes) is not list:
        valueindexes = [valueindexes]

    # Determine if using column names (strings) or indices (integers)
    # Check all index parameters to ensure consistency
    all_indices = [datetimeindex, timeindex, dateindex] + valueindexes
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
            try:
                temp = pd.read_excel(
                    fp, 
                    engine='xlrd', 
                    header=header,
                    usecols=usecols,
                    **kwargs
                    )
            except XLRDError:
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
        
        df = pd.concat([df, temp])

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
                        slm_type, 
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
                slm_type, 
                timezone)
    else:
        df = _parse_data(
            df, 
            datetimeindex, 
            timeindex, 
            dateindex, 
            slm_type, 
            timezone)

    # Ensure the index is unique and sorted
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    # Resample the data to fill gaps based on the interval between rows
    if len(df) > 2:
        interval = df.index[2] - df.index[1]
        resample_freq = f'{int(interval.total_seconds())}s'
        df = df.resample(resample_freq).asfreq()
        
    return df

def _parse_data(chunk, datetimeindex, timeindex, dateindex,
                    slm_type, timezone):
    """Process a chunk of data to convert it into a DataFrame suitable for
    sound level analysis."""
    try:
        # Only try to remove Unnamed columns if columns are string type
        if hasattr(chunk.columns, 'str'):
            chunk = chunk.loc[:, ~chunk.columns.str.contains('^Unnamed')]
    except (TypeError, AttributeError):
        pass

    # Convert column names (strings) to column indices (integers)
    if datetimeindex is not None and isinstance(datetimeindex, str):
        datetimeindex = chunk.columns.get_loc(datetimeindex)
    if dateindex is not None and isinstance(dateindex, str):
        dateindex = chunk.columns.get_loc(dateindex)
    if timeindex is not None and isinstance(timeindex, str):
        timeindex = chunk.columns.get_loc(timeindex)

    if datetimeindex is not None:
        if not isinstance(chunk.iloc[0, datetimeindex], pd.Timestamp):
            chunk.iloc[:, datetimeindex] = chunk.iloc[:, datetimeindex].map(
                lambda a: parser.parse(a))
    elif all(ind is not None for ind in [dateindex, timeindex]): 
        chunk.iloc[:, dateindex] = chunk.iloc[:, dateindex].map(
            lambda a: parser.parse(a).date())
        chunk.iloc[:, timeindex] = chunk.iloc[:, timeindex].map(
            lambda a: parser.parse(a).time())
        chunk.iloc[:, dateindex] = chunk.apply(
            lambda a: datetime.combine(
                a.iloc[dateindex], a.iloc[timeindex]), axis=1)
        datetimeindex = dateindex
    else:
        raise Exception("You must provide either a datetime "
                        "index or time and date indexes.")
    
    chunk = chunk.rename(columns={chunk.columns[datetimeindex]: 'datetime'})

    # Convert datetime column to avoid dtype inference
    chunk['datetime'] = pd.to_datetime(chunk['datetime'])
    chunk = chunk.set_index('datetime')

    if timezone is not None:
        chunk.index = chunk.index.tz_convert(
            timezone).tz_localize(None)

    # Apply NoiseSentry formatting only to value columns, not datetime
    if slm_type == 'NoiseSentry':
        for col in chunk.columns:
            # Convert European decimal format only for numeric columns
            chunk[col] = chunk[col].map(
                lambda a: (
                    locale.atof(str(a).replace(',', '.'))
                    if pd.notna(a) else a
                )
            )

    return chunk




