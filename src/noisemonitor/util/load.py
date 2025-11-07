""" Functions used to load and parse data into a dataframe suitable
for analysis.
"""

import os
import locale
import pandas as pd
import numpy as np

from datetime import datetime
from dateutil import parser
from xlrd import XLRDError
from typing import Optional, List, Union
from concurrent.futures import ProcessPoolExecutor

def load(
    path: Union[str, List[str]], 
    datetimeindex: Optional[int] = None, 
    timeindex: Optional[int] = None, 
    dateindex: Optional[int] = None, 
    valueindexes: Optional[Union[int, List[int]]] = 1, 
    header: Optional[int] = 0, 
    sep: str = '\t', 
    slm_type: Optional[str] = None, 
    timezone: Optional[str] = None,
    use_chunks: bool = True,
    chunksize: int = 10000
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
    datetimeindex: int
        column index for date and time if combined in a single column. Not to 
        be indicated if date and time are in different columns.
    timeindex: int
        column index for time. Not to be indicated if date and time are
        combined in a single column. Must be entered conjointly 
        with dateindex.
    dateindex: int
        column index for date. Not to be indicated if date and time are
        combined in a single column. Must be entered conjointly 
        with timeindex.
    valueindexes: int or list of int, default 1.
        column index or list of column indices for sound level values to which 
        averages are to be computed. The columns should contain sound levels 
        values, either weighted or unweighted and integrated over a period 
        corresponding to the refresh rate of the sound level meter (typically 
        between 1 second and several minutes, though the module will work with
        smaller or higher refresh rates). Example of relevant indices: LAeq, 
        LCeq, LZeq, LAmax, LAmin, LCpeak, etc.
    header: int, None, default 0
        row index for datasheet header. If None, the datasheet has 
        no header.
    sep: str, default '\t'
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

    Returns
    ---------- 
    DataFrame: dataframe formatted for sound level analysis when associated
    with a LevelMonitor class. Contains a datetime (or pandas Timestamp) array
    as index and corresponding equivalent sound level as first column.
    """

    df = pd.DataFrame()

    if type(path) is not list:
        path = [path]
    if type(valueindexes) is not list:
        valueindexes = [valueindexes]

    for fp in path:   
        ext = os.path.splitext(fp)[-1].lower()

        # Reading datasheet with pandas
        if ext == '.xls':
            try:
                temp = pd.read_excel(
                    fp, 
                    engine='xlrd', 
                    header=header
                    )
            except XLRDError:
                temp = pd.read_csv(
                    fp, 
                    sep=sep, 
                    header=header
                )
        elif ext == '.xlsx':
            temp = pd.read_excel(
                fp, 
                engine='openpyxl', 
                header=header
                )
        elif ext in ['.csv', '.txt']:
            temp = pd.read_csv(
                fp, 
                sep=sep, 
                header=header,
                engine='c'
                )
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        df = pd.concat([df, temp])

    if use_chunks:
        # Process chunks in parallel
        with ProcessPoolExecutor() as executor:
            chunk_size = max(1, len(df) // chunksize)
            slices = [(i * chunk_size, (i + 1) * chunk_size) 
                      for i in range((len(df) + chunk_size - 1) // chunk_size)]
            
            futures = [
                executor.submit(
                    _parse_data, 
                    df.iloc[start:stop],  # Manual slicing
                    datetimeindex, 
                    timeindex, 
                    dateindex, 
                    valueindexes, 
                    slm_type, 
                    timezone
                ) for start, stop in slices
            ]
            df = pd.concat([future.result() for future in futures])
    else:
        # Process the entire file at once
        df = _parse_data(
            df, 
            datetimeindex, 
            timeindex, 
            dateindex, 
            valueindexes, 
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

def _parse_data(chunk, datetimeindex, timeindex, dateindex, valueindexes,
                    slm_type, timezone):
    """Process a chunk of data to convert it into a DataFrame suitable for
    sound level analysis."""
    try:
        chunk = chunk.loc[:, ~chunk.columns.str.contains('^Unnamed')]
    except TypeError:
        pass

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
                a.iloc[:, dateindex], a.iloc[:, timeindex]))
        datetimeindex = dateindex
    else:
        raise Exception("You must provide either a datetime "
                        "index or time and date indexes.")
    
    chunk = chunk.rename(columns={chunk.columns[datetimeindex]: 'datetime'})

    # Convert datetime column toavoid dtype inference
    chunk['datetime'] = pd.to_datetime(chunk['datetime'])
    chunk = chunk.set_index('datetime')

    if timezone is not None:
        chunk.index = chunk.index.tz_convert(
            timezone).tz_localize(None)

    
    for valueindex in valueindexes:
        if slm_type == 'NoiseSentry':
            chunk.iloc[:, valueindex-1] = chunk.iloc[:, valueindex-1].map(
                lambda a: locale.atof(a.replace(',', '.')))

    chunk = chunk.iloc[:, [i-1 for i in valueindexes]]
    return chunk




