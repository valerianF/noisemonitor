import os
import locale
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime, time
from dateutil import parser
from xlrd import XLRDError
from typing import Optional, List, Union
from concurrent.futures import ProcessPoolExecutor

def compare_plots(
    dfs: List[pd.DataFrame], 
    labels: List[str], 
    *args: str, 
    ylabel: str = "Sound Level (dBA)",
    weighting: str = "A", 
    step: bool = False, 
    figsize: tuple = (10,8), 
    fill_between: bool = None, 
    **kwargs
) -> None:
    """Compare multiple DataFrames by plotting their columns in the same plot.

    Parameters
    ---------- 
    dfs: list of DataFrames
        list of compatible DataFrames (typically generated with functions 
        load_data(), NoiseMonitor.daily() or NoiseMonitor.weekly() ),
        with a datetime, time or pandas.Timestamp index.
    labels: list of str
        list of labels for each DataFrame.
    step: bool, default False
        if set to True, will plot the data as a step function.
    figsize: tuple, default (10,8)
        figure size in inches.
    fill_between: list of tuples, default None
        list of tuples specifying the columns to use for filling in-between
        values. Each tuple should contain three column names: (lower_bound, 
        upper_bound, column_to_plot).
    *args: str
        column name(s) to be plotted.
    weighting: str, default "A"
        type of sound level data, typically A, C or Z. 
    **kwargs: any
        ylim and title arguments can be passed to matplotlib. 
    """
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=figsize)

    for df, label in zip(dfs, labels):
        for arg in args:
            ax = level_plot(df, arg, ylabel=ylabel, weighting=weighting, step=step,
                            figsize=figsize, ax=ax, fill_between=fill_between,
                            **kwargs)
            ax.lines[-1].set_label(f"{label} - {arg}")

    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return

def convert_datetime_index(df: pd.DataFrame) -> np.ndarray:
    """Converts time index from dataframe to datetime array for plotting.

    Parameters
    ---------- 
    df: DataFrame
        DataFrame with a datetime, time, or pandas.Timestamp index.
    """

    if df.index[-1] < df.index[0]:
        x1 = df.iloc[(df.index >= df.index[0])].index.map(
            lambda a: datetime.combine(datetime(1800, 10, 9), a))
        x2 = df.iloc[(df.index < df.index[0])].index.map(
            lambda a: datetime.combine(datetime(1800, 10, 10), a))
        x = x1.union(x2)
    else:
        x = df.index.map(lambda a: datetime.combine(datetime(1800, 10, 10), a))
    
    return x.to_pydatetime()

def filter_data(
    df: pd.DataFrame, 
    start_datetime: datetime, 
    end_datetime: datetime, 
    between: bool = False
) -> pd.DataFrame:
    """Filter a datetime index DataFrame (typically the output of load_data())
    between two particular dates. By setting between = True, you can filter 
    out the times that are between the two dates.

    Parameters
    ---------- 
    df: DataFrame
        a compatible DataFrame (typically generated with functions load_data(),
        NoiseMonitor.daily() or NoiseMonitor.weekly() ),
        with a datetime, time or pandas.Timestamp index.
    start_datetime: datetime object
        first date from which to filter the data. 
        Must be earlier than end_datetime.
    end_datetime: datetime object
        second date from which to filter the data. 
        Must be later than start_datetime.
    between: bool, default False
        If set to True, will filter out data that is between the two dates.
        Else and by default, will filter data that is outside the two dates.

    Returns
    ---------- 
    DataFrame: input dataframe filtered to the specified dates range.
    """
    try:
        if between: 
            return df.loc[(df.index < start_datetime) | (df.index > end_datetime)]
        return df.loc[(df.index >= start_datetime) & (df.index <= end_datetime)]
    except TypeError as e:
        raise TypeError("Invalid comparison. Please check whether a timezone "
                        "argument should be indicated when creating the "
                        "NoiseMonitor instance.") from e

def get_datetime_index(df: pd.DataFrame) -> np.ndarray:
    """Get the datetime index for plotting.

    Parameters
    ---------- 
    df: DataFrame
        DataFrame with a datetime, time, or pandas.Timestamp index.

    Returns
    ---------- 
    x: array-like
        Array of datetime objects for plotting.
    """
    if isinstance(df.index[0], pd.Timestamp):
        return df.index.to_pydatetime()
    elif isinstance(df.index[0], time):
        return convert_datetime_index(df)
    elif not isinstance(df.index[0], datetime):
        raise TypeError(f'DataFrame index must be of type datetime, \
                        time or pd.Timestamp not {type(df.index[0])}')
    
def level_plot(
        df: pd.DataFrame, 
        *args: str, 
        ylabel: str = "Sound Level (dBA)", 
        step: bool = False, 
        figsize: tuple = (10,8), 
        ax: matplotlib.axes.Axes = None,
        fill_between: bool = None, 
        **kwargs) -> matplotlib.axes.Axes:
    """Plot columns of a dataframe according to the index, using matplotlib.

    Parameters
    ---------- 
    df: DataFrame
        a compatible DataFrame (typically generated with functions load_data(),
        NoiseMonitor.daily() or NoiseMonitor.weekly() ),
        with a datetime, time or pandas.Timestamp index.
    *args: str
        column name(s) to be plotted.
    ylabel: str, default "Sound Level (dBA)"
        label for the y-axis.
    step: bool, default False
        if set to True, will plot the data as a step function.
    figsize: tuple, default (10,8)
        figure size in inches.
    ax: matplotlib.axes.Axes, default None
        Axes object to plot on. If None, a new figure and axes are created.
    fill_between: list of tuples, default None
        list of tuples specifying the columns to use for filling in-between
        values. Each tuple should contain three column names: (lower_bound, 
        upper_bound, column_to_plot).
    weighting: str, default "A"
        type of sound level data, typically A, C or Z. 
    **kwargs: any
        ylim and title arguments can be passed to matplotlib. 
    """
    x = get_datetime_index(df)
    
    if ax is None:
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize=figsize)

    for i in range(0, len(args)):
        if step:
            ax.step(x, df.loc[:, args[i]], label=args[i])
        else:
            ax.plot(x, df.loc[:, args[i]], label=args[i])

    if fill_between:
        for lower, upper, column in fill_between:
            if lower in df.columns and upper in df.columns and column in df.columns:
                ax.fill_between(x, df[lower], df[upper], alpha=0.15,
                                label=f"{column} - {lower} to {upper}")

    if any(isinstance(df.index[0], t) for t in [pd.Timestamp, datetime]):
        ax.figure.autofmt_xdate()
        ax.set_xlabel('Date (y-m-d)')
    elif isinstance(df.index[0], time):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_xlabel('Time (h:m)')

    ax.set_ylabel(ylabel)

    if "ylim" in kwargs:
        ax.set_ylim(kwargs["ylim"])
    if "title" in kwargs:
        ax.set_title(kwargs["title"])

    ax.grid(linestyle='--')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return ax

def load_data(
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
        higher or smaller refresh rates). Example of relevant indices: LAeq, 
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
    use_chunks: bool, default False
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
            futures = [executor.submit(
                parse_data, 
                chunk, 
                datetimeindex, 
                timeindex, 
                dateindex, 
                valueindexes, 
                slm_type, 
                timezone
                ) for chunk in np.array_split(df, max(1, len(df) // chunksize))]
            df = pd.concat([future.result() for future in futures])
    else:
        # Process the entire file at once
        df = parse_data(
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
        resample_freq = f'{interval.total_seconds()}S'
        df = df.resample(resample_freq).asfreq()
        
    return df

def parse_data(chunk, datetimeindex, timeindex, dateindex, valueindexes,
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
            chunk = chunk.rename(
                columns={chunk.columns[datetimeindex]: 'datetime'})
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
    
    if timezone is not None:
        chunk['datetime'] = chunk['datetime'].dt.tz_convert(
            timezone).dt.tz_localize(None)
    
    chunk = chunk.rename(columns={chunk.columns[datetimeindex]: 'datetime'})
    chunk = chunk.set_index('datetime')

    for valueindex in valueindexes:
        if slm_type == 'NoiseSentry':
            chunk.iloc[:, valueindex-1] = chunk.iloc[:, valueindex-1].map(
                lambda a: locale.atof(a.replace(',', '.')))

    chunk = chunk.iloc[:, [i-1 for i in valueindexes]]
    chunk = chunk.dropna()
    return chunk




