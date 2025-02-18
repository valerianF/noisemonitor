import os
import locale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime, time
from dateutil import parser
from xlrd import XLRDError

def load_data(path, datetimeindex=None, timeindex=None, dateindex=None, 
    valueindex=1, header=0, sep='\t', slm_type=None, timezone=None):
    """Take one or severall datasheets with date and time indicators
    in combined in one column or across to columns, and sound level measured 
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
    valueindex: int, default 1
        column index for sound level values to which averages are to be
        computed. The column should contain equivalent sound levels (Leq),
        weighted or unweighted and integrated over a period corresponding to
        the refresh rate of the sound level meter (typically between 1 second
        and one minute, though the module will work with higher or smaller 
        refresh rates).
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

    Returns
    ---------- 
    DataFrame: dataframe formatted for sound level analysis when associated
    with a LevelMonitor class. Contains a datetime (or pandas Timestamp) array
    as index and corresponding equivalent sound level as first column.
    """

    df = pd.DataFrame()

    if type(path) is not list:
        path = [path]

    for fp in path:   
        ext = os.path.splitext(fp)[-1].lower()

        # Reading datasheet with pandas
        if ext == '.xls':
            try:
                temp = pd.read_excel(fp, engine='xlrd', header=header)
            except XLRDError:
                temp = pd.read_csv(fp, sep=sep, header=header)
        elif ext == '.xlsx':
            temp = pd.read_excel(fp, engine='openpyxl', header=header)
        elif ext in ['.csv', '.txt']:
            temp = pd.read_csv(fp, sep=sep, header=header)


        try:
            temp = temp.loc[:, ~temp.columns.str.contains('^Unnamed')]
        except TypeError:
            pass

        if datetimeindex is not None:
            if not isinstance(temp.iloc[0, datetimeindex], pd.Timestamp):
                temp.iloc[:, datetimeindex] = temp.iloc[:, datetimeindex].map(
                    lambda a: parser.parse(a))
                temp = temp.rename(columns={temp.columns[datetimeindex]: 'datetime'})
        elif all(ind is not None for ind in [dateindex, timeindex]): 
            temp.iloc[:, dateindex] = temp.iloc[:, dateindex].map(
                lambda a: parser.parse(a).date())
            temp.iloc[:, timeindex] = temp.iloc[:, timeindex].map(
                lambda a: parser.parse(a).time())
            temp.iloc[:, dateindex] = temp.apply(
                lambda a: datetime.combine(
                    a.iloc[:, dateindex], a.iloc[:, timeindex]))
            datetimeindex = dateindex
        else:
            raise Exception("You must provide either a datetime index \
                            or time and date indexes.")
        
        if timezone is not None:
            temp['datetime'] = temp['datetime'].dt.tz_convert(timezone).dt.tz_localize(None)
        

        temp = temp.rename(columns={temp.columns[datetimeindex]: 'datetime', 
                                    temp.columns[valueindex]: 'Leq'})
        temp = temp.set_index('datetime')

        if slm_type == 'NoiseSentry':
            temp.iloc[:, valueindex-1] = temp.iloc[:, valueindex-1].map(
                lambda a: locale.atof(a.replace(',', '.'))
            )

        temp = temp[[temp.columns[valueindex-1]]]
        temp = temp.dropna()
        df = pd.concat([df, temp])

    df = df.sort_index()
        
    return df

def filter_data(df, start_datetime, end_datetime, between=False):
    """Filter a datetime index DataFrame (typically the output of load_data())
    between two particular dates. By setting between = True, you can filter 
    out the times that are between the two dates.

    Parameters
    ---------- 
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
    
    if between: 
        return df.loc[(df.index < start_datetime) | (df.index > end_datetime)]
    return df.loc[(df.index >= start_datetime) & (df.index <= end_datetime)]



def level_plot(df, *args, weighting="A", step=False, figsize=(10,8), **kwargs):
    """Plot columns of a dataframe according to the index, using matplotlib.

    Parameters
    ---------- 
    df: DataFrame
        a compatible DataFrame (typically generated with functions load_data(),
        LevelMonitor.daily() or LevelMonitor.weekly()),
        with a datetime, time or pd.Timestamp index.
    step: bool, default False
        if set to True, will plot the data as a step function.
    figsize: tuple, default (10,8)
        figure size in inches.
    *args: str
        column name(s) to be plotted.
    weighting: str, default "A"
        type of sound level data, typically A, C or Z. 
    **kwargs: any
        ylim argument can be passed to matplotlib. 
    """
    if isinstance(df.index[0], pd.Timestamp):
        x = df.index.to_pydatetime()
    elif isinstance(df.index[0], time):
        x = datetime_index(df)
    elif not isinstance(df.index[0], datetime):
        raise TypeError(f'DataFrame index must be of type datetime,\
                        time or pd.Timestamp not {type(df.index[0])}')
    
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=figsize)

    colors = ["#E51079", "#6344F0", "#5A89FF", "#F3C900"]

    for i in range(0, len(args)):
        if step:
            plt.step(x, df.loc[:, args[i]], label=args[i], color=colors[i])
        else:
            plt.plot(x, df.loc[:, args[i]], label=args[i], color=colors[i])

    if any(isinstance(df.index[0], t) for t in [pd.Timestamp, datetime]):
        plt.gcf().autofmt_xdate()
        plt.xlabel('Date (y-m-d)')
    elif isinstance(df.index[0], time):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xlabel('Time (h:m)')

    plt.ylabel(f'Sound Level (dB{weighting})')

    if "ylim" in kwargs:
        plt.ylim(kwargs["ylim"])

    plt.grid(linestyle='--')
    plt.xticks(rotation = 45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return

def datetime_index(df):
    """Converts time index from dataframe to datetime array for plotting."""

    if df.index[-1] < df.index[0]:
        x1 = df.iloc[(df.index >= df.index[0])].index.map(
            lambda a: datetime.combine(datetime(1800, 10, 9), a))
        x2 = df.iloc[(df.index < df.index[0])].index.map(
            lambda a: datetime.combine(datetime(1800, 10, 10), a))
        x = x1.union(x2)

    else:
        x = df.index.map(lambda a: datetime.combine(datetime(1800, 10, 10), a))
    
    return x.to_pydatetime() 





