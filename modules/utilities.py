import pandas as pd
import os
import locale
from datetime import datetime
from dateutil import parser
from xlrd import XLRDError

def load_data(files, datetimeindex=None, timeindex=None, dateindex=None, 
    valueindex=1, header=0, sep='\t', type=None):
    """Take one or severall datasheets with date and time indicators
    in combined in one column or across to columns, and sound level measured 
    with a sound level monitor as input and return a DataFrame suitable for 
    sound level averaging and descriptors computation provided in
    the LevelMonitor class. Date and time are automatically parsed with 
    dateutil package. 

    Parameters
    ---------- 
    files: list of str
        list of absolute or relative pathnames to convert as a sound level
        DataFrame. If more than one pathname is in the list, they will be
        sequentially combined in the same DataFrame. File(s) format can either
        be .csv, .xls or .xlsx.
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
    type: str, default None
        performs specific parsing operation for known sound level monitor
        manufacturers. For now will only respond to 'NoiseSentry' as input, 
        replacing the commas by dots in the sound level data to convert sound
        levels to floats.

    Returns
    ---------- 
    DataFrame: dataframe formatted for sound level analysis when associated
    with a LevelMonitor class. Contains a datetime (or pandas Timestamp) array
    as index and corresponding equivalent sound level as first column.
    """

    df = pd.DataFrame()

    for fp in files:   
        ext = os.path.splitext(fp)[-1].lower()

        # Reading datasheet with pandas
        if ext == '.xls':
            try:
                temp = pd.read_excel(fp, engine='xlrd', header=header)
            except XLRDError:
                temp = pd.read_csv(fp, sep=sep, header=header)
        elif ext == '.xlsx':
            temp = pd.read_excel(fp, engine='openpyxl', header=header)
        elif ext == '.csv':
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

        temp = temp.rename(columns={temp.columns[datetimeindex]: 'datetime', 
                                    temp.columns[valueindex]: 'Leq'})
        temp = temp.set_index('datetime')

        if type == 'NoiseSentry':
            temp.iloc[:, valueindex-1] = temp.iloc[:, valueindex-1].map(
                lambda a: locale.atof(a.replace(',', '.'))
            )

        temp = temp[[temp.columns[valueindex-1]]]
        temp = temp.dropna()
        df = pd.concat([df, temp])
        
    return df


