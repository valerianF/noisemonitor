import pandas as pd
import os
import locale
from datetime import datetime
from dateutil import parser
from xlrd import XLRDError

def load_data(files, datetimeindex=None, timeindex=None, dateindex=None, header=0, valueindex=1, sep='\t', type=None):

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
                temp.iloc[:, datetimeindex] = temp.iloc[:, datetimeindex].map(lambda a: parser.parse(a))
                temp = temp.rename(columns={temp.columns[datetimeindex]: 'datetime'})
        elif all(ind is not None for ind in [dateindex, timeindex]): 
            temp.iloc[:, dateindex] = temp.iloc[:, dateindex].map(lambda a: parser.parse(a).date())
            temp.iloc[:, timeindex] = temp.iloc[:, timeindex].map(lambda a: parser.parse(a).time())
            temp.iloc[:, dateindex] = temp.apply(lambda a: datetime.combine(a.iloc[:, dateindex], a.iloc[:, timeindex]))
            datetimeindex = dateindex
        else:
            raise Exception("You must provide either a datetime index or time and date indexes.")

        temp = temp.rename(columns={temp.columns[datetimeindex]: 'datetime', 
                                    temp.columns[valueindex]: 'sound level'})
        temp = temp.set_index('datetime')

        if type == 'NoiseSentry':
            temp.iloc[:, valueindex-1] = temp.iloc[:, valueindex-1].map(lambda a: locale.atof(a.replace(',', '.')))

        temp = temp[[temp.columns[valueindex-1]]]
        temp = temp.dropna()
        df = pd.concat([df, temp])
        
    return df


