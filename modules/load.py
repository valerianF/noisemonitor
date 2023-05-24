import pandas as pd
import os
import locale
from datetime import datetime
from xlrd import XLRDError

def load_data(files, format="%Y/%m/%d %H:%M:%S,%f", sep='\t', header=0, datetimeind=0, type=None):

    df = pd.DataFrame()

    for fp in files:   
        ext = os.path.splitext(fp)[-1].lower()

        if ext == '.xls':
            try:
                temp = pd.read_excel(fp, engine='xlrd', header=header)
            except XLRDError:
                temp = pd.read_csv(fp, sep=sep, header=header)
        elif ext == '.xlsx':
            temp = pd.read_excel(fp, engine='openpyxl', header=header)
        elif ext == '.csv':
            temp = pd.read_csv(fp, sep=sep, header=header)

        temp = temp.loc[:, ~temp.columns.str.contains('^Unnamed')]
        temp.iloc[:, datetimeind] = temp.iloc[:, datetimeind].map(lambda a: datetime.strptime(a, format))

        if type == 'NoiseSentry':
            temp.iloc[:, 1:4] = temp.iloc[:, 1:4].applymap(lambda a: locale.atof(a.replace(',', '.')))

        temp = temp.rename(columns={temp.columns[datetimeind]: 'datetime'})
        temp = temp.set_index('datetime')

        df = pd.concat([df, temp])
        
    return df


