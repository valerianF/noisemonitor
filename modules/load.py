import pandas as pd
import os
import locale
from datetime import datetime
from xlrd import XLRDError

def load_data(files, format="%Y/%m/%d %H:%M:%S,%f", sep='\t', header=0, datetimeind=0, type=None):

    datasheets = []

    for fp in files:   
        ext = os.path.splitext(fp)[-1].lower()

        if ext == '.xls':
            try:
                df = pd.read_excel(fp, engine='xlrd', header=header)
            except XLRDError:
                df = pd.read_csv(fp, sep=sep, header=header)
        elif ext == '.xlsx':
            df = pd.read_excel(fp, engine='openpyxl', header=header)
        elif ext == '.csv':
            df = pd.read_csv(fp, sep=sep, header=header)

        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.iloc[:, datetimeind] = df.iloc[:, datetimeind].map(lambda a: datetime.strptime(a, '%Y/%m/%d %H:%M:%S,%f'))

        if type == 'NoiseSentry':
            df.iloc[:, 1:3] = df.iloc[:, 1:4].applymap(lambda a: locale.atof(a.replace(',', '.')))


        datasheets.append(df)
    return datasheets


