import pandas as pd
import warnings
import os
from datetime import time, date, datetime
from xlrd import XLRDError

def load_data(files, format="%Y/%m/%d %H:%M:%S,fff", sep='\t', header=0):

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

        datasheets.append(df)
    
    return datasheets


