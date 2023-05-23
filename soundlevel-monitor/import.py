import pandas as pd
import os
from datetime import time, date, datetime

def load_data(filepaths, format="%Y/%m/%d %H:%M:%S,fff"):

    datasheets = []

    for fp in filepaths:   
        ext = os.path.splitext(fp)[-1].lower()

        if ext in ['.xls', '.xlsx']:
            df = pd.read_excel(fp)
        elif ext == '.csv':
            df = pd.read_csv(fp)

        datasheets.append(df)
    
    return datasheets


