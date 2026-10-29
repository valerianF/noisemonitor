"""Build the Accra test data (Clark et al., 2021) from its Zenodo archive.

Writes ext_data_accra_h.csv and ext_data_accra_d.csv next to this file,
keeping only the days whose published LAeq24hr is reproducible from the
published hourly values (see README.md). Requires pandas and
acoustic-toolbox (Python >= 3.11); neither the script nor acoustic-toolbox
is needed to run the tests.

Usage: python tests/data/external/data_parser.py
"""
import hashlib
import io
import urllib.request
from pathlib import Path

import pandas as pd
from acoustic_toolbox.decibel import dbmean  # energetic mean

RECORD = 'https://zenodo.org/records/11223686/files/'
MD5 = {'Hr.long.csv': '3f515bde8f126e1b9809521483916c10',
       'Day.long.csv': 'cf464b25b70093d044e2d3c82f82939b'}
SITES = ['EL', 'JT']  # the two most complete of the year-long sites
TOL = 1e-4  # dB
OUT = Path(__file__).parent


def download(name):
    """Read an archived file, checked against its md5."""
    with urllib.request.urlopen(RECORD + name + '?download=1') as response:
        raw = response.read()
    assert hashlib.md5(raw).hexdigest() == MD5[name], f'{name} has changed'
    return pd.read_csv(io.BytesIO(raw))


hourly = download('Hr.long.csv')
daily = download('Day.long.csv')
hourly = hourly[hourly['ID'].isin(SITES)].dropna(subset=['LAeqhr'])
daily = daily[daily['ID'].isin(SITES)]
hourly['date_hour'] = pd.to_datetime(hourly['date_hour'], format='%d/%m/%Y %H:%M')
hourly['date'] = hourly['date_hour'].dt.normalize()
daily['date'] = pd.to_datetime(daily['date'], format='%d/%m/%Y')

# 1. Reproducible days: published LAeq24hr within TOL of the energetic mean
#    of the published hourly values of that day
laeq24 = hourly.groupby(['ID', 'date'])['LAeqhr'].apply(dbmean)
daily = daily.join(laeq24.rename('from_hourly'), on=['ID', 'date'])
daily = (daily[(daily['LAeq24hr'] - daily['from_hourly']).abs() < TOL]
         .drop(columns='from_hourly').sort_values(['ID', 'date']))

# 2. Hourly values of the reproducible days only, one column per site
hourly = hourly.merge(daily[['ID', 'date']], on=['ID', 'date'])
hourly = hourly.pivot(index='date_hour', columns='ID', values='LAeqhr')[SITES]
hourly.columns.name = None

hourly.to_csv(OUT / 'ext_data_accra_h.csv')
daily.to_csv(OUT / 'ext_data_accra_d.csv', index=False, date_format='%Y-%m-%d')
print(daily.groupby('ID').size().to_string())
