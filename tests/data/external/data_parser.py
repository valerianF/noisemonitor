"""Build the external test data from its sources (see README.md).

Dataset 1, Accra (Clark et al., 2021), from its Zenodo archive: writes
ext_data_accra_h.csv and ext_data_accra_d.csv, keeping only the days whose
published LAeq24hr is reproducible from the published hourly values.

Dataset 2, Bruitparif: extracts two days from files provided by Bruitparif 
(LAeq,1s and the matching hourly indicators computed by
Bruitparif, CC BY 4.0), plus the 10 min of LAeq,1s before the first hour,
which the HARMONICA of that hour needs, and the first five hours of local
time of the next day, which span the end of daylight saving time. 
The source files are not downloadable: set BRUITPARIF_DIR to the folder
holding them. Writes ext_data_bruitparif_s.csv.xz and
ext_data_bruitparif_h.csv.

Dataset 3, Tartu (Baclet and Rumpler, 2024): extracts three of the twenty
receivers of files provided by S. Baclet (simulated outdoor LAeq,1s over
one day and the hourly numbers of noise events computed from them, CC BY
4.0). The source files are not downloadable: set TARTU_DIR to the folder
holding them. Writes ext_data_tartu_s.csv.xz and ext_data_tartu_h.csv.

Requires pandas, plus acoustic-toolbox (Python >= 3.11) for dataset 1;
neither the script nor acoustic-toolbox is needed to run the tests.

Usage: python tests/data/external/data_parser.py
"""
import hashlib
import io
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent

# Dataset 1: Accra
RECORD = 'https://zenodo.org/records/11223686/files/'
MD5 = {'Hr.long.csv': '3f515bde8f126e1b9809521483916c10',
    'Day.long.csv': 'cf464b25b70093d044e2d3c82f82939b'}
SITES = ['EL', 'JT']  # the two most complete of the year-long sites
TOL = 1e-4  # dB

# Dataset 2: Bruitparif
BRUITPARIF_DIR = Path('path/to/bruitparif/files')
BRUITPARIF_FILES = {'s': 'ext_data_bruitparif_s.csv',  # LAeq,1s
                    'h': 'ext_data_bruitparif_h.csv'}  # hourly indicators
# Friday 23 and Saturday 24 October 2026: one weekday and one weekend day
DAYS = ('2026-10-23', '2026-10-24')
LA95_LEAD = ('2026-10-22 23:50:00', '2026-10-22 23:59:59')  # 10 min before the first hour
# End of daylight saving time in Paris: 03:00 (UTC+2) -> 02:00 (UTC+1) on
# 25 October 2026. Local times, matching both occurrences of the 02:xx hour.
DST_SPAN = ('2026-10-25 00:00:00', '2026-10-25 04:59:59')
OFFSETS = ('+02:00', '+01:00')

# Dataset 3: Tartu
TARTU_DIR = Path('path/to/tartu/files')
TARTU_FILES = {'s': 'timeseries_24h_1s_20receivers.csv',  # outdoor LAeq,1s
               'h': 'hourly_event_counts_wide.csv'}  # hourly noise events
# Receivers (pk) at the quantiles 0.125, 0.525 and 0.925 of the source's
# 20, ranked by outdoor LAeq,24h: 39.1, 46.6 and 59.7 dB(A)
TARTU_RECEIVERS = ('10428', '13914', '5080')


def download(name):
    """Read an archived file, checked against its md5."""
    with urllib.request.urlopen(RECORD + name + '?download=1') as response:
        raw = response.read()
    assert hashlib.md5(raw).hexdigest() == MD5[name], f'{name} has changed'
    return pd.read_csv(io.BytesIO(raw))


def accra():
    from acoustic_toolbox.decibel import dbmean  # energetic mean

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


def bruitparif():
    sources = {k: BRUITPARIF_DIR / name for k, name in BRUITPARIF_FILES.items()}
    if not all(p.exists() for p in sources.values()):
        print(f'Bruitparif source files not found in {BRUITPARIF_DIR}: skipped')
        return
    # Read and write as text, so values and timestamps (with their UTC offset)
    # are copied unchanged; missing seconds stay empty
    for k, p in sources.items():
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        keep = (df['ts'].str[:10].isin(DAYS)
                | df['ts'].str[:19].between(*DST_SPAN))
        if k == 's':
            keep |= df['ts'].str[:19].between(*LA95_LEAD)
        df = df[keep]
        assert df['ts'].str[-6:].isin(OFFSETS).all()
        out = OUT / (p.name + ('.xz' if k == 's' else ''))
        df.to_csv(out, index=False, lineterminator='\n',
                compression={'method': 'xz', 'preset': 9} if k == 's' else None)
        print(f'{out.name}: {len(df)} rows, {(df.iloc[:, 1] == "").sum()} missing')


def tartu():
    sources = {k: TARTU_DIR / name for k, name in TARTU_FILES.items()}
    if not all(p.exists() for p in sources.values()):
        print(f'Tartu source files not found in {TARTU_DIR}: skipped')
        return
    # Read and write as text, so values are copied unchanged; empty cells
    # (no vehicle contributing) stay empty
    series = pd.read_csv(sources['s'], dtype=str, keep_default_na=False)
    series = series[['second_of_day', 'hour', 'timestep']
                    + [f'R{pk}' for pk in TARTU_RECEIVERS]]
    assert len(series) == 86400
    series.to_csv(OUT / 'ext_data_tartu_s.csv.xz', index=False, lineterminator='\n',
                  compression={'method': 'xz', 'preset': 9})

    counts = pd.read_csv(sources['h'], dtype=str, keep_default_na=False)
    counts = counts[counts['pk'].isin(TARTU_RECEIVERS)]
    assert len(counts) == 24 * len(TARTU_RECEIVERS)
    counts.to_csv(OUT / 'ext_data_tartu_h.csv', index=False, lineterminator='\n')
    print(f'ext_data_tartu_s.csv.xz: {len(series)} rows; ext_data_tartu_h.csv: '
          f'{len(counts)} rows, {counts.shape[1] - 2} indicators')


if __name__ == '__main__':
    accra()
    bruitparif()
    tartu()
