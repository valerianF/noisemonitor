# External test data

Reference values from outside `noisemonitor`, used by the external validation tests (`TestExternalAccra` in `tests/test_core.py`, `test_summary.py` and `test_profile.py`, marker `external`; fixtures, constants and helpers in `tests/conftest.py`).

## Dataset 1: `ext_data_accra_h.csv`, `ext_data_accra_d.csv`

**Source:** Clark, S. N. et al. (2021). Spatial modelling and inequalities of environmental noise in Accra, Ghana. *Scientific Reports* 11, 11113. https://doi.org/10.1038/s41598-021-90454-6. Data: Zenodo record 11223686, https://zenodo.org/records/11223686 (CC BY 4.0), files `Hr.long.csv` (md5 `3f515bde8f126e1b9809521483916c10`) and `Day.long.csv` (md5 `cf464b25b70093d044e2d3c82f82939b`): one-year measurements at 10 sites, April 2019 to March 2020, logged every minute and published as hourly and daily values.

**Purpose:** Daily averages are computed from hourly data (`ext_data_accra_h.csv`) using `noisemonitor` functions and compared to the published daily values for reference (`ext_data_accra_d.csv`). The following functions are tested using this data for daily sound level averaging : 
- `core.equivalent_level`
- `core.lden`
- `summary.periodic`
- `summary.lden`
- `summary.leq`
- `profile.periodic`
- `profile.series`. 

**Processing:** Both files are built by `data_parser.py`:
1. Download `Hr.long.csv` and `Day.long.csv` from Zenodo and check their md5.
2. Keep two sites, `EL` and `JT`: the two most complete of the 10.
3. Keep the reproducible days only. The source computed its daily values from the 1-min data, so a
   day with partial hours cannot be reproduced from the hourly values. A day is kept when its published `LAeq24hr` is within 1e-4 dB of the energetic mean of its hourly values, computed with the external function`acoustic_toolbox.decibel.dbmean`.
4. Write the kept days for the selected sites, values unchanged:
   - `ext_data_accra_h.csv`: hourly LAeq (`LAeqhr`), one column per site, ISO timestamps
     (`YYYY-MM-DD HH:MM:SS`).
   - `ext_data_accra_d.csv`: the rows of `Day.long.csv`, all columns, ISO dates, sorted by site and date.

To rebuild them: `python tests/data/external/data_parser.py` (needs pandas and `acoustic-toolbox`, Python >= 3.11).

**Conventions from the source:** `LAeq24hr` covers 00–24 h, `LDay` 06–22 h, `LNight` 22–06 h of the same date (00–06 h and 22–24 h). `LDEN` uses day 06–19 h, evening 19–22 h (+5 dB) and night 22–06 h (+10 dB). It is empty when a period has no data. `IR.day` / `IR.night` are intermittency ratios (not used).

**Tolerance criterion:** 1e-4 dB, plus 0.005 dB for the functions that round their output to 0.01 dB (`summary.lden`, `summary.leq`).
