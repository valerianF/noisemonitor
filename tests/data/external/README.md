# External test data

Reference values from outside `noisemonitor`, used by the external validation tests (`TestExternalAccra` and `TestExternalBruitparif` in `tests/test_core.py`, `test_summary.py`, `test_profile.py` and `test_load.py`, marker `external`; fixtures and constants in `tests/conftest.py`). Both datasets are built by `data_parser.py` (`python tests/data/external/data_parser.py`; needs pandas, plus `acoustic-toolbox` and Python >= 3.11 for dataset 1), with the source values unchanged.

## Dataset 1: Accra (`ext_data_accra_h.csv`, `ext_data_accra_d.csv`)

**Source:** Clark, S. N. et al. (2021). Spatial modelling and inequalities of environmental noise in Accra, Ghana. *Scientific Reports* 11, 11113. https://doi.org/10.1038/s41598-021-90454-6. Data: Zenodo record 11223686, https://zenodo.org/records/11223686 (CC BY 4.0), files `Hr.long.csv` (md5 `3f515bde8f126e1b9809521483916c10`) and `Day.long.csv` (md5 `cf464b25b70093d044e2d3c82f82939b`): one year of measurements at 10 sites (April 2019 to March 2020), logged every minute and published as hourly and daily values.

**Tested:** daily values computed from the hourly LAeq, against the published daily values: `core.equivalent_level`, `core.lden`, `summary.periodic`, `summary.lden`, `summary.leq`, `profile.periodic`, `profile.series`.

**Processing:**
1. Download both files from Zenodo and check their md5.
2. Keep the two most complete sites, `EL` and `JT`.
3. Keep the reproducible days only. The source computed its daily values from the 1-min data, so a day with partial hours cannot be reproduced from the hourly values: a day is kept when its published `LAeq24hr` is within 1e-4 dB of the energetic mean of its hourly values (`acoustic_toolbox.decibel.dbmean`).
4. Write `ext_data_accra_h.csv` (hourly `LAeqhr`, one column per site) and `ext_data_accra_d.csv` (the rows of `Day.long.csv` for these days, all columns).

**Source conventions:** `LAeq24hr` covers 00–24 h, `LDay` 06–22 h, `LNight` 22–06 h of the same date (00–06 h and 22–24 h). `LDEN` uses day 06–19 h, evening 19–22 h (+5 dB) and night 22–06 h (+10 dB), and is empty when a period has no data.

**Tolerance:** 1e-4 dB, plus 0.005 dB for `summary.lden` and `summary.leq`, which round to 0.01 dB.

## Dataset 2: Bruitparif (`ext_data_bruitparif_s.csv.xz`, `ext_data_bruitparif_h.csv`)

**Source:** Bruitparif (CC BY 4.0). Provided by Bruitparif on request: LAeq,1s and the hourly LAeq, L10, L50, L90, and the HARMONICA index with its BGN and EVT components (Mietlicki et al., 2015).

**Tested:** hourly indicators computed from the LAeq,1s, against Bruitparif's: LAeq, L10, L50 and L90 with `core.equivalent_level` (LAeq), `summary.leq`, `profile.periodic`, `profile.series`; HARMONICA, BGN and EVT with `core.harmonica`, `summary.harmonica_periodic`. Also used by an internal check of the Traffic Noise Index of `profile.periodic`, against its formula applied to Bruitparif's hourly L10 and L90 (`TestCalculations` in `tests/test_profile.py`). Sub-second data: each LAeq,1s of the first three hours (and the 10 min before them) is split into two 0.5 s levels whose energetic mean is the LAeq,1s; LAeq and HARMONICA computed from them are checked against Bruitparif's (`bruitparif_halfsecond` in `tests/conftest.py`). End of daylight saving time: `load()` with `timezone='Europe/Paris'` on 25 October 00:00–05:00, hourly LAeq against Bruitparif's (`tests/test_load.py`).

**Processing:** The source files (one week of data) were processed to contain only two days, Friday 23 and Saturday 24 October 2026 (plus the 10 min of LAeq,1s before them), followed by the first five hours of local time of Sunday 25 October, which span the end of daylight saving time (03:00 UTC+2 → 02:00 UTC+1): the 02:00 hour occurs twice.
- `ext_data_bruitparif_s.csv.xz`: source data with LAeq,1s values compressed with xz; the tests unpack it to a temporary folder. The tests of 23 and 24 October read its first 173,400 rows only.
- `ext_data_bruitparif_h.csv`: source data for the 48 hourly rows of 23 and 24 October and the 6 of 25 October (both 02:00 hours), all columns.

**Source conventions:** Timestamps are written with their UTC offset (+02:00, then +01:00 from the second 02:00 hour of 25 October). A hourly timestamp marks the start of its integration period, as in `noisemonitor`; Bruitparif gives a value for each of the two 02:00 hours of 25 October. Hourly LAeq and percentiles use the valid seconds of the hour; statistical levels are interpolated linearly between samples. HARMONICA, BGN and EVT are given in single precision.

**Tolerance:** 1e-4 dB, plus 0.005 dB for `summary.leq`, which rounds to 0.01 dB. HARMONICA, BGN and EVT: 1e-4 for the 30 hours with no missing data (`noisemonitor` matches within 1e-6); 0.005 for the other 18 (within 3e-3). For `summary.harmonica_periodic`, 1e-4 applies to the hours of the day with no missing second in their span on both days.
