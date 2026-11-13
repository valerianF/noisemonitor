# External test data

Reference values from outside `noisemonitor`, used by the external validation tests (`TestExternalAccra`, `TestExternalBruitparif`, `TestExternalTartu`, `TestExternalBands` and `TestExternalLowFrequency` in `tests/test_core.py`, `test_summary.py`, `test_profile.py` and `test_load.py`, marker `external`; fixtures and constants in `tests/conftest.py`). The data files of datasets 1 to 3 are built by `data_parser.py` (`python tests/data/external/data_parser.py`; needs pandas, plus `acoustic-toolbox` and Python >= 3.11 for dataset 1), with the source values unchanged.

## Dataset 1: Accra (`ext_data_accra_h.csv`, `ext_data_accra_d.csv`)

**Source:** Clark, S. N. et al. (2021). Spatial modelling and inequalities of environmental noise in Accra, Ghana. *Scientific Reports* 11, 11113. https://doi.org/10.1038/s41598-021-90454-6. Data: Zenodo record 11223686, https://zenodo.org/records/11223686 (CC BY 4.0), files `Hr.long.csv` (md5 `3f515bde8f126e1b9809521483916c10`) and `Day.long.csv` (md5 `cf464b25b70093d044e2d3c82f82939b`): one year of measurements at 10 sites (April 2019 to March 2020), logged every minute and published as hourly and daily values.

**Tested:** daily values computed from the hourly $L_{Aeq}$, against the published daily values: `core.equivalent_level`, `core.lden`, `summary.periodic`, `summary.lden`, `summary.leq`, `profile.periodic`, `profile.series`.

**Processing:**
1. Download both files from Zenodo and check their md5.
2. Keep the two most complete sites, `EL` and `JT`.
3. Keep the reproducible days only. The source computed its daily values from the 1-min data, so a day with partial hours cannot be reproduced from the hourly values: a day is kept when its published `LAeq24hr` is within 1e-4 dB of the energetic mean of its hourly values (`acoustic_toolbox.decibel.dbmean`).
4. Write `ext_data_accra_h.csv` (hourly `LAeqhr`, one column per site) and `ext_data_accra_d.csv` (the rows of `Day.long.csv` for these days, all columns).

**Source conventions:** `LAeq24hr` covers 00–24 h, `LDay` 06–22 h, `LNight` 22–06 h of the same date (00–06 h and 22–24 h). `LDEN` uses day 06–19 h, evening 19–22 h (+5 dB) and night 22–06 h (+10 dB), and is empty when a period has no data.

**Tolerance:** 1e-4 dB, plus 0.005 dB for `summary.lden` and `summary.leq`, which round to 0.01 dB.

## Dataset 2: Bruitparif (`ext_data_bruitparif_s.csv.xz`, `ext_data_bruitparif_h.csv`)

**Source:** Bruitparif (CC BY 4.0). Provided by Bruitparif on request: $L_{Aeq,1s}$ and the hourly $L_{Aeq}$, $L_{10}$, $L_{50}$, $L_{90}$, and the HARMONICA index with its BGN and EVT components (Mietlicki et al., 2015).

**Tested:** hourly indicators computed from the $L_{Aeq,1s}$, against Bruitparif's: $L_{Aeq}$, $L_{10}$, $L_{50}$ and $L_{90}$ with `core.equivalent_level` ($L_{Aeq}$), `summary.leq`, `profile.periodic`, `profile.series`; HARMONICA, BGN and EVT with `core.harmonica`, `summary.harmonica_periodic`. Also used by an internal check of the Traffic Noise Index of `profile.periodic`, against its formula applied to Bruitparif's hourly $L_{10}$ and $L_{90}$ (`TestCalculations` in `tests/test_profile.py`). Sub-second data: each $L_{Aeq,1s}$ of the first three hours (and the 10 min before them) is split into two 0.5 s levels whose energetic mean is the $L_{Aeq,1s}$; $L_{Aeq}$ and HARMONICA computed from them are checked against Bruitparif's (`bruitparif_halfsecond` in `tests/conftest.py`). End of daylight saving time: `load()` with `timezone='Europe/Paris'` on 25 October 00:00–05:00, hourly $L_{Aeq}$ against Bruitparif's (`tests/test_load.py`).

**Processing:** The source files (one week of data) were processed to contain only two days, Friday 23 and Saturday 24 October 2026 (plus the 10 min of $L_{Aeq,1s}$ before them), followed by the first five hours of local time of Sunday 25 October, which span the end of daylight saving time (03:00 UTC+2 → 02:00 UTC+1): the 02:00 hour occurs twice.
- `ext_data_bruitparif_s.csv.xz`: source data with $L_{Aeq,1s}$ values compressed with xz; the tests unpack it to a temporary folder. The tests of 23 and 24 October read its first 173,400 rows only.
- `ext_data_bruitparif_h.csv`: source data for the 48 hourly rows of 23 and 24 October and the 6 of 25 October (both 02:00 hours), all columns.

**Source conventions:** Timestamps are written with their UTC offset (+02:00, then +01:00 from the second 02:00 hour of 25 October). A hourly timestamp marks the start of its integration period, as in `noisemonitor`; Bruitparif gives a value for each of the two 02:00 hours of 25 October. Hourly $L_{Aeq}$ and percentiles use the valid seconds of the hour; statistical levels are interpolated linearly between samples. HARMONICA, BGN and EVT are given in single precision.

**Tolerance:** 1e-4 dB, plus 0.005 dB for `summary.leq`, which rounds to 0.01 dB. HARMONICA, BGN and EVT: 1e-4 for the 30 hours with no missing data (`noisemonitor` matches within 1e-6); 0.005 for the other 18 (within 3e-3). For `summary.harmonica_periodic`, 1e-4 applies to the hours of the day with no missing second in their span on both days.

## Dataset 3: Tartu (`ext_data_tartu_s.csv.xz`, `ext_data_tartu_h.csv`)

**Source:** Baclet, S., & Rumpler, R. (2024). Performance of a generalised algorithm for the detection of noise events from road traffic in a real urban area: A simulation study. *Applied Acoustics*, 228, 110337. https://doi.org/10.1016/j.apacoust.2024.110337. Data provided by the first author on request (CC BY 4.0), from the modelling chain of the paper: calibrated microscopic traffic simulation of Tartu, Estonia, over one day, and CNOSSOS-EU emission and propagation (NoiseModelling). Simulated outdoor $L_{Aeq,1s}$ at 20 façade receivers over 24 h, and the hourly numbers of noise events computed from them by the authors' implementation of the generalised algorithm of Brown and De Coensel (2018), for 128 parameter sets.

**Tested:** hourly numbers of noise events with `profile.nne` (and `core.noise_events`): 1 h windows over the day, every parameter set, at three receivers, against the source's counts (`TestExternalTartu` in `tests/test_profile.py`).

**Processing:** Three of the 20 receivers, at the source's quantiles 0.125, 0.525 and 0.925 of the outdoor $L_{Aeq,24h}$ (`pk` 10428, 13914 and 5080: 39.1, 46.6 and 59.7 dB(A)), values unchanged.
- `ext_data_tartu_s.csv.xz`: columns `second_of_day`, `hour`, `timestep` (second of the hour) and `R<pk>` (outdoor $L_{Aeq,1s}$ in dB(A)), compressed with xz.
- `ext_data_tartu_h.csv`: the 24 hourly rows of the three receivers (`pk`, `hour`), one column per parameter set, all columns.

**Source conventions:**
- Source data is outdoor sound levels, but events are counted on indoor levels (Eqs. 3 and 4 of the paper): $L_{indoors} = 10 \log_{10} \left( 10^{(L_{outdoors} - A_{building})/10} + 10^{35/10} \right)$, with a façade attenuation $A_{building}$ = 5 dB (window open, `OP`) or 25 dB (window closed, `CL`) and 35 dB(A) of indoor noise. Seconds with no vehicle are taken as $L_{outdoors}$ = 0 dB(A).
- Each clock hour [hh:00, hh+1:00) is a record of its own, with a threshold computed from its indoor levels: fixed (40 to 55 dB(A)), or $L_{Aeq}$, $L_{50}$ or $L_{90}$ (linear interpolation between samples) plus an exceedance of 3, 5, 10 or 15 dB.
- An exceedance is a run of seconds strictly above the threshold. The first exceedance of the hour is an event, even when the hour starts within it.
- Parameter sets are named `<threshold>E<exceedance>G<gap><window>` as in Table 2 of the paper (e.g., `L90E10G30CL`; `T45E00G05OP` for a fixed threshold of 45 dB(A)). The paper itself uses minimum gaps of 3 to 30 s and fixed thresholds of 45 to 75 dB(A).

**Tolerance:** none, the counts are equal.

## Dataset 4: Published tables (constants in `tests/conftest.py`)

Values copied from published tables, without data files.

**ISO 266:1997** (Acoustics - Preferred frequencies), Table 1: preferred frequencies (`ISO_266_PREFERRED`) and calculated frequencies (`ISO_266_CALCULATED`), 1.00 to 800, read in Hz and in kHz. **Tested:** the band labels of `core._parse_freq_columns` (`TestExternalBands` in `tests/test_core.py`).

**Montano, W. A., & Gushiken, E. I. (2023).** Summary of "Predicting and assessing low-frequency sound using its own bandwidth". *Proceedings of Meetings on Acoustics*, 52, 040010. https://doi.org/10.1121/2.0001880. Table 2 (`MONTANO_TABLE2`): Z-weighted levels of the bands from 16 Hz to 200 Hz. **Tested:** `summary.lf_periodic`, `profile.lf_series` and `profile.lf_periodic` with `method='montano'` (`TestExternalLowFrequency` in `tests/test_summary.py` and `tests/test_profile.py`). **Source conventions:** the A- and C-weighted energetic sums of the table's levels are the values the paper gives for its Fig. 3 ($C_{LF}$ = 68.1 dB, $A_{LF}$ = 51.4 dB, $C_{LF} - A_{LF}$ = 16.7 dB, `MONTANO_FIG3C`).

**Generalitat de Catalunya (2012).** Guies per a la gestió i avaluació de la contaminació acústica, 10. Motor que genera soroll a baixa freqüència (A6), Departament de Territori i Sostenibilitat, September 2012. https://mediambient.repositori.gencat.cat/bitstream/handle/20.500.14343/794/2012_10._avaluacio_baixa_frequencia.pdf?sequence=1&isAllowed=y. Tables 3 and 5 (`GENCAT_A6`, keys `'T3'` and `'T5'`): Z-weighted levels of the bands from 20 Hz to 160 Hz (energetic means of three measurements), and the guide's $L_{Aeq,20-160Hz}$, $L_{Ceq,20-160Hz}$ and $L_f$. **Tested:** `summary.lf_periodic`, `profile.lf_series` and `profile.lf_periodic` with `method='barcelona'` (`TestExternalLowFrequency`).

**Tolerance:** 0.05 dB (values published to 0.1 dB), plus 1e-4 dB.
