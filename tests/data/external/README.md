# External test data

## `ext_data_accra_h.csv`, `ext_data_accra_d.csv`

**Source.** Clark, S. N. et al. (2021). Spatial modelling and inequalities of environmental noise in
Accra, Ghana. *Scientific Reports* 11, 11113. https://doi.org/10.1038/s41598-021-90454-6.
Data: Zenodo record 11223686, https://zenodo.org/records/11223686 (CC BY 4.0), files `Hr.long.csv`
(md5 `3f515bde8f126e1b9809521483916c10`) and `Day.long.csv` (md5 `cf464b25b70093d044e2d3c82f82939b`):
one-year measurements at 10 sites, April 2019 to March 2020.

**Processing.** Two sites kept: `Nima` and `Ash`. Values are unchanged; only the layout and the date format were changed:
- `ext_data_accra_h.csv`: hourly LAeq (`LAeqhr`) reshaped to one column per site, ISO timestamps
  (`YYYY-MM-DD HH:MM:SS`, the start of each hour). Missing hours are empty cells. 7916 hours (Nima), 7424 (Ash).
- `ext_data_accra_d.csv`: daily rows of `Day.long.csv` for the two sites, with all their columns,
  ISO dates, sorted by site and date. 337 days (Nima), 319 (Ash).

**Definitions in the source** (checked against the hourly values): calendar days. `LAeq24hr`
covers 00–24 h, `LDay` 06–22 h, `LNight` 22–06 h of the same date (00–06 h and 22–24 h).
`LDEN` uses day 06–19 h, evening 19–22 h (+5 dB) and night 22–06 h (+10 dB). It is empty when a
period has no data. `IR.day` / `IR.night` are intermittency ratios (not used).