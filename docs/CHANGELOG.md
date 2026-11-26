# Changelog

## 1.1.0

- **External validation.** Most indicators now match reference values from outside the package: they are validated against independent implementations ([status of each function](https://github.com/valerianF/noisemonitor/blob/main/docs/usage.md#software-testing-and-external-validation)).
- **Some results change.** Several fixes change computed values. If you used one of the functions below with previous versions, run it again.
- **Some code may need updating.** A few outputs were renamed and a few options removed (see [Changes that may break your code](#changes-that-may-break-your-code)).

### New features
- **Low-frequency noise**: C−A criteria from one-third-octave bands, in `summary.lf_periodic`, `profile.lf_series` and `profile.lf_periodic` (methods `'botsford'`, `'montano'`, `'barcelona'`).
- **Lden periods**: set your own day, evening and night start hours (`start_day`, `start_evening`, `start_night`), and start days at the day period with `start_at_day` (to avoid cutting night periods).
- **Loading** (`load`) has been substantially improved:
  - detects the separator of text files;
  - converts decimal commas;
  - reads compressed files (`.gz`, `.bz2`, `.xz`, `.zip`), Excel date and time cells, and sub-second data;
  - converts timestamps with a UTC offset to local time (`timezone`), across daylight saving time changes.
- **Filtering**: `filter.extreme_values(return_filtered=True)` also returns the rows removed.
- **Warnings** are now more exhaustive: you are now told when data are missing for a result, when values are filtered out, and when samples are moved or dropped to fit the time grid. They use two classes you can filter: `core.CoverageWarning` and `core.IntegrationTimeWarning`.
- **Plots**: every `display` function returns its Axes.

### Results that may change

| Function(s) | What changes |
|---|---|
| `profile.periodic` (roughness) | $dL_{min,90}$ is now the mean of the lowest 90 % of level changes, as in the source paper (it was the lowest 10 %): values are much higher. |
| `core.harmonica`, `summary.harmonica_periodic` | Follow Bruitparif's method (background level over the 10 minutes before each second). EVT could be off in the presence of missing data before. |
| `profile.nne` | An event in progress at the start of a window now counts in it (up to +1 event per window). A night past midnight is one record, and `day1`/`day2` refer to the day it starts. A window without data gives NaN (it used to give 0). |
| `summary.periodic`, `nday`, `freq_periodic` | Weeks run Monday to Sunday (they used to run Tuesday to Monday). |
| All periods and windows | A sample stamped exactly at the end of a period is no longer counted in it (it was counted in both neighbouring periods). This includes `filter.all_data`: the end date is now excluded. |
| Every $L_{den}$ (`summary.lden`, `periodic`, `freq_periodic`, `freq_indicators`, `nday`) | Day, evening and night levels are no longer rounded to 0.01 dB before being combined, which changes $L_{den}$ by up to about 0.01 dB. Only `summary.lden` still rounds, and only its final values; the others now return unrounded values. |
| `load` | The time step is the most common step in the file, not anymore the step between the 2nd and 3rd rows. Files with an irregular start no longer lose samples. |
| `weather.merge_weather` | Weather times are converted to local time (`timezone='America/Toronto'` by default), which removes a bug that sometimes lead to one-hour shift in summer. Another bug when computing rain over previous windows was fixed. Drizzle and ice pellets are now parsed too and count as rain. Stations that report conditions but no amounts now get rain and snow flags. |
| Any function with `coverage_check=True` | A first or last period only partly covered by the data now counts its missing samples, and may become NaN. |

### Changes that may break your code
- Requires Python ≥ 3.9 and pandas ≥ 2.2. `xlrd` (recently deprecated in pandas) is replaced by `python-calamine`.
- Removed: `load(slm_type=...)`, which is no longer needed, and `weathercan.flip_daterange`, since dates are now accepted in either order.
- `get_historical_stations` returns one column per climate ID (columns were station names), with a new `'name'` row.
- Some variables and parameters were renamed for API consistency:
  - `'All Flags'` / `'Neither Flags'` → `'Any Flag'` / `'No Flag'`;
  - the `nday` column is `'Number of Weeks'` or `'Number of Months'` for weekly and monthly counts instead of `'Number of Days'`;
  - `chunks` → `use_chunks` in `profile.freq_periodic` and `freq_series`. The old name still works, with a warning, until 2.0.
- Now raise a `ValueError`:
  - hours outside 0–23 in `profile.periodic`;
  - numeric timestamps (they will be accepted in a future release);
  - `timezone` given for timestamps without a UTC offset;
  - `profile.series(start_at_midnight=True)` with less than one window after midnight.

### Other fixes
- `profile.nne` accepts `(0, 24)` for a full day.
- `summary.harmonica_periodic` now also works with less than 24 h of data.
- `filter.extreme_values` no longer modifies its input and returns a copy instead.
- Stations sharing a name are no longer lost in `get_historical_stations`.
- Contingency tables keep rows that have an $L_{eq}$ but no $L_{den}$.
- Results computed in parallel keep their order.
- Display functions now accept any datetime index.

### Tests
- 531 tests on Python 3.9–3.15 (Windows, macOS, Linux), in three groups: external validation, internal validation and behaviour.
- Five reference datasets: Clark et al. (2021), Bruitparif (2026), Baclet & Rumpler (2024), Montano & Gushiken (2023), Generalitat de Catalunya (2012).
- Optional live tests against the Environment Canada weather service.
