# API Reference

Complete reference for all noisemonitor functions and modules.

**Note:** This documentation is auto-generated from source code docstrings.

## Table of Contents
- [Loading Module](#loading-module)
- [Filter Module](#filter-module)
- [Summary Module](#summary-module)
- [Profile Module](#profile-module)
- [Display Module](#display-module)
- [Core Module](#core-module)
- [Weather Module](#weather-module)

## Loading Module

### `noisemonitor.load()`

Take one or several datasheets with date and time indicators
combined in one column or across two columns, and sound level measured 
with a sound level monitor as input (e.g., equivalent levels and/or
third-octave levels) and return a pandas DataFrame suitable for 
use in the noisemonitor package. Date and time given as text are
automatically parsed with dateutil package; date and time cells of
Excel files are also accepted. Numeric timestamps (e.g. Unix epoch or
Excel serial numbers) are not supported and raise a ValueError.

Columns with an empty header (e.g., trailing column created by a 
separator at the end of each line) are dropped when their first 100 rows 
are empty; otherwise they are kept and a UserWarning is issued. 
Value columns read as text are converted to floats.

The data are put on a regular time grid: its interval is the most
common step between consecutive timestamps, and its times the most
common position of the samples within this interval, so that missing
samples, gaps or a first sample out of step do not change it. Missing
samples are rows of NaN. Samples off the grid (e.g., timestamps with
jitter, or after an adjustment of the clock) are moved to the nearest
grid time, with a UserWarning; a sample whose nearest grid time holds a
sample closer to it is dropped, with a UserWarning. Of the samples
sharing a timestamp, the first one is kept, with a UserWarning.

**Parameters:**
- `path`:  str or list of str \
  absolute or relative pathname (or list of pathnames) to convert as a \
  sound level DataFrame. File(s) format can either be .csv, .xls, .xlsx, \
  or .txt; .csv and .txt files can be compressed (.xz, .gz, .bz2 or \
  .zip, e.g. 'data.csv.xz'). Files with an .xls extension are read as \
  Excel workbooks or as text files (e.g. NoiseSentry exports), \
  depending on their content. When several files are given, they must \
  share the same format (typically, files exported by the same \
  instrument).
- `datetimeindex`:  int or str \
  column index or column name for date and time if combined in a single \
  column. Not to be indicated if date and time are in different columns.
- `timeindex`:  int or str \
  column index or column name for time. Not to be indicated if date and \
  time are combined in a single column. Must be entered conjointly \
  with dateindex.
- `dateindex`:  int or str \
  column index or column name for date. Not to be indicated if date and \
  time are combined in a single column. Must be entered conjointly \
  with timeindex.
- `valueindexes`:  int, str, list of int/str, or None, default 1. \
  column index/name or list of column indices/names for sound level \
  values to which averages are to be computed. The columns should contain \
  sound levels values, either weighted or unweighted and integrated over \
  a period corresponding to the refresh rate of the sound level meter \
  (typically between 1 second and several minutes, though the module will \
  work with smaller or higher refresh rates, e.g. 100 ms). If None, all \
  columns are loaded. \
  Example of relevant indices: LAeq, LCeq, LZeq, LAmax, LAmin, LCpeak, etc.
- `header`:  int, None, default 0 \
  row index for datasheet header. If None, the datasheet has \
  no header. Blank lines are not counted in text files (.csv, .txt, \
  and text files with an .xls extension) but are counted in Excel \
  files.
- `sep`:  str, default None \
  separator of text files (.csv, .txt, and text files with an .xls \
  extension). If None, it is detected in each file from its first \
  100 lines from the header row: the first of tab, ';', ',' and '|' \
  found on each of these lines and the same number of times on each \
  data line; ',' if none is.
- `timezone`:  str, default None \
  timezone of the measurement site (e.g. 'Europe/Paris'), for \
  timestamps written with a UTC offset (e.g. \
  '2026-10-25T02:00:00+01:00'): they are converted to the local time \
  of this timezone, without timezone information. Timestamps without \
  a UTC offset are already local times: leave timezone to None. \
  Timestamps with several UTC offsets (e.g., an export spanning a \
  change to or from daylight saving time) require a timezone. \
  When daylight saving time ends, one hour of local time occurs twice: \
  only its first occurrence is kept, with a UserWarning. When it starts, \
  the hour skipped by the clocks gives rows without values.
- `use_chunks`:  bool, default True \
  whether to process the data in chunks for large datasets.
- `chunksize`:  int, default 10000 \
  number of rows to read at a time for large datasets.
- `**kwargs`:  dict \
  additional keyword arguments to pass to pandas read functions \
  (read_csv, read_excel). Note: 'usecols' and 'index_col' parameters \
  are not allowed as they conflict with 'valueindexes' and \
  'datetimeindex'/'timeindex'/'dateindex' respectively.

**Returns:**
- DataFrame: dataframe for sound level analysis with noisemonitor
functions: a DatetimeIndex on a regular time grid (missing samples are
rows of NaN) and the selected sound level columns.

## Filter Module

### `noisemonitor.filter.all_data()`

Filter a DataFrame (typically the output of noisemonitor.load())
with a datetime index between two particular dates. By setting
between = True, you can filter out the times that are between the two
dates. The range is half-open, [start_datetime, end_datetime), as
every period of the package: a timestamp marks the start of its
integration period, so a sample stamped end_datetime lies after the
range (e.g., start_datetime=datetime(2025, 6, 1) and
end_datetime=datetime(2025, 8, 1) keep June and July whole).

**Parameters:**
- `df`:  DataFrame \
  a DataFrame with a datetime index (e.g., the output of \
  noisemonitor.load() or noisemonitor.summary.periodic()).
- `start_datetime`:  datetime object \
  start of the range (included). \
  Must be earlier than end_datetime.
- `end_datetime`:  datetime object \
  end of the range (excluded). \
  Must be later than start_datetime.
- `between`:  bool, default False \
  If set to True, will filter out data that is between the two dates. \
  Else and by default, will filter data that is outside the two dates.

**Returns:**
- DataFrame: input dataframe filtered to the specified dates range.

### `noisemonitor.filter.extreme_values()`

Replace values in the specified column that are below min_value or
above max_value with NaN. A UserWarning gives the number of values
replaced, if any (otherwise a message is printed): an extreme level can
be a genuine noise event rather than a measurement error, so these
values should be checked.

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame containing the data.
- `column`:  Union[int, str], default None \
  The column name or index to use for calculations. If None, the first column \
  will be used.
- `min_value`:  int, default 30 \
  The minimum value threshold.
- `max_value`:  int, default 100 \
  The maximum value threshold.
- `return_filtered`:  bool, default False \
  If True, also return the rows whose value was replaced, with their \
  original values.

**Returns:**
- pd.DataFrame: copy of df with extreme values replaced by NaN (df is
left unchanged). With return_filtered=True, a tuple of two DataFrames:
this copy and the rows of df whose value was replaced.

### `noisemonitor.filter.weather_flags()`

Replace values in the specified column with NaN based on weather flags.

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame containing the data.
- `column`:  Union[int, str], default None \
  The column name or index to use for calculations. \
  If None, the first column will be used.
- `filter_wind_flag`:  bool, default True \
  Whether to filter data with Wind Speed Flag.
- `filter_rain_flag`:  bool, default True \
  Whether to filter data with Rain Flag.
- `filter_temp_flag`:  bool, default False \
  Whether to filter data with Temperature Flag.
- `filter_rel_hum_flag`:  bool, default False \
  Whether to filter data with Relative Humidity Flag.
- `filter_snow_flag`:  bool, default False \
  Whether to filter data with Snow Flag.

**Returns:**
- pd.DataFrame: DataFrame with flagged values replaced by NaN.

## Summary Module

### `noisemonitor.summary.freq_indicators()`

Compute overall Leq and Lden for each frequency band. Lden follows the
Directive 2002/49/EC (see noisemonitor.summary.lden).

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame containing octave or third-octave frequency bands as \
  columns. Column names are read as frequencies (e.g. \
  '63', '125 Hz' or '1.25k'); exact or base-2 one-third-octave \
  mid-band frequencies (e.g. 31.62 or 31.25) are labelled with the \
  nominal frequency of their band (31.5).
- `hour1`:  int, default 0 \
  Starting hour of the Leq average (0-24), as in \
  noisemonitor.summary.leq.
- `hour2`:  int, default 24 \
  Ending hour (excluded) of the Leq average (0-24). If hour1 > hour2, \
  the period wraps past midnight.
- `day1`:  str, optional \
  First day of the week in English, case-insensitive, \
  to include in the computation.
- `day2`:  str, optional \
  Last day of the week to include in the computation (wrapping past \
  Sunday if day1 comes later in the week).
- `stats`:  bool, default False \
  If set to True, the function will include statistical \
  indicators (L10, L50, L90) in addition to Leq.
- `values`:  bool, default True \
  If set to True, the function will include individual \
  day, evening, and night values in addition to Lden. On by default, \
  as in noisemonitor.summary.lden.
- `coverage_check`:  bool, default False \
  if set to True, assess data coverage and automatically filter periods \
  with insufficient data coverage and emit warnings.
- `coverage_threshold`:  float, default 0.5 \
  minimum data coverage ratio required (0.0 to 1.0).
- `start_day`:  int, default 7 \
  hour (0-23) at which the Lden day period starts.
- `start_evening`:  int, default 19 \
  hour (0-23) at which the Lden evening period starts.
- `start_night`:  int, default 23 \
  hour (0-23) at which the Lden night period starts.

**Returns:**
- pd.DataFrame:
    DataFrame with rows corresponding to indicators
    (e.g., Leq, Lden, etc.) and columns corresponding to
    frequency bands.

### `noisemonitor.summary.freq_periodic()`

Compute periodic levels for each frequency band (e.g. octave
band or third octave bands) in the input DataFrame. Lden follows the
Directive 2002/49/EC (see noisemonitor.summary.periodic).

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame containing octave or third-octave frequency bands as \
  columns. Column names are read as frequencies (e.g. \
  '63', '125 Hz' or '1.25k'); exact or base-2 one-third-octave \
  mid-band frequencies (e.g. 31.62 or 31.25) are labelled with the \
  nominal frequency of their band (31.5).
- `freq`:  str, default 'D' \
  Frequency for the computation. 'D' for daily, 'W' for weekly \
  (Monday to Sunday), and 'MS' for monthly (calendar months). \
  Periods are labelled by their start.
- `values`:  bool, default False \
  If set to True, the function will return individual day, \
  evening, and night values in addition to the Lden. Off by default, \
  as in noisemonitor.summary.periodic.
- `coverage_check`:  bool, default False \
  if set to True, assess data coverage and automatically filter periods \
  with insufficient data coverage and emit warnings. Samples of a \
  period missing from the data (first or last period cut by the start \
  or end of the data) count as missing.
- `coverage_threshold`:  float, default 0.5 \
  minimum data coverage ratio required (0.0 to 1.0).
- `start_day`:  int, default 7 \
  hour (0-23) at which the Lden day period starts.
- `start_evening`:  int, default 19 \
  hour (0-23) at which the Lden evening period starts.
- `start_night`:  int, default 23 \
  hour (0-23) at which the Lden night period starts.
- `start_at_day`:  bool, default False \
  if True, days, weeks and months start at start_day instead of \
  midnight (see noisemonitor.summary.periodic).

**Returns:**
- pd.DataFrame: DataFrame with weekly or daily levels for each
    frequency band.

### `noisemonitor.summary.harmonica_periodic()`

Compute the average HARMONICA indicators for each hour of a day:
the hourly values of core.harmonica, averaged over the days for each
hour of the day (arithmetic mean). Optionally, can return the average
values for specific days of the week. Coverage rules are those of
core.harmonica: a 10 min LA95 window with fewer than 50% of its
expected values gives no LA95, and an hour with less than 80% of its
expected values is NaN; NaN hours are left out of the mean, with a
CoverageWarning. Sub-second data are first averaged over each whole
second (LAeq,1s), as in core.harmonica.

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame with a datetime index and LAeq values at 1 s or finer.
- `column`:  int or str, default 0 \
  column index (int) or column name (str) to use for calculations. \
  If None, the first column of the DataFrame will be used.
- `use_chunks`:  bool default True \
  whether to process the data in chunks for large datasets.
- `day1`:  Optional[str], default None \
  First day of the week to include in the calculation. The 10 min \
  before each selected day, when available, are used for the LA95 of \
  its first seconds, as for any other hour.
- `day2`:  Optional[str], default None \
  Last day of the week to include in the calculation.

**Returns:**
- pd.DataFrame: DataFrame with time index and 24 BGN, EVT, and HARMONICA
values. Hours of the day without data are NaN.

### `noisemonitor.summary.lden()`

Return the Lden, an indicator of noise level based on Leq over
a whole day with a penalty of 5 dB for evening (default 19h-23h) and
10 dB for night (default 23h-7h), following the EU Directive
2002/49/EC. Each period is weighted by its duration in hours
(ISO 1996-1:2016). The Directive defines Lden over a year; here
an average Lden encompassing all the dataset is computed by default.
Can return a Lden value corresponding to specific days of the week.

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame with a datetime index and sound level values.
- `day1`:  str, optional \
  first day of the week included in the computation, in English, \
  case-insensitive (e.g., 'monday').
- `day2`:  str, optional \
  last day of the week included in the computation. If day1 comes \
  later in the week than day2, the days wrap past Sunday (e.g., \
  'friday' to 'tuesday').
- `column`:  int or str, default 0 \
  column index (int) or column name (str) to use for calculations. \
  If None, the first column of the DataFrame will be used.
- `values`:  bool, default True \
  If set to True, the function will return individual day, evening \
  and night values in addition to the lden. On by default, for a \
  single period (noisemonitor.summary.periodic, which returns one row \
  per period, leaves them out by default).
- `coverage_check`:  bool, default False \
  if set to True, assess data coverage and automatically filter periods \
  with insufficient data coverage. Coverage counts the samples present \
  in the data. Without any value in the day, evening or night period \
  (all NaN, or no sample), the Lden is NaN in any case, with a \
  CoverageWarning.
- `coverage_threshold`:  float, default 0.5 \
  minimum data coverage ratio required (0.0 to 1.0).
- `start_day`:  int, default 7 \
  hour (0-23) at which the day period starts.
- `start_evening`:  int, default 19 \
  hour (0-23) at which the evening period starts.
- `start_night`:  int, default 23 \
  hour (0-23) at which the night period starts. Member States may \
  choose start hours other than the Directive's defaults.

**Returns:**
- pd.DataFrame: dataframe with daily or weekly lden rounded to two decimals.
    Associated day, evening and night values are returned if values
    is set to True.

### `noisemonitor.summary.leq()`

Return the equivalent level (and optionally statistical indicators)
between two hours of the day. Can return a value corresponding to 
specific days of the week.

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame with a datetime index and sound level values.
- `hour1`:  int, between 0 and 24 \
  hour for the starting time of the daily average.
- `hour2`:  int, between 0 and 24 \
  hour for the ending time of the daily average: the \
  average covers [hour1, hour2), a timestamp marking the start of \
  its integration period. If hour1 > hour2, the period wraps past \
  midnight ([hour1, 24) and [0, hour2)).
- `day1`:  str, optional \
  first day of the week included in the computation, in English, \
  case-insensitive (e.g., 'monday').
- `day2`:  str, optional \
  last day of the week included in the computation. If day1 comes \
  later in the week than day2, the days wrap past Sunday (e.g., \
  'friday' to 'tuesday').
- `column`:  int or str, default 0 \
  column index (int) or column name (str) to use for calculations. \
  If None, the first column of the DataFrame will be used.
- `stats`:  bool, default True \
  If set to True, the function will return L10, L50 and L90 \
  together with the Leq (percentiles by linear interpolation between \
  samples, numpy's default method). For statistical levels, data \
  should be sampled at 1 s or finer: they are computed from the \
  samples as given, so their values depend on the integration time \
  of the data.
- `coverage_check`:  bool, default False \
  if set to True, assess data coverage and automatically filter periods \
  with insufficient data coverage and emit warnings. Coverage counts \
  the samples present in the data. Without any value (all NaN, or no \
  sample), the levels are NaN in any case, with a CoverageWarning.
- `coverage_threshold`:  float, default 0.5 \
  minimum data coverage ratio required (0.0 to 1.0).

**Returns:**
- float: daily or weekly equivalent level rounded to two decimals.
    Statistical indicators are included if stats is set to True.

### `noisemonitor.summary.lf_periodic()`

Compute low-frequency criteria on a periodic basis: the A- and
C-weighted equivalent levels of a range of one-third-octave bands over
each day, week or month, their difference C-A, and whether it flags
low-frequency content.

The input levels must be Z-weighted (unweighted) one-third-octave
band levels, one column per band named by its frequency. Methods:
- 'botsford': whole spectrum, 20 Hz to 20 kHz; low-frequency content
    when C-A > 10 dB (WHO, Berglund et al., 1999).
- 'montano': 16 Hz to 200 Hz, the low-frequency range of ISO 1996-2:2017;
    low-frequency content when C-A > 15 dB, an empirical trigger (Montano &
    Gushiken, 2023).
- 'barcelona': 20 Hz to 160 Hz; significant low-frequency components when
    C-A >= 20 dB (Generalitat de Catalunya, 2012).
Restricted to a band range, C-A depends on the shape of the spectrum
within the range, not on its level: it does not tell whether the low
frequencies are audible.

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame with a datetime index and Z-weighted one-third-octave band \
  levels as columns. Each column is named by its frequency (e.g., \
  '20', '25 Hz' or '1.25k'; exact or base-2 mid-band frequencies \
  such as 31.62 or 31.25 are read as the nominal band, 31.5). Each \
  method needs all the bands of its range (ValueError otherwise); \
  other bands are ignored. The levels of each sample are weighted \
  and summed over the bands. A sample with a missing band is left out.
- `freq`:  str, default 'D' \
  frequency for the computation. 'D' for daily, 'W' for weekly \
  (Monday to Sunday), and 'MS' for monthly (calendar months). Periods \
  are labelled by their start.
- `method`:  str or list of str, default 'montano' \
  low-frequency criterion or criteria: 'botsford', 'montano' or \
  'barcelona'.
- `coverage_check`:  bool, default False \
  if set to True, assess data coverage and automatically filter periods \
  with insufficient data coverage and emit warnings.
- `coverage_threshold`:  float, default 0.5 \
  minimum data coverage ratio required (0.0 to 1.0).
- `start_day`:  int, default 7 \
  hour (0-23) at which days, weeks and months start if start_at_day \
  is True.
- `start_at_day`:  bool, default False \
  if True, days, weeks and months start at start_day instead of \
  midnight (see noisemonitor.summary.periodic).

**Returns:**
- pd.DataFrame: DataFrame indexed by the start of each period, with
    MultiIndex columns (Indicator, Method): 'LA' and 'LC', the A- and
    C-weighted equivalent levels of the method's bands, 'C-A', their
    difference, in dB, and 'Lowfreq_Flag', True when C-A flags
    low-frequency content (nullable boolean, <NA> where C-A is NaN).

### `noisemonitor.summary.nday()`

Compute the number of days (or weeks, or months) in a dataset for
which the indicators are between given values of decibels.

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame with a datetime index and sound level values.
- `indicator`:  str, default 'Leq' \
  Indicator to use for the computation. Options are 'Leq', 'Lden', \
  'Lday', 'Levening', and 'Lnight'.
- `bins`:  list of int, optional \
  List of decibel values to define the bins. By default: <40 dBA \
  and every 5 dBA until >=80 dBA.
- `freq`:  str, default 'D' \
  Frequency for the computation. 'D' for daily, 'W' for weekly \
  (Monday to Sunday) and 'MS' for monthly (calendar months).
- `column`:  int or str, default 0 \
  column index (int) or column name (str) to use for calculations. \
  If None, the first column of the DataFrame will be used.
- `coverage_check`:  bool, default False \
  if set to True, assess data coverage and automatically filter periods \
  with insufficient data coverage and emit warnings. Samples of a \
  period missing from the data (first or last period cut by the start \
  or end of the data) count as missing.
- `coverage_threshold`:  float, default 0.5 \
  minimum data coverage ratio required (0.0 to 1.0).
- `start_day`:  int, default 7 \
  hour (0-23) at which the Lden day period starts.
- `start_evening`:  int, default 19 \
  hour (0-23) at which the Lden evening period starts.
- `start_night`:  int, default 23 \
  hour (0-23) at which the Lden night period starts.
- `start_at_day`:  bool, default False \
  if True, days and weeks start at start_day instead of midnight \
  (see noisemonitor.summary.periodic).

**Returns:**
- tuple[pd.DataFrame, List[int]] : Tuple containing a DataFrame with the
    number of periods for each decibel range, in a column named
    'Number of Days', 'Number of Weeks' or 'Number of Months' after
    freq, and the list of bins used.

### `noisemonitor.summary.periodic()`

Compute Leq and Lden on a periodic basis. Lden follows the
Directive 2002/49/EC (Annex I), computed here over each period instead
of a year (see noisemonitor.util.core.lden).

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame with a datetime index and sound level values.
- `freq`:  str, default 'D' \
  frequency for the computation. 'D' for daily, 'W' for weekly \
  (Monday to Sunday), and 'MS' for monthly (calendar months). Periods \
  are labelled by their start.
- `column`:  int or str, default 0 \
  column index (int) or column name (str) to use for calculations. \
  If None, the first column of the DataFrame will be used.
- `values`:  bool, default False \
  if set to True, the function will return individual day, evening \
  and night values in addition to the lden. Off by default, to keep \
  one Leq and one Lden column per period (noisemonitor.summary.lden \
  returns them by default, for a single period).
- `coverage_check`:  bool, default False \
  if set to True, assess data coverage and automatically filter periods \
  with insufficient data coverage and emit warnings. Samples of a \
  period missing from the data (first or last period cut by the start \
  or end of the data) count as missing. Without any value in its day, \
  evening or night period, the Lden of a period is NaN in any case, \
  with a single CoverageWarning per call.
- `coverage_threshold`:  float, default 0.5 \
  minimum data coverage ratio required (0.0 to 1.0).
- `start_day`:  int, default 7 \
  hour (0-23) at which the Lden day period starts.
- `start_evening`:  int, default 19 \
  hour (0-23) at which the Lden evening period starts.
- `start_night`:  int, default 23 \
  hour (0-23) at which the Lden night period starts.
- `start_at_day`:  bool, default False \
  if False, days, weeks and months start at midnight (calendar \
  periods): a day's night then joins the end of one night (00:00 to \
  start_day) and the start of the next (start_night to 24:00). If \
  True, they start at start_day (e.g. 07:00 to 07:00 the next day), \
  so that each night is kept whole. Periods are labelled by their \
  start time. That this affects both Leq and Lden computation.

**Returns:**
- pd.DataFrame: DataFrame with Leq and Lden values for each day or week.

## Profile Module

### `noisemonitor.profile.freq_periodic()`

Compute daily or weekly profiles for each frequency band in the
DataFrame, as noisemonitor.profile.periodic. For statistical levels,
data should be sampled at 1 s or finer.

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame with a datetime index and sound level values for each \
  frequency band, columns named by their frequency as in \
  noisemonitor.summary.freq_periodic.
- `hour1`:  int, between 0 and 23 \
  Hour at which the first window starts.
- `hour2`:  int, between 0 and 23 \
  Hour at which the last window starts (included), as in \
  noisemonitor.profile.periodic. If hour1 > hour2, the windows wrap \
  past midnight.
- `day1`:  Optional[str], default None \
  First day of the week included in the weekly average.
- `day2`:  Optional[str], default None \
  Last day of the week included in the weekly average.
- `win`:  int, default 3600 \
  Window size for the averaging function, in whole seconds. Each \
  window covers the half-open interval [start, start + win).
- `step`:  int, default 0 \
  Step size to compute a sliding average, in whole seconds. If set to \
  0 (default value), the function will compute non-sliding averages.
- `use_chunks`:  bool, default True \
  If set to True, the function will use parallel processing to compute \
  weekly levels for each frequency band (one after the other, with a \
  RuntimeWarning, if parallel processing is unavailable).
- `stat`:  Optional[Union[int, list]], default None \
  statistical level(s) to compute (between 1 and 99). If provided, \
  computes the corresponding percentile(s) (e.g., stat=5 computes L5).
- `coverage_check`:  bool, default False \
  if set to True, assess data coverage and automatically filter periods \
  with insufficient data coverage and emit warnings.
- `coverage_threshold`:  float, default 0.5 \
  minimum data coverage ratio required (0.0 to 1.0).
- `chunks`:  bool, optional \
  Deprecated alias of use_chunks, to be removed in noisemonitor \
  2.0.

**Returns:**
- pd.DataFrame
    DataFrame with the levels of each window for each frequency band.

### `noisemonitor.profile.freq_series()`

Compute time series of sound levels for each frequency band in the
DataFrame, as noisemonitor.profile.series. For statistical levels, 
data should be sampled at 1 s or finer.

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame with a datetime index and sound level values for each \
  frequency band, columns named by their frequency as in \
  noisemonitor.summary.freq_periodic.
- `win`:  int, default 3600 \
  Window size for the averaging function, in whole seconds.
- `step`:  int, default 0 \
  Step size to compute a rolling average, in whole seconds. If set to \
  0 (default value), the function will compute non-rolling averages.
- `use_chunks`:  bool, default True \
  If set to True, the function will use parallel processing to compute \
  the series for each frequency band (one after the other, with a \
  RuntimeWarning, if parallel processing is unavailable).
- `start_at_midnight`:  bool, default False \
  if True, the first window starts at the first midnight of the data, \
  as in noisemonitor.profile.series.
- `coverage_check`:  bool, default False \
  if set to True, assess data coverage and automatically filter periods \
  with insufficient data coverage and emit warnings.
- `coverage_threshold`:  float, default 0.5 \
  minimum data coverage ratio required (0.0 to 1.0).
- `chunks`:  bool, optional \
  Deprecated alias of use_chunks, to be removed in noisemonitor \
  2.0.

**Returns:**
- pd.DataFrame
    DataFrame with time series for each frequency band.

### `noisemonitor.profile.lf_periodic()`

Daily or weekly profile of low-frequency criteria: the A- and
C-weighted equivalent levels of a range of one-third-octave bands and
their difference C-A, in windows pooled over the days as in
noisemonitor.profile.periodic. Input bands and methods are those of
noisemonitor.summary.lf_periodic (see its description).

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame with a datetime index and Z-weighted one-third-octave band \
  levels as columns, as in noisemonitor.summary.lf_periodic.
- `hour1`:  int, between 0 and 23 \
  hour at which the first window starts.
- `hour2`:  int, between 0 and 23 \
  hour at which the last window starts (included), as in \
  noisemonitor.profile.periodic. If hour1 > hour2, the windows wrap \
  past midnight.
- `day1`:  Optional[str], default None \
  First day of the week included in the weekly average.
- `day2`:  Optional[str], default None \
  Last day of the week included in the weekly average.
- `method`:  str or list of str, default 'montano' \
  low-frequency criterion or criteria: 'botsford', 'montano' or \
  'barcelona' (see noisemonitor.summary.lf_periodic).
- `win`:  int, default 3600 \
  window size, in whole seconds, as in noisemonitor.profile.periodic.
- `step`:  int, default 0 \
  step size of sliding windows, in whole seconds (0: consecutive \
  windows).
- `coverage_check`:  bool, default False \
  if set to True, assess data coverage and automatically filter periods \
  with insufficient data coverage and emit warnings.
- `coverage_threshold`:  float, default 0.5 \
  minimum data coverage ratio required (0.0 to 1.0).

**Returns:**
- pd.DataFrame: DataFrame with the time of day of each window and
    MultiIndex columns (Indicator, Method): 'LA' and 'LC', the A- and
    C-weighted equivalent levels of the method's bands, and 'C-A', their
    difference, in dB.

### `noisemonitor.profile.lf_series()`

Time series of low-frequency criteria: the A- and C-weighted
equivalent levels of a range of one-third-octave bands and their
difference C-A, over consecutive or sliding windows of the whole data,
as noisemonitor.profile.series. Input bands and methods are those of
noisemonitor.summary.lf_periodic (see its description).

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame with a datetime index and Z-weighted one-third-octave band \
  levels as columns, as in noisemonitor.summary.lf_periodic.
- `method`:  str or list of str, default 'montano' \
  low-frequency criterion or criteria: 'botsford', 'montano' or \
  'barcelona' (see noisemonitor.summary.lf_periodic).
- `win`:  int, default 3600 \
  window size, in whole seconds, as in noisemonitor.profile.series.
- `step`:  int, default 0 \
  step size of sliding windows, in whole seconds (0: consecutive \
  windows).
- `start_at_midnight`:  bool, default False \
  if True, the first window starts at the first midnight of the data, \
  as in noisemonitor.profile.series.
- `coverage_check`:  bool, default False \
  if set to True, assess data coverage and automatically filter periods \
  with insufficient data coverage and emit warnings.
- `coverage_threshold`:  float, default 0.5 \
  minimum data coverage ratio required (0.0 to 1.0).

**Returns:**
- pd.DataFrame: DataFrame with a datetime index (centre of each window) and
    MultiIndex columns (Indicator, Method): 'LA' and 'LC', the A- and
    C-weighted equivalent levels of the method's bands, and 'C-A', their
    difference, in dB.

### `noisemonitor.profile.nne()`

Compute the Number of Noise Events (NNE) following the algorithm 
proposed in (Brown and De Coensel, 2018). The function computes the 
average NNE using sliding windows, computing daily or weekly profiles.
Note that this function is computationally expensive as noise NNEs are
separately computed for each individual day and then averaged since
background levels are relative to each day. For noise events, data
should be sampled at 1 s or finer (e.g., 125 ms). Events are detected on
the samples as given, so their number depends on the time weighting and
sampling rate of the data. The days are taken from hour1, so that windows
past midnight belong to the night that starts at hour1 and hold
contiguous data (e.g., a 22-06 window on Mondays will run
from Mondays 22h to Tuesdays 6h). Each window of each day is counted as a
record of its own, with its own threshold: its first exceedance is an
event even when the window starts within it, so an exceedance spanning
two windows counts in both. Following exceedances are new events if
they start at least min_gap seconds after the end of the previous one.

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame with a datetime index and sound level values.
- `hour1`:  int, between 0 and 23 \
  hour at which the first window starts.
- `hour2`:  int, between 0 and 24 \
  hour at which the last window ends (excluded): the windows cover \
  [hour1, hour2), and hour1=0, hour2=24 give the full day. If \
  hour1 > hour2, the windows wrap past midnight.
- `background_type`:  int or str, default 'Leq' \
  Type of background level indicator for computing the threshold to \
  use for defining a noise event. Can be 'Leq', 'L50', 'L90' or int \
  for a constant value.
- `exceedance`:  int, default 5 \
  Exceedance value in dB to add to the background level to define the \
  detection threshold, when the background level is adaptive.
- `min_gap`:  int \
  Minimum time gap in seconds between successive noise events: the \
  time spent at or below the threshold between two exceedances.
- `win`:  int, default 3600 \
  Window size for the averaging function, in whole seconds. Each \
  window covers the half-open interval [start, start + win).
- `step`:  int, default 0 \
  Step size to compute a sliding average, in whole seconds. If set to \
  0 (default value), the function will compute non-sliding averages.
- `column`:  int or str, default 0 \
  column index (int) or column name (str) to use for calculations. \
  If None, the first column of the DataFrame will be used.
- `day1`:  Optional[str], default None \
  First day of the week to include in the calculation: the day at \
  whose hour1 the windows start (e.g., with hour1=22, hour2=6 and \
  day1=day2='friday', the night from Friday 22:00 to Saturday 06:00).
- `day2`:  Optional[str], default None \
  Last day of the week to include in the calculation, as day1.
- `coverage_check`:  bool, default True \
  if set to True, assess data coverage and automatically filter periods \
  with insufficient data coverage and emit warnings. We recommand to use \
  coverage checks for NNEs computation, as gaps in data significantly affect \
  the counts of noise events. Samples of a window missing from the \
  data (window cut by the start or end of the data) count as missing. \
  Windows without any value (all NaN, or no sample) have no count in \
  any case (left out of the average), with a CoverageWarning.
- `coverage_threshold`:  float, default 0.8 \
  minimum data coverage ratio required (0.0 to 1.0).

**Returns:**
- DataFrame: DataFrame with the number of noise events for each sliding 
window.

### `noisemonitor.profile.periodic()`

Compute daily or weekly rolling averages of the sound level, in 
terms of equivalent level (Leq), statistical levels (L10, L50 and 
L90), and optionally other indicators (e.g., Traffic Noise Index). 
Percentiles are computed by linear interpolation between samples
(numpy's default method).

For statistical and other percentile-based levels, data should be sampled
at 1 s or finer (e.g., 100 ms) although they are computed
from the samples as given, so their values depend on the integration time
of the data.

Each window pools the samples of all the selected days whose time of
day falls in it: its levels are computed over these pooled samples, not
as the mean of daily values (e.g., the L90 of the 08:00-09:00 window is
the 10th percentile of all the samples between 08:00 and 09:00, and its
Leq their energetic mean).

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame with a datetime index and sound level values.
- `hour1`:  int, between 0 and 23 \
  hour at which the first window starts.
- `hour2`:  int, between 0 and 23 \
  hour at which the last window starts (included): with win=3600, \
  hour1=0 and hour2=23 give the 24 hours of the day. If \
  hour1 > hour2, the windows wrap past midnight.
- `day1`:  Optional[str], default None \
  First day of the week included in the weekly average.
- `day2`:  Optional[str], default None \
  Last day of the week included in the weekly average.
- `column`:  int or str, default 0 \
  column index (int) or column name (str) to use for calculations. \
  If None, the first column of the DataFrame will be used.
- `win`:  int, default 3600 \
  window size for the averaging function, in whole seconds. Each \
  window covers the half-open interval [start, start + win): a \
  timestamp marks the start of its integration period.
- `step`:  int, default 0 \
  step size to compute a rolling average, in whole seconds. If set to \
  0 (default value), the function will compute non-rolling averages.
- `traffic_noise_indicators`:  bool, default False \
  if set to True, the function will compute traffic noise indicators \
  Traffic Noise Index (Griffiths and Langdon, 1968) as well as the \
  Noise Pollution Level (Robinson, 1971) in addition to the \
  equivalent levels and percentiles, for each window: \
  TNI = 4 (L10 - L90) + L90 - 30 and NPL = Leq + 2.56 σ, σ being the \
  standard deviation of the levels of the window (sample standard \
  deviation, n - 1). Griffiths and Langdon defined TNI on the means of \
  24 hourly L10 and L90 values; the formula being linear, their daily \
  TNI is the mean of the 24 hourly values of 1 h windows over one day.
- `roughness_indicators`:  bool, default False \
  if set to True, the function will compute roughness indicators \
  based on difference between consecutive LAeq,1s values according \
  to (DeFrance et al., 2010): from the absolute differences |dL|, \
  dLav is their mean, dLmax,1 the mean of the largest 1 % and dLmin,90 \
  the mean of the lowest 90 % (percentiles by linear interpolation, \
  values at the percentile included). Sub-second data are first averaged \
  energetically over each whole second (LAeq,1s), assuming that their \
  interval divides one second (e.g., 100, 125, 250 or 500 ms). Only \
  samples one sampling interval apart are differenced: pairs with a \
  missing value, split by a gap, or across the junction between days \
  pooled in a window are left out. A window without any such pair \
  gives NaN.
- `stat`:  int, list of ints, or None, default None \
  statistical level(s) to compute (between 1 and 99). If an int is provided, \
  computes the corresponding percentile (e.g., stat=10 computes L10). If a list \
  is provided, computes all specified percentiles. Results are added to the output \
  DataFrame with column names like 'L10', 'L5', etc.
- `coverage_check`:  bool, default False \
  if set to True, assess data coverage and automatically filter periods \
  with insufficient data coverage and emit warnings. Coverage counts \
  the samples present in the data: days outside the data range are \
  not counted as missing. Windows without any value (all NaN, or no \
  sample) are NaN in any case, with a CoverageWarning.
- `coverage_threshold`:  float, default 0.5 \
  minimum data coverage ratio required (0.0 to 1.0).

**Returns:**
- DataFrame: dataframe containing time index and daily averaged
    Leq, L10, L50 and L90 at the corresponding columns

### `noisemonitor.profile.series()`

Sliding average of the entire sound level array, in terms of
equivalent level (LEQ), and statistical levels (L10, L50 and L90).
Percentiles are computed by linear interpolation between samples
(numpy's default method). For statistical levels, data should 
be sampled at 1 s or finer. Percentiles are computed from the 
samples as given, so their values depend on the integration time 
of the data.

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame with a datetime index and sound level values.
- `win`:  int, default 3600 \
  window size (in whole seconds) for the averaging function. For daily, \
  weekly or monthly levels, we recommend using \
  noisemonitor.summary.periodic instead.
- `step`:  int, default 0 \
  step size (in whole seconds) to compute a sliding average. If set \
  to 0 (default value), the function will compute non-sliding \
  averages.
- `column`:  int or str, default 0 \
  column index (int) or column name (str) to use for calculations. \
  If None, the first column of the DataFrame will be used.
- `start_at_midnight`:  bool, default False \
  if set to True, the first window starts at the first sample at or \
  after the first midnight of the data (ValueError if fewer than one \
  window of samples follow it).
- `coverage_check`:  bool, default False \
  if set to True, assess data coverage and automatically filter periods \
  with insufficient data coverage and emit warnings. Windows without \
  any value (all NaN) are NaN in any case, with a CoverageWarning.
- `coverage_threshold`:  float, default 0.5 \
  minimum data coverage ratio required (0.0 to 1.0).

**Returns:**
- DataFrame: dataframe containing datetime index and averaged
    Leq, L10, L50 and L90 at the corresponding columns

## Display Module

### `noisemonitor.display.compare()`

Compare multiple DataFrames by plotting their columns in the same plot.

**Parameters:**
- `dfs`:  list of DataFrames \
  list of DataFrames (e.g., outputs of noisemonitor.load(), \
  noisemonitor.summary.periodic() or noisemonitor.profile functions), \
  with a datetime, time or pandas.Timestamp index.
- `labels`:  list of str \
  list of labels for each DataFrame.
- `*args`:  str \
  column name(s) to be plotted.
- `ylabel`:  str, default "Sound Level (dBA)" \
  label for the y-axis.
- `step`:  bool, default False \
  if set to True, will plot the data as a step function.
- `show_points`:  bool, default False \
  if True, scatter points will be added on top of the line plots.
- `figsize`:  tuple, default (10,8) \
  figure size in inches.
- `show`:  bool, default True \
  if True, display the plot using plt.show().
- `threshold`:  float, optional \
  if provided, plots a horizontal line at this sound level value.
- `**kwargs`:  any \
  Additional keyword arguments passed to display.line() and \
  matplotlib functions. Common examples: fill_background, \
  ylim, xlim, title, color, linewidth, alpha, etc.

**Returns:**
- ax: matplotlib.axes.Axes
    Axes object containing the plot.

### `noisemonitor.display.compare_weather_daily()`

Compare daily level profiles with or without flags.

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame containing the data.
- `column`:  Union[int, str], default 0 \
  The column name or index for sound levels. If None, the first column \
  will be used.
- `show_col`:  str, default 'Leq' \
  Column names to be plotted.
- `include_wind_flag`:  bool, default True \
  Whether to include the Wind Speed Flag.
- `include_rain_flag`:  bool, default True \
  Whether to include the Rain Flag.
- `include_temp_flag`:  bool, default False \
  Whether to include the Temperature Flag.
- `include_rel_hum_flag`:  bool, default False \
  Whether to include the Relative Humidity Flag.
- `include_snow_flag`:  bool, default False \
  Whether to include the Snow Flag.
- `win`:  int, default 3600 \
  The window size for rolling average.
- `step`:  int, default 1200 \
  The step size for rolling average.
- `coverage_check`:  bool, default False \
  Whether to check data coverage when computing levels.
- `coverage_threshold`:  float, default 0.5 \
  Minimum coverage ratio required (0.0-1.0).
- `title`:  str, default "Daily Leq Profiles for Different Conditions" \
  The title of the plot.
- `figsize`:  tuple, default (12, 8) \
  The size of the plot.
- `show`:  bool, default True \
  if True, display the plot using plt.show().
- `threshold`:  float, optional \
  if provided, plots a horizontal line at this sound level value.
- `**kwargs`:  any \
  Additional keyword arguments passed to compare function. \
  Common examples: ylim, color, linewidth, alpha, etc.

**Returns:**
- ax: matplotlib.axes.Axes
    Axes object containing the plot.

### `noisemonitor.display.freq_line()`

Plot the overall levels for each frequency band.

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame containing overall levels for each frequency band, with \
  rows as indicators (e.g., Leq, Lden) and columns as frequency bands.
- `ylabel`:  str, default "Sound Level (dBA)" \
  Label for the y-axis.
- `title`:  str, default "Overall Frequency Bands" \
  Title for the plot.
- `figsize`:  tuple, default (12, 8) \
  Figure size in inches.
- `show`:  bool, default True \
  if True, display the plot using plt.show().
- `**kwargs`:  any \
  Additional keyword arguments passed to the plot function.

**Returns:**
- ax: matplotlib.axes.Axes
    Axes object containing the plot.

### `noisemonitor.display.freq_map()`

Plot a heatmap of sound levels across frequency bands over time.

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame containing sound levels with frequency bands as columns \
  and datetime index as rows.
- `title`:  str, default "Frequency Bands Heatmap" \
  Title for the heatmap.
- `ylabel`:  str, default "Frequency Band" \
  Label for the y-axis.
- `cblabel`:  str, default "Sound Level (dBA)" \
  Label for the color bar.
- `figsize`:  tuple, default (12, 8) \
  Figure size in inches.
- `show`:  bool, default True \
  if True, display the plot using plt.show().
- `**kwargs`:  any \
  Additional keyword arguments passed to pcolormesh. \
  Common examples: cmap, vmin, vmax, shading, etc.

**Returns:**
- ax: matplotlib.axes.Axes
    Axes object containing the plot.

### `noisemonitor.display.harmonica()`

Plot the HARMONICA index.

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame containing the HARMONICA indicators.
- `title`:  str, default "HARMONICA Index Plot" \
  Title for the plot.
- `show`:  bool, default True \
  if True, display the plot using plt.show().
- `**kwargs`:  any \
  Additional keyword arguments passed to matplotlib functions. \
  Common examples: ylim, figsize, etc.

**Returns:**
- ax: matplotlib.axes.Axes
    Axes object containing the plot.

### `noisemonitor.display.line()`

Plot columns of a dataframe according to the index, using matplotlib.

**Parameters:**
- `df`:  DataFrame \
  a DataFrame (e.g., the output of noisemonitor.load(), \
  noisemonitor.summary.periodic() or a noisemonitor.profile function), \
  with a datetime, time or pandas.Timestamp index.
- `*args`:  str \
  column name(s) to be plotted.
- `ylabel`:  str, default "Sound Level (dBA)" \
  label for the y-axis.
- `step`:  bool, default False \
  if True, will plot the data as a step function.
- `show_points`:  bool, default False \
  if True, scatter points will be added on top of the line plots.
- `fill_background`:  bool, default False \
  If True, fills the background with colors based on the time of day: \
  - Day (7-19h): Light yellow \
  - Evening (19-23h): Light orange \
  - Night (23-7h): Light blue \
  Only applied for temporal resolutions > 6/day.
- `figsize`:  tuple, default (10,8) \
  figure size in inches.
- `show`:  bool, default True \
  if True, display the plot using plt.show(). Set it to False to \
  add to the plot afterwards (e.g. when drawing on a given ax).
- `ax`:  matplotlib.axes.Axes, default None \
  Axes object to plot on. If None, a new figure and axes are created.
- `threshold`:  float, optional \
  if provided, plots a horizontal line at this sound level value.
- `**kwargs`:  any \
  Additional keyword arguments passed to matplotlib plot/step functions. \
  Common examples: ylim, xlim, title, color, linewidth, alpha, \
  linestyle, marker, etc.

**Returns:**
- ax: matplotlib.axes.Axes
    Axes object containing the plot.

### `noisemonitor.display.line_weather()`

Plot sound levels with weather flags.

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame containing the data.
- `column`:  Union[int, str], default 0 \
  The column name or index for sound levels. If None, the first column \
  will be used.
- `win`:  int, optional \
  window size in seconds: levels are averaged over consecutive \
  windows (noisemonitor.profile.series). If None, the samples are \
  plotted as they are.
- `include_wind_flag`:  bool, default True \
  Whether to show the Wind Speed Flag.
- `include_rain_flag`:  bool, default True \
  Whether to show the Rain Flag.
- `include_temp_flag`:  bool, default False \
  Whether to show the Temperature Flag.
- `include_rel_hum_flag`:  bool, default False \
  Whether to show the Relative Humidity Flag.
- `include_snow_flag`:  bool, default False \
  Whether to show the Snow Flag.
- `coverage_check`:  bool, default False \
  Whether to check data coverage when computing levels.
- `coverage_threshold`:  float, default 0.5 \
  Minimum coverage ratio required (0.0-1.0).
- `show`:  bool, default True \
  if True, display the plot using plt.show().
- `threshold`:  float, optional \
  if provided, plots a horizontal line at this sound level value.
- `**kwargs`:  any \
  Additional keyword arguments passed to line() and plt.step(). \
  Common examples: title, figsize, ylim, color, linewidth, etc.

**Returns:**
- ax: matplotlib.axes.Axes
    Axes object containing the plot.

### `noisemonitor.display.nday()`

Plots a histogram from the output of the noisemonitor.summary.nday()
function with the number of days (or weeks, or months) in each decibel
range and a color coding corresponding to the indicated thresholds.

**Parameters:**
- `nday_df`:  pd.DataFrame \
  DataFrame output from the nday() function with the number \
  of days for each decibel range.
- `bins`:  list of int \
  List of decibel values to define the bins.
- `freq`:  str, default 'D' \
  Frequency for the label. 'D' for daily, 'W' for weekly and 'MS' \
  for monthly.
- `thresholds`:  list of int, default [55, 60, 65] \
  Thresholds for color coding.
- `title`:  str, optional \
  Title for the plot.
- `figsize`:  tuple, default (10,8) \
  Figure size in inches.
- `show`:  bool, default True \
  if True, display the plot using plt.show().
- `**kwargs`:  any \
  Additional keyword arguments passed to matplotlib bar plot. \
  Common examples: ylim, xlim, xlabel, ylabel, etc.

**Returns:**
- ax: matplotlib.axes.Axes
    Axes object containing the plot.

## Core Module

### `noisemonitor.util.core.check_coverage()`

Check if data coverage meets the specified threshold.

Assesses the proportion of valid (non-NaN) values in an array and
determines if it meets the minimum coverage requirement.

**Parameters:**
- `array`:  np.array \
  Input array to assess for data coverage.
- `threshold`:  float, default 0.5 \
  Minimum data coverage ratio required (0.0 to 1.0).
- `emit_warning`:  bool, default False \
  If True, emit a warning when coverage is insufficient.

**Returns:**
- bool: True if coverage meets threshold, False otherwise (always False
for an empty array)

### `noisemonitor.util.core.equivalent_level()`

Compute the equivalent sound level from the input array.

**Parameters:**
- `array`:  np.array \
  Input array of sound levels in decibels.

**Returns:**
- float
    Equivalent sound level in decibels.

### `noisemonitor.util.core.get_interval()`

Compute the sampling interval of the data in seconds, as a float
(e.g., 0.5 for 500 ms data, 86400 for daily data): the most common step
between consecutive timestamps, so that missing samples, gaps or a first
sample out of step do not change it.

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame with a datetime index of at least two distinct timestamps.

**Returns:**
- float: sampling interval, in seconds.

### `noisemonitor.util.core.harmonica()`

Compute the hourly HARMONICA index and its EVT and BGN components, 
as proposed by Bruitparif (Mietlicki et al., 2015):
BGN = 0.2 (LA95eq - 30), EVT = 0.25 (LAeq - LA95eq) and
HARMONICA = BGN + EVT, where LAeq is the equivalent level of the hour
and LA95eq the equivalent level of its LA95 values. One LA95 is computed
every second t of the hour over the 10 min before it, [t - 600 s, t),
as the 5th percentile taken as the next higher sample (numpy method
'higher'), from the values available in the window; a window holding
fewer than 50% of its expected values gives no LA95, so an hour without
data before it uses the LA95 values from 5 min into the hour. An hour
with less than 80% of its expected values returns NaN. HARMONICA is not
capped: values above 10 are possible.

**Parameters:**
- `df`:  pd.DataFrame \
  DataFrame containing LAeq values at 1 s or finer with a datetime \
  index. The index was designed on LAeq,1s: sub-second levels are \
  first averaged energetically over each whole second (LAeq,1s), \
  assuming that their interval divides one second (e.g., 100, 125, \
  250 or 500 ms).
- `column`:  int \
  Column index containing the LAeq values.
- `use_chunks`:  bool default True \
  whether to process the data in chunks for large datasets.

**Returns:**
- DataFrame: DataFrame with EVT, BGN, and HARMONICA indicators.

### `noisemonitor.util.core.hourly_harmonica()`

Compute the HARMONICA indicators of one hour (see harmonica).

### `noisemonitor.util.core.lden()`

Compute the Lden value for a given DataFrame, following the
EU Directive 2002/49/EC: Lden = 10 lg[(t_day 10^(Lday/10) +
t_evening 10^((Levening+5)/10) + t_night 10^((Lnight+10)/10)) / 24],
where t_day, t_evening and t_night are the durations of the periods in
hours (ISO 1996-1:2016). The Directive defines Lden over a year;
here it is computed over all the data in df. Defaults are the
Directive's 07:00-19:00, 19:00-23:00 and 23:00-07:00 (12, 4 and 8
hours); other start hours may be chosen. Periods are half-open, e.g.,
[07:00, 19:00).

**Parameters:**
- `df`:  DataFrame \
  DataFrame with a datetime index and sound level values.
- `column`:  int or str \
  column index (int) or column name (str) to use for calculations. \
  Should contain LAeq values.
- `values`:  bool, default False \
  If set to True, the function will return individual day, evening \
  and night values in addition to the lden.
- `coverage_check`:  bool, default False \
  if set to True, assess data coverage and automatically filter periods \
  with insufficient data coverage and emit warnings. Without any value \
  in the day, evening or night period (all NaN, or no sample), the \
  Lden is NaN in any case, with a CoverageWarning.
- `coverage_threshold`:  float, default 0.5 \
  minimum data coverage ratio required (0.0 to 1.0).
- `start_day`:  int, default 7 \
  hour (0-23) at which the day period starts.
- `start_evening`:  int, default 19 \
  hour (0-23) at which the evening period starts.
- `start_night`:  int, default 23 \
  hour (0-23) at which the night period starts.

**Returns:**
- DataFrame: Lden value and optionally day, evening, and night levels.

### `noisemonitor.util.core.noise_events()`

Compute the Number of Noise Events (NNE) in a DataFrame slice.

According to the algorithm proposed in (Brown and De Coensel, 2018).
Events are detected on the sound level time series as given, at its
own resolution (e.g. LAeq,1s, or sub-second levels). Please note that
the number of events depends on the time weighting and on the sampling
rate of the data.

An exceedance is a run of samples strictly above the threshold (NaN
values are not above it). The slice is treated as a record of its own:
its first exceedance is always an event, even when the slice starts
within it. Each following exceedance is a new event if the time from
the end of the previous exceedance (its first sample at or below the
threshold) to its first sample is at least min_gap; otherwise it
extends the current event (Baclet and Rumpler, 2024).

**Parameters:**
- `df`:  DataFrame \
  DataFrame with a datetime index and sound level values.
- `column`:  int \
  Column index to use for calculations.
- `threshold`:  float \
  Threshold level in dB to define a noise event.
- `min_gap`:  int \
  Minimum time gap in seconds between successive noise events.

**Returns:**
- int: Number of noise events.

## Weather Module

### `noisemonitor.weather.weathercan.contingency_weather_flags()`

Create a contingency table for LAeq, Lden, Lday, Levening, and Lnight
with and without the different weather flags. Lden follows the
Directive 2002/49/EC (see noisemonitor.summary.lden).

**Parameters:**
- `df`:  DataFrame \
  Input DataFrame with a datetime index. Typically, the output of \
  merge_weather().
- `column`:  Optional[int or str], default 0 \
  The column index (int) or column name (str) to use for calculations.
- `include_wind_flag`:  bool, default True \
  Whether to include the Wind Speed Flag in the contingency table.
- `include_rain_flag`:  bool, default True \
  Whether to include the Rain Flag in the contingency table.
- `include_temp_flag`:  bool, default False \
  Whether to include the Temperature Flag in the contingency table.
- `include_rel_hum_flag`:  bool, default False \
  Whether to include the Relative Humidity Flag in the contingency table.
- `include_snow_flag`:  bool, default False \
  Whether to include the Snow Flag in the contingency table.
- `start_day`:  int, default 7 \
  hour (0-23) at which the Lden day period starts.
- `start_evening`:  int, default 19 \
  hour (0-23) at which the Lden evening period starts.
- `start_night`:  int, default 23 \
  hour (0-23) at which the Lden night period starts.

**Returns:**
- pd.DataFrame:
    Contingency table with the Leq and Lden of all the data, of the
    rows with and without each included flag, of the rows with any
    included flag raised ('Any Flag') and of those with none ('No
    Flag'), their differences from all the data, and the proportion of
    the data in each row. Rows with fewer than 3 samples are left out.
    The Lden of a row without data in the day, evening or night period
    (e.g., a flag raised only at night) is NaN, and its Leq is kept.

### `noisemonitor.weather.weathercan.get_historical_data()`

Get historical weather data from Environment Canada in
the given range for the given station.

**Parameters:**
- `station_id`:  str \
  the climate ID of the station, as given in the 'id' row of \
  get_historical_stations (e.g., '702S006').
- `daterange`:  tuple of datetime \
  the dates between which the data are retrieved, in either order
- `timeframe`:  str \
  selection of granularity : 'hourly' or 'daily'

**Returns:**
- DataFrame: All data in the range

### `noisemonitor.weather.weathercan.get_historical_stations()`

Get list of all historical stations from Environment Canada.

**Parameters:**
- `coordinates`:  list of float \
  List of two floats for latitude and longitude
- `radius`:  int, default 25 \
  Radius in kilometers to search for stations surrounding the specified \
  coordinates (must be between 25 and 100)
- `start_year`:  int, default 1840 \
  Starting year for the database query
- `end_year`:  int, default present year \
  Ending year for the database query
- `limit`:  int, default 25 \
  limit of weather stations to list

**Returns:**
- DataFrame: one column per station, labelled by its climate ID and
ordered by proximity, with rows 'name', 'prov' (province), 'proximity'
(distance to the point in kilometers), 'id' (climate ID, as taken by
get_historical_data and merge_weather) and 'hlyRange', 'dlyRange',
'mlyRange' (first and last dates of the hourly, daily and monthly data,
as 'start|end'). Stations sharing a name are all listed.

### `noisemonitor.weather.weathercan.merge_weather()`

Async function. Merge hourly Environment Canada weather data into df
and add weather flags (True when their condition is met).

Samples at 1 h or less take the nearest hourly row. Coarser samples
(e.g., daily levels) take the hourly rows of their own period, with a
UserWarning: values are averaged (precipitation summed) and the rain
and snow flags are raised if raised in any hour.

Rain and snow are flagged in hours with precipitation whose 'Weather'
condition names rain (drizzle and ice pellets included) or snow. A
UserWarning is given when drizzle or ice pellets alone raise the rain
flag, when a condition without a precipitation amount is flagged
alone, when precipitation without a condition is not flagged, and when
a station reporting no condition gets rain (>= 0 °C) or snow (< 0 °C)
from the temperature. Details: usage guide, Weather Integration.

**Parameters:**
- `df`:  DataFrame \
  Input DataFrame with a datetime index.
- `station_id`:  str \
  The climate ID of the weather station, as given in the 'id' row of \
  get_historical_stations (e.g., '702S006').
- `wind_speed_flag`:  int, default 18 \
  Threshold for wind speed flag.
- `temp_range_flag`:  tuple, default (-10, 30) \
  Temperature range for temperature flag.
- `hum_flag`:  int, default 90 \
  Threshold for relative humidity flag.
- `rolling_window_hours`:  int, default 48 \
  Size of the rolling window for rain and snow flags in hours.
- `timezone`:  str, optional \
  Time zone of the times of df (default: 'America/Toronto'). \
  Environment Canada times are in Local Standard Time (LST) all year; \
  they are converted to the local time of this zone, daylight saving \
  time included (the repeated hour at its end keeps its first row). \
  If None, they are kept in LST.

**Returns:**
- DataFrame: Merged DataFrame with weather data and flags.

### `noisemonitor.weather.weathercan.monthlist()`

List the months requested from Environment Canada for a date range.

**Parameters:**
- `daterange`:  tuple of datetime \
  the dates of the range, in either order.

**Returns:**
- list of tuple: (year, month, 1) of each month from the month of the
    first date to the month of the last date.

