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

Take one or several datasheets with date and time indicators in combined in one column or across two columns, and sound level measured with a sound level monitor as input and return a DataFrame suitable for sound level averaging and descriptors computation provided in the LevelMonitor class. Date and time are automatically parsed with dateutil package.

**Parameters:**
- `path`: str or list of str absolute or relative pathname (or list of pathnames) to convert as a sound level DataFrame. File(s) format can either be .csv, .xls, .xlsx, or .txt
- `datetimeindex`: int column index for date and time if combined in a single column. Not to be indicated if date and time are in different columns.
- `timeindex`: int column index for time. Not to be indicated if date and time are combined in a single column. Must be entered conjointly with dateindex.
- `dateindex`: int column index for date. Not to be indicated if date and time are combined in a single column. Must be entered conjointly with timeindex.
- `valueindexes`: int or list of int, default 1. column index or list of column indices for sound level values to which averages are to be computed. The columns should contain sound levels values, either weighted or unweighted and integrated over a period corresponding to the refresh rate of the sound level meter (typically between 1 second and several minutes, though the module will work with
- `smaller or higher refresh rates). Example of relevant indices`: LAeq, LCeq, LZeq, LAmax, LAmin, LCpeak, etc.
- `header`: int, None, default 0 row index for datasheet header. If None, the datasheet has no header.
- `sep`: str, default ',' separator if reading .csv file(s).
- `slm_type`: str, default None performs specific parsing operation for known sound level monitor manufacturers. For now will only respond to 'NoiseSentry' as input, replacing the commas by dots in the sound level data to convert sound levels to floats.
- `timezone`: str, default None when indicated, will convert the datetime index from the specified timezone to a timezone unaware format.
- `use_chunks`: bool, default True whether to process the data in chunks for large datasets.
- `chunksize`: int, default 10000 number of rows to read at a time for large datasets.

**Returns:**
- DataFrame: dataframe formatted for sound level analysis when associated with a LevelMonitor class. Contains a datetime (or pandas Timestamp) array as index and corresponding equivalent sound level as first column.

## Filter Module

### `noisemonitor.filter.all_data()`

Filter a datetime index DataFrame (typically the output of load_data()) between two particular dates. By setting between = True, you can filter out the times that are between the two dates.

**Parameters:**
- `df`: DataFrame a compatible DataFrame (typically generated with functions load_data(), NoiseMonitor.daily() or NoiseMonitor.weekly() ), with a datetime, time or pandas.Timestamp index.
- `start_datetime`: datetime object first date from which to filter the data. Must be earlier than end_datetime.
- `end_datetime`: datetime object second date from which to filter the data. Must be later than start_datetime.
- `between`: bool, default False If set to True, will filter out data that is between the two dates. Else and by default, will filter data that is outside the two dates.

**Returns:**
- DataFrame: input dataframe filtered to the specified dates range.

### `noisemonitor.filter.extreme_values()`

Replace values in the specified column that are below min_value or above max_value with NaN.

**Parameters:**
- `df`: pd.DataFrame DataFrame containing the data.
- `column`: Union[int, str], default None The column name or index to use for calculations. If None, the first column will be used.
- `min_value`: int, default 30 The minimum value threshold.
- `max_value`: int, default 95 The maximum value threshold.

**Returns:**
- pd.DataFrame: DataFrame with extreme values replaced by NaN.

### `noisemonitor.filter.weather_flags()`

Replace values in the specified column with NaN based on weather flags.

**Parameters:**
- `df`: pd.DataFrame DataFrame containing the data.
- `column`: Union[int, str], default None The column name or index to use for calculations. If None, the first column will be used.
- `filter_wind_flag`: bool, default False Whether to filter data with Wind Speed Flag.
- `filter_rain_flag`: bool, default True Whether to filter data with Rain Flag.
- `filter_temp_flag`: bool, default False Whether to filter data with Temperature Flag.
- `filter_rel_hum_flag`: bool, default False Whether to filter data with Relative Humidity Flag.
- `filter_snow_flag`: bool, default False Whether to filter data with Snow Flag.

**Returns:**
- pd.DataFrame: DataFrame with flagged values replaced by NaN.

## Summary Module

### `noisemonitor.summary.freq_indicators()`

Compute overall Leq and Lden for each frequency band.

**Parameters:**
- `hour1 (optional)`: int, default 0 Starting hour for the daily Leq average (0-24).
- `hour2 (optional)`: int, default 24 Ending hour for the daily Leq average (0-24).
- `day1 (optional)`: str, default None First day of the week in English, case-insensitive, to include in the computation.
- `day2 (optional)`: str, default None Last day of the week in English, case-insensitive, to include in the computation.
- `stats`: bool, default False If set to True, the function will include statistical indicators (L10, L50, L90) in addition to Leq.
- `values`: bool, default True If set to True, the function will include individual day, evening, and night values in addition to Lden.
- `coverage_check`: bool, default False if set to True, assess data coverage and automatically filter periods with insufficient data coverage and emit warnings.
- `coverage_threshold`: float, default 0.5 minimum data coverage ratio required (0.0 to 1.0).

**Returns:**
- pd.DataFrame: DataFrame with rows corresponding to indicators (e.g., Leq, Lden, etc.) and columns corresponding to frequency bands.

### `noisemonitor.summary.freq_periodic()`

Compute periodic levels for each frequency band (e.g. octave band or third octave bands) in the input DataFrame.

**Parameters:**
- `df`: pd.DataFrame DataFrame containing octave or third-octave frequency bands as columns.
- `freq`: str, default 'D' Frequency for the computation. 'D' for daily, 'W' for weekly, and 'MS' for monthly.
- `values`: bool, default False If set to True, the function will return individual day, evening, and night values in addition to the Lden.
- `coverage_check`: bool, default False if set to True, assess data coverage and automatically filter periods with insufficient data coverage and emit warnings.
- `coverage_threshold`: float, default 0.5 minimum data coverage ratio required (0.0 to 1.0).

**Returns:**
- pd.DataFrame: DataFrame with weekly or daily levels for each frequency band.

### `noisemonitor.summary.harmonica_periodic()`

Compute the average HARMONICA indicators for each hour of a day. Optionally, can return the average values for specific days of the week.

**Parameters:**
- `df`: pd.DataFrame DataFrame with a datetime index and sound level values.
- `column`: int or str, default 0 column index (int) or column name (str) to use for calculations. If None, the first column of the DataFrame will be used.
- `use_chunks`: bool default True whether to process the data in chunks for large datasets.
- `day1`: Optional[str], default None First day of the week to include in the calculation.
- `day2`: Optional[str], default None Last day of the week to include in the calculation.

**Returns:**
- pd.DataFrame: DataFrame with time index and 24 BGN, EVT, and HARMONICA values.

### `noisemonitor.summary.lden()`

Return the Lden, an indicator of noise level based on Leq over a whole day with a penalty for evening (19h-23h) and night (23h-7h) time noise. By default, an average Lden encompassing all the dataset is computed. Can return a Lden value corresponding to specific days of the week.

**Parameters:**
- `df`: pd.DataFrame DataFrame with a datetime index and sound level values.
- `day1 (optional)`: str, a day of the week in english, case-insensitive first day of the week included in the Lden computation.
- `day2 (optional)`: str, a day of the week in english, case-insensitive last (included) day of the week in the Lden computation. If day2 happens later in the week than day1 the average will be computed outside of these days.
- `column`: int or str, default 0 column index (int) or column name (str) to use for calculations. If None, the first column of the DataFrame will be used.
- `values`: bool, default False If set to True, the function will return individual day, evening and night values in addition to the lden.
- `coverage_check`: bool, default False if set to True, assess data coverage and automatically filter periods with insufficient data coverage.
- `coverage_threshold`: float, default 0.5 minimum data coverage ratio required (0.0 to 1.0).

**Returns:**
- pd.DataFrame: dataframe with daily or weekly lden rounded to two decimals. Associated day, evening and night values are returned if values is set to True.

### `noisemonitor.summary.leq()`

Return the equivalent level (and optionally statistical indicators) between two hours of the day. Can return a value corresponding to specific days of the week.

**Parameters:**
- `df`: pd.DataFrame DataFrame with a datetime index and sound level values.
- `hour1`: int, between 0 and 24 hour for the starting time of the daily average.
- `hour2`: int, between 0 and 24 hour for the ending time of the daily average. If hour2 > hour1 the average will be computed outside of these hours.
- `day1 (optional)`: str, a day of the week in english, case-insensitive first day of the week included in the Lden computation.
- `day2 (optional)`: str, a day of the week in english, case-insensitive last (included) day of the week in the Lden computation. If day2 happens later in the week than day1 the average will be computed outside of these days.
- `column`: int or str, default 0 column index (int) or column name (str) to use for calculations. If None, the first column of the DataFrame will be used.
- `stats`: bool, default True If set to True, the function will return L10, L50 and L90 together with the Leq.
- `coverage_check`: bool, default False if set to True, assess data coverage and automatically filter periods with insufficient data coverage and emit warnings.
- `coverage_threshold`: float, default 0.5 minimum data coverage ratio required (0.0 to 1.0).

**Returns:**
- float: daily or weekly equivalent level rounded to two decimals. Statistical indicators are included if stats is set to True.

### `noisemonitor.summary.nday()`

Compute the number of days in a dataset for which the indicators are between given values of decibels.

**Parameters:**
- `indicator`: str, default 'Leq' Indicator to use for the computation. Options are 'Leq', 'Lden', 'Lday', 'Levening', and 'Lnight'.
- `bins`: list of int, optional
- `List of decibel values to define the bins. By default`: <40 dBA and every 5 dBA until >=80 dBA.
- `freq`: str, default 'D' Frequency for the computation. 'D' for daily and 'W' for weekly.
- `column`: int or str, default 0 column index (int) or column name (str) to use for calculations. If None, the first column of the DataFrame will be used.
- `coverage_check`: bool, default False if set to True, assess data coverage and automatically filter periods with insufficient data coverage and emit warnings.
- `coverage_threshold`: float, default 0.5 minimum data coverage ratio required (0.0 to 1.0).

**Returns:**
- DataFrame : DataFrame with the number of days for each decibel range.

### `noisemonitor.summary.periodic()`

Compute Leq and Lden on a periodic basis.

**Parameters:**
- `df`: pd.DataFrame DataFrame with a datetime index and sound level values.
- `freq`: str, default 'D' frequency for the computation. 'D' for daily, 'W' for weekly, and 'MS' for monthly.
- `column`: int or str, default 0 column index (int) or column name (str) to use for calculations. If None, the first column of the DataFrame will be used.
- `values`: bool, default False if set to True, the function will return individual day, evening and night values in addition to the lden.
- `coverage_check`: bool, default False if set to True, assess data coverage and automatically filter periods with insufficient data coverage and emit warnings.
- `coverage_threshold`: float, default 0.5 minimum data coverage ratio required (0.0 to 1.0).

**Returns:**
- pd.DataFrame: DataFrame with Leq and Lden values for each day or week.

## Profile Module

### `noisemonitor.profile.freq_periodic()`

Compute weekly levels for each frequency band in the DataFrame.

**Parameters:**
- `df`: pd.DataFrame, DataFrame with a datetime index and sound level values for each frequency band.
- `hour1`: int, between 0 and 23 Hour for the starting time of the daily average.
- `hour2`: int, between 0 and 23 Hour (included) for the ending time of the daily average.
- `day1`: Optional[str], default None First day of the week included in the weekly average.
- `day2`: Optional[str], default None Last day of the week included in the weekly average.
- `win`: int, default 3600 Window size for the averaging function, in seconds.
- `step`: int, default 0 Step size to compute a sliding average. If set to 0 (default value), the function will compute non-sliding averages.
- `chunks`: bool, default True If set to True, the function will use parallel processing to compute weekly levels for each frequency band.
- `coverage_check`: bool, default False if set to True, assess data coverage and automatically filter periods with insufficient data coverage and emit warnings.
- `coverage_threshold`: float, default 0.5 minimum data coverage ratio required (0.0 to 1.0).

**Returns:**
- pd.DataFrame DataFrame with weekly levels for each frequency band.

### `noisemonitor.profile.freq_series()`

Compute time series of sound levels for each frequency band in the DataFrame.

**Parameters:**
- `df`: pd.DataFrame DataFrame with a datetime index and sound level values for each frequency band.
- `win`: int, default 3600 Window size for the averaging function, in seconds.
- `step`: int, default 0 Step size to compute a rolling average. If set to 0 (default value), the function will compute non-rolling averages.
- `chunks`: bool, default True If set to True, the function will use parallel processing to compute the series for each frequency band.
- `start_at_midnight`: bool, default False If set to True, the computation will start at midnight.
- `coverage_check`: bool, default False if set to True, assess data coverage and automatically filter periods with insufficient data coverage and emit warnings.
- `coverage_threshold`: float, default 0.5 minimum data coverage ratio required (0.0 to 1.0).

**Returns:**
- pd.DataFrame DataFrame with time series for each frequency band.

### `noisemonitor.profile.nne()`

Compute the Number of Noise Events (NNE) following the algorithm proposed in (Brown and De Coensel, 2018). The function computes the average NNE using sliding windows, computing daily or weekly profiles. Note that this function is computationally expensive as noise NNEs are separately computed for each individual day and then averaged since background levels are relative to each day.

**Parameters:**
- `hour1`: int, between 0 and 23 hour for the starting time of the daily average.
- `hour2`: int, between 0 and 23 hour for the ending time of the daily average. If hour1 > hour2 the average will be computed outside of these hours.
- `background_type`: str Type of background level indicator for computing the threshold to use for defining a noise event. Can be 'Leq', 'L50', 'L90' or int for a constant value.
- `exceedance`: int, default 5 Exceedance value in dB to add to the background level to define the detection threshold, when the background level is adaptive.
- `min_gap`: int Minimum time gap in seconds between successive noise events.
- `win`: int, default 3600 Window size for the averaging function, in seconds.
- `step`: int, default 0 Step size to compute a sliding average. If set to 0 (default value), the function will compute non-sliding averages.
- `column`: int or str, default 0 column index (int) or column name (str) to use for calculations. If None, the first column of the DataFrame will be used.
- `day1`: Optional[str], default None First day of the week to include in the calculation.
- `day2`: Optional[str], default None Last day of the week to include in the calculation.
- `coverage_check`: bool, default False if set to True, assess data coverage and automatically filter periods with insufficient data coverage and emit warnings.
- `coverage_threshold`: float, default 0.5 minimum data coverage ratio required (0.0 to 1.0).

**Returns:**
- DataFrame: DataFrame with the number of noise events for each sliding window.

### `noisemonitor.profile.periodic()`

Compute daily or weekly rolling averages of the sound level, in terms of equivalent level (Leq), and percentiles (L10, L50 and L90).

**Parameters:**
- `hour1`: int, between 0 and 23 hour for the starting time of the daily average.
- `hour2`: int, between 0 and 23 hour (included) for the ending time of the daily average. If hour2 > hour1 the average will be computed outside of these hours.
- `day1`: Optional[str], default None First day of the week included in the weekly average.
- `day2`: Optional[str], default None Last day of the week included in the weekly average.
- `column`: int or str, default 0 column index (int) or column name (str) to use for calculations. If None, the first column of the DataFrame will be used.
- `win`: int, default 3600 window size for the averaging function, in seconds.
- `step`: int, default 0 step size to compute a rolling average. If set to 0 (default value), the function will compute non-rolling averages.
- `traffic_noise_indicators`: bool, default False if set to True, the function will compute traffic noise indicators Traffic Noise Index (Griffiths and Langdon, 1968) as well as the Noise Pollution Level (Robinson, 1971) in addition to the equivalent levels and percentiles.
- `roughness_indicators`: bool, default False if set to True, the function will compute roughness indicators based on difference between consecutive LAeq,1s values according to (DeFrance et al., 2010).
- `stat`: int, list of ints, or None, default None statistical level(s) to compute (between 1 and 99). If an int is provided, computes the corresponding percentile (e.g., stat=10 computes L10). If a list is provided, computes all specified percentiles. Results are added to the output DataFrame with column names like 'L10', 'L5', etc.
- `coverage_check`: bool, default False if set to True, assess data coverage and automatically filter periods with insufficient data coverage and emit warnings.
- `coverage_threshold`: float, default 0.5 minimum data coverage ratio required (0.0 to 1.0).

**Returns:**
- DataFrame: dataframe containing time index and daily averaged Leq, L10, L50 and L90 at the corresponding columns

### `noisemonitor.profile.series()`

Sliding average of the entire sound level array, in terms of equivalent level (LEQ), and percentiles (L10, L50 and L90).

**Parameters:**
- `win`: int, default 3600 window size (in seconds) for the averaging function. For averages at a daily or weekly window, we recommend using the daily_weekly_indicators function instead.
- `step`: int, default 0 step size (in seconds) to compute a sliding average. If set to 0 (default value), the function will compute non-sliding averages.
- `column`: int or str, default 0 column index (int) or column name (str) to use for calculations. If None, the first column of the DataFrame will be used.
- `start_at_midnight`: bool, default False if set to True, the computation will start at midnight.
- `coverage_check`: bool, default False if set to True, assess data coverage and automatically filter periods with insufficient data coverage and emit warnings.
- `coverage_threshold`: float, default 0.5 minimum data coverage ratio required (0.0 to 1.0).

**Returns:**
- DataFrame: dataframe containing datetime index and averaged Leq, L10, L50 and L90 at the corresponding columns

## Display Module

### `noisemonitor.display.compare()`

Compare multiple DataFrames by plotting their columns in the same plot.

**Parameters:**
- `dfs`: list of DataFrames list of compatible DataFrames (typically generated with functions load_data(), NoiseMonitor.daily() or NoiseMonitor.weekly() ), with a datetime, time or pandas.Timestamp index.
- `*args`: str column name(s) to be plotted.
- `labels`: list of str list of labels for each DataFrame.
- `step`: bool, default False if set to True, will plot the data as a step function.
- `show_points`: bool, default False if True, scatter points will be added on top of the line plots.
- `figsize`: tuple, default (10,8) figure size in inches.
- `weighting`: str, default "A" type of sound level data, typically A, C or Z.
- `**kwargs`: any ylim and title arguments can be passed to matplotlib.

**Returns:**
- ax: matplotlib.axes.Axes Axes object containing the plot.

### `noisemonitor.display.compare_weather_daily()`

Compare daily level profiles with or without flags.

**Parameters:**
- `df`: pd.DataFrame DataFrame containing the data.
- `column`: Union[int, str], default 0 The column name or index for sound levels. If None, the first column will be used.
- `show`: str, default 'Leq' Column names to be plotted.
- `include_wind_flag`: bool, default True Whether to include the Wind Speed Flag.
- `include_rain_flag`: bool, default True Whether to include the Rain Flag.
- `include_temp_flag`: bool, default False Whether to include the Temperature Flag.
- `include_rel_hum_flag`: bool, default False Whether to include the Relative Humidity Flag.
- `include_snow_flag`: bool, default False Whether to include the Snow Flag.
- `win`: int, default 3600 The window size for rolling average.
- `step`: int, default 1200 The step size for rolling average.
- `coverage_check`: bool, default False Whether to check data coverage when computing levels.
- `coverage_threshold`: float, default 0.5 Minimum coverage ratio required (0.0-1.0).
- `title`: str, default "Daily Leq Profiles for Different Conditions" The title of the plot.
- `figsize`: tuple, default (12, 8) The size of the plot.

**Returns:**
- None

### `noisemonitor.display.freq_line()`

Plot the overall levels for each frequency band.

**Parameters:**
- `df`: pd.DataFrame DataFrame containing overall levels for each frequency band, with rows as indicators (e.g., Leq, Lden) and columns as frequency bands.
- `weighting`: str, default "A" Type of sound level data, typically A, C or Z.
- `title`: str, default "Overall Frequency Bands" Title for the plot.
- `figsize`: tuple, default (10, 8) Figure size in inches.
- `ax`: matplotlib.axes.Axes, optional Axes object to plot on. If None, a new figure and axes are created.
- `**kwargs`: any Additional keyword arguments passed to the plot function.

**Returns:**
- None

### `noisemonitor.display.freq_map()`

Plot a heatmap of sound levels across frequency bands over time.

**Parameters:**
- `df`: pd.DataFrame DataFrame containing sound levels with frequency bands as columns and datetime index as rows.
- `title`: str, default "Frequency Bands Heatmap" Title for the heatmap.
- `ylabel`: str, default "Frequency Band" Label for the y-axis.
- `figsize`: tuple, default (12, 8) Figure size in inches.

**Returns:**
- None

### `noisemonitor.display.harmonica()`

Plot the HARMONICA index.

**Parameters:**
- `df`: pd.DataFrame DataFrame containing the HARMONICA indicators.
- `title`: str, default "HARMONICA Index Plot" Title for the plot.

### `noisemonitor.display.line()`

Plot columns of a dataframe according to the index, using matplotlib.

**Parameters:**
- `df`: DataFrame a compatible DataFrame (typically generated with functions load_data(), NoiseMonitor.daily() or NoiseMonitor.weekly() ), with a datetime, time or pandas.Timestamp index.
- `*args`: str column name(s) to be plotted.
- `ylabel`: str, default "Sound Level (dBA)" label for the y-axis.
- `step`: bool, default False if True, will plot the data as a step function.
- `show_points`: bool, default False if True, scatter points will be added on top of the line plots.
- `fill_background`: bool, default False
- `If True, fills the background with colors based on the time of day`:
- `- Day (7-19h)`: Light yellow
- `- Evening (19-23h)`: Light orange
- `- Night (23-7h)`: Light blue Only applied for temporal resolutions > 6/day.
- `figsize`: tuple, default (10,8) figure size in inches.
- `ax`: matplotlib.axes.Axes, default None Axes object to plot on. If None, a new figure and axes are created.
- `weighting`: str, default "A" type of sound level data, typically A, C or Z.
- `**kwargs`: any ylim and title arguments can be passed to matplotlib.

**Returns:**
- ax: matplotlib.axes.Axes Axes object containing the plot.

### `noisemonitor.display.line_weather()`

Plot sound levels with weather flags.

**Parameters:**
- `df`: pd.DataFrame DataFrame containing the data.
- `column`: Union[int, str], default 0 The column name or index for sound levels. If None, the first column will be used.
- `window_size`: int, optional The window size for rolling average. If None, no rolling average is applied.
- `show_wind_spd_flag`: bool, default True Whether to show the Wind Speed Flag.
- `show_rain_flag`: bool, default True Whether to show the Rain Flag.
- `show_rel_hum_flag`: bool, default True Whether to show the Relative Humidity Flag.
- `show_temp_flag`: bool, default False Whether to show the Temperature Flag.
- `show_snow_flag`: bool, default True Whether to show the Snow Flag.
- `coverage_check`: bool, default False Whether to check data coverage when computing levels.
- `coverage_threshold`: float, default 0.5 Minimum coverage ratio required (0.0-1.0).

**Returns:**
- None

### `noisemonitor.display.nday()`

Plots a histogram from the output of noisemonitor.indicators.nday() function with the number of days under a certain decibel range and a color coding corresponding to the indicated thresholds.

**Parameters:**
- `nday_df`: pd.DataFrame DataFrame output from the nday() function with the number of days for each decibel range.
- `bins`: list of int List of decibel values to define the bins.
- `freq`: str, default 'D' Frequency for the label. 'D' for daily and 'W' for weekly.
- `thresholds`: list of int, default [55, 60, 65] Thresholds for color coding.
- `title`: str, optional Title for the plot.
- `figsize`: tuple, default (10,8) Figure size in inches.

**Returns:**
- None

## Core Module

### `noisemonitor.util.core.check_coverage()`

Check if data coverage meets the specified threshold. Assesses the proportion of valid (non-NaN) values in an array and determines if it meets the minimum coverage requirement.

**Parameters:**
- `array`: np.array Input array to assess for data coverage.
- `threshold`: float, default 0.5 Minimum data coverage ratio required (0.0 to 1.0).
- `emit_warning`: bool, default False If True, emit a warning when coverage is insufficient.

**Returns:**
- bool: True if coverage meets threshold, False otherwise

### `noisemonitor.util.core.equivalent_level()`

Compute the equivalent sound level from the input array.

**Parameters:**
- `array`: np.array Input array of sound levels in decibels.

**Returns:**
- float Equivalent sound level in decibels.

### `noisemonitor.util.core.get_interval()`

Compute the interval in seconds between rows 2 and 3. In some datasets, the first row index may not be representative of the interval.

### `noisemonitor.util.core.harmonica()`

Compute the HARMONICA indicator and return a DataFrame with EVT, BGN, and HARMONICA indicators as proposed in (Mietlicki et al., 2015).

**Parameters:**
- `df`: pd.DataFrame DataFrame containing the LAeq,1s values with a time or datetime index.
- `column`: int Column index containing the LAeq,1s values.
- `use_chunks`: bool default True whether to process the data in chunks for large datasets.

**Returns:**
- DataFrame: DataFrame with EVT, BGN, and HARMONICA indicators.

### `noisemonitor.util.core.hourly_harmonica()`

Compute a single hour of data to compute HARMONICA indicators.

### `noisemonitor.util.core.lden()`

Compute the Lden value for a given DataFrame.

**Parameters:**
- `df`: DataFrame DataFrame with a datetime index and sound level values.
- `column`: int or str column index (int) or column name (str) to use for calculations. Should contain LAeq values.
- `values`: bool, default False If set to True, the function will return individual day, evening and night values in addition to the lden.
- `coverage_check`: bool, default False if set to True, assess data coverage and automatically filter periods with insufficient data coverage and emit warnings.
- `coverage_threshold`: float, default 0.5 minimum data coverage ratio required (0.0 to 1.0).

**Returns:**
- DataFrame: Lden value and optionally day, evening, and night levels.

### `noisemonitor.util.core.noise_events()`

Compute the Number of Noise Events (NNE) in a DataFrame slice. According to the algorithm proposed in (Brown and De Coensel, 2018). Please note that this indicator is highly dependent on the refresh rate of the data. Usually, NNEs are computed with LAeq,1s for traffic noise.

**Parameters:**
- `df`: DataFrame DataFrame with a datetime index and sound level values.
- `column`: int Column index to use for calculations.
- `threshold`: float Threshold level in dB to define a noise event.
- `min_gap`: int Minimum time gap in seconds between successive noise events.

**Returns:**
- int: Number of noise events.

## Weather Module

### `noisemonitor.weather.weathercan.contingency_weather_flags()`

Create a contingency table for LAeq, Lden, Lday, Levening, and Lnight with and without the different weather flags.

**Parameters:**
- `df`: DataFrame Input DataFrame with a datetime index. Typically, the output of merge_weather().
- `column`: Optional[int or str], default 0 The column index (int) or column name (str) to use for calculations.
- `include_wind_flag`: bool, default True Whether to include the Wind Speed Flag in the contingency table.
- `include_rain_flag`: bool, default True Whether to include the Rain Flag in the contingency table.
- `include_temp_flag`: bool, default False Whether to include the Temperature Flag in the contingency table.
- `include_rel_hum_flag`: bool, default False Whether to include the Relative Humidity Flag in the contingency table.
- `include_snow_flag`: bool, default False Whether to include the Snow Flag in the contingency table.

**Returns:**
- pd.DataFrame: Contingency table with LAeq,24h, Lden, Lday, Levening, Lnight, and other indicators. and the proportion of data covered from the initial dataset.

### `noisemonitor.weather.weathercan.get_historical_data()`

Get historical weather data from Environment Canada in the given range for the given station.

**Parameters:**
- `station_id`: int the ID of the station found with get_historical_stations
- `daterange`: tuple of datetime the dates between which the data are retrieved
- `timeframe`: str
- `selection of granularity`: 'hourly' or 'daily'

**Returns:**
- DataFrame: All data in the range

### `noisemonitor.weather.weathercan.get_historical_stations()`

Get list of all historical stations from Environment Canada.

**Parameters:**
- `coordinates`: list of float List of two floats for latitude and longitude
- `radius`: int, default 25 Radius in kilometers to search for stations surrounding the specified coordinates (must be between 25 and 100)
- `start_year`: int, default 1840 Starting year for the database query
- `end_year`: int, default present year Ending year for the database query
- `limit`: int, default 25 limit of weather stations to list

**Returns:**
- DataFrame: includes all available weather stations, their proximity to the point in kilometers, id number, and the time range of covered data and associated time granulation (hourly, daily or monthly).

### `noisemonitor.weather.weathercan.merge_weather()`

Async function. Merge weather data with the input DataFrame based on the datetime index. Will also include flags for weather conditions.

**Parameters:**
- `df`: DataFrame Input DataFrame with a datetime index.
- `station_id`: int The ID of the weather station.
- `wind_speed_flag`: int, default 18 Threshold for wind speed flag.
- `temp_range_flag`: tuple, default (-10, 30) Temperature range for temperature flag.
- `hum_flag`: int, default 90 Threshold for relative humidity flag.
- `rolling_window_hours`: int, default 48 Size of the rolling window for rain and snow flags in hours.

**Returns:**
- DataFrame: Merged DataFrame with weather data and flags.

