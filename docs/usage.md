# Usage Guide

This guide provides detailed examples for using noisemonitor's key features.

## Table of Contents
- [Data Loading](#data-loading)
- [Data Filtering](#data-filtering)
- [Summary Indicators](#summary-indicators)
- [Profile Indicators](#profile-indicators)
- [Coverage Checking](#coverage-checking)
- [Weather Integration (Canada only)](#weather-integration-canada-only)
- [References](#references)

## Data Loading

The `noisemonitor.load()` function imports data from various file formats and automatically handles datetime parsing, timezone conversion, and data resampling, using pandas and dateutil. For easy custom analyses, data is imported into a `pandas.DataFrame` object, as is the output of most noisemonitor functions.

```python
import noisemonitor as nm

# Basic loading
df_1s = nm.load(
    'tests/data/test_data_laeq1s.csv', # Example dataset with 1-second LAeq data
    datetimeindex=0,      # Column index for datetime
    valueindexes=1,       # Column index for sound levels
    header=0,             # Header row index
    sep=','
)

# Load multiple files at once (automatically concatenated)
df2 = nm.load(
    ['day1.csv', 'day2.csv', 'day3.csv'],
    datetimeindex=0,
    valueindexes=[1, 2, 3],    # Multiple sound level columns
    header=0,
    sep=','
)

# Separate date and time columns
df3 = nm.load(
    'data.xlsx',
    dateindex=0,              # Separate date column
    timeindex=1,              # Separate time column
    valueindexes=2,
    timezone='CET',           # Convert from specified timezone
    header=0
)

# Multiple columns (e.g., octave band levels, LAeq vs. LCeq, etc.)
df_freq = nm.load(
    'tests/data/test_data_freq.csv', # Load data with octave bands levels
    datetimeindex=0, # Column index for datetime
    valueindexes=list(range(1, 9)), # Columns with octave band levels
    header=0,
    sep=','
)
```

**Supported formats:** `.csv`, `.xls`, `.xlsx`, `.txt`

**Note:** Reading large files with datetime parsing can be computationally intensive. Parallel processing is enabled by default (can be deactivated) to reduce loading time.

## Data Filtering

The `noisemonitor.filter` module provides functions to subset data by datetime range, but also to filter extreme values, or according to weather conditions (see the [Weather Integration](#weather-integration) for detail).

```python
from datetime import datetime

# Filter by date range
df_filtered = nm.filter.all_data(
    df_1m,
    start_datetime=datetime(2025, 3, 23),
    end_datetime=datetime(2025, 3, 29)
)

# Remove extreme outliers (filtered values are set to NaN)
df_filtered = nm.filter.extreme_values(
    df_1m,
    min_value=30,    # Remove values below 30 dBA
    max_value=100    # Remove values above 100 dBA
)
```

## Summary Indicators

The `noisemonitor.summary` module computes **discrete sound level indicators** that summarize the entire dataset or specific periods (days, weeks, months) with single values. These functions do not compute time series (which are provided in `noisemonitor.profile`, see [Profile Indicators](#profile-indicators)) but rather return aggregate indicators.

### Equivalent level and statistical levels

The `noisemonitor.summary.leq()` function computes the overall or time-filtered (by the time of the day and optionaly day of the week) equivalent level together with statistical sound levels.

```python
# Overall Leq for entire dataset
overall_leq = nm.summary.leq(df_1s, hour1=0, hour2=24)
```

| Leq   | L10   | L50   | L90   |
|-------|-------|-------|-------|
| 49.74 | 52.19 | 47.09 | 42.89 |

```python
# Daytime weekday Leq (7am-7pm)
weekday_leq = nm.summary.leq(
    df_1m,
    hour1=7,
    hour2=19,
    day1='monday',
    day2='friday',
    stats=False  # Disable statistical levels
)
```

**Note:** To be accurate, statistical levels (L10, L50, L90) require a minimum data refresh rate of 1 second.

### Day-Evening-Night Level (Lden)

The `noisemonitor.summary.lden()` function computes the Day-Evening-Night Level (Lden) following the 2002 European standard (European Parliament and Council, 2002).

```python
# Overall Lden
overall_lden = nm.summary.lden(df_1m)
```

| Lden | Lday  | Levening | Lnight |
|------|-------|----------|--------|
| 56.1 | 51.75 | 50.08    | 49.23  |

```python
# Weekdays only
weekday_lden = nm.summary.lden(
    df_1m,
    day1='monday',
    day2='friday'
)
```

### Periodic indicators

The `noisemonitor.summary.periodic()` function computes Leq and Lden (with the option to compute day, evening, and night levels) and at a periodic rate (daily, weekly, or monthly).

```python
# Daily Lden and Leq for each day in the dataset
daily_summary = nm.summary.periodic(
    df_1m,
    freq='D'    # 'D' (daily), 'W' (weekly), or 'MS' (monthly)
)

# Visualization
nm.display.line(
    daily_summary,
    'Lden', 'Leq',
    show_points=True,
    title="Daily Lden and Leq Levels",
    threshold=55 # Option to plot threshold
)
```

![Daily Lden and Leq Levels](images/daily_leq_and_lden.png)

### HARMONICA indicator

The `noisemonitor.summary.harmonica_periodic()` function computes and plots the HARMONICA index based on Mietlicki et al. (2015). Optionally, this can be computed for specific days of the week (e.g., weekdays only).

```python
# Hourly HARMONICA profile
harmonica_profile = nm.summary.harmonica_periodic(df_1s)
```

| Time     | EVT  | BGN  | HARMONICA |
|----------|------|------|-----------|
| 00:00:00 | 2.64 | 0.50 | 3.13      |
| 01:00:00 | 2.38 | 0.34 | 2.72      |
| 02:00:00 | 2.27 | 0.33 | 2.59      |

```python
# Visualization
nm.display.harmonica(harmonica_profile)
```

![HARMONICA Index Plot](images/harmonica_index_plot.png)

**Note:** This function requires data with a refresh rate equal to or below 1 second.

### Frequency band analysis

Summary indicators (e.g., Lden and Leq) can be computed along octave or third-octave sound levels using the `noisemonitor.summary.freq_indicators()` function.

```python
# Summary indicators per frequency band
freq_ind = nm.summary.freq_indicators(
    df_freq,
    values=True,  # Include day, evening, night levels
    stats=False   # Exclude statistical levels
)

# Visualize
nm.display.freq_line(freq_ind, title="Frequency Levels")
```

![Frequency Levels](images/freq_levels.png)

These indicators can also be computed on a periodic basis (i.e., daily, weekly, or monthly) using the `noisemonitor.summary.freq_periodic()` function.

```python
# Periodic frequency analysis
freq_per = nm.summary.freq_periodic(
    df_freq,
    freq='D',
    values=True
)

# Visualize on heatmap
nm.display.freq_map(freq_per["Leq"])
```

![LAeq,24h Frequency Heatmap](images/freq_heatmap.png)

### Sound level distribution histogram

Count days/weeks by sound level ranges using the `noisemonitor.summary.nday()` function.

```python
# Histogram of days by Lden ranges
histogram, bins = nm.summary.nday(
    df_1m,
    indicator='Lden', # Other options:  'Leq', 'Lday', 'Levening', and 'Lnight'
    bins=[50, 55, 60, 65, 70],  # Bin boundaries
    freq='D',                    # Daily ('D') or weekly ('W')
    column=0
)
histogram.head()
```
|   |	Decibel Range |	Number of Days |
|---|-----------------|----------------|
| 0	| [-inf, 50.0)    |	0              |
| 1	| [50.0, 55.0)    |	4              |
| 2	| [55.0, 60.0)    |	6              |
| 3	| [60.0, 65.0)    |	1              |
| 4	| [65.0, 70.0)    |	0              |

```python
# Visualize with color-coded thresholds
nm.display.nday(
    histogram,  # Use the DataFrame from the tuple
    bins=bins,
    thresholds=[55, 60, 65],     # Color boundaries: green/yellow/orange/red
    title="Days by Lden Range"
)
```

![Histogram of Lden values](images/histogram_lden.png)

## Profile indicators (Time Series)

The `nm.profile` module computes **time-varying noise profiles** using rolling windows. In contrast with `noisemonitor.summary` (see - [Summary indicators](#summary-indicators-discrete-values)), these functions return time serie (DataFrames with datetime or time index) showing how noise levels evolve over time.

### Complete Time Series

Compute rolling average Leq, L10, L50, and L90 across the entire dataset with the `noisemonitor.profile.series()` function.

```python
# Hourly sliding averages with 20-minute steps
time_series = nm.profile.series(
    df_1m,
    win=7200,           # Window size in seconds (2 hour)
    step=2400,          # Step size in seconds (40 minutes)
    start_at_midnight=True  # Option to start rolling windows at midnight (typically for computing LAeq,24h levels)
)

# Visualize the time series
nm.display.line(
    time_series, 
    'Leq', 'L10', 'L50', 'L90', 
    step=True,
    title='Complete time series'
    )
```
![Time series](images/profile_series.png)

**Note:** L10, L50, and L90 are shown for the example but shouldn't be computed with refresh times above one second. 

### Daily/Weekly Profiles

Compute average profiles representing daily or weekly patterns with the `noisemonitor.profile.periodic()` function.

```python
# Weekday profile (Mon-Fri)
weekday_profile = nm.profile.periodic(
    df_1m,
    hour1=23, # If hour1 > hour2, profile will be computed outside those hours
    hour2=22,
    day1='monday',   # Use day names, not numbers
    day2='friday',
    win=3600,
    step=1200
)

# Weekend profile (Sat-Sun)
weekend_profile = nm.profile.periodic(
    df_1m,
    hour1=0,
    hour2=24,
    day1='saturday',
    day2='sunday',
    win=3600,
    step=1200
)

# Plot weekday values
nm.display.line(
    weekday_profile,
    'Leq', 'L90',
    fill_background=True, # Option to fill background depending on the time of the day.
    title='Weekday Noise Profile' 
)
```

![Weekday Profile](images/weekday_profile.png)

```python
# Compare weekday vs. weekend
nm.display.compare(
    [weekday_profile, weekend_profile],
    ['Weekdays', 'Weekend'],
    'L90', 'Leq'
)
```

![Comparing Weekday vs. Weekend Profiles](images/weekend_vs_weekday_profile.png)

### Number of Noise Events (NNE)

The function `noisemonitor.profile.nne()` can compute the Number of Noise Events (NNE) following the algorithm proposed in (Brown and De Coensel, 2018). The function computes the average NNE using rolling windows, computing daily or weekly profiles. Note that this function is computationally expensive as noise NNEs are separately computed for each individual day and then averaged since background levels are relative to each day.

```python
# Average daily noise event profile
nne_profile = nm.profile.nne(
    df_1s,
    hour1=23,
    hour2=22,
    background_type='L50',     # Use L50 as background reference
    exceedance=5,              # Events must exceed background by 10 dB
    min_gap=5,                 # Minimum 5 seconds between events
    win=3600,
    step=1200
)

# Visualize event frequency
nm.display.line(
    nne_profile, 
    'Average NNEs',
    title='Noise Events Profile',
    ylabel='Noise Events (L50 + 5dBA)',
    fill_background=True
)
```
![Number of Noise Events Profile](images/nne_profile.png)

**Note:** As emergence indicators, NNEs shouldn't be computed with refresh times above one second.

### Advanced indicators

Other advanced acoustic indicators can be computed with the `noisemonitor.profile.periodic()` function, such as roughness indicators (DeFrance et al., 2010), Traffic Noise Index (Griffiths and Langdon, 1968) as well as the Noise Pollution Level (Robinson, 1971).

```python
# Daily profile with advanced indicators
advanced_profile = nm.profile.periodic(
    df_1s,
    hour1=7,
    hour2=19,
    win=3600,
    step=1200,
    traffic_noise_indicators=True,
    roughness_indicators=True
)

# Plot roughness indicators
nm.display.line(
    advanced_profile,
    'dLav', 'dLmax,1', 'dLmin,90',
    title='Roughness Indicators'
)
```

![Roughness Profile](images/roughness_profile.png)

```python
# Plot traffic noise indexes and noise pollution levels
nm.display.line(
    advanced_profile,
    'TNI', 'NPL',
    title='Traffic Noise Index and Noise Pollution Level'
)
```

![Traffic Noise Index and Noise Pollution Level Profile](images/TNI_NPL_profile.png)

**Note:** Because they are based on statistical indicators, these indicators shouldn't be computed with refresh rates above one second.

### Frequency Band Profiles

As with `noisemonitor.summary` functions, you can compute frequency-wise time series (`noisemonitor.profile.freq_series()`), as well as weekly or daily profiles (`noisemonitor.profile.freq_periodic()`).

```python
# Time series for all frequency bands
freq_series = nm.profile.freq_series(
    df_freq,
    win=7200,
    step=2400
)

# Visualize as heatmap
nm.display.freq_map(freq_series['Leq'], title="Frequency Evolution Over Time")
```
![Frequency Bands Time Series](images/freq_series_heatmap.png)

```python
# Daily profiles for each frequency band
freq_daily = nm.profile.freq_periodic(
    df_freq,
    hour1=23,
    hour2=22,
    win=3600,
    step=1200
)

# Visualize daily frequency patterns
nm.display.freq_map(freq_daily['Leq'], title="Daily Frequency Profile")
```

![Frequency Bands Daily Profile](images/freq_daily_profile.png)

**Note:** being computationnaly expensive, these functions use parallel processing by default. For big datasets, we recommend using `noisemonitor.summary.freq_indicators()` over `noisemonitor.profile.freq_series()`.

### Custom statistical levels

```python
# Compute custom Lx levels
custom_stat_profile = nm.profile.periodic(
    df_1s,
    hour1=7,
    hour2=19,
    win=3600,
    step=1200,
    stat=[1, 5, 99]  # L1, L5, L99
)

# Create custom indicators
custom_stat_profile['L5-L99'] = custom_stat_profile['L5'] - custom_stat_profile['L99']
custom_stat_profile['L1-Leq'] = custom_stat_profile['L1'] - custom_stat_profile['Leq']

nm.display.line(
    custom_stat_profile,
    'L5-L99', 'L1-Leq',
    title="Custom Statistical Levels"
)
```

![Daily Profile for Custom Statistical Levels](images/daily_custom_stats.png)

## Coverage Checking

Most functions support data coverage validation (deactivated by default). When activated (`coverage_check=True`), coverage (i.e. number of complete cases relative to the size of the dataset) will be assessed based on the computation period (e.g. daily for daily summary indicators; hourly for hourly time series), and results for the corresponding period will be filtered out if coverage requirement (`coverage_threshold` argument) is not met.

```python
with_coverage_leq = nm.summary.leq(
    df_1m,
    hour1=12,
    hour2=16,
    coverage_check=True,
    coverage_threshold=0.8  # Minimum 80% data coverage required
)
```

**Note:** A warning is raised if data coverage is not met for any of the computed periods. For Lden computation, coverage is assessed separately for each period (day, evening, and night).

## Weather Integration

The `nm.weather.weathercan` module integrates historical weather data from Environment Canada.

**Installation:**
```bash
pip install noisemonitor[weather]
```

### Finding weather stations

```python
# Find nearby stations
coordinates = [45.505571, -73.598987]  # [lat, lng]

stations = nm.weather.weathercan.get_historical_stations(
    coordinates=coordinates,
    radius=25,
    start_year=2020,
    end_year=2025
)
```

### Merging weather data

```python
# Merge weather data (async function)
df_with_weather = await nm.weather.weathercan.merge_weather(
    df_1m,
    station_id=30165,
    wind_speed_flag=18,
    temp_range_flag=(-10, 30),
    hum_flag=90,
    rolling_window_hours=48
df_with_weather.head()
```

| datetime | LEQ dB -A | Wind Spd (km/h) | Precip. Amount (mm) | Weather | Temp (°C) | Rel Hum (%) | Wind_Spd_Flag | Rain_Flag | Snow_Flag | Temp_Flag | Rel_Hum_Flag | Rain_Flag_Roll | Snow_Flag_Roll |
|----------|-----------|-----------------|---------------------|---------|-----------|-------------|---------------|-----------|-----------|-----------|--------------|----------------|----------------|
| 2025-03-21 00:00:00 | 47.36 | 12.0 | 0.0 | NaN | 6.6 | 81.0 | True | False | False | False | False | False | False |
| 2025-03-21 00:01:00 | 48.36 | 12.0 | 0.0 | NaN | 6.6 | 81.0 | True | False | False | False | False | False | False |
| 2025-03-21 00:02:00 | 47.60 | 12.0 | 0.0 | NaN | 6.6 | 81.0 | True | False | False | False | False | False | False |
| 2025-03-21 00:03:00 | 48.74 | 12.0 | 0.0 | NaN | 6.6 | 81.0 | True | False | False | False | False | False | False |
| 2025-03-21 00:04:00 | 49.75 | 12.0 | 0.0 | NaN | 6.6 | 81.0 | True | False | False | False | False | False | False |

The merged DataFrame now includes weather data together with boolean flags indicating whether inclusion criteria are fulfilled for each row (e.g., wind speed below 18 km/h).

### Weather contingency analysis

Assess the potential impact of weather conditions on sound levels using the `nm.weather.weathercan.contingency_weather_flags()` function.

```python
# Assess weather impact
contingency = nm.weather.weathercan.contingency_weather_flags(
    df_with_weather
)

contingency.head()
```

| Condition | Leq | Lden | Diff Leq | Diff Lden | Covered data (%) |
|-----------|-----|------|----------|-----------|------------------|
| All Data | 50.76 | 56.10 | 0.00 | 0.00 | 100.0 |
| Wind speed >= 18 km/h | 51.04 | 56.12 | 0.28 | 0.02 | 44.8 |
| Rain in the last 48h | 51.37 | 56.65 | 0.61 | 0.55 | 61.9 |
| No Flag - Wind | 50.53 | 56.14 | -0.23 | 0.04 | 55.2 |
| No Flag - Rain | 49.54 | 55.08 | -1.22 | -1.02 | 38.1 |

**Note:** Sound level differences below 1 dB are generally considered negligible.

### Visualizing weather impact

```python
# Compare profiles by weather condition
nm.display.compare_weather_daily(
    df_with_weather,
    column=0,
    include_wind_flag=True,
    include_rain_flag=True,
    win=3600,
    step=1200,
    title="Daily Profiles: Weather Comparison"
)
```

![Daily Profile With and Without Weather Flags](images/daily_profile_weather.png)

```python
# Time series with weather flags
nm.display.line_weather(
    df_with_weather,
    win=3600,
    include_wind_flag=True,
    include_rain_flag=True
)
```

![Time Series with Weather Flags Overlaid](images/series_weather.png)

### Weather-based filtering

Once the potential impact of weather conditions (e.g. wind, rain, snow) on sound levels have been analyzed, data can be filtered based on weather conditions to ensure data reliability. 

```python
# Filter by weather conditions
df_filtered = nm.filter.weather_flags(
    df_with_weather,
    filter_wind_flag=True,
    filter_rain_flag=False,
    filter_temp_flag=False,
    filter_rel_hum_flag=False,
    filter_snow_flag=False
)
```

Output:
```
Proportion of values filtered out: 44.80%
```

**Note:** Weather data retrieval is asynchronous. Use `await` with `merge_weather()` or run it in an async context.

## References

- Brown, A. L., & De Coensel, B. (2018). A study of the performance of a generalized exceedance algorithm for detecting noise events caused by road traffic. *Applied Acoustics*, 138, 101-114. https://doi.org/10.1016/j.apacoust.2018.03.031
- DeFrance, J., Palacino. J., & Baulac, M. (2010). Auscultation acoustique des aménagements cyclables en milieu urbain. *Proceedings of the 10ème Congrès Français d’Acoustique*, Lyon, France. https://hal.science/hal-00551186v1
- European Parliament and Council. (2002). Directive 2002/49/EC of the European Parliament and of the Council of 25 June 2002 relating to the assessment and management of environmental noise. *EUR-Lex*. http://data.europa.eu/eli/dir/2002/49/2021-07-29
- Griffiths, I. M., & Langdon, F. J. (1968). Subjective response to road traffic noise. *Journal of Sound and Vibration*, 1, 16-32, 375-378. https://doi.org/10.1016/0022-460X(68)90191-0
- Mietlicki, C., Mietlicky, F., Ribeiro, C., and Gaudiber, P. (2015). The HARMONICA project, new tools to assess environmental noise and better inform the public. *Proceedings of Forum Acusticum*, Krakow, Poland. https://www.bruitparif.fr/pages/En-tete/500%20Innovation%20et%20recherche/700%20Publications%20scientifiques/2014%20-%20The%20HARMONICA%20project.pdf
- Robinson, D. W. (1971). The Concept of Noise Pollution Level. *Journal of Occupational Medicine*, 13(12), 602. https://journals.lww.com/joem/toc/1971/12000
