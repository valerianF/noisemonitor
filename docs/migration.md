# Migration Guide: Version 0.0.3 → 1.0.0

Version 1.0.0 introduces breaking changes. The class-based API has been replaced with a functional API organized into modules.

## Key Changes

| Version 0.0.3 | Version 1.0.0 |
|---------------|---------------|
| `nm.load_data()` | `nm.load()` |
| `nm.filter_data()` | `nm.filter.all_data()` |
| `average = nm.NoiseMonitor(df)` | No class instantiation needed |
| `average.leq()` | `nm.summary.leq(df, ...)` |
| `average.lden()` | `nm.summary.lden(df, ...)` |
| `average.sliding_average()` | `nm.profile.series(df, ...)` |
| `average.daily()` | `nm.profile.periodic(df, ...)` |
| `average.weekly()` | `nm.profile.periodic(df, ...)` |
| `nm.level_plot()` | `nm.display.line()` |

## Module Organization

Version 1.0.0 organizes functions into logical modules:

- **`nm.load()`** - Data loading
- **`nm.filter`** - Data filtering and subsetting
- **`nm.summary`** - Discrete summary indicators (daily, monthly, yearly Leq, Lden, etc.)
- **`nm.profile`** - Time-varying profiles (rolling averages with daily, weekly patterns, etc.)
- **`nm.display`** - Visualization functions
- **`nm.weather`** - Weather integration (Canada)

## Migration Examples

### Loading Data

**Before (0.0.3):**
```python
df = nm.load_data('data.csv', datetime_col=0, value_col=1)
```

**After (1.0.0):**
```python
df = nm.load('data.csv', datetimeindex=0, valueindexes=1, header=0, sep=',')
```

### Filtering Data

**Before:**
```python
df_filtered = nm.filter_data(df, start='2025-01-01', end='2025-01-31')
```

**After:**
```python
from datetime import datetime
df_filtered = nm.filter.all_data(
    df,
    start_datetime=datetime(2025, 1, 1),
    end_datetime=datetime(2025, 1, 31)
)
```

### Computing Indicators

**Before:**
```python
average = nm.NoiseMonitor(df)
leq_result = average.leq(hour1=7, hour2=19)
lden_result = average.lden()
```

**After:**
```python
leq_result = nm.summary.leq(df, hour1=7, hour2=19)
lden_result = nm.summary.lden(df)
```

### Time Series

**Before:**
```python
average = nm.NoiseMonitor(df)
series = average.sliding_average(window=3600, step=1200)
```

**After:**
```python
series = nm.profile.series(df, win=3600, step=1200)
```

### Daily/Weekly Profiles

**Before:**
```python
average = nm.NoiseMonitor(df)
daily = average.daily(hour1=0, hour2=23, window=3600)
weekly = average.weekly(day1='monday', day2='friday', hour1=7, hour2=19)
```

**After:**
```python
# Daily profile (all days)
daily = nm.profile.periodic(df, hour1=0, hour2=23, win=3600)

# Weekly profile (weekdays only)
weekly = nm.profile.periodic(
    df,
    hour1=7,
    hour2=19,
    day1='monday',
    day2='friday',
    win=3600
)
```

### Visualization

**Before:**
```python
nm.level_plot(df, 'Leq', 'L10', title='Noise Levels')
```

**After:**
```python
nm.display.line(df, 'Leq', 'L10', title='Noise Levels')
```
