# soundmonitor

Python package for sound level monitor (long-term sound level meter) data analysis. Contains various fonctions to analyze and plot sliding averages, weekly and daily averages for noise descriptors such as the Lden, Leq and statistical indicators such as L10, L50 and L90 (see this [paper](https://hal.science/hal-01373857v3/file/doc00025834.pdf) for more details on these descriptors). The package works with equivalent sound level data (weighted or un-weighted) captured at regular intervals, typically ranging from 1 second to 1 minute.

## Installation

TBD

## Usage

### Read sound level monitor data

A function is included to read data in the form of either .csv, .xls or .xlsx files from a sound level monitor and convert them to such a DataFrame with datetime or pandas TimeStamp index. Multiple files can be read at once, and the resulting data will be concatenated in a single DataFrame. Note that you must indicate the datasheet's indexes corresponding to date, time and captured equivalent sound level. Reading .xls or .xls files with pandas and automatic parsing into datetime values is computationaly expensive and the process can last a few minutes, depending on the input data.

```python
import soundmonitor as sm
from datetime import datetime

# Load .csv data
df = sm.utilities.load_data(["test.csv"], datetimeindex=0, valueindex=1)

# Filter out data between specified dates if required
df = sm.utilities.filter_data(df, datetime(2020,19,05,10,12), datetime(2020,19,06,10,12))
```
