"""Top-level module"""

# Import main classes
from noisemonitor.modules.noisemonitor import NoiseMonitor
from noisemonitor.modules.rolling import Rolling
from noisemonitor.modules.indicators import Indicators
from noisemonitor.modules.weather_can import (
    get_historical_stations_can, 
    get_historical_data_can,
    merge_weather_can,
    contingency_weather_flags
)

# Import utility functions
from noisemonitor.utilities.loading import load_data
from noisemonitor.utilities.compute import (
    equivalent_level, 
    harmonica, 
    hourly_harmonica, 
    lden, 
    noise_events
)
from noisemonitor.utilities.plotting import (
    plot_compare,
    plot_freqs_overall,
    plot_freqs_heatmap,
    plot_harmonica,
    plot_levels,
    plot_nday,
    plot_with_weather,
    plot_compare_weather_daily
)
from noisemonitor.utilities.process import (
    filter_data,
    filter_extreme_values,
    filter_weather_flags
)

# Import decorators
from noisemonitor.utilities.decorators import (
    validate_interval
)