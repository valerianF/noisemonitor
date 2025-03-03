import pandas as pd

from noisemonitor.modules.rolling import Rolling
from noisemonitor.modules.indicators import Indicators

class NoiseMonitor:
    """Compute discrete values and different types of sliding mean averages
    for various kinds of sound level descriptors, including Leq, L10, L50,
    L90, Lden, Number of Noise Events, overall or at daily or weekly rates, 
    from sound level monitor data, weighted or unweighted. 

    Parameters
    ---------- 
    df: DataFrame
        a compatible DataFrame (typically generated with function load_data()),
        with a datetime, time or pd.Timestamp index and corresponding sound 
        level values in the first column.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.interval = (self.df.index[2] - self.df.index[1]).seconds
        self.rolling = Rolling(self)
        self.indicators = Indicators(self)