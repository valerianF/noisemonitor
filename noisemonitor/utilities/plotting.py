""" Functions used to plot data.
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.patches import Polygon
from datetime import datetime, time
from typing import List

def get_datetime_index(df: pd.DataFrame) -> np.ndarray:
    """Get the datetime index for plotting.

    Parameters
    ---------- 
    df: DataFrame
        DataFrame with a datetime, time, or pandas.Timestamp index.

    Returns
    ---------- 
    x: array-like
        Array of datetime objects for plotting.
    """
    if isinstance(df.index[0], pd.Timestamp):
        return df.index.to_pydatetime()
    elif isinstance(df.index[0], time):
        return convert_datetime_index(df)
    elif not isinstance(df.index[0], datetime):
        raise TypeError(f'DataFrame index must be of type datetime, \
                        time or pd.Timestamp not {type(df.index[0])}')
    
def plot_compare(
    dfs: List[pd.DataFrame], 
    labels: List[str], 
    *args: str, 
    ylabel: str = "Sound Level (dBA)",
    weighting: str = "A", 
    step: bool = False, 
    figsize: tuple = (10,8), 
    fill_between: bool = None, 
    **kwargs
) -> None:
    """Compare multiple DataFrames by plotting their columns in the same plot.

    Parameters
    ---------- 
    dfs: list of DataFrames
        list of compatible DataFrames (typically generated with functions 
        load_data(), NoiseMonitor.daily() or NoiseMonitor.weekly() ),
        with a datetime, time or pandas.Timestamp index.
    labels: list of str
        list of labels for each DataFrame.
    step: bool, default False
        if set to True, will plot the data as a step function.
    figsize: tuple, default (10,8)
        figure size in inches.
    fill_between: list of tuples, default None
        list of tuples specifying the columns to use for filling in-between
        values. Each tuple should contain three column names: (lower_bound, 
        upper_bound, column_to_plot).
    *args: str
        column name(s) to be plotted.
    weighting: str, default "A"
        type of sound level data, typically A, C or Z. 
    **kwargs: any
        ylim and title arguments can be passed to matplotlib. 
    """
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=figsize)

    for df, label in zip(dfs, labels):
        for arg in args:
            ax = plot_levels(df, arg, ylabel=ylabel, weighting=weighting, step=step,
                            figsize=figsize, ax=ax, fill_between=fill_between,
                            **kwargs)
            ax.lines[-1].set_label(f"{label} - {arg}")

    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return

def convert_datetime_index(df: pd.DataFrame) -> np.ndarray:
    """Converts time index from dataframe to datetime array for plotting.

    Parameters
    ---------- 
    df: DataFrame
        DataFrame with a datetime, time, or pandas.Timestamp index.
    """

    if df.index[-1] < df.index[0]:
        x1 = df.iloc[(df.index >= df.index[0])].index.map(
            lambda a: datetime.combine(datetime(1800, 10, 9), a))
        x2 = df.iloc[(df.index < df.index[0])].index.map(
            lambda a: datetime.combine(datetime(1800, 10, 10), a))
        x = x1.union(x2)
    else:
        x = df.index.map(lambda a: datetime.combine(datetime(1800, 10, 10), a))
    
    return x.to_pydatetime()
    
def plot_harmonica(df: pd.DataFrame) -> None:
    """Plot the HARMONICA index.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the HARMONICA indicators.
    """
    # Raise an error if the DataFrame length is not 24
    if len(df) != 24:
        raise ValueError("The input DataFrame must contain daily HARMONICA "
                         "indexes computed with the daily_weekly_harmonica "
                         "function.")


    # Plot the HARMONICA index
    plt.figure(figsize=(12, 5))
    plt.rcParams.update({'font.size': 16})

    ax = plt.gca()
    ax.grid(linestyle='--', zorder=0) 

    for hour, row in df.iterrows():
        hour = hour.hour
        if 6 < hour < 22:
            if row['HARMONICA'] < 4:
                color = 'green'
            elif 4 <= row['HARMONICA'] < 8:
                color = 'orange'
            else:
                color = 'red'
        else:
            if row['HARMONICA'] < 3:
                color = 'green'
            elif 3 <= row['HARMONICA'] < 7:
                color = 'orange'
            else:
                color = 'red'

        plt.bar(hour, row['BGN'], color=color, width=0.81, zorder=2)
        triangle = Polygon(
            [[hour - 0.4, row['BGN'] + 0.1],
             [hour + 0.4, row['BGN'] + 0.1],
             [hour, row['HARMONICA']]],
            closed=True, 
            color=color,
            zorder=3
        )
        ax.add_patch(triangle)

    # Set x-ticks every 3 hours with ':00' added to every tick
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f'{hour}h' for hour in range(0, 24, 3)])

    if df['HARMONICA'].max() <= 10:
        plt.ylim(0, 10)

    plt.xlabel('Hour')
    plt.ylabel('HARMONICA Index')
    plt.title('HARMONICA Index Plot')
    plt.grid(linestyle='--', zorder=3)
    plt.tight_layout()
    plt.show()

def plot_levels(
        df: pd.DataFrame, 
        *args: str, 
        ylabel: str = "Sound Level (dBA)", 
        step: bool = False, 
        figsize: tuple = (10,8), 
        ax: matplotlib.axes.Axes = None,
        fill_between: bool = None, 
        **kwargs
) -> matplotlib.axes.Axes:
    """Plot columns of a dataframe according to the index, using matplotlib.

    Parameters
    ---------- 
    df: DataFrame
        a compatible DataFrame (typically generated with functions load_data(),
        NoiseMonitor.daily() or NoiseMonitor.weekly() ),
        with a datetime, time or pandas.Timestamp index.
    *args: str
        column name(s) to be plotted.
    ylabel: str, default "Sound Level (dBA)"
        label for the y-axis.
    step: bool, default False
        if set to True, will plot the data as a step function.
    figsize: tuple, default (10,8)
        figure size in inches.
    ax: matplotlib.axes.Axes, default None
        Axes object to plot on. If None, a new figure and axes are created.
    fill_between: list of tuples, default None
        list of tuples specifying the columns to use for filling in-between
        values. Each tuple should contain three column names: (lower_bound, 
        upper_bound, column_to_plot).
    weighting: str, default "A"
        type of sound level data, typically A, C or Z. 
    **kwargs: any
        ylim and title arguments can be passed to matplotlib. 
    """
    x = get_datetime_index(df)
    
    if ax is None:
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize=figsize)

    for i in range(0, len(args)):
        if step:
            ax.step(x, df.loc[:, args[i]], label=args[i])
        else:
            ax.plot(x, df.loc[:, args[i]], label=args[i])

    if fill_between:
        for lower, upper, column in fill_between:
            if lower in df.columns and upper in df.columns and column in df.columns:
                ax.fill_between(x, df[lower], df[upper], alpha=0.15,
                                label=f"{column} - {lower} to {upper}")

    if any(isinstance(df.index[0], t) for t in [pd.Timestamp, datetime]):
        ax.figure.autofmt_xdate()
        ax.set_xlabel('Date (y-m-d)')
    elif isinstance(df.index[0], time):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_xlabel('Time (h:m)')

    ax.set_ylabel(ylabel)

    if "ylim" in kwargs:
        ax.set_ylim(kwargs["ylim"])
    if "title" in kwargs:
        ax.set_title(kwargs["title"])

    ax.grid(linestyle='--')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return ax

def plot_nday(
    nday_df: pd.DataFrame, 
    bins: List[int], 
    freq: str = 'D',
    thresholds: List[int] = [55, 60, 65],
    title: str = None,
    figsize: tuple = (10,8)
) -> None:
    """Plots a histogram from the output of noisemonitor.indicators.nday() 
    function with the number of days under a certain decibel range 
    and a color coding corresponding to the indicated thresholds.

    Parameters
    ----------
    nday_df: pd.DataFrame
        DataFrame output from the nday() function with the number 
        of days for each decibel range.
    bins: list of int
        List of decibel values to define the bins.
    freq: str, default 'D'
        Frequency for the label. 'D' for daily and 'W' for weekly.
    thresholds: list of int, default [55, 60, 65]
        Thresholds for color coding. 
    title: str, optional
        Title for the plot.
    figsize: tuple, default (10,8)
        Figure size in inches.

    Returns
    ----------
    None
    """
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=figsize)


    bars = nday_df.plot(
        kind='bar', 
        width=0.8, 
        zorder=3,
        ax=ax,
        legend=False
    )

    # Manually set the colors for each bar
    for bar, value in zip(bars.patches, [0] + bins):
        if value < thresholds[0]:
            bar.set_color('#89dc35')
        elif value < thresholds[1]:
            bar.set_color('#dcdc35')
        elif value < thresholds[2]:
            bar.set_color('#dc8935')
        else:
            bar.set_color('#dc3535')

    ax.grid(True, linestyle='--', zorder=0)
    plt.xlabel('Decibel Range (dBA)')

    if freq == 'D':
        plt.ylabel('Number of Days')
    elif freq == 'W':
            plt.ylabel('Number of Weeks')   

    # Set custom x-tick labels
    ytick_labels = [f'<{bins[0]}'] + \
        [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)] + \
        [f'>{bins[-1]}']
    ax.set_xticklabels(ytick_labels)

    if title is not None:
        plt.title(title)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return

