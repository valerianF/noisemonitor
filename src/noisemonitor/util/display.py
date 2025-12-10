""" Functions used to plot data.
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.patches import Polygon
from datetime import datetime, time
from typing import List, Optional, Union
from itertools import cycle

from noisemonitor.profile import series, periodic
from . import core
    
def compare(
    dfs: List[pd.DataFrame], 
    labels: List[str], 
    *args: str, 
    ylabel: str = "Sound Level (dBA)",
    weighting: str = "A", 
    step: bool = False, 
    show_points: bool = False, 
    figsize: tuple = (10,8), 
    **kwargs
) -> matplotlib.axes.Axes:
    """Compare multiple DataFrames by plotting their columns in the same plot.

    Parameters
    ---------- 
    dfs: list of DataFrames
        list of compatible DataFrames (typically generated with functions 
        load_data(), NoiseMonitor.daily() or NoiseMonitor.weekly() ),
        with a datetime, time or pandas.Timestamp index.
    *args: str
        column name(s) to be plotted.
    labels: list of str
        list of labels for each DataFrame.
    step: bool, default False
        if set to True, will plot the data as a step function.
    show_points: bool, default False
        if True, scatter points will be added on top of the line plots.
    figsize: tuple, default (10,8)
        figure size in inches.
    weighting: str, default "A"
        type of sound level data, typically A, C or Z. 
    **kwargs: any
        ylim and title arguments can be passed to matplotlib. 

    Returns
    ----------
    ax: matplotlib.axes.Axes
        Axes object containing the plot.
    """
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=figsize)

    for df, label in zip(dfs, labels):
        for arg in args:
            ax = line(df, arg, ylabel=ylabel, weighting=weighting, 
                            step=step, show_points=show_points,
                            figsize=figsize, ax=ax,
                            **kwargs)
            ax.lines[-1].set_label(f"{label} - {arg}")

    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return ax

def freq_line(
    df: pd.DataFrame,
    weighting: str = "A",
    title: str = "Overall Frequency Bands",
    figsize: tuple = (12, 8),
    **kwargs
) -> None:
    """
    Plot the overall levels for each frequency band.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing overall levels for each frequency band, with
        rows as indicators (e.g., Leq, Lden) and columns as frequency bands.
    weighting: str, default "A"
        Type of sound level data, typically A, C or Z.
    title: str, default "Overall Frequency Bands"
        Title for the plot.
    figsize: tuple, default (10, 8)
        Figure size in inches.
    ax: matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.
    **kwargs: any
        Additional keyword arguments passed to the plot function.

    Returns
    ----------
    None
    """

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Plot each row (indicator) in the DataFrame
    for indicator in df.index:
        ax.plot(
            df.columns,  # Convert frequency bands to floats
            df.loc[indicator],
            '-o',
            label=indicator,
            **kwargs
        )


    # Add labels, title, and legend
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"Sound Level (dB{weighting})")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, zorder=0, linestyle=(0, (2.5, 5)))

    plt.xticks( rotation=45)
    plt.tight_layout()
    plt.show()

def freq_map(
    df: pd.DataFrame,
    title: str = "Frequency Bands Heatmap",
    ylabel: str = "Frequency Band",
    figsize: tuple = (12, 8),
    weighting: str = "A"
) -> None:
    """
    Plot a heatmap of sound levels across frequency bands over time.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing sound levels with frequency bands as columns
        and datetime index as rows.
    title: str, default "Frequency Bands Heatmap"
        Title for the heatmap.
    ylabel: str, default "Frequency Band"
        Label for the y-axis.
    figsize: tuple, default (12, 8)
        Figure size in inches.

    Returns
    ----------
    None
    """
    x = _get_datetime_index(df)

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=figsize)
    ax = plt.gca()

    pcm = ax.pcolormesh(
        x,
        df.columns.astype(str),  # Convert frequency bands to strings
        df.T,  # Transpose to align frequency bands on the y-axis
        shading='auto',
        cmap='magma'
    )

    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label(f"Sound Level (dB{weighting})")

    _format_time_axis(ax, x, df)

    y_ticks = ax.get_yticks()
    if len(y_ticks) > 15:
        ax.set_yticks(y_ticks[::2]) 

    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    plt.show()

def harmonica(
        df: pd.DataFrame,
        title: str = "HARMONICA Index Plot"
) -> None:
    """Plot the HARMONICA index.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the HARMONICA indicators.
    title: str, default "HARMONICA Index Plot"
        Title for the plot.
    """
    # Raise an error if the DataFrame length is not 24
    if len(df) != 24:
        raise ValueError("The input DataFrame must contain daily HARMONICA "
                        "indexes computed with the daily_weekly_harmonica "
                        "function.")


    # Plot the HARMONICA index
    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 16})

    ax = plt.gca()
    ax.grid(linestyle='--', zorder=0) 

    df = pd.concat([
        df[(df.index >= time(22, 0)) & (df.index <= time(23, 59))],
        df[(df.index >= time(0, 0)) & (df.index < time(22, 0))]
    ])

    for position, (hour, row) in enumerate(df.iterrows()):
        hour = hour.hour
        if 6 <= hour < 22:
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

        plt.bar(position, row['BGN'], color=color, width=0.81, zorder=2)
        triangle = Polygon(
            [[position - 0.4, row['BGN'] + 0.1],
            [position + 0.4, row['BGN'] + 0.1],
            [position, row['HARMONICA']]],
            closed=True, 
            color=color,
            zorder=3
        )
        ax.add_patch(triangle)

    reordered_hours = [22] + list(range(0, 22, 2))
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f'{hour}h' for hour in reordered_hours])

    ax.axvspan(-0.5, 7.5, color="lightskyblue", alpha=1, zorder=0)

    if df['HARMONICA'].max() <= 10:
        plt.ylim(0, 10)

    plt.xlim(-0.5, 23.5)
    plt.xlabel('Hour')
    plt.ylabel('HARMONICA Index')
    plt.title(title)
    plt.grid(linestyle='--', zorder=3)
    plt.tight_layout()
    plt.show()

def line(
        df: pd.DataFrame, 
        *args: str, 
        ylabel: str = "Sound Level (dBA)", 
        step: bool = False, 
        show_points: bool = False, 
        fill_background: bool = False, 
        figsize: tuple = (10,8), 
        ax: matplotlib.axes.Axes = None,
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
        if True, will plot the data as a step function.
    show_points: bool, default False
        if True, scatter points will be added on top of the line plots.
    fill_background: bool, default False
        If True, fills the background with colors based on the time of day:
        - Day (7-19h): Light yellow
        - Evening (19-23h): Light orange
        - Night (23-7h): Light blue
        Only applied for temporal resolutions > 6/day.
    figsize: tuple, default (10,8)
        figure size in inches.
    ax: matplotlib.axes.Axes, default None
        Axes object to plot on. If None, a new figure and axes are created.
    weighting: str, default "A"
        type of sound level data, typically A, C or Z. 
    **kwargs: any
        ylim and title arguments can be passed to matplotlib. 

    Returns
    ----------
    ax: matplotlib.axes.Axes
        Axes object containing the plot.
    """
    x = _get_datetime_index(df)
    
    if ax is None:
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize=figsize)

    # Check temporal resolution and raise a warning if > 4 hours
    resolution = (x[1] - x[0]).total_seconds() / 3600
    if fill_background and resolution >= 4:
        warnings.warn("Background filling is only applied for temporal "
                    "resolutions > 6/day.")
        fill_background = False

    if fill_background:
        day_patch = None
        evening_patch = None
        night_patch = None
        for i in range(len(x) - 1):
            start_time = x[i].time()
            if time(7, 0) <= start_time < time(19, 0):
                day_patch = ax.axvspan(x[i], x[i + 1], 
                                    color="ivory", alpha=1, zorder=0)
            elif time(19, 0) <= start_time < time(23, 0):
                evening_patch = ax.axvspan(x[i], x[i + 1], 
                                    color="papayawhip", alpha=1, zorder=0)
            else:
                night_patch = ax.axvspan(x[i], x[i + 1], 
                                    color="aliceblue", alpha=1, zorder=0)

    for i in range(0, len(args)):
        if step:
            line, = ax.step(x, df.loc[:, args[i]], label=args[i])
        else:
            line, = ax.plot(x, df.loc[:, args[i]], label=args[i])

        if show_points:
            ax.scatter(x, df.loc[:, args[i]], color=line.get_color(), s=15,
                zorder=3)

    _format_time_axis(ax, x, df)

    ax.set_ylabel(ylabel)


    if "ylim" in kwargs:
        ax.set_ylim(kwargs["ylim"])
    if "title" in kwargs:
        ax.set_title(kwargs["title"])

    ax.grid(linestyle='--', zorder=1)

    handles, labels = ax.get_legend_handles_labels()

    # Add a legend for the background colors
    if fill_background:
        custom_patches = []
        if day_patch:
            custom_patches.append(matplotlib.patches.Patch(
                color="ivory", label="Day"))
        if evening_patch:
            custom_patches.append(matplotlib.patches.Patch(
                color="papayawhip", label="Evening"))
        if night_patch:
            custom_patches.append(matplotlib.patches.Patch(
                color="aliceblue", label="Night"))
        if custom_patches != []:    
            handles.extend(custom_patches)

    ax.legend(handles=handles)

    plt.xticks(rotation=45)
    plt.tight_layout()
    return ax

def nday(
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

def line_weather(
    df: pd.DataFrame, 
    column: Union[int, str] = 0,
    win: int = None,
    include_wind_flag: bool = True,
    include_rain_flag: bool = True,
    include_temp_flag: bool = False,
    include_rel_hum_flag: bool = False,
    include_snow_flag: bool = False,
    coverage_check: bool = False,
    coverage_threshold: float = 0.5
):
    """
    Plot sound levels with weather flags.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the data.
    column: Union[int, str], default 0
        The column name or index for sound levels. If None, the first column
        will be used.
    window_size: int, optional
        The window size for rolling average. If None, no rolling average is applied.
    show_wind_spd_flag: bool, default True
        Whether to show the Wind Speed Flag.
    show_rain_flag: bool, default True
        Whether to show the Rain Flag.
    show_rel_hum_flag: bool, default True
        Whether to show the Relative Humidity Flag.
    show_temp_flag: bool, default False
        Whether to show the Temperature Flag.
    show_snow_flag: bool, default True
        Whether to show the Snow Flag.
    coverage_check: bool, default False
        Whether to check data coverage when computing levels.
    coverage_threshold: float, default 0.5
        Minimum coverage ratio required (0.0-1.0).

    Returns
    ----------
    None
    """
    column = core._column_to_index(df, column)

    if win:
        levels_df = series(df, column=column, win=win, step=0,
                        coverage_check=coverage_check,
                        coverage_threshold=coverage_threshold)
        _column_p = 0
        # Resample the weather data to match the window size
        resample_rule = f'{win}s'
        df = df.resample(resample_rule).mean(numeric_only=True)
    else:
        levels_df = df
        _column_p = column

    line(
        levels_df,
        levels_df.columns[_column_p],
        step=True,
        title="Sound Levels with Weather Flags",
        figsize=(12, 8)
    )

    sound_min = levels_df.iloc[:, _column_p].min()
    sound_max = levels_df.iloc[:, _column_p].max()

    flags = {
        'Wind_Spd_Flag': include_wind_flag,
        'Rain_Flag_Roll': include_rain_flag,
        'Rel_Hum_Flag': include_rel_hum_flag,
        'Snow_Flag_Roll': include_snow_flag,
        'Temp_Flag': include_temp_flag
    }

    linestyles = cycle(['--', ':', '-.', '-'])

    for flag, show in flags.items():
        if show:
            normalized_flag = df[flag].astype(int) * (
                sound_max - sound_min) + sound_min
            plt.step(
                df.index,
                normalized_flag,
                label=flag,
                linestyle=next(linestyles),
                alpha=0.7
            )

    # Add legend
    plt.legend()

    # Show plot
    plt.show()

def compare_weather_daily(
    df: pd.DataFrame,
    column: Union[int, str] = 0,
    show: str = 'Leq', 
    include_wind_flag: bool = True,
    include_rain_flag: bool = True,
    include_temp_flag: bool = False,
    include_rel_hum_flag: bool = False,
    include_snow_flag: bool = False,
    win: int = 3600,
    step: int = 1200,
    coverage_check: bool = False,
    coverage_threshold: float = 0.5,
    title: str = "Daily Leq Profiles for Different Weather Conditions",
    figsize: tuple = (12, 8)
):
    """
    Compare daily level profiles with or without flags.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the data.
    column: Union[int, str], default 0
        The column name or index for sound levels. If None, the first column
        will be used.
    show: str, default 'Leq'
        Column names to be plotted.
    include_wind_flag: bool, default True
        Whether to include the Wind Speed Flag.
    include_rain_flag: bool, default True
        Whether to include the Rain Flag.
    include_temp_flag: bool, default False
        Whether to include the Temperature Flag.
    include_rel_hum_flag: bool, default False
        Whether to include the Relative Humidity Flag.
    include_snow_flag: bool, default False
        Whether to include the Snow Flag.
    win: int, default 3600
        The window size for rolling average.
    step: int, default 1200
        The step size for rolling average.
    coverage_check: bool, default False
        Whether to check data coverage when computing levels.
    coverage_threshold: float, default 0.5
        Minimum coverage ratio required (0.0-1.0).
    title: str, default "Daily Leq Profiles for Different Conditions"
        The title of the plot.
    figsize: tuple, default (12, 8)
        The size of the plot.

    Returns
    ----------
    None
    """
    column = core._column_to_index(df, column)

    flags = {
        'Wind_Spd_Flag': include_wind_flag,
        'Rain_Flag_Roll': include_rain_flag,
        'Temp_Flag': include_temp_flag,
        'Rel_Hum_Flag': include_rel_hum_flag,
        'Snow_Flag_Roll': include_snow_flag
    }

    _active_flags = {
        flag: include for flag, include in flags.items()
        if include
    }

    subsets = {
        'All Data': df
    }

    for flag in _active_flags:
        subsets[f'No {flag}'] = df[~df[flag]]
        subsets[flag] = df[df[flag]]

    subsets['All Flags'] = df[df[list(_active_flags.keys())].any(axis=1)]
    subsets['Neither Flags'] = df[~df[list(_active_flags.keys())].any(axis=1)]

    weekly_levels_dict = {}
    for key, subset_df in subsets.items():
        weekly_levels_dict[key] = periodic(
            subset_df,
            1,
            23,
            column=column,
            win=win,
            step=step,
            coverage_check=coverage_check,
            coverage_threshold=coverage_threshold
        )

    # Plot daily Leq profiles using nm.plot_compare
    compare(
        list(weekly_levels_dict.values()),
        list(weekly_levels_dict.keys()),
        show,
        title=title,
        figsize=figsize
    )

def _convert_datetime_index(df: pd.DataFrame) -> np.ndarray:
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

def _get_datetime_index(df: pd.DataFrame) -> np.ndarray:
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
        return _convert_datetime_index(df)
    elif not isinstance(df.index[0], datetime):
        raise TypeError(f'DataFrame index must be of type datetime, \
                        time or pd.Timestamp not {type(df.index[0])}')
    
def _format_time_axis(ax, x, df) -> None:
    """
    Format the time axis for a plot.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        The axes object to format.
    x: array-like
        The datetime or time index for the x-axis.
    df: DataFrame
        The DataFrame containing the data.

    Returns
    ----------
    None
    """
    if any(isinstance(df.index[0], t) for t in [pd.Timestamp, datetime]):
        ax.figure.autofmt_xdate()
        if (x[1] - x[0]).days < 30:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%B %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.set_xlabel('Date')
    elif isinstance(df.index[0], time):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel('Time')