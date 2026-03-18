import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.core.groupby.generic import DataFrameGroupBy
from functools import singledispatch
from matplotlib.axes import Axes

from ..data.experiment import Experiment
from ..analysis.data_quality import DataQuality
from ..palette import NEIColorPalette
from .plotresult import PlotResult
from ..settings import Settings
from ..config import FIGURE_SETTINGS, SNS_LINEPLOT_DEFAULT_SETTINGS


def __prepare_groupby(
    data: pd.DataFrame, kwargs, include_frequency: bool = False
) -> DataFrameGroupBy:
    SNS_Grouping_Args = ["hue", "style", "size"]
    grouping = [kwargs.get(arg) for arg in SNS_Grouping_Args if arg in kwargs]

    grouping += ["Frequency"] if include_frequency else []

    if not grouping:  # Check if empty -> return full groupby object
        return data.groupby(np.ones(len(data)))
    else:
        return data.groupby(grouping, sort=False)


def __prepare_palette(data: pd.DataFrame, kwargs) -> list[NEIColorPalette]:
    palette: list[NEIColorPalette] = list()

    for pal, group in data.groupby("Palette", sort=False):
        n = __prepare_groupby(
            group, {"hue": kwargs["hue"]} if "hue" in kwargs else {}
        ).ngroups
        palette += pal.shade(n)  # pyright: ignore

    return palette


def __combine_data_frames(
    data: list[Experiment | pd.DataFrame], kwargs: dict
) -> pd.DataFrame:
    if len(data) == 0:
        raise ValueError("Data list is empty.")

    data_frames = [d.data if isinstance(d, Experiment) else d for d in data]
    combined = pd.concat(data_frames, ignore_index=True)

    if (hue_group := kwargs.get("hue", None)) is not None:
        # Multiple data sets and hue differentiation
        combined[hue_group] = combined["Experiment Name"] + " - " + combined[hue_group]

    return combined


def __plot_clean_kwargs(kwargs: dict) -> dict:
    """Cleans the kwargs dictionary by removing plot related keys.

    Args:
        kwargs: Original kwargs dictionary

    Returns:
        Cleaned kwargs dictionary
    """
    keys_to_remove = {"data", "title", "Rmin", "Rspan", "offset_correct"}
    return {k: v for k, v in kwargs.items() if k not in keys_to_remove}


@singledispatch
def plot(data, x: str, y: str, title: str | None = None, **kwargs) -> PlotResult:
    raise TypeError(
        f"Unsupported data type: {type(data).__name__}. Expected ImpedanceSpectrumExperiment."
    )


@plot.register(Experiment)
def __plot_single_eis(
    data: Experiment, x: str, y: str, title: str | None = None, **kwargs
) -> PlotResult:
    return plot(data.data, x, y, title, **kwargs)


@plot.register(pd.DataFrame)
def __plot_single_eis_data(
    data: pd.DataFrame, x: str, y: str, title: str | None = None, **kwargs
) -> PlotResult:
    config = {
        "data": data,
        "x": x,
        "y": y,
    }

    # Assemble correct palette
    palette = __prepare_palette(data, kwargs)

    # Add default arguments
    kwargs = SNS_LINEPLOT_DEFAULT_SETTINGS | kwargs
    sns_args = Settings.Clean_Kwargs(kwargs)
    sns_args = PlotResult.Clean_Kwargs(sns_args)

    sns_args |= config | {"palette": palette}

    with plt.rc_context(FIGURE_SETTINGS):
        # Create new figure if necessary
        ax: Axes = kwargs.get("ax") or plt.figure().gca()
        fig = ax.figure

        # Plot
        sns.lineplot(**sns_args, sort=False)

        # Temporary Solution
        Series_Info = Experiment.Series_Info | DataQuality.Series_Info

        ax.set_xlabel(f"{x} {Series_Info[x].symbol} [{Series_Info[x].unit}]")
        ax.set_ylabel(f"{y} {Series_Info[y].symbol} [{Series_Info[y].unit}]")

        ax.set_xscale(Series_Info[x].scale)
        ax.set_yscale(Series_Info[y].scale)

        # Set title or super title
        if title is not None:
            ax.set_title(
                title,
                fontsize=FIGURE_SETTINGS[
                    "axes.titlesize" if len(fig.axes) == 1 else "legend.title_fontsize"
                ],
            )

        # Turn on grid
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

        return PlotResult(title, fig, **__plot_clean_kwargs(kwargs))  # pyright: ignore


@plot.register(list)
def __plot_multiple_eis(
    data: list[Experiment | pd.DataFrame],
    x: str,
    y: str,
    title: str | None = None,
    **kwargs,
) -> PlotResult:
    return plot(__combine_data_frames(data, kwargs), x, y, title, **kwargs)


@singledispatch
def bode(data: Experiment, title: str | None = None, **kwargs) -> PlotResult:
    return bode(data.data, title, **kwargs)


@bode.register(pd.DataFrame)
def __bode_data(data: pd.DataFrame, title: str | None = None, **kwargs) -> PlotResult:
    if (ax := kwargs.pop("ax", None)) is None:
        with plt.rc_context(SNS_LINEPLOT_DEFAULT_SETTINGS):
            ax = plt.subplots(2, 1, sharex=True)[1]

    if len(ax) != 2:
        raise ValueError("axs must be a list of two Axes for Bode plot.")

    fig = ax[0].figure

    kwargs_intermed = kwargs.copy()
    kwargs_intermed["noSave"] = True

    plot_args = kwargs | {"data": data, "x": "Frequency"}

    with plt.rc_context(FIGURE_SETTINGS):
        # Get two axes
        if (ax := kwargs.pop("ax", None)) is None:
            ax = plt.subplots(2, 1, sharex=True)[1]
        elif len(ax) != 2:
            raise ValueError("Two axes must be defined for Bode plot.")

        fig = ax[0].figure

        # Set title beforhand because buggy otherwise
        if title is not None:
            fig.suptitle(
                title,
                fontsize=FIGURE_SETTINGS["axes.titlesize"],
                fontweight=FIGURE_SETTINGS["axes.titleweight"],
                y=0.98,
            )

        plot(**(plot_args | {"ax": ax[0], "y": "Impedance", "title": "Magnitude"}))
        res = plot(data, x="Frequency", ax=ax[1], y="Phase", title="Phase", **kwargs)
        res.title = title

    return res


@bode.register(list)
def __bode_multiple(
    data: list[Experiment | pd.DataFrame], title: str | None = None, **kwargs
) -> PlotResult:
    return bode(__combine_data_frames(data, kwargs), title, **kwargs)


@singledispatch
def fresponse(
    data: Experiment, y: str, title: str | None = None, **kwargs
) -> PlotResult:
    return plot(data.data, x="Frequency", y=y, title=title, **kwargs)


@fresponse.register(pd.DataFrame)
def __fresponse_data(
    data: pd.DataFrame, y: str, title: str | None = None, **kwargs
) -> PlotResult:
    return plot(data, x="Frequency", y=y, title=title, **kwargs)


@fresponse.register(list)
def __fresponse_multiple(
    data: list[Experiment | pd.DataFrame], y: str, title: str | None = None, **kwargs
) -> PlotResult:
    return fresponse(__combine_data_frames(data, kwargs), y, title, **kwargs)
