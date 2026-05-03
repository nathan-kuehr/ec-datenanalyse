import numpy as np
import pandas as pd
import seaborn as sns

from copy import deepcopy
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from pandas.core.groupby.generic import DataFrameGroupBy

from ..settings import Settings
from .plotresult import PlotResult
from ..palette import NEIColorPalette
from ..config import (
    FIGURE_SETTINGS,
    DEFAULT_LINEPLOT_SETTINGS,
    DEFAULT_JOINT_DISTRIBUTION_PLOT_SETTINGS,
    DataSeriesInfo,
)


def _active_groupby_cols(kwargs: dict, additional_groups: list[str] = []) -> list[str]:
    active = [kwargs.get(arg) for arg in {"hue", "style", "size"} if arg in kwargs]
    return active + additional_groups


def _clean_args(kwargs: dict, to_remove: list[str]) -> dict:
    return {k: v for k, v in kwargs.items() if k not in to_remove}


def _clean_plot_args(kwargs: dict) -> dict:
    return _clean_args(kwargs, ["data", "x", "y", "title", "series_info"])


def _merge_kwargs(default: dict, to_merge: dict) -> dict:
    config = deepcopy(default)

    for key, value in to_merge.items():
        if isinstance(value, dict) and isinstance(default.get(key, None), dict):
            config[key] |= value
        else:
            config[key] = value

    return config


def _iterate_legend(ax: Axes, dummy: bool):
    if not (legend := ax.get_legend()):
        return

    def is_dummy(artist: Artist) -> bool:
        if hasattr(artist, "get_markersize"):
            return artist.get_markersize() == 0.0
        else:
            return False

    handles = ax.get_legend_handles_labels()[0]
    texts = legend.get_texts()

    for handle, text in zip(handles, texts):
        if dummy == is_dummy(handle):
            yield handle, text


def _make_axes_label(name: str, info: DataSeriesInfo) -> str:
    unit = info.unit or "$-$"

    if "$" in unit:
        unit = unit.strip("$")
    else:
        unit = "\\mathrm{" + unit.replace(" ", r"\ ").replace("%", r"\%") + "}"

    return f"{name} {info.symbol} $\\left[{unit}\\right]$"


def _prepare_groupby(
    data: pd.DataFrame,
    kwargs,
    additional_groups: list[str] = [],
) -> DataFrameGroupBy:
    grouping = _active_groupby_cols(kwargs, additional_groups)

    if not grouping:  # Check if empty -> return full groupby object
        return data.groupby(np.ones(len(data)))
    else:
        return data.groupby(grouping, sort=False)


def _prepare_palette(data: pd.DataFrame, kwargs) -> dict[str, list[NEIColorPalette]]:
    palette: list[NEIColorPalette] = list()

    for pal, group in data.groupby("Palette", sort=False):
        n = _prepare_groupby(
            group, {"hue": kwargs["hue"]} if "hue" in kwargs else {}
        ).ngroups
        palette += pal.shade(n)  # pyright: ignore

    if "hue" in kwargs:
        return {"palette": palette}
    else:
        return {"color": palette[0]}


def _set_axes_from_series_info(
    ax: Axes, x: str | None, y: str | None, series_info: dict[str, DataSeriesInfo] = {}
) -> None:
    config = {}

    if x is not None:
        config |= {
            "xlabel": _make_axes_label(x, series_info[x]),
            "xscale": series_info[x].scale,
        }
    if y is not None:
        config |= {
            "ylabel": _make_axes_label(y, series_info[y]),
            "yscale": series_info[y].scale,
        }

    ax.set(**config)


def lineplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str | None = None,
    series_info: dict[str, DataSeriesInfo] = {},
    **kwargs,
) -> PlotResult:
    config = {"data": data, "x": x, "y": y}

    # Assemble correct palette
    palette = _prepare_palette(data, kwargs)

    # Add default arguments
    sns_args = DEFAULT_LINEPLOT_SETTINGS | kwargs | config | palette
    sns_args = Settings.Clean_Kwargs(sns_args)
    sns_args = PlotResult.Clean_Kwargs(sns_args)

    # Adapt title size
    title_config = {}
    if "ax" in kwargs and len(kwargs["ax"].figure.axes) > 1:
        title_config["axes.title_size"] = FIGURE_SETTINGS["legend.title_fontsize"]

    with plt.rc_context(FIGURE_SETTINGS | title_config):
        # Create new figure if necessary
        ax: Axes = sns_args.pop("ax", None) or plt.figure().gca()
        assert isinstance(fig := ax.figure, Figure)

        # Plot
        sns.lineplot(**sns_args, ax=ax)

        # Set axes correctly
        _set_axes_from_series_info(ax, x, y, series_info)

        # If multiple active groups, titles are not well handled by sns
        for _, text in _iterate_legend(ax, dummy=True):
            text.set_fontsize(plt.rcParams["legend.title_fontsize"])
            text.set_ha("center")

        # Set title or super title
        if title is not None:
            ax.set_title(title)

        return PlotResult(title, fig, **kwargs)  # pyright: ignore


def joint_distribution_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str | None = None,
    series_info: dict[str, DataSeriesInfo] = {},
    **kwargs,
) -> PlotResult:
    config = {
        "x": x,
        "y": y,
        "joint_kws": {  # Need to add here cuz the joint plot swallows the data
            "data": data
        },
        "marginal_kws": {  # Need to add here cuz the joint plot swallows the data
            "data": data
        },
    }
    # Assemble correct palette
    palette = _prepare_palette(data, kwargs)

    # Add default arguments
    sns_args = _merge_kwargs(
        DEFAULT_JOINT_DISTRIBUTION_PLOT_SETTINGS,
        _merge_kwargs(kwargs, config | palette),
    )

    sns_args = Settings.Clean_Kwargs(sns_args)
    sns_args = PlotResult.Clean_Kwargs(sns_args)

    if "ax" in sns_args:
        raise ValueError("'ax' parameter not supported for joint distribution plots!")

    with plt.rc_context(FIGURE_SETTINGS | {"figure.constrained_layout.use": False}):
        joint = sns.jointplot(data, **sns_args)

        # Get handles
        fig = joint.figure
        joint_ax = fig.axes[0]
        marginal_axes = fig.axes[1:]

        # Set axes correctly
        _set_axes_from_series_info(joint_ax, x, y, series_info)

        for _, text in _iterate_legend(joint_ax, dummy=True):
            text.set_fontsize(plt.rcParams["legend.title_fontsize"])
            text.set_ha("center")

        for m_ax in marginal_axes:
            m_ax.grid(False)

        # Add title if wanted
        if title is not None:
            fig.suptitle(title)

    return PlotResult(title, fig, **kwargs).add_meta({"jointplot": joint})
