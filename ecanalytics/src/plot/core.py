import numpy as np
import pandas as pd
import seaborn as sns

from copy import deepcopy
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from pandas.core.groupby.generic import DataFrameGroupBy
from seaborn import FacetGrid
from typing import Iterable

from ..settings import Settings
from .plotresult import PlotResult
from ..palette import NEIColorPalette
from ..config import (
    FIGURE_SETTINGS,
    DEFAULT_LINEPLOT_SETTINGS,
    DEFAULT_JOINT_DISTRIBUTION_PLOT_SETTINGS,
    DataSeriesInfo,
    PARAMETER_PLOT_FIGURE_SETTINGS,
    DEFAULT_PARAMETER_PLOT_SETTINGS,
    DEFAULT_LINEPLOT_GRID_SETTINGS
)

## ===================== GROUPING OF DATA ===================== 

def _active_groupby_cols(data: pd.DataFrame, kwargs: dict, additional_groups: set[str] = set()) -> list[str]:
    active = {kwargs[arg] for arg in {"hue", "style", "size", "tile"} if arg in kwargs} | additional_groups
    return [col for col in data.columns if col in active]

def _prepare_groupby(
    data: pd.DataFrame,
    kwargs,
    additional_groups: set[str] = set(),
) -> DataFrameGroupBy:
    grouping = _active_groupby_cols(data, kwargs, additional_groups)

    if not grouping:  # Check if empty -> return full groupby object
        return data.groupby(np.ones(len(data)))
    else:
        return data.groupby(grouping, sort=False)
    
def _prepare_palette(data: pd.DataFrame, kwargs) -> dict:
    palette: list[NEIColorPalette] = list()

    for pal, group in data.groupby("Palette", sort=False):
        n = _prepare_groupby(
            group, {"hue": kwargs["hue"]} if "hue" in kwargs else {}
        ).ngroups
        palette += pal.shade(n)  # pyright: ignore

    if len(palette) > 1 or kwargs.get("hue") is not None:
        return {"palette": palette}
    else:
        return {"color": palette[0]}


## ===================== ARGS ===================== 

def _clean_args(kwargs: dict, to_remove: list[str]) -> dict:
    return {k: v for k, v in kwargs.items() if k not in to_remove}

def _clean_plot_args(kwargs: dict) -> dict:
    return _clean_args(kwargs, ["data", "x", "y", "title", "series_info", "catplot_kws", "stripplot_kws"])

def _merge_kwargs(default: dict, to_merge: dict) -> dict:
    config = deepcopy(default)

    for key, value in to_merge.items():
        if isinstance(value, dict) and isinstance(default.get(key, None), dict):
            config[key] = _merge_kwargs(config[key], value)
        else:
            config[key] = value

    return config

## ===================== AXIS & LEGEND ===================== 

def _make_axes_label(name: str, info: DataSeriesInfo, separated: bool = False) -> str | tuple[str, str, str]:
    label, unit, _ = info

    # Label
    GREEK_LETTERS = {"tau", "varphi"}
    if not "$" in label:
        if "_" in label:
            symbol, index = label.split("_", 1)

            if symbol in GREEK_LETTERS:
                symbol = rf"\{symbol}"

            label = rf"${symbol}_\mathrm{{{index}}}$"
        else:
            label = f"${label}$"

    # Unit
    unit = unit or "$-$"
    if "$" in unit:
        # Already latex -> remove the inline math indicators
        unit = unit.strip("$") 
    else:
        unit = unit.replace(" ", r"\ ") \
                   .replace("%", r"\%")
        
    unit = rf"$\left[\mathrm{{{unit}}}\right]$"

    return (name, label, unit) if separated else f"{name} {label} {unit}"

def _set_axes_from_series_info(
    ax: Axes | FacetGrid, x: str | None, y: str | None, series_info: dict[str, DataSeriesInfo] = {}
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

def _iterate_legend(target: Axes | Figure, dummy: bool):
    if isinstance(target, Axes):
        legend = target.get_legend()
    elif isinstance(target, Figure):
        legend = target.legends[0] if target.legends else None
    else:
        return
    
    if not legend:
        return

    def is_dummy(artist: Artist | None) -> bool:
        if hasattr(artist, "get_markersize"):
            return artist.get_markersize() == 0.0
        else:
            return False

    handles = legend.legend_handles
    texts = legend.get_texts()

    for handle, text in zip(handles, texts):
        if dummy == is_dummy(handle):
            yield handle, text



def lineplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str | None = None,
    series_info: dict[str, DataSeriesInfo] = {},
    **kwargs,
) -> PlotResult:
    config = {"data": data, "x": x, "y": y}

    # Add default arguments
    sns_args = _merge_kwargs(DEFAULT_LINEPLOT_SETTINGS, kwargs) | config
    sns_args = Settings.Clean_Kwargs(sns_args)
    sns_args = PlotResult.Clean_Kwargs(sns_args)

    # Figure setting adaptations
    figure_settings = FIGURE_SETTINGS.copy()

    # Check if we need to make a grid
    tile_col: str = sns_args.pop("tile", None)
    if (make_grid := tile_col is not None):
        if "ax" in kwargs:
            raise ValueError("Cannot pass 'ax' parameter along with 'tile'!")
        
        if not (make_legend := "hue" in sns_args):
            sns_args["hue"] = tile_col
        
        # Adapt sns args for grid
        sns_args = _merge_kwargs(DEFAULT_LINEPLOT_GRID_SETTINGS | {
            "col_wrap": min(3, data[tile_col].nunique())
        }, sns_args) | {"col": tile_col, "legend": make_legend}

        # Adapt figure settings
        figure_settings["figure.constrained_layout.use"] = False
        
    
    # Assemble correct palette
    sns_args |= _prepare_palette(data, sns_args)

    # Number of experiments
    nexp = data["Experiment Name"].nunique()

    # Adapt title size if multiple axes passed
    if ("ax" in kwargs and len(kwargs["ax"].figure.axes)) or make_grid:
        figure_settings["axes.titlesize"] = FIGURE_SETTINGS["legend.title_fontsize"]

    with plt.rc_context(figure_settings):
        if make_grid:
            grid = sns.relplot(**sns_args)
            fig = grid.figure
            
            # Assign the axis config via the grid handle
            grid.set_titles(col_template="{col_name}")
            _set_axes_from_series_info(grid, x, y, series_info)

            if make_legend:
                sns.move_legend(grid, "outside lower center", ncol=nexp)

                for _, text in _iterate_legend(fig, dummy=True):
                    text.set_fontsize(plt.rcParams["legend.title_fontsize"])
                    text.set_ha("center")

            # Set title or super title
            if title is not None:
                fig.suptitle(title)

            # Turn back on constrained
            fig.set_layout_engine("constrained")

            return PlotResult(title, fig, **kwargs).add_meta({"grid": grid})  # pyright: ignore

        else:
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

        fig.set_layout_engine("constrained")

    return PlotResult(title, fig, **kwargs).add_meta({"jointplot": joint})

def parameter_plot(data: pd.DataFrame, x: str, parameters: Iterable[str], title: str | None = None, series_info: dict[str, DataSeriesInfo] = {}, **kwargs) -> PlotResult:
    # Argument work, add default args
    if "hue" in kwargs:
        raise ValueError("'hue' parameter not allowed for parameter plots! Put it in the x argument.")
    kwargs = _merge_kwargs(DEFAULT_PARAMETER_PLOT_SETTINGS, kwargs)

    # Assemble correct palette
    palette = _prepare_palette(data, kwargs | {"hue": x})
    config = {
        "x": x,
        "hue": x,
        "y": "Value"
    } | palette

    # Prepare the args for the two functions
    catplot_config = kwargs["catplot_kws"] | config | {
        "data": data[data["Parameter"].isin(parameters)],
        "col": "Parameter",
        "kind": "box",
    }
    catplot_config = Settings.Clean_Kwargs(catplot_config)
    catplot_config = PlotResult.Clean_Kwargs(catplot_config)

    stripplot_config = kwargs["stripplot_kws"] | config | {"func": sns.stripplot}
    stripplot_config = Settings.Clean_Kwargs(stripplot_config)
    stripplot_config = PlotResult.Clean_Kwargs(stripplot_config)

    # We need the map from the symbol to the 
    remapped_series_info = {info.symbol: (name, info) for name, info in series_info.items()}

    # Get lengths
    nexp = data["Experiment Name"].nunique()
    nhue = data[x].nunique() // nexp

    with plt.rc_context(FIGURE_SETTINGS | PARAMETER_PLOT_FIGURE_SETTINGS):
        # Main plots
        grid = sns.catplot(**catplot_config)
        grid.map_dataframe(**stripplot_config)

        # Set titles to raw name first
        grid.set_titles(col_template="{col_name}")

        ax: Axes
        for ax in grid.axes.flat:
            name, info = remapped_series_info[ax.get_title()]
            _, label, unit = _make_axes_label(name, info, separated=True)

            # Adjust title & y label
            ax.set(title=f"{name} {label}", ylabel=unit, xlabel="", xticks=[])

            # Add lines separating experiments
            for i in range(1, nexp):
                ax.axvline(nhue * i - 0.5, color="gray", linestyle="-.", alpha=0.5, zorder=0, linewidth=0.5)

        sns.move_legend(grid, "outside lower center", ncol=nexp)

        fig = grid.figure

        # Add title if wanted
        if title is not None:
            fig.suptitle(title)


        # Turn back on constrained
        fig.set_layout_engine("constrained")

        PlotResult(title, fig, **_clean_plot_args(kwargs)).add_meta({"grid": grid})