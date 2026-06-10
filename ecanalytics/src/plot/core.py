import numpy as np
import pandas as pd
import seaborn as sns

from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D


class _EngScalarFormatter(ScalarFormatter):
    def __init__(self, **kwargs):
        kwargs.setdefault("useMathText", True)
        kwargs.setdefault("useOffset", False)
        super().__init__(**kwargs)
        self.set_scientific(True)
        self.set_powerlimits((-3, 3))

    def _set_order_of_magnitude(self):
        super()._set_order_of_magnitude()
        self.orderOfMagnitude = 3 * (self.orderOfMagnitude // 3)

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
    DEFAULT_LINEPLOT_GRID_SETTINGS,
)

_GREEK_LETTERS = {"tau", "varphi"}
_MAX_GRID_COL_WRAP = 3
_EXPERIMENT_SEPARATOR_LINE_KWARGS = {
    "color": "gray",
    "linestyle": "-.",
    "alpha": 0.5,
    "zorder": 0,
    "linewidth": 0.5,
}

# ===================== GROUPING OF DATA =====================


def _active_groupby_cols(
    data: pd.DataFrame, kwargs: dict, additional_groups: set[str] = set()
) -> list[str]:
    active = {
        kwargs[arg] for arg in {"hue", "style", "size", "tile"} if kwargs.get(arg) is not None
    } | additional_groups
    return [col for col in data.columns if col in active]


def _prepare_groupby(
    data: pd.DataFrame,
    kwargs: dict,
    additional_groups: set[str] = set(),
) -> DataFrameGroupBy:
    grouping = _active_groupby_cols(data, kwargs, additional_groups)

    if not grouping:
        return data.groupby(np.ones(len(data)))
    return data.groupby(grouping, sort=False)


def _prepare_palette(data: pd.DataFrame, kwargs: dict) -> dict:
    mapping = dict()
    hue_col = kwargs.get("hue")

    for pal, group in data.groupby("Palette", sort=False):
        grouped = _prepare_groupby(group, {"hue": hue_col})

        keys = grouped.groups.keys()
        shades = pal.shade(len(keys))  # pyright: ignore

        mapping |= dict(zip(keys, shades))

    if hue_col is None or len(mapping) <= 1:
        kwargs.pop("hue", None)

        val, *_ = mapping.values()
        return {"color": val}
    else:
        return {"palette": mapping}


# ===================== ARGS =====================


def _clean_args(kwargs: dict, to_remove: list[str]) -> dict:
    return {k: v for k, v in kwargs.items() if k not in to_remove}


def _clean_plot_args(kwargs: dict) -> dict:
    return _clean_args(
        kwargs,
        ["data", "x", "y", "title", "series_info", "catplot_kws", "stripplot_kws"],
    )


def _merge_kwargs(default: dict, to_merge: dict) -> dict:
    config = deepcopy(default)

    for key, value in to_merge.items():
        if isinstance(value, dict) and isinstance(default.get(key, None), dict):
            config[key] = _merge_kwargs(config[key], value)
        else:
            config[key] = value

    return config


# ===================== AXIS & LEGEND =====================


def _make_axes_label(
    name: str, info: DataSeriesInfo, separated: bool = False
) -> str | tuple[str, str, str]:
    label, unit, _ = info

    if "$" not in label:
        if "_" in label:
            symbol, index = label.split("_", 1)

            if symbol in _GREEK_LETTERS:
                symbol = rf"\{symbol}"

            label = rf"${symbol}_\mathrm{{{index}}}$"
        else:
            if label in _GREEK_LETTERS:
                label = rf"\{label}"
            label = f"${label}$"

    unit = unit or "$-$"
    if "$" in unit:
        # Already latex -> remove the inline math indicators
        unit = unit.strip("$")
    else:
        unit = unit.replace(" ", r"\ ").replace("%", r"\%")

    unit = rf"$\left[\mathrm{{{unit}}}\right]$"

    return (name, label, unit) if separated else f"{name} {label} {unit}"


def _set_axes_from_series_info(
    ax: Axes | FacetGrid,
    x: str | None,
    y: str | None,
    series_info: dict[str, DataSeriesInfo] = {},
) -> None:
    config = {}

    if x is not None and x != "Value":
        config |= {
            "xlabel": _make_axes_label(x, series_info[x]),
            "xscale": series_info[x].scale,
        }
    if y is not None and y != "Value":
        config |= {
            "ylabel": _make_axes_label(y, series_info[y]),
            "yscale": series_info[y].scale,
        }
        
    ax.set(**config)


def _iterate_legend(target: Axes | FacetGrid, dummy: bool):
    if isinstance(target, Axes):
        legend = target.get_legend()
    elif isinstance(target, FacetGrid):
        legend = target.legend
    else:
        return

    if not legend:
        return

    def is_dummy(artist: Artist | None) -> bool:
        if isinstance(artist, Line2D):
            return artist.get_linewidth() == 0.0
        return False

    handles = legend.legend_handles
    texts = legend.get_texts()

    for handle, text in zip(handles, texts):
        if dummy == is_dummy(handle):
            yield handle, text


def _improve_legend(host: Axes | FacetGrid) -> None:
    if isinstance(host, Axes):
        legend = host.get_legend()
        loc = "best"
    elif isinstance(host, FacetGrid):
        legend = host.legend
        loc = "outside lower center"
    else:
        raise ValueError("Host must be an Axes or FacetGrid!")
    
    if legend is None:
        return

    ncols = 0
    for _, text in _iterate_legend(host, dummy=True):
        text.set_fontsize(plt.rcParams["legend.title_fontsize"])
        text.set_ha("center")
        ncols += 1
    
    ncols = max(ncols, len(legend.legend_handles) // 5, 1)
    ncols = min(ncols, 4)

    sns.move_legend(host, loc, ncol=ncols)


def lineplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str | None = None,
    series_info: dict[str, DataSeriesInfo] = {},
    **kwargs,
) -> PlotResult:
    config = {"data": data, "x": x, "y": y}

    # Prepare args
    sns_args = _merge_kwargs(DEFAULT_LINEPLOT_SETTINGS, kwargs) | config
    sns_args = Settings.clean_kwargs(sns_args)
    sns_args = PlotResult.clean_kwargs(sns_args)

    figure_settings = FIGURE_SETTINGS.copy()

    _FACET_KEYS = ["row", "col"]

    # =============== TILE ===============
    # Shortcut for col / col_wrap
    tile_col: str | None = sns_args.pop("tile", None)
    if tile_col is not None:
        if not sns_args.keys().isdisjoint(_FACET_KEYS):
            raise ValueError("Cannot pass 'tile' together with 'row'/'col'!")

        
        ncols = sns_args.pop("col_wrap", None) or min(_MAX_GRID_COL_WRAP, data[tile_col].nunique())
        sns_args |= { "col": tile_col, "col_wrap": ncols }
    
    # Check if any other grouping args passed
    sns_args.setdefault("legend", len(_active_groupby_cols(data, sns_args)) > 0)
        
    # =============== GRID ===============
    make_grid = not sns_args.keys().isdisjoint(_FACET_KEYS)
    if make_grid:
        if sns_args.get("ax") is not None:
            raise ValueError("Cannot pass 'ax' parameter along with facet args!")
        
        sns_args = _merge_kwargs(DEFAULT_LINEPLOT_GRID_SETTINGS, sns_args)

        # By default, set a color if columns are specified
        sns_args["hue"] = sns_args.get("hue") or sns_args.get("col")
        
        # Turn it of for the grid
        figure_settings["figure.constrained_layout.use"] = False
        
    # Set palette (overridable)
    sns_args = _prepare_palette(data, sns_args) | sns_args

    nexps = data["Experiment Name"].nunique()

    # Adapt title size if multiple axes passed
    if ("ax" in kwargs and len(kwargs["ax"].figure.axes) > 1) or make_grid:
        figure_settings["axes.titlesize"] = FIGURE_SETTINGS["legend.title_fontsize"]

    with plt.rc_context(figure_settings):
        if make_grid:
            grid = sns.relplot(**sns_args)
            fig = grid.figure

            grid.set_titles(col_template="{col_name}")
            _set_axes_from_series_info(grid, x, y, series_info)

            _improve_legend(grid)

            if title is not None:
                fig.suptitle(title)

            fig.set_layout_engine("constrained")

            return PlotResult(title, fig, **_clean_plot_args(kwargs)).add_meta({"grid": grid})  # pyright: ignore
        else:
            ax: Axes = sns_args.pop("ax", None) or plt.figure().gca()
            assert isinstance(fig := ax.figure, Figure)

            sns.lineplot(**sns_args, ax=ax)

            _set_axes_from_series_info(ax, x, y, series_info)

            _improve_legend(ax)

            if title is not None:
                ax.set_title(title)

            return PlotResult(title, fig, **_clean_plot_args(kwargs))  # pyright: ignore



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
        # Need to add here because the joint plot swallows the data
        "joint_kws": {"data": data},
        "marginal_kws": {"data": data},
    }
    palette = _prepare_palette(data, kwargs)

    sns_args = _merge_kwargs(
        DEFAULT_JOINT_DISTRIBUTION_PLOT_SETTINGS,
        _merge_kwargs(kwargs, config | palette),
    )

    sns_args = Settings.clean_kwargs(sns_args)
    sns_args = PlotResult.clean_kwargs(sns_args)

    if "ax" in sns_args:
        raise ValueError("'ax' parameter not supported for joint distribution plots!")

    with plt.rc_context(FIGURE_SETTINGS | {"figure.constrained_layout.use": False}):
        joint = sns.jointplot(data, **sns_args)

        fig = joint.figure
        joint_ax = fig.axes[0]
        marginal_axes = fig.axes[1:]

        _set_axes_from_series_info(joint_ax, x, y, series_info)

        for _, text in _iterate_legend(joint_ax, dummy=True):
            text.set_fontsize(plt.rcParams["legend.title_fontsize"])
            text.set_ha("center")

        for m_ax in marginal_axes:
            m_ax.grid(False)

        if title is not None:
            fig.suptitle(title)

        fig.set_layout_engine("constrained")

    return PlotResult(title, fig, **kwargs).add_meta({"jointplot": joint})


def parameter_plot(
    data: pd.DataFrame,
    x: str,
    parameters: Iterable[str],
    title: str | None = None,
    series_info: dict[str, DataSeriesInfo] = {},
    **kwargs,
) -> PlotResult:
    if "hue" in kwargs:
        raise ValueError("'hue' parameter not allowed for parameter plots! Put it in the x argument.")
    if kwargs.get("tile") is not None:
        raise ValueError("'tile' parameter not supported for parameter plots - the grid is already used for the parameters (col='Parameter').")
    kwargs = _merge_kwargs(DEFAULT_PARAMETER_PLOT_SETTINGS, kwargs)

    palette = _prepare_palette(data, kwargs | {"hue": x})
    config = {
        "x": x,
        "hue": x,
        "y": "Value",
    } | palette

    catplot_config = kwargs["catplot_kws"] | config | {
        "data": data[data["Parameter"].isin(parameters)],
        "col": "Parameter",
        "kind": "box",
    }
    catplot_config = Settings.clean_kwargs(catplot_config)
    catplot_config = PlotResult.clean_kwargs(catplot_config)

    stripplot_config = kwargs["stripplot_kws"] | config | {"func": sns.stripplot}
    stripplot_config = Settings.clean_kwargs(stripplot_config)
    stripplot_config = PlotResult.clean_kwargs(stripplot_config)

    # Map from symbol to (name, info)
    remapped_series_info = {info.symbol: (name, info) for name, info in series_info.items()}

    nexps = data["Experiment Name"].nunique()
    nhue = data[x].nunique() // nexps

    with plt.rc_context(FIGURE_SETTINGS | PARAMETER_PLOT_FIGURE_SETTINGS):
        grid = sns.catplot(**catplot_config)
        grid.map_dataframe(**stripplot_config)

        grid.set_titles(col_template="{col_name}")

        ax: Axes
        for ax in grid.axes.flat:
            name, info = remapped_series_info[ax.get_title()]
            _, label, unit = _make_axes_label(name, info, separated=True)

            ax.set(title=f"{name} {label}", ylabel=unit, xlabel="", xticks=[], yscale=info.scale)
            ax.yaxis.set_major_formatter(_EngScalarFormatter())

            # Add lines separating experiments
            for i in range(1, nexps):
                ax.axvline(nhue * i - 0.5, **_EXPERIMENT_SEPARATOR_LINE_KWARGS)

        sns.move_legend(grid, "outside lower center", ncol=nexps)

        fig = grid.figure

        if title is not None:
            fig.suptitle(title)

        fig.set_layout_engine("constrained")

        return PlotResult(title, fig, **_clean_plot_args(kwargs)).add_meta({"grid": grid})
