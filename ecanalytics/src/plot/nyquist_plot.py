import pandas as pd
import seaborn as sns
import numpy as np

from matplotlib import colors, pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
from pandas.core.groupby.generic import DataFrameGroupBy
from seaborn import FacetGrid
from typing import Iterable

import matplotlib.ticker as ticker

from . import core
from .covariance import CovarianceVisualization
from .basics import _combine_experiment_data, _listify
from .plotresult import PlotResult
from ..analysis.regions import Regions
from ..data.experiment import Experiment, SimulatedExperiment
from ..config import (
    FIGURE_SETTINGS,
    DEFAULT_LINEPLOT_SETTINGS,
)

_INSET_AXES_BOUNDS = (0.09, 0.67, 0.3, 0.3)
_INSET_TICK_LABELSIZE = 7
_ERRORBAR_FILL_ALPHA = 0.1
_ERRORBAR_EDGE_LINEWIDTH = 0.5

def _auto_zoom(data: pd.DataFrame, region_data: pd.DataFrame, *, offset_correct: bool, gran: float | None = None, ax: Axes | None = None) -> tuple[float, float]:
    dco_data = Regions.select_in(data, region_data, "Diffusive-Capacitive Onset")
    if dco_data.empty:
        return 60, 50 # Some good values for my data

    x_axis = "Resistance"
    if offset_correct:
        x_axis = f"Offset-Corrected {x_axis}"

    xmin = data[x_axis].min()
    xmax = dco_data[x_axis].max()

    if gran is None:
        gran = 5 * 10 ** max(0, int(np.floor(np.log10(xmax))) - 2)

    xmin = gran * (np.floor(xmin / gran) - 1)
    xmax = gran * (np.ceil(xmax / gran) + 1)
    xspan = xmax - xmin

    if ax is not None:
        ax.set(xlim=(xmin, xmax), ylim=(0, xspan))

    return xmin, xspan

    

def _draw_inset_axes(data: pd.DataFrame, ax: Axes, kwargs: dict) -> Axes:
    with plt.rc_context(FIGURE_SETTINGS):
        inset: Axes = ax.inset_axes(_INSET_AXES_BOUNDS)

        kwargs.pop("tile", None)
        kwargs.pop("ax", None)
        kwargs.pop("title", None)
        kwargs["legend"] = False

        core.lineplot(data, ax=inset, **kwargs)

        # Copy colors from main axis to inset
        for inset_line, line in zip(inset.lines, ax.lines):
            inset_line.set_color(line.get_color())


        patch_width = _INSET_AXES_BOUNDS[2]
        patch_height = _INSET_AXES_BOUNDS[3]
        phantom = Rectangle(
            (_INSET_AXES_BOUNDS[0] - patch_width*0.1, _INSET_AXES_BOUNDS[1]-patch_height*0.1), 
            patch_width*1.2, 
            patch_height*1.2,
            transform=ax.transAxes,
            alpha=0.0,
            zorder=0
        )
        ax.add_patch(phantom)

        inset.set_aspect("equal", adjustable="datalim")

        inset.set(xlabel=None, ylabel=None, xscale="linear", yscale="linear")
        inset.tick_params(axis="both", which="major", labelsize=_INSET_TICK_LABELSIZE)
        inset.xaxis.set_major_formatter(ticker.EngFormatter(places=0))
        inset.yaxis.set_major_formatter(ticker.EngFormatter(places=0))

    return inset


def _draw_nyquist_errorbars(
    ax: Axes, covvis: list[CovarianceVisualization], kwargs: dict
) -> None:
    for line, cv in zip(ax.lines, covvis):
        color = line.get_color()
        linestyle = line.get_linestyle()

        line.set_zorder(2)

        if kwargs.get("err_style") == "bars":
            draw_func = CovarianceVisualization.draw_ellipses
        else:
            draw_func = CovarianceVisualization.draw_hull

        if "style" in kwargs:
            config = {
                "facecolor": colors.to_rgba(color, _ERRORBAR_FILL_ALPHA),
                "edgecolor": colors.to_rgba(color, 1.0),
                "linewidth": _ERRORBAR_EDGE_LINEWIDTH,
                "linestyle": linestyle,
                "zorder": 1,
            }
        else:
            config = {
                "color": color,
                "alpha": _ERRORBAR_FILL_ALPHA,
                "zorder": 1,
            }

        draw_func(cv, ax, config)
        if len(ax.child_axes) > 0:
            draw_func(cv, ax.child_axes[0], config)


def _prepare_covvis(
    tile_grouped: DataFrameGroupBy, kwargs: dict
) -> list[list[CovarianceVisualization]]:
    covvis, dfs, ns = [], [], []

    for _, tile_group in tile_grouped:
        grouped = core._prepare_groupby(tile_group, kwargs)

        dfs += [group for _, group in grouped]
        ns.append(len(grouped))

    covvis_flat = CovarianceVisualization.calculate_covariances(dfs, kwargs)
    for n in ns:
        covvis.append(covvis_flat[:n])
        covvis_flat = covvis_flat[n:]

    return covvis


def nyquist(
    data: Experiment | list[Experiment],
    title: str | None = None,
    Rmin: float | None = None,
    Rspan: float | None = None,
    offset_correct: bool = True,
    add_inset: bool = True,
    add_frequency_labels: bool = False,
    **kwargs,
) -> PlotResult:
    if (Rmin is None and Rspan is None):
        df, region_df = _combine_experiment_data(
            data,
            lambda x: x.data,
            lambda x: x.analysis.regions.data,
            kwargs=kwargs,
        )
        Rmin, Rspan = _auto_zoom(df, region_df, offset_correct=offset_correct)
    else:
        df = _combine_experiment_data(data, lambda x: x.data, kwargs=kwargs)
        

    # Determine x axis
    x_axis = "Resistance"
    if offset_correct:
        x_axis = f"Offset-Corrected {x_axis}"

    # Setup the plot config
    config = {
        "x": x_axis,
        "y": "Neg. Reactance",
        "title": title,
        "series_info": Experiment.SERIES_INFO,
    }
    kwargs = DEFAULT_LINEPLOT_SETTINGS | kwargs | config
    
    # Prepare mean calculation
    grouped = core._prepare_groupby(
        df, kwargs, additional_groups={"Frequency", "Experiment Name", "Palette"}
    )

    # Base aggregations
    agg_info = {kwargs[v]: "mean" for v in ("x", "y")}

    # Aggregate sample names in a list
    if sample_aggregate := "Sample Name" not in grouped.keys:
        agg_info["Sample Name"] = list

    mean_data = grouped.agg(agg_info).reset_index()

    # Add sample names column
    if sample_aggregate:
        mean_data["Sample Names"] = mean_data["Sample Name"].apply(tuple)
        mean_data.drop(columns=["Sample Name"], inplace=True)
    else:
        mean_data["Sample Names"] = mean_data["Sample Name"].apply(lambda x: (x,))


    tile_config = {"tile": kwargs.get("tile")}
    tile_grouped = core._prepare_groupby(df, tile_config)

    if (errorbar := kwargs.get("errorbar")) is not None:
        covvis = _prepare_covvis(tile_grouped, kwargs)

    res = core.lineplot(mean_data, **kwargs)

    with res as (fig, ax):
        if res.has_meta("grid"):
            grid: FacetGrid = res.get_meta("grid")

            fig.set_layout_engine("tight")
            grid.set(
                xscale="linear",
                yscale="linear",
                aspect=1,
                xlim=(Rmin, Rmin + Rspan),
                ylim=(0, Rspan),
            )

            if add_inset:
                def _add_facet_inset_axes(data: pd.DataFrame, **_) -> None:
                    _draw_inset_axes(data, plt.gca(), kwargs)
                grid.map_dataframe(_add_facet_inset_axes)

            fig.set_layout_engine("constrained")

            axes = list(grid.axes.flat)
        else:
            assert isinstance(ax, Axes)

            ax.set(
                xscale="linear",
                yscale="linear",
                xlim=(Rmin, Rmin + Rspan),
                ylim=(0, Rspan),
            )

            if add_inset:
                _draw_inset_axes(mean_data, ax, kwargs)

            if (legend := ax.get_legend()) is not None:
                sns.move_legend(ax, "best", ncol=legend._ncols)

            axes = [ax]

        if errorbar is not None:
            for ax, cv in zip(axes, covvis):
                _draw_nyquist_errorbars(ax, cv, kwargs)

    return res
