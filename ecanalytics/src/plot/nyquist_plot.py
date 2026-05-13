import numpy as np
import pandas as pd
import seaborn as sns

from copy import copy
from matplotlib import colors, pyplot as plt
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from pandas.core.groupby.generic import DataFrameGroupBy
from seaborn import FacetGrid

import matplotlib.ticker as ticker

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from . import core
from ..analysis import parallel
from .covariance_visualization import CovarianceVisualization
from .basics import _combine_experiment_data
from .plotresult import PlotResult
from ..data.experiment import Experiment
from ..config import (
    EIS_EXPERIMENT_FREQUENCY_TOLERANCE,
    FIGURE_SETTINGS,
    DEFAULT_LINEPLOT_SETTINGS,
)


def _clean_nyquist_args(kwargs: dict) -> dict:
    return core._clean_args(
        kwargs, ["data", "title", "Rmin", "Rspan", "offset_correct"]
    )

def _draw_inset_axes(data: pd.DataFrame, ax: Axes, kwargs: dict):
    with plt.rc_context(FIGURE_SETTINGS):
        inset: Axes = ax.inset_axes((0.09, 0.67, 0.3, 0.3))

        kwargs.pop("tile", None)
        kwargs.pop("ax", None)
        kwargs.pop("title", None)
        kwargs["legend"] = False

        core.lineplot(data, ax=inset, **kwargs)

        # Copy colors
        for isline, line in zip(inset.lines, ax.lines):
            isline.set_color(line.get_color())

        # Center 
        inset.set_aspect("equal", adjustable="datalim")

        inset.set(xlabel=None, ylabel=None, xscale="linear", yscale="linear")#, xlim=lims[0], ylim=lims[1])
        inset.tick_params(axis="both", which="major", labelsize=7)
        inset.xaxis.set_major_formatter(ticker.EngFormatter(places=0))
        inset.yaxis.set_major_formatter(ticker.EngFormatter(places=0))

    return inset

def _draw_nyquist_errorbars(
    ax: Axes, covvis: list[CovarianceVisualization], kwargs: dict
) -> None:
    for line, cv in zip(ax.lines, covvis):
        c = line.get_color()
        s = line.get_linestyle()

        line.set_zorder(2)
        
        # Which type ?
        if kwargs.get("err_style") == "bars":
            func = CovarianceVisualization.draw_ellipses 
        else: 
            func = CovarianceVisualization.draw_hull

        # Do we need to include the line style ? 
        if "style" in kwargs:
            config = {
                "facecolor": colors.to_rgba(c, 0.1),
                "edgecolor": colors.to_rgba(c, 1),
                "linewidth": 0.5,
                "linestyle": s,
                "zorder": 1,
            }
        else:
            config = {
                "color": c,
                "alpha": 0.1,
                "zorder": 1,
            }

        # Plot in main axis
        func(cv, ax, config)
        if len(ax.child_axes) > 0:
            # Plot in inset
            func(cv, ax.child_axes[0], config)


def _prepare_covvis(tile_grouped: DataFrameGroupBy, kwargs: dict):
    covvis, dfs, ns = [], [], []

    for _, tile_group in tile_grouped:
        grouped = core._prepare_groupby(tile_group, kwargs)

        dfs += [group for _, group in grouped]
        ns.append(len(grouped))

    covvis_flat = CovarianceVisualization.Calculate_Covariances(dfs, kwargs)
    for n in ns:
        covvis.append(covvis_flat[:n])
        covvis_flat = covvis_flat[n:]
    
    return covvis

def nyquist(
    data: Experiment | list[Experiment],
    title: str | None = None,
    Rmin: float = 60,
    Rspan: float = 50,
    offset_correct: bool = True,
    add_inset: bool = True,
    add_frequency_labels: bool = False,
    **kwargs,
) -> PlotResult:
    df: pd.DataFrame = _combine_experiment_data(data, lambda x: x.data, kwargs=kwargs)

    # Choose correct x-axis
    x_axis = ("Offset-Corrected " if offset_correct else "") + "Resistance"
    config = {
        "x": x_axis,
        "y": "Neg. Reactance",
        "title": title,
        "no_save": True,
        "series_info": Experiment.Series_Info
    }

    kwargs = DEFAULT_LINEPLOT_SETTINGS | kwargs | config

    # Aggregate means
    grouped = core._prepare_groupby(df, kwargs, additional_groups={"Frequency", "Experiment Name"})
    agg = {kwargs["x"]: "mean", kwargs["y"]: "mean", "Palette": "first"}
    if not "Sample Name" in core._active_groupby_cols(df, kwargs):
        agg["Sample Name"] = list

    mean_data = grouped.agg(agg).reset_index()
    mean_data["Sample Names"] = mean_data["Sample Name"].apply(lambda x: tuple(x) if isinstance(x, list) else (x,))

    if "Sample Name" in agg:
        mean_data.pop("Sample Name")

    # Prepare the covariances
    tile_grouped = core._prepare_groupby(df, {"tile": kwargs["tile"]} if "tile" in kwargs else {})
    if (errorbar := kwargs.get("errorbar")) is not None:
        covvis = _prepare_covvis(tile_grouped, kwargs)

    with (res := core.lineplot(mean_data, **kwargs)) as (fig, ax):
        # Check if the plot is tiled
        if (grid := res.get_meta("grid")) is not None:
            assert isinstance(grid, FacetGrid)

            # Setup axes correctly
            fig.set_layout_engine("tight")
            grid.set(xscale="linear", yscale="linear", aspect=1, xlim=(Rmin, Rmin + Rspan), ylim=(0, Rspan))

            if add_inset:
                def _add_facet_inset_axes(data: pd.DataFrame, **_):
                    _draw_inset_axes(data, plt.gca(), kwargs)
                grid.map_dataframe(_add_facet_inset_axes)

            # Return to constrained layout
            fig.set_layout_engine("constrained")

            # Get iterator over large scale axes
            axes = grid.axes.flat
        else:
            assert isinstance(ax, Axes)

            # Setup axes correctly
            ax.set(xscale="linear", yscale="linear", xlim=(Rmin, Rmin + Rspan), ylim=(0, Rspan))

            if add_inset:
                _draw_inset_axes(mean_data, ax, kwargs)

            # Only this axis is belonging to the nyquist plot
            axes = [ax]

    if errorbar is not None:
        for ax, cv in zip(axes, covvis):
            _draw_nyquist_errorbars(ax, cv, kwargs)

            # if add_frequency_labels:
            #     freqs = mean_data["Frequency"].unique()
            #     decades = 10.0 ** np.round(np.log10(freqs))

            #     target_freqs = set(
            #         freqs[
            #             np.isclose(freqs, decades, rtol=EIS_EXPERIMENT_FREQUENCY_TOLERANCE)
            #         ]
            #     )
            #     if not target_freqs:
            #         raise ValueError("No frequencies found near decade values!")

            #     def freq_label_formatter(f):
            #         if f >= 1e6:
            #             return f"{f / 1e6:.0f} MHz"
            #         if f >= 1e3:
            #             return f"{f / 1e3:.0f} kHz"
            #         if f >= 1:
            #             return f"{f:.0f} Hz"
            #         else:
            #             return f"{f:.2f} Hz"

            #     with plt.rc_context(FIGURE_SETTINGS):
            #         grouped = core._prepare_groupby(mean_data, kwargs)
            #         for line, (_, group) in zip(ax.lines, grouped):
            #             c = line.get_color()
            #             scatter_color = tuple(c * 0.5 for c in colors.to_rgb(c))

            #             mask = group["Frequency"].isin(target_freqs)

            #             freqs = group["Frequency"][mask].to_numpy()
            #             pos = group[[kwargs["x"], kwargs["y"]]][mask].to_numpy()

            #             # Plot scatter points at target frequencies
            #             ax.scatter(
            #                 pos[:, 0],
            #                 pos[:, 1],
            #                 s=DEFAULT_LINEPLOT_SETTINGS["markersize"] ** 2,
            #                 color=scatter_color,
            #                 linewidths=0,
            #                 zorder=10,
            #             )

            #         # Add annotation for each target frequency
            #         for f, p in zip(freqs, pos):
            #             ax.annotate(
            #                 freq_label_formatter(f),
            #                 xy=p,
            #                 fontsize=plt.rcParams[
            #                     "legend.fontsize"
            #                 ],  # Greift die aktuelle Legenden-Schriftgröße ab
            #                 ha="center",  # Horizontal zentriert
            #                 va="center",
            #             )

        
        # # Set axes scaling now after insetting
        # if grid is not None:
        #     grid.set(xlim=(Rmin, Rmin + Rspan), ylim=(0, Rspan))
        # else:
        #     axes[0].set(xlim=(Rmin, Rmin + Rspan), ylim=(0, Rspan))


    kwargs = _clean_nyquist_args(kwargs)
    kwargs = core._clean_plot_args(kwargs)

    if grid is not None:
        return PlotResult(title, fig, **kwargs).add_meta({"grid": grid})
    else:
        return PlotResult(title, fig, **kwargs)
