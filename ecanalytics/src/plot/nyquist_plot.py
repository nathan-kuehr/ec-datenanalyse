import pandas as pd

from matplotlib import colors, pyplot as plt
from matplotlib.axes import Axes
from pandas.core.groupby.generic import DataFrameGroupBy
from seaborn import FacetGrid

import matplotlib.ticker as ticker

from . import core
from .covariance import CovarianceVisualization
from .basics import _combine_experiment_data
from .plotresult import PlotResult
from ..data.experiment import Experiment
from ..config import (
    FIGURE_SETTINGS,
    DEFAULT_LINEPLOT_SETTINGS,
)


_INSET_AXES_BOUNDS = (0.09, 0.67, 0.3, 0.3)
_INSET_TICK_LABELSIZE = 7
_ERRORBAR_FILL_ALPHA = 0.1
_ERRORBAR_EDGE_LINEWIDTH = 0.5


def _clean_nyquist_args(kwargs: dict) -> dict:
    return core._clean_args(
        kwargs, ["data", "title", "Rmin", "Rspan", "offset_correct"]
    )


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
                "edgecolor": colors.to_rgba(color, 1),
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
    Rmin: float = 60,
    Rspan: float = 50,
    offset_correct: bool = True,
    add_inset: bool = True,
    add_frequency_labels: bool = False,
    **kwargs,
) -> PlotResult:
    df: pd.DataFrame = _combine_experiment_data(data, lambda x: x.data, kwargs=kwargs)

    x_axis = ("Offset-Corrected " if offset_correct else "") + "Resistance"
    config = {
        "x": x_axis,
        "y": "Neg. Reactance",
        "title": title,
        "no_save": True,
        "series_info": Experiment.SERIES_INFO,
    }

    kwargs = DEFAULT_LINEPLOT_SETTINGS | kwargs | config

    # Aggregate means
    grouped = core._prepare_groupby(
        df, kwargs, additional_groups={"Frequency", "Experiment Name"}
    )
    agg = {kwargs["x"]: "mean", kwargs["y"]: "mean", "Palette": "first"}
    if "Sample Name" not in core._active_groupby_cols(df, kwargs):
        agg["Sample Name"] = list

    mean_data = grouped.agg(agg).reset_index()
    mean_data["Sample Names"] = mean_data["Sample Name"].apply(
        lambda x: tuple(x) if isinstance(x, list) else (x,)
    )

    if "Sample Name" in agg:
        mean_data.pop("Sample Name")

    tile_grouped = core._prepare_groupby(
        df, {"tile": kwargs["tile"]} if "tile" in kwargs else {}
    )
    if (errorbar := kwargs.get("errorbar")) is not None:
        covvis = _prepare_covvis(tile_grouped, kwargs)

    with (res := core.lineplot(mean_data, **kwargs)) as (fig, ax):
        if (grid := res.get_meta("grid")) is not None:
            assert isinstance(grid, FacetGrid)

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

            axes = grid.axes.flat
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

            axes = [ax]

    if errorbar is not None:
        for ax, cv in zip(axes, covvis):
            _draw_nyquist_errorbars(ax, cv, kwargs)

    kwargs = _clean_nyquist_args(kwargs)
    kwargs = core._clean_plot_args(kwargs)

    if grid is not None:
        return PlotResult(title, fig, **kwargs).add_meta({"grid": grid})
    return PlotResult(title, fig, **kwargs)
