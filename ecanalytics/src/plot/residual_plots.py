import numpy as np
import pandas as pd
import seaborn as sns

from seaborn import JointGrid, FacetGrid
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
from typing import Callable, Iterable

from . import core, region_plots
from .basics import _combine_experiment_data
from .plotresult import PlotResult
from ..analysis.analysis import Analysis
from ..analysis.kkt import compile_residual_stats
from ..data.experiment import Experiment, SimulatedExperiment
from ..config import LARGE_FIGURE_SIZE, RESIDUAL_PLOT_SETTINGS


_RESIDUAL_BOUND_PADDING = 1.05
_MIN_RESIDUAL_BOUND = 1.0
_BINWIDTH_RESIDUAL_DIVISOR = 35
_GOOD_DATA_REFERENCE_LINE_WIDTH = 0.75


def _add_stats_overview(ax: Axes, data: pd.DataFrame, kwargs: dict) -> None:
    if len(core._active_groupby_cols(data, kwargs)) > 0:
        grouped = core._prepare_groupby(data, kwargs)
        legend = core._iterate_legend(ax, dummy=False)

        for (_, text), (_, group) in zip(legend, grouped):
            stats = np.squeeze(compile_residual_stats(group, pool=True))
            text.set_text(text.get_text() + f" [{_make_stat_label(stats)}]")
    else:
        # No legend drawn yet
        phantom = Rectangle((0, 0), 1, 1, visible=False)
        stats = np.squeeze(compile_residual_stats(data, pool=True))

        ax.legend(
            handles=[phantom],
            labels=[_make_stat_label(stats)],
            loc="best",
            frameon=False,
        )

def _make_stat_label(stat_row: np.ndarray) -> str:
    rms, _, _, rho = stat_row
    return rf"$\Delta_{{\mathrm{{rms}}}}$ = {rms:.2f} %, $\rho$ = {rho:.4f}"


def residuals(
    exps: Experiment | list[Experiment],
    title: str | None,
    show_regions: bool | Iterable[str] = False, 
    **kwargs,
) -> PlotResult:
    data = _combine_experiment_data(exps, lambda e: e.analysis.kkt.data_long, kwargs=kwargs)
    if show_regions:
        region_data = _combine_experiment_data(exps, lambda e: e.analysis.regions.data, kwargs=kwargs)
        assert isinstance(region_data, pd.DataFrame)
    assert isinstance(data, pd.DataFrame)

    config = kwargs | RESIDUAL_PLOT_SETTINGS | {
        "title": title,
        "series_info": Analysis.SERIES_INFO,
    }

    res = core.lineplot(data, **config)
    with res as (fig, ax):
        if (grid := res.get_meta("grid")) is not None:
            assert isinstance(grid, FacetGrid)

            fig.set_layout_engine("tight")

            # Equilibrated y axis
            ylim_max = abs(max(fig.axes[0].get_ylim(), key=abs))
            grid.set(ylim=(-ylim_max, ylim_max))

            # Good data borders
            for y in (-1, +1):
                grid.refline(y=y, linewidth=_GOOD_DATA_REFERENCE_LINE_WIDTH, zorder=0, linestyle="-.", color="k")

            fig.set_layout_engine("constrained")
        else:
            assert isinstance(ax, Axes)
            fig.set_size_inches(LARGE_FIGURE_SIZE)

            # Equilibrated y axis
            ylim_max = max(ax.get_ylim(), key=abs)
            ax.set_ylim((-abs(ylim_max), abs(ylim_max)))

            # Good data borders
            for y in (-1, +1):
                ax.axhline(y, linewidth=_GOOD_DATA_REFERENCE_LINE_WIDTH, zorder=0, linestyle="-.", color="k")

            sns.move_legend(ax, "best", ncol=2)
        
        if show_regions:
            region_plots._draw_markers(fig.axes, data, region_data, show_regions, "Frequency", kwargs)
        
    return res


def residual_distribution(
    exp: Experiment | list[Experiment],
    title: str | None,
    min_bound: float = _MIN_RESIDUAL_BOUND,
    include_stats: bool = True,
    **kwargs,
) -> PlotResult:
    data = _combine_experiment_data(exp, lambda e: e.analysis.kkt.data, kwargs=kwargs)
    assert isinstance(data, pd.DataFrame)

    config = {
        "x": "Real Residual",
        "y": "Imag. Residual",
        "data": data,
        "title": title,
        "series_info": Analysis.SERIES_INFO,
    }

    bound = _RESIDUAL_BOUND_PADDING * np.max(
        np.abs(data[["Real Residual", "Imag. Residual"]].to_numpy()), initial=min_bound
    )

    # If only one group: bar plot
    if len(core._active_groupby_cols(data, kwargs)) == 0:
        config["marginal_kws"] = kwargs.pop("marginal_kws", {}) | {
            "binwidth": bound / _BINWIDTH_RESIDUAL_DIVISOR
        }

    res = core.joint_distribution_plot(**config, **kwargs)
    with res as (fig, axes):
        assert isinstance(axes, list) and len(axes) == 3

        joint_ax = axes[0]

        # Draw line cross through the origin
        joint: JointGrid = res.get_meta("jointplot")
        joint.refline(x=0, y=0, linewidth=_GOOD_DATA_REFERENCE_LINE_WIDTH, marginal=True, zorder=0, linestyle="-.")

        if include_stats:
            _add_stats_overview(joint_ax, data, kwargs)

        joint_ax.set(xlim=(-bound, bound), ylim=(-bound, bound))

    return PlotResult(title, fig, **kwargs)