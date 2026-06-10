import numpy as np
import pandas as pd
import seaborn as sns

from seaborn import FacetGrid
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from typing import Iterable

from . import core, region_plots
from .basics import _combine_experiment_data
from .plotresult import PlotResult
from ..analysis.analysis import Analysis
from ..analysis.kkt import compile_residual_stats
from ..data.experiment import Experiment
from ..config import LARGE_FIGURE_SIZE, RESIDUAL_PLOT_SETTINGS


REFERENCE_LINE_STYLE = dict(linewidth=0.5, zorder=0, linestyle="-.", color="k")

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
        if res.has_meta("grid"):
            grid: FacetGrid = res.get_meta("grid")

            fig.set_layout_engine("tight")

            # Equilibrated y axis
            ylim_max = abs(max(fig.axes[0].get_ylim(), key=abs))
            grid.set(ylim=(-ylim_max, ylim_max))

            for y in (-1, +1):
                grid.refline(y=y, **REFERENCE_LINE_STYLE)

            fig.set_layout_engine("constrained")
        else:
            assert isinstance(ax, Axes)
            fig.set_size_inches(LARGE_FIGURE_SIZE)

            ylim_max = abs(max(ax.get_ylim(), key=abs))
            ax.set_ylim((-ylim_max, ylim_max))

            for y in (-1, +1):
                ax.axhline(y, **REFERENCE_LINE_STYLE)

            sns.move_legend(ax, "best", ncol=2)

        if show_regions:
            region_plots._draw_markers(fig.axes, data, region_data, show_regions, "Frequency", kwargs)

    return res


def residual_distribution(
    exp: Experiment | list[Experiment],
    title: str | None,
    **kwargs,
) -> PlotResult:
    data = _combine_experiment_data(exp, lambda e: e.analysis.kkt.data, kwargs=kwargs)
    assert isinstance(data, pd.DataFrame)

    res = core.joint_distribution_plot(
        data=data,
        x="Real Residual",
        y="Imag. Residual",
        title=title,
        series_info=Analysis.SERIES_INFO,
        **kwargs,
    )

    joint_axes: list[Axes] = res.get_meta("joint_axes")
    marginal_axes: list[tuple[Axes, Axes]] = res.get_meta("marginal_axes")

    for joint, (margx, margy) in zip(joint_axes, marginal_axes):
        joint.set(box_aspect=1)

        # Center cross on joint + marginals
        joint.axhline(0, **REFERENCE_LINE_STYLE)
        joint.axvline(0, **REFERENCE_LINE_STYLE)
        margx.axvline(0, **REFERENCE_LINE_STYLE)
        margy.axhline(0, **REFERENCE_LINE_STYLE)

    return res
