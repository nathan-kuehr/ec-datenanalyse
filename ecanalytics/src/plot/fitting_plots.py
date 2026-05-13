import numpy as np
import pandas as pd

from matplotlib.axes import Axes
from matplotlib import colors

from typing import Callable

from . import core
from .basics import _combine_experiment_data, _listify_experiments
from .nyquist_plot import nyquist
from .plotresult import PlotResult
from ..config import DataSeriesInfo
from ..data.experiment import Experiment
from ..analysis.fitting import Fit


_FIT_FREQ_GRID_POINTS = 1000


def _mix_colors(
    a: tuple | str, b: tuple | str = (0, 0, 0), factor: float = 0.5
) -> tuple[float, float, float]:
    a_rgb = colors.to_rgb(a)
    b_rgb = colors.to_rgb(b)

    return tuple((1 - factor) * ac + factor * bc for ac, bc in zip(a_rgb, b_rgb))


def fitted_parameters(
    exp: Experiment | list[Experiment],
    title: str | None = None,
    series_info: dict[str, DataSeriesInfo] = {},
    **kwargs,
) -> PlotResult | None:
    data = _combine_experiment_data(
        _listify_experiments(exp), lambda e: e.analysis.fit.params_long, kwargs=kwargs
    )

    return core.parameter_plot(
        data,
        kwargs.pop("hue", "Experiment Name"),
        data["Parameter"].unique(),
        title,
        Fit.SERIES_INFO | series_info,
        **kwargs,
    )


def show_fit(
    exps: Experiment | list[Experiment],
    kind: Callable,
    title: str | None = None,
    series_info: dict[str, DataSeriesInfo] = {},
    **kwargs,
) -> PlotResult | None:
    exps = _listify_experiments(exps)

    if kind is nyquist:
        freqs = np.vstack([exp.data["Frequency"].unique() for exp in exps])
        f_max, f_min = freqs.max(), freqs.min()
        freq_grid = np.logspace(np.log10(f_min), np.log10(f_max), _FIT_FREQ_GRID_POINTS)

        sims = [exp.analysis.fit.simulate_experiment(freq_grid) for exp in exps]
        res = nyquist(exps + sims, title, series_info=series_info, legend=True, **kwargs)

        with res as (fig, axes):
            if isinstance(axes, Axes):
                axes = [axes]

            ax: Axes
            for ax in axes:
                lines = ax.lines
                if len(ax.child_axes) > 0:
                    lines += ax.child_axes[0].lines
                for line in lines:
                    if len(line.get_xdata()) == len(freq_grid):
                        line.set_markersize(0)
                        line.set_color(_mix_colors(line.get_color()))

        return res

    return None
