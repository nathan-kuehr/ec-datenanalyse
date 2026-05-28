import numpy as np
import pandas as pd

from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from seaborn import FacetGrid

from typing import Callable, Iterable

from . import core
from .basics import _combine_experiment_data, _listify, bode, fresponse

from .residual_plots import residuals, residual_distribution
from .nyquist_plot import nyquist
from .plotresult import PlotResult
from ..config import DataSeriesInfo
from ..data.experiment import Experiment
from ..analysis.fitting import Fit
from ..config import DEFAULT_FIT_LINEPLOT_SETTINGS


_FIT_FREQ_GRID_POINTS = 1000

def _color_blend(c1, c2, alpha: float = 0.5):
    c1 = np.array(colors.to_rgba(c1))
    c2 = np.array(colors.to_rgba(c2))
    return (1 - alpha) * c1 + alpha * c2


def fitted_parameters(
    exp: Experiment | list[Experiment],
    title: str | None = None,
    series_info: dict[str, DataSeriesInfo] = {},
    **kwargs,
) -> PlotResult:
    data = _combine_experiment_data(exp, lambda e: e.analysis.fit.params_long, kwargs=kwargs)
    assert isinstance(data, pd.DataFrame)

    return core.parameter_plot(
        data,
        kwargs.pop("hue", "Experiment Name"),
        data["Parameter"].unique(),
        title,
        Fit.SERIES_INFO | series_info,
        **kwargs,
    )

def _get_all_lines(res: PlotResult) -> Iterable[Line2D]:
    ax: Axes

    if (grid := res.get_meta("grid")) is not None:
        assert isinstance(grid, FacetGrid)

        for ax in grid.axes.flat:
            yield from ax.get_lines()
            for child_ax in ax.child_axes:
                yield from child_ax.get_lines()
    else:
        with res as (fig, _):
            for ax in fig.axes:
                yield from ax.get_lines()

def _adapt_fit_linestyle(res: PlotResult) -> None:
    sizes = DEFAULT_FIT_LINEPLOT_SETTINGS["sizes"]
    fitted_size = sizes["Fitted"]

    for line in _get_all_lines(res):
        if line.get_linewidth() == fitted_size:
            c = line.get_color()

            blend = "w" if np.mean(colors.to_rgb(c)) < 0.5 else "k"

            line.set(markersize=0, color=_color_blend(c, blend))
            

def show_fit(
    exps: Experiment | list[Experiment],
    kind: Callable,
    title: str | None = None,
    series_info: dict[str, DataSeriesInfo] = {},
    **kwargs,
) -> PlotResult:
    
    if kwargs.get("size") is not None:
        raise ValueError("Cannot passe 'size' argument to show_fit - this data discrimination is reserved for internal use!")

    if kind in (residuals, residual_distribution):
        return kind([exp.analysis.fit.simulate_experiment() for exp in _listify(exps)], title=title, **kwargs)
    elif kind in (nyquist, bode, fresponse):
        exps = _listify(exps)

        # Span the simulation grid across all experiment frequencies
        all_freqs = np.concatenate([exp.data["Frequency"].unique() for exp in exps])
        freq_grid = np.logspace(
            np.log10(all_freqs.min()),
            np.log10(all_freqs.max()),
            _FIT_FREQ_GRID_POINTS,
        )

        # Create the simulated fitted data
        sims = [exp.analysis.fit.simulate_experiment(freq_grid) for exp in exps]

        # Add differentiation via the line size
        kwargs |= DEFAULT_FIT_LINEPLOT_SETTINGS
        kwargs.setdefault("errorbar", None)

        res = kind(exps + sims, title=title, series_info=series_info, **kwargs)

        _adapt_fit_linestyle(res)
        return res

    else:
        raise ValueError("Unsupported plotting function passed as 'kind'!")
