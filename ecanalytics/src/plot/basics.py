import numpy as np
import pandas as pd

from matplotlib.axes import Axes
from seaborn import FacetGrid
from typing import Callable

from . import core
from ..data.experiment import Experiment
from .plotresult import PlotResult


_BODE_PHASE_DEFAULT_YLIM = (-90.0, 0.0)
_BODE_COMPONENTS = ["Impedance", "Phase"]


def _listify_experiments(exp: Experiment | list[Experiment]) -> list[Experiment]:
    if isinstance(exp, Experiment):
        return [exp]
    if isinstance(exp, list):
        if not exp:
            raise ValueError("No experimental data passed!")
        return exp
    raise TypeError("Unsupported data type passed!")


def _combine_experiment_data(
    experiments: Experiment | list[Experiment],
    *extractors: Callable[[Experiment], pd.DataFrame],
    kwargs: dict,
) -> pd.DataFrame | list[pd.DataFrame]:
    experiments = _listify_experiments(experiments)

    def transform(data: pd.DataFrame) -> None:
        data[diff_col] = data["Experiment Name"] + " - " + data[diff_col].astype(str)

    do_transform = False
    if len(experiments) > 1:
        diff_col = kwargs.get("hue") or kwargs.get("tile")
        if diff_col is None:
            kwargs["hue"] = "Experiment Name"
        else:
            do_transform = diff_col != "Experiment Name"

    combined = []
    for extractor in extractors:
        df = pd.concat([extractor(exp) for exp in experiments], axis=0, ignore_index=True)
        if do_transform:
            transform(df)
        combined.append(df)

    return combined[0] if len(extractors) == 1 else combined


def plot(
    data: Experiment | list[Experiment],
    x: str,
    y: str,
    title: str | None = None,
    **kwargs,
) -> PlotResult:
    df = _combine_experiment_data(data, lambda x: x.data, kwargs=kwargs)
    assert isinstance(df, pd.DataFrame)
    return core.lineplot(df, x, y, title, Experiment.SERIES_INFO, **kwargs)


def fresponse(
    data: Experiment | list[Experiment], y: str, title: str | None = None, **kwargs
) -> PlotResult:
    return plot(data, x="Frequency", y=y, title=title, **kwargs)


def _prepare_bode_grid_data(
    data: pd.DataFrame, tile_col: str | None, tile_vals
) -> pd.DataFrame:
    id_cols = [c for c in data.columns if c not in _BODE_COMPONENTS]
    melted = data.melt(
        id_vars=id_cols,
        value_vars=_BODE_COMPONENTS,
        var_name="Component",
        value_name="Value",
    )

    if tile_col is None:
        tidc = pd.Series(0, index=melted.index)
        ncols = 1
    else:
        tidc = melted[tile_col].map({t: i for i, t in enumerate(tile_vals)})
        ncols = min(core._MAX_GRID_COL_WRAP, len(tile_vals))

    sub_row = np.where(melted["Component"] == "Impedance", 0, 1)
    return melted.assign(
        _grid_col=tidc % ncols,
        _grid_row=(tidc // ncols) * 2 + sub_row,
    )


def _adjust_bode_axes(grid: FacetGrid, tile_vals):
    # Limits
    imp_lim, phase_lim = [np.inf, -np.inf], [np.inf, -np.inf]

    _, ncols = grid.axes.shape

    ax: Axes
    for i, ax in enumerate(grid.axes.flat):
        if not ax.has_data():
            ax.set_visible(False) # Hide the ones w/o data
            continue

        r = i // ncols
        is_magnitude = (r % 2 == 0)
        y = "Impedance" if is_magnitude else "Phase"

        # Set axes - before reading lims for the log scale 
        core._set_axes_from_series_info(ax, "Frequency", y, Experiment.SERIES_INFO)

        # Update limits
        ymin, ymax = ax.get_ylim()
        lim = imp_lim if is_magnitude else phase_lim
        lim[0] = min(lim[0], ymin)
        lim[1] = max(lim[1], ymax)
    
    # Apply now the lims + titles 
    for i, ax in enumerate(grid.axes.flat):
        if not ax.has_data():
            continue

        r, c = i // ncols, i % ncols

        is_magnitude = (r % 2 == 0)
        title = tile_vals[(r // 2) * ncols + c]

        ax_config = {
            "title": title if len(tile_vals) and is_magnitude else "",
            "ylim": imp_lim if is_magnitude else phase_lim
        }

        # Suppress inner ylabels & ticks
        if c > 0:
            ax_config |= { "ylabel": "", "yticklabels": []}

        ax.set(**ax_config)

def bode(
    data: Experiment | list[Experiment],
    title: str | None = None,
    phase_ylim: tuple[float, float] | None = _BODE_PHASE_DEFAULT_YLIM,
    **kwargs,
) -> PlotResult:
    df = _combine_experiment_data(data, lambda x: x.data, kwargs=kwargs)
    assert isinstance(df, pd.DataFrame)

    if (tile_col := kwargs.pop("tile", None)) is not None:
        tile_vals = df[tile_col].unique()
    else:
        tile_vals = []

    grid_data = _prepare_bode_grid_data(df, tile_col, tile_vals)

    config = kwargs | {
        "data": grid_data,
        "x": "Frequency",
        "y": "Value",
        "row": "_grid_row",
        "col": "_grid_col",
        "style": "Component",
        "series_info": Experiment.SERIES_INFO,
        "title": title,
        "facet_kws": {"sharex": True, "sharey": False},
        **core._prepare_palette(df, kwargs),
        "hue": kwargs.get("hue") or tile_col
    }

    res = core.lineplot(**config)
    with res as (fig, _):
        assert isinstance(grid := res.get_meta("grid"), FacetGrid)

        # Manually adapt the axes
        _adjust_bode_axes(grid, tile_vals)

        # Half the height 
        w, h = fig.get_size_inches()
        fig.set_size_inches(w, h / 2)

        fig.set_layout_engine("constrained")
    return res
