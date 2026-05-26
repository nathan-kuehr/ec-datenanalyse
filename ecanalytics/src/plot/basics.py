import numpy as np
import pandas as pd

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from seaborn import FacetGrid
from typing import Callable, Iterable

from . import core, region_plots
from ..data.experiment import Experiment, SimulatedExperiment
from .plotresult import PlotResult
from ..config import DataSeriesInfo


def _listify[T](obj: T | list[T]) -> list[T]:
    return obj if isinstance(obj, list) else [obj]

def _combine_experiment_data(
    experiments: Experiment | list[Experiment],
    *extractors: Callable[[Experiment], pd.DataFrame],
    kwargs: dict,
) -> pd.DataFrame | list[pd.DataFrame]:
    experiments = _listify(experiments)

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
    exps: Experiment | list[Experiment], 
    y: str, 
    title: str | None = None, 
    show_regions: bool | Iterable[str] = False, 
    **kwargs
) -> PlotResult:
    res = plot(exps, x="Frequency", y=y, title=title, **kwargs)

    if show_regions:
        real_exps = [exp for exp in _listify(exps) if not isinstance(exp, SimulatedExperiment)]
        data, region_data = _combine_experiment_data(
            real_exps, 
            lambda e: e.data,
            lambda e: e.analysis.regions.data,
            kwargs=kwargs)
        assert isinstance(data, pd.DataFrame) and isinstance(region_data, pd.DataFrame)

        with res as (fig, _):
            region_plots._draw_markers(fig.axes, data, region_data, show_regions, "Frequency", kwargs)

    return res

def _prepare_bode_grid_data(data: pd.DataFrame, components: list[str], tile_col: str | None) -> pd.DataFrame:
    id_cols = [c for c in data.columns if c not in components]
    melted = data.melt(
        id_vars=id_cols,
        value_vars=components,
        var_name="Component",
        value_name="Value",
    )

    if tile_col is None:
        tidc = pd.Series(0, index=melted.index)
        ncols = 1
    else:
        tile_vals = data[tile_col].unique()
        tidc = melted[tile_col].map({t: i for i, t in enumerate(tile_vals)})
        ncols = min(core._MAX_GRID_COL_WRAP, len(tile_vals))

    sub_row = np.where(melted["Component"].str.contains("Impedance"), 0, 1)
    return melted.assign(
        _grid_col=tidc % ncols,
        _grid_row=(tidc // ncols) * 2 + sub_row,
    )

def _bode_adjust_axes_kind(axes: np.ndarray, y_axis: str, tile_vals):
    lims = [np.inf, -np.inf]

    is_magnitude = y_axis.count("Impedance") > 0
    if not is_magnitude:
        tile_vals = [""] * len(tile_vals)
    
    
    _, ncols = axes.shape 
    suppress_y_axis_annotation = {
        "ylabel": "", 
        "yticklabels": []
    }

    ax: Axes
    for ax in axes.flat:
        if not ax.has_data():
            ax.set_visible(False) # Hide the ones w/o data
        
        core._set_axes_from_series_info(ax, "Frequency", y_axis, Experiment.SERIES_INFO)

        ymin, ymax = ax.get_ylim()
        lims[0] = min(lims[0], ymin)
        lims[1] = max(lims[1], ymax)

    for i, ax in enumerate(axes.flat):
        if not ax.has_data():
            continue

        ax.set(
            title=tile_vals[i],
            ylim=lims,
            **({} if not i % ncols else suppress_y_axis_annotation)
        )


def bode(
    exps: Experiment | list[Experiment],
    title: str | None = None,
    offset_correct: bool = False,
    show_regions: bool | Iterable[str] = False,
    phase_clip: tuple[float, float] | bool = False,
    **kwargs,
) -> PlotResult:
    
    # Prepare the data
    data = _combine_experiment_data(exps, lambda x: x.data, kwargs=kwargs)
    if show_regions:
        region_data = _combine_experiment_data(exps, lambda e: e.analysis.regions.data, kwargs=kwargs)
    assert isinstance(data, pd.DataFrame)

    # Check which components to plot 
    components = ["Impedance", "Phase"]
    if offset_correct:
        components = [f"Offset-Corrected {c}" for c in components]

    # Grab the tiling columns
    if (tile_col := kwargs.pop("tile", None)) is not None:
        tile_vals = list(data[tile_col].unique())
        kwargs.setdefault("hue", tile_col)
    else:
        tile_vals = [""] * 2

    # Prepare the special grid data
    grid_data = _prepare_bode_grid_data(data, components, tile_col)

    BODE_DEFAULT_CONFIG = {
        "x": "Frequency",
        "y": "Value",
        "row": "_grid_row",
        "col": "_grid_col",
        "style": "Component",
        "facet_kws": {"sharex": True, "sharey": False},
    }

    config = kwargs | BODE_DEFAULT_CONFIG | {
        "data": grid_data,
        "series_info": Experiment.SERIES_INFO,
        "title": title,
        "hue": kwargs.get("hue") or tile_col
    } | core._prepare_palette(data, kwargs)


    res = core.lineplot(**config)
    with res as (fig, _):
        assert isinstance(grid := res.get_meta("grid"), FacetGrid)

        # Half the height
        w, h = fig.get_size_inches()
        fig.set_size_inches(w, h / 2)

        # Get axes handles
        magn_axes, phase_axes = grid.axes[::2], grid.axes[1::2]

        for axes, y_axis in zip((magn_axes, phase_axes), components):
            _bode_adjust_axes_kind(axes, y_axis, tile_vals)

        # Add lines indicating region boundaries        
        if show_regions:
            data = data[data["Data Origin"] == "Measured"]

            real_exps = [exp for exp in _listify(exps) if not isinstance(exp, SimulatedExperiment)]
            region_data = _combine_experiment_data(real_exps, lambda e: e.analysis.regions.data, kwargs=kwargs)
            assert isinstance(region_data, pd.DataFrame)

            for axes in (magn_axes, phase_axes):
                region_plots._draw_markers(axes.flat, data, region_data, show_regions, "Frequency", kwargs | {"tile": tile_col})

        # Phase clip
        if isinstance(phase_clip, bool) and phase_clip:
            phase_clip = (-90, 0)
        if isinstance(phase_clip, tuple):
            for ax in phase_axes.flat:
                ax.set_ylim(phase_clip)

        fig.set_layout_engine("constrained")

    return res
