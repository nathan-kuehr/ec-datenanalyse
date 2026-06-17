import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from . import core, nyquist_plot
from seaborn import FacetGrid

from .basics import _combine_experiment_data
from .plotresult import PlotResult
from ..analysis.regions import Regions
from ..config import DEFAULT_LINEPLOT_SETTINGS, FIGURE_SETTINGS
from ..data.experiment import Experiment


def _decorate_facet(data: pd.DataFrame, *, fspan: tuple[float, float], curve_data: pd.DataFrame, region_data: pd.DataFrame, **__):
    ax = plt.gca()
    panel = data["Panel"].iloc[0]
    name = data["Sample Name"].iloc[0]
    fmin, fmax = fspan

    # Set axes correctly
    if panel == "Nyquist":
        X, Y = "Offset-Corrected Resistance", "Neg. Reactance"
    else:
        X, Y = "Frequency", panel
    core._set_axes_from_series_info(ax, X, Y, Experiment.SERIES_INFO)


    pos = {row["Point"]: (row["x"], row["y"]) for _, row in data.iterrows()}
    if panel == "Nyquist":
        ax.set(xscale="linear", yscale="linear")
        nyquist_plot._auto_zoom(curve_data[curve_data["Panel"] == "Nyquist"].rename(columns={"x": X}), region_data, offset_correct=True, ax=ax)
    elif panel == "Tangent Angle":
        ax.invert_xaxis()
        for h in (0, 45, 70): # TODO: 70° here is argument dependent
            ax.axhline(h, ls="-.", color="k", lw=0.5)
        if "HF Artefact Onset" in pos:
            ax.axvspan(fmax, pos["HF Artefact Onset"][0], alpha=0.1)
        if "LF Artefact Onset" in pos:
            ax.axhline(pos["LF Artefact Onset"][1], ls="-.", color="k", lw=0.5)
            ax.axvspan(pos["LF Artefact Onset"][0], fmin, alpha=0.1)
    else:
        ax.invert_xaxis()
        f_A = pos.get("Kinetic Limit", (fmax,))[0]
        f_B = pos.get("Diffusive-Capacitive Onset", (fmin,))[0]
        if f_A > f_B:
            has_mass_transport = region_data[region_data["Sample Name"] == name]["Mass Transport Resolvable"].iloc[0]
            ax.axvspan(f_A, f_B, color="#10b981" if has_mass_transport else "#d73818", alpha=0.1)


def _prepare_regions_dfs(exp: Experiment | list[Experiment], kwargs: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nyquist, region, angles = _combine_experiment_data(
        exp,
        lambda x: x.data,
        lambda x: x.analysis.regions.data,
        lambda x: x.analysis.regions.curve_data,
        kwargs=kwargs
    )

    name_col_loc: int = region.columns.get_loc("Sample Name")
    meta_cols = region.columns[name_col_loc + 1:].to_list()

    curve_dfs, point_dfs = [], []
    for panel in ("Nyquist", "Tangent Angle", "Tangent Angle Derivative"):
        if panel == "Nyquist":
            base = nyquist
            rename = {"Offset-Corrected Resistance": "x", "Neg. Reactance": "y"}
        else:
            base = angles
            rename = {"Frequency": "x", panel: "y"}

        curve_dfs.append(base.rename(columns=rename).assign(Panel=panel))
        point_dfs.append(Regions.select_in(base, region) \
                         .rename(columns=rename).assign(Panel=panel))


    curve_df = pd.concat(curve_dfs, axis=0, ignore_index=True)
    point_df = pd.concat(point_dfs, axis=0, ignore_index=True)

    curve_df = curve_df[["x", "y", "Frequency", "Panel", "Sample Name"] + meta_cols]
    point_df = point_df[["x", "y", "Panel", "Point", "Sample Name"] + meta_cols]

    return curve_df, point_df, region

def regions(
    exp: Experiment | list[Experiment],
    title: str | None = None,
    **kwargs,
) -> PlotResult:
    # Disallow grouping args
    if not {"hue", "tile", "style", "size", "col", "row"}.isdisjoint(kwargs): 
        raise ValueError("No data grouping parameter allowed!")
    
    curve_data, point_data, region_data = _prepare_regions_dfs(exp, kwargs)
    
    config = kwargs | {
        "x": "x", "y": "y",
        "row": "Sample Name", "col": "Panel", "style": "Point",
        "col_order": ["Nyquist", "Tangent Angle", "Tangent Angle Derivative"],
        "edgecolor": "white",
        "facecolor": "#292929",
        "s": 80,
        "zorder": 5,
        "hue": None,
        "legend": True,
        "title": title,
        "facet_kws": {"sharex": False, "sharey": False},
    } 

    res = core.scatterplot(point_data, **config)
    with plt.rc_context(FIGURE_SETTINGS):
        with res as (fig, _):
            grid: FacetGrid = res.get_meta("grid")

            fig.set_layout_engine("tight")

            grid.set_titles("{col_name}") 
            core._improve_legend(grid, ncols=grid.data["Point"].nunique())

            # Prepare the config for the curves
            line_config = {
                "zorder": 1,
                "legend": False,
                "hue": "Sample Name",
                "x": "x", "y": "y"
            } | DEFAULT_LINEPLOT_SETTINGS
            line_config |= core._prepare_palette(curve_data, line_config)

            # Add the curve graphs
            for (name, panel), ax in grid.axes_dict.items():
                data = curve_data[(curve_data["Sample Name"] == name) & (curve_data["Panel"] == panel)]
                sns.lineplot(data, ax=ax, **line_config)
            
            fmin, *_, fmax = np.sort(curve_data["x"].unique()) # Assume descending
            grid.map_dataframe(_decorate_facet, fspan=(fmin, fmax), curve_data=curve_data, region_data=region_data)

            fig.set_layout_engine("constrained")
    
    return res


