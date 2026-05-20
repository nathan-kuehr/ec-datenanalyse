import numpy as np
import pandas as pd

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from typing import Callable

from . import core, kkt_plots
from .basics import _combine_experiment_data, _listify, bode, fresponse

from .kkt_plots import _residuals_plot, _residual_distribution_plot, residuals, residual_distribution
from .nyquist_plot import nyquist
from .plotresult import PlotResult
from ..config import DataSeriesInfo
from ..data.experiment import Experiment
from ..analysis.fitting import Fit


_FIT_FREQ_GRID_POINTS = 1000
_FIT_LABEL_SUFFIX = " (Fit)"
_FIT_LINESTYLE = (0, (5, 2))
_FIT_PHANTOM_COLOR = "0.3"
_FIT_LEGEND_LABEL = "Fit"

_OVERLAY_KINDS = (nyquist, bode, fresponse)
_RESIDUAL_KINDS = (residuals, residual_distribution)


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


# ===================== show_fit helpers =====================


def _resolve_hue_column(kwargs: dict) -> str:
    return kwargs.get("hue") or "Experiment Name"


def _is_only_tile_grouping(kwargs: dict) -> bool:
    """True iff `tile` is set and no other grouping (hue/style/size) is."""
    return kwargs.get("tile") is not None and not any(
        kwargs.get(k) is not None for k in ("hue", "style", "size")
    )


def _build_fit_palette(real_data: pd.DataFrame, hue_col: str) -> dict[str, str]:
    """Map hue values (real and sim) onto the same shade per Palette group."""
    palette: dict[str, str] = {}
    for pal, group in real_data.groupby("Palette", sort=False):
        hue_vals = group[hue_col].drop_duplicates().tolist()
        shades = pal.shade(len(hue_vals))  # pyright: ignore
        for val, shade in zip(hue_vals, shades):
            palette[str(val)] = shade
            palette[f"{val}{_FIT_LABEL_SUFFIX}"] = shade
    return palette


def _tag_sim_data(sims: list[Experiment], hue_col: str) -> None:
    """Append the fit suffix to the hue column of each sim experiment in place."""
    for sim in sims:
        sim.data[hue_col] = sim.data[hue_col].astype(str) + _FIT_LABEL_SUFFIX


def _iter_plot_axes(fig: Figure) -> list[Axes]:
    """Return all axes containing plotted data (skipping legend-only axes)."""
    return [ax for ax in fig.axes if ax.has_data() or ax.lines or ax.collections]


def _restyle_fit_lines(fig: Figure) -> None:
    """Hide markers and apply dashed style on lines whose label carries the fit suffix."""
    for ax in _iter_plot_axes(fig):
        lines = list(ax.lines)
        for child in ax.child_axes:
            lines += list(child.lines)

        for line in lines:
            label = line.get_label() or ""
            if isinstance(label, str) and label.endswith(_FIT_LABEL_SUFFIX):
                line.set_markersize(0)
                line.set_linestyle(_FIT_LINESTYLE)


def _replace_fit_legend(fig: Figure) -> None:
    """Drop fit-suffixed entries from each legend; add a single "Fit" phantom."""
    phantom = Line2D(
        [0], [0],
        color=_FIT_PHANTOM_COLOR,
        linestyle=_FIT_LINESTYLE,
        linewidth=2.0,
        marker="",
    )

    targets: list[tuple[Axes | Figure, object]] = []
    for ax in fig.axes:
        if (leg := ax.get_legend()) is not None:
            targets.append((ax, leg))
    for leg in fig.legends:
        targets.append((fig, leg))

    for target, legend in targets:
        handles_keep = []
        labels_keep = []
        for handle, text in zip(legend.legend_handles, legend.get_texts()):  # pyright: ignore
            label = text.get_text()
            if not label.endswith(_FIT_LABEL_SUFFIX):
                handles_keep.append(handle)
                labels_keep.append(label)

        handles_keep.append(phantom)
        labels_keep.append(_FIT_LEGEND_LABEL)

        title_text = legend.get_title().get_text() or None  # pyright: ignore
        loc = getattr(legend, "_loc", "best")
        target.legend(handles=handles_keep, labels=labels_keep, title=title_text, loc=loc)


def _show_fit_residuals(
    exps: list[Experiment],
    kind: Callable,
    title: str | None,
    series_info: dict[str, DataSeriesInfo],
    min_bound: float,
    include_stats: bool,
    kwargs: dict,
) -> PlotResult:
    extractor: Callable[[Experiment], pd.DataFrame] = lambda e: e.analysis.fit.data
    if kind is residuals:
        return _residuals_plot(exps, extractor, title, kwargs)
    return _residual_distribution_plot(
        exps, extractor, title, min_bound, include_stats, kwargs
    )


def show_fit(
    exps: Experiment | list[Experiment],
    kind: Callable,
    title: str | None = None,
    series_info: dict[str, DataSeriesInfo] = {},
    **kwargs,
) -> PlotResult:
    
    if kwargs.get("style") is not None:
        raise ValueError("Cannot passe 'style' argument to show_fit - this data discrimination is reserved for internal use!")

    if kind in (residuals, residual_distribution):
        extractor = lambda e: e.analysis.fit.data

        if kind is residuals:
            return kkt_plots._residuals_plot(exps, extractor, title, kwargs)
        else:
            return kkt_plots._residual_distribution_plot(exps, extractor, title, kwargs.pop("min_bound", kkt_plots._MIN_RESIDUAL_BOUND), kwargs.pop("include_stats", True), kwargs)
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

        # Add differentiation via the linestyle
        kwargs |= {
            "style": "Data Origin",
            "dashes": {"Measured": "", "Fitted": (5, 2)},
            "markers": {"Measured": "o", "Fitted": ","},
        }

        return kind(exps + sims, title=title, series_info=series_info, **kwargs)

    else:
        raise ValueError("Unsupported plotting function passed as 'kind'!")



    # # When tile is the only grouping, facet titles already name the groups
    # if "legend" not in kwargs and _is_only_tile_grouping(kwargs):
    #     kwargs["legend"] = False

    # # Force an explicit hue column. Without this, e.g. bode would default
    # # hue=tile_col, collapsing real+sim into a single line per facet.
    # hue_col = _resolve_hue_column(kwargs)
    # kwargs["hue"] = hue_col

    # # Span the simulation grid across all experiment frequencies
    # all_freqs = np.concatenate([exp.data["Frequency"].unique() for exp in exps])
    # freq_grid = np.logspace(
    #     np.log10(all_freqs.min()),
    #     np.log10(all_freqs.max()),
    #     _FIT_FREQ_GRID_POINTS,
    # )

    # sims = [exp.analysis.fit.simulate_experiment(freq_grid) for exp in exps]

    # # Force sim into its own hue group via suffix; share colors via explicit palette
    # real_data = pd.concat([exp.data for exp in exps], ignore_index=True)
    # _tag_sim_data(sims, hue_col)

    # kwargs.setdefault("palette", _build_fit_palette(real_data, hue_col))

    # res = kind(exps + sims, title=title, series_info=series_info, **kwargs)

    # with res as (fig, _):
    #     _restyle_fit_lines(fig)
    #     if kwargs.get("legend") is not False:
    #         _replace_fit_legend(fig)

    # return res
