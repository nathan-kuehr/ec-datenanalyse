import numpy as np
import pandas as pd

from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from seaborn import FacetGrid

from . import core
from .plotresult import PlotResult
from ..analysis.analysis import Analysis
from ..analysis.drt import DRT
from ..config import LARGE_FIGURE_SIZE
from ..data.experiment import Experiment


_TAU_GRID_POINTS = 1000
_NEGATIVE_POLARIZATION_CLIP = -1


def _combine_drt_data_frames(
    data: list[Experiment], peaks_to_draw: list[float], kwargs: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(data) == 0:
        raise ValueError("Data list is empty.")

    combined_drt = pd.concat([exp.analysis.drt.data for exp in data], ignore_index=True)
    combined_peaks = pd.concat(
        [exp.analysis.drt.peak_select(peaks_to_draw) for exp in data],
        ignore_index=True,
    )

    hue_group = kwargs.get("hue") or kwargs.get("tile")
    if hue_group is None:
        kwargs["hue"] = "Experiment Name"
    elif hue_group != "Experiment Name":
        combined_drt[hue_group] = (
            combined_drt["Experiment Name"] + " - " + combined_drt[hue_group]
        )
        combined_peaks[hue_group] = (
            combined_peaks["Experiment Name"] + " - " + combined_peaks[hue_group]
        )

    return combined_drt, combined_peaks


def _draw_sampled_peaks(
    ax: Axes, drt_data: pd.DataFrame, peak_data: pd.DataFrame, kwargs: dict
) -> None:
    taus = drt_data["Time Constant"].to_numpy()
    tau_grid = np.logspace(np.log10(taus.min()), np.log10(taus.max()), _TAU_GRID_POINTS)
    sampled_peaks = DRT.sample_peak_data(peak_data, tau_grid)

    nsamples, npeaks, ntau = sampled_peaks.shape

    grouped = core._prepare_groupby(peak_data, kwargs)
    for line, (_, group) in zip(ax.lines, grouped):
        color = line.get_color()
        ngroup_samples = len(group) // npeaks

        # Get indices and convert to 2D indices of sample peak array
        group_indices = group.index.to_numpy()
        sample_indices, peak_indices = np.unravel_index(group_indices, (nsamples, npeaks))

        # Extract & average the gammas
        extracted = np.reshape(
            sampled_peaks[sample_indices, peak_indices, :], (ngroup_samples, npeaks, ntau)
        )
        gammas = np.mean(extracted, axis=0)

        # Extract & average the polarisations
        extracted = np.reshape(
            group["Polarisation"].to_numpy(), (ngroup_samples, npeaks)
        )
        polarisations = np.mean(extracted, axis=0)

        for gamma, pol in zip(gammas, polarisations):
            max_idx = np.argmax(gamma)

            ax.fill_between(tau_grid, gamma, alpha=0.3, color=color)
            ax.annotate(
                f"{pol:.3g} $\\Omega$",
                xy=(tau_grid[max_idx], gamma[max_idx]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                color=color,
            )


def drt(
    exp: Experiment | list[Experiment],
    title: str | None = None,
    peaks_to_draw: list[float] = [],
    **kwargs,
) -> PlotResult:
    if isinstance(exp, Experiment):
        data = exp.analysis.drt.data
        peak_data = exp.analysis.drt.peak_select(peaks_to_draw)
    elif isinstance(exp, list):
        data, peak_data = _combine_drt_data_frames(exp, peaks_to_draw, kwargs)
    else:
        raise TypeError("Unsupported data type passed!")

    kwargs["errorbar"] = None

    res = core.lineplot(
        data=data,
        x="Time Constant",
        y="Polarization Density",
        title=title,
        series_info=Analysis.SERIES_INFO,
        **kwargs,
    )
    with res as (fig, ax):
        if (grid := res.get_meta("grid")) is not None:
            assert isinstance(grid, FacetGrid)

            def _draw_facet_sampled_peaks(data: pd.DataFrame, **_) -> None:
                samples = data["Sample Name"].unique()
                facet_peak_data = peak_data[peak_data["Sample Name"].isin(samples)].reset_index(drop=True)
                _draw_sampled_peaks(plt.gca(), data, facet_peak_data, kwargs)

            if len(peaks_to_draw) > 0:
                fig.set_layout_engine("tight")
                grid.map_dataframe(_draw_facet_sampled_peaks)
                fig.set_layout_engine("constrained")

        else:
            assert isinstance(ax, Axes)
            fig.set_size_inches(LARGE_FIGURE_SIZE)

            if len(peaks_to_draw) > 0:
                _draw_sampled_peaks(ax, data, peak_data, kwargs)

        # Cut off negative polarisation
        for ax in fig.axes:
            ymin, ymax = ax.get_ylim()
            if ymin < _NEGATIVE_POLARIZATION_CLIP:
                ax.set_ylim((_NEGATIVE_POLARIZATION_CLIP, ymax))

    return res
