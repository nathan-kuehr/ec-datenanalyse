import numpy as np
import pandas as pd

from matplotlib.axes import Axes

from . import core
from .plotresult import PlotResult
from ..analysis.analysis import Analysis
from ..analysis.drt import DRT
from ..config import LARGE_FIGURE_SIZE
from ..data.experiment import Experiment


def _combine_drt_data_frames(
    data: list[Experiment], peaks_to_draw, kwargs: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(data) == 0:
        raise ValueError("Data list is empty.")

    combined_drt = pd.concat([exp.analysis.drt.data for exp in data], ignore_index=True)
    combined_peaks = pd.concat(
        [exp.analysis.drt.peak_select(peaks_to_draw) for exp in data], ignore_index=True
    )

    if (hue_group := kwargs.get("hue", None)) is not None:
        # Multiple data sets and hue differentiation
        combined_drt[hue_group] = (
            combined_drt["Experiment Name"] + " - " + combined_drt[hue_group]
        )
        combined_peaks[hue_group] = (
            combined_peaks["Experiment Name"] + " - " + combined_peaks[hue_group]
        )
    else:
        kwargs["hue"] = "Experiment Name"

    return combined_drt, combined_peaks


def _draw_sampled_peaks(
    ax: Axes, drt_data: pd.DataFrame, peak_data: pd.DataFrame, kwargs: dict
):
    # Sample the peaks
    taus = drt_data["Time Constant"].to_numpy()
    tau_grid = np.logspace(np.log10(taus.min()), np.log10(taus.max()), 1000)
    sampled_peaks = DRT.Sample_Peak_Data(peak_data, tau_grid)

    # Get dimension sizes
    nsamples, npeaks, ntau = sampled_peaks.shape

    grouped = core._prepare_groupby(peak_data, kwargs)
    for line, (_, group) in zip(ax.lines, grouped):
        # Get color for later
        c = line.get_color()
        ngroup_samples = len(group) // npeaks

        # Get indices and convert to 2D indices of sample peak array
        group_idc = group.index.to_numpy()
        sample_idc, peak_idc = np.unravel_index(group_idc, (nsamples, npeaks))

        # Extract & average the gammas
        extracted = np.reshape(
            sampled_peaks[sample_idc, peak_idc, :], (ngroup_samples, npeaks, ntau)
        )
        gammas = np.mean(extracted, axis=0)

        # Extract & average the polarisations
        extracted = np.reshape(
            group["Polarisation"].to_numpy(), (ngroup_samples, npeaks)
        )
        polarisations = np.mean(extracted, axis=0)

        # Draw
        for gamma, pol in zip(gammas, polarisations):
            max_idx = np.argmax(gamma)

            ax.fill_between(tau_grid, gamma, alpha=0.3, color=c)
            ax.annotate(
                f"{pol:.3g} $\\Omega$",
                xy=(tau_grid[max_idx], gamma[max_idx] / 4),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                color=c,
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

    res = core.lineplot(
        data,
        "Time Constant",
        "Polarization Density",
        title,
        Analysis.Series_Info,
        **kwargs,
    )
    with res as (fig, ax):
        assert isinstance(ax, Axes)
        fig.set_size_inches(LARGE_FIGURE_SIZE)

        if len(peaks_to_draw) > 0:
            _draw_sampled_peaks(ax, data, peak_data, kwargs)

        # Cut off negative polarisation
        ymin, ymax = ax.get_ylim()
        if ymin < -1:
            ax.set_ylim((-1, ymax))

    return res
