import numpy as np
import pandas as pd
from functools import singledispatch
from scipy.linalg import logm, expm

from .plotresult import PlotResult
from .plot import plot, combineDataFrames
from ..eis import EIS
from .covariance_visualization import CovarianceVisualization


def covInterpolation(sigma1: np.ndarray, sigma2: np.ndarray, t: float) -> np.ndarray:
    """
    Log Euclidean interpolation between two covariance matrices
    """
    log_sigma1 = logm(sigma1)[0]  # logm returns (matrix, info)
    log_sigma2 = logm(sigma2)[0]
    interpolated = (1 - t) * log_sigma1 + t * log_sigma2
    return expm(interpolated)[0]  # expm also returns (matrix, info)


@singledispatch
def nyquist(
    data: EIS,
    title: str | None = None,
    Rmin: float = 60,
    Rspan: float = 50,
    offsetCorrect: bool = True,
    **kwargs,
) -> PlotResult:
    return __nyquist_data(
        data.data,
        title=title,
        Rmin=Rmin,
        Rspan=Rspan,
        offsetCorrect=offsetCorrect,
        **kwargs,
    )


@nyquist.register(pd.DataFrame)
def __nyquist_data(
    data: pd.DataFrame,
    title: str | None = None,
    Rmin: float = 60,
    Rspan: float = 50,
    offsetCorrect: bool = True,
    **kwargs,
) -> PlotResult:
    config = {"x": "Resistance", "y": "Neg. Reactance"}

    kwargs.setdefault("errorbar", ("ci", 95))
    kwargs.setdefault("err_style", "band")

    # See if offset correction is desired
    if offsetCorrect:
        config["x"] = "Offset-Corrected Resistance"

    # Prepare data for mean & covs
    hueGroup = kwargs.get("hue", "Experiment Group")

    GroupingArgs = {"hue", "style", "size"}
    grouping = ["Frequency"] + [
        kwargs.get(arg) for arg in GroupingArgs if arg in kwargs
    ]
    grouped = data.groupby(grouping)

    meanData = grouped.agg(
        {config["x"]: "mean", config["y"]: "mean", "Palette": "first"}
    ).reset_index()

    kwargsIntermed = kwargs.copy()
    kwargsIntermed["noSave"] = True

    with plot(meanData, **config, title=title, **kwargsIntermed) as (fig, axes):
        ax = axes[0]
        ax.set_xscale("linear")
        ax.set_yscale("linear")

        ax.set_xlim(left=Rmin, right=Rmin + Rspan)
        ax.set_ylim(bottom=0, top=Rspan)

        if kwargs.get("errorbar") is not None:
            grouped = data.groupby(hueGroup)
            for (_, group), line in zip(data.groupby(hueGroup), ax.lines):
                color = line.get_color()
                covVis = CovarianceVisualization(
                    group, kwargs.get("errorbar"), config["x"]
                )

                if covVis.N == 1:
                    continue  # No covariance to plot

                if kwargs.get("err_style") == "band":
                    hull = covVis.hull(ax)
                    ax.fill(
                        hull[:, 0],
                        hull[:, 1],
                        color=color,
                        alpha=0.1,
                        label="Hüllkurve",
                        zorder=1,
                    )
                elif kwargs.get("err_style") == "bars":
                    covVis.draw(ax, color=color)
                else:
                    raise ValueError(
                        f"Unknown err_style '{kwargs.get('err_style')}'. Supported styles are 'band' and 'bars'."
                    )

                line.set_zorder(2)  # Bring lines to front

    return PlotResult(title, fig, **kwargs)


@nyquist.register(list)
def __nyquist_multiple(
    data: list[EIS | pd.DataFrame],
    title: str | None = None,
    Rmin: float = 60,
    Rspan: float = 50,
    offsetCorrect: bool = True,
    **kwargs,
) -> PlotResult:
    if len(data) == 0:
        raise ValueError("Data list is empty.")

    combinedData = combineDataFrames(data, **kwargs)
    kwargs["hue"] = "Experiment Group"

    return nyquist(
        combinedData,
        title,
        Rmin=Rmin,
        Rspan=Rspan,
        offsetCorrect=offsetCorrect,
        **kwargs,
    )
