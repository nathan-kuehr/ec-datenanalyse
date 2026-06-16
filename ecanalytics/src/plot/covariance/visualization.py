from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from ... import parallel
from . import geometry


# Epsilon added to the diagonal to keep covariance matrices invertible
_SINGULARITY_EPSILON = 1e-12
_DEFAULT_ERRORBAR_SCALE = 95
_CHI2_DOF = 2


@parallel.CACHE.cache
def _cached_covariance_calculation(
    freqs: np.ndarray, x: np.ndarray, y: np.ndarray, factor: float, calc_hull: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray | list | None]:
    data = pd.DataFrame({"f": freqs, "x": x, "y": y})
    grouped = data.groupby("f", sort=False)[["x", "y"]]

    positions = grouped.mean().to_numpy()

    nfreqs = data["f"].nunique()
    nsamples = len(data) // nfreqs

    if nsamples <= 1:
        return positions, np.full((nfreqs, 2, 2), np.nan), None

    covs = grouped.cov().to_numpy().reshape(nfreqs, 2, 2)
    covs = covs + _SINGULARITY_EPSILON * np.eye(2)  # Fallback against singular matrices

    covs = covs * factor

    visualization = geometry.hull if calc_hull else geometry.ellipse_parameters
    return positions, covs, visualization(positions, covs)


def _cov_scaling_factor(errorbar: Any, nsamples: int) -> float:
    measure, scale = ("se", _DEFAULT_ERRORBAR_SCALE)

    if isinstance(errorbar, tuple) and len(errorbar) == 2:
        measure, scale = errorbar
    elif isinstance(errorbar, (int, float)):
        scale = errorbar
    elif isinstance(errorbar, str):
        measure = errorbar
    else:
        raise ValueError(
            "Unsupported errorbar specification for parametric uncertainty visualization."
        )

    factor = float(chi2.ppf(scale / 100, df=_CHI2_DOF))

    if measure == "sd":
        return factor
    if measure == "se":
        return factor / nsamples

    raise ValueError(
        "Unsupported errorbar specification for parametric uncertainty visualization."
    )


class CovarianceVisualization:
    def __init__(
        self,
        positions: np.ndarray,
        covs: np.ndarray,
        vis: np.ndarray | list | None,
    ) -> None:
        self.positions = positions
        self.covs = covs

        if isinstance(vis, np.ndarray):
            self.hull = vis
            self.ellipses = None
        elif isinstance(vis, list):
            self.hull = None
            self.ellipses = vis
        else:
            self.hull = None
            self.ellipses = None

    def draw_hull(self, ax: Axes, config: dict) -> None:
        if self.hull is not None:
            ax.fill(self.hull[:, 0], self.hull[:, 1], **config)

    def draw_ellipses(self, ax: Axes, config: dict) -> None:
        if self.ellipses is not None:
            for pos, a, b, angle in self.ellipses:
                ax.add_patch(Ellipse(pos, width=a, height=b, angle=angle, **config))

    @staticmethod
    def calculate_covariances(
        groups: list[pd.DataFrame], kwargs: dict, calc_hull: bool = True
    ) -> list["CovarianceVisualization"]:
        x_col = kwargs["x"]
        y_col = kwargs["y"]
        errorbar = kwargs.get("errorbar", ("se", _DEFAULT_ERRORBAR_SCALE))

        mp_inputs = []
        mp_args = []

        for group in groups:
            nsamples = group["Sample Name"].nunique()
            factor = _cov_scaling_factor(errorbar, nsamples)

            mp_inputs.append(group["Frequency"].to_numpy())
            mp_args.append(
                {
                    "x": group[x_col].to_numpy(),
                    "y": group[y_col].to_numpy(),
                    "factor": factor,
                    "calc_hull": calc_hull,
                }
            )

        raw_results = parallel.multiprocess(
            method=_cached_covariance_calculation,
            inputs=mp_inputs,
            args=mp_args,
            tqdm_note="Calculating covariances...",
        )

        return [CovarianceVisualization(*res) for res in raw_results]
