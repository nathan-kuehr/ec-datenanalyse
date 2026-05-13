import warnings
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from scipy.interpolate import interp1d
from scipy.linalg import expm, logm
from scipy.stats import chi2
from shapely.geometry import MultiPoint
from shapely.ops import unary_union

from .. import parallel
from ..config import COVVIS_ANGLE_STEPS, COVVIS_INTERPOLATION_POINTS


# Epsilon added to the diagonal to keep covariance matrices invertible
_SINGULARITY_EPSILON = 1e-12
_DEGREES_PER_CIRCLE = 360
_DEFAULT_ERRORBAR_SCALE = 95
_CHI2_DOF = 2


# Signature matched to parallel.multiprocess: method(sample, **arg).
# Here `freqs` corresponds to the `sample` input; `x`, `y`, `factor`,
# `calc_hull` come from `**arg`.
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
    covs += _SINGULARITY_EPSILON * np.eye(2)  # Fallback against singular matrices

    covs *= factor

    visualization = (
        CovarianceVisualization._hull
        if calc_hull
        else CovarianceVisualization._ellipse_parameters
    )
    return positions, covs, visualization(positions, covs)


class CovarianceVisualization:
    _THETA = np.linspace(0, 2 * np.pi, _DEGREES_PER_CIRCLE // COVVIS_ANGLE_STEPS)
    _UNIT_CIRCLE = np.stack((np.cos(_THETA), np.sin(_THETA)), axis=1)

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

    @staticmethod
    def _ellipse_parameters(positions: np.ndarray, covs: np.ndarray) -> list:
        ellipses = []
        for pos, cov in zip(positions, covs):
            eigvals, eigvecs = np.linalg.eigh(cov)
            axes = 2 * np.sqrt(eigvals)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            ellipses.append((tuple(pos), float(axes[0]), float(axes[1]), float(angle)))
        return ellipses

    @staticmethod
    def _interp_cov(covs: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="logm result may be inaccurate")
            log_covs = [logm(cov) for cov in covs]
            interp_log_covs = interp1d(np.arange(len(covs)), log_covs, axis=0)(alpha)
            return np.array([expm(cov) for cov in interp_log_covs])

    @staticmethod
    def _hull(positions: np.ndarray, covs: np.ndarray) -> np.ndarray:
        nfreqs = len(covs)

        alpha = np.linspace(
            0, nfreqs - 1, (nfreqs - 1) * COVVIS_INTERPOLATION_POINTS + 1
        )
        interp_positions = interp1d(np.arange(nfreqs), positions, axis=0)(alpha)
        interp_covs = CovarianceVisualization._interp_cov(covs, alpha)

        cholesky = np.linalg.cholesky(interp_covs)
        scaled_unit_circle = np.einsum("nij,zj->nzi", cholesky, CovarianceVisualization._UNIT_CIRCLE)
        x = scaled_unit_circle + interp_positions[:, None, :]

        pairs = [
            MultiPoint(np.vstack((x[i], x[i + 1]))).convex_hull
            for i in range(len(x) - 1)
        ]

        if (union := unary_union(pairs)).geom_type != "Polygon":
            raise RuntimeError(
                "The union of covariance pairs must always be connected!"
            )

        return np.array(union.exterior.coords)

    def draw_hull(self, ax: Axes, config: dict) -> None:
        if self.hull is not None:
            ax.fill(self.hull[:, 0], self.hull[:, 1], **config)

    def draw_ellipses(self, ax: Axes, config: dict) -> None:
        if self.ellipses is not None:
            for pos, a, b, angle in self.ellipses:
                ax.add_patch(Ellipse(pos, width=a, height=b, angle=angle, **config))

    @staticmethod
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
        elif measure == "se":
            return factor / nsamples
        else:
            raise ValueError(
                "Unsupported errorbar specification for parametric uncertainty visualization."
            )

    @staticmethod
    def calculate_covariances(
        groups: list[pd.DataFrame], kwargs: dict, calc_hull: bool = True
    ) -> list["CovarianceVisualization"]:
        x_col = kwargs["x"]
        y_col = kwargs["y"]
        errorbar = kwargs.get("errorbar", ("se", _DEFAULT_ERRORBAR_SCALE))

        # Inputs list: only contains the first argument (freqs)
        mp_inputs = []

        # Args list: contains dictionaries for the remaining arguments (**arg)
        mp_args = []

        for group in groups:
            nsamples = group["Sample Name"].nunique()
            factor = CovarianceVisualization._cov_scaling_factor(errorbar, nsamples)

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
