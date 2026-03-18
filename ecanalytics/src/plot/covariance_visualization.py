import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from scipy.linalg import logm, expm

from shapely.geometry import MultiPoint
from shapely.ops import unary_union

from scipy.interpolate import interp1d

from ..config import COVVIS_ANGLE_STEPS, COVVIS_INTERPOLATION_POINTS


class CovarianceVisualization:
    __Theta = np.linspace(0, 2 * np.pi, 360 // COVVIS_ANGLE_STEPS)
    __Z = np.stack((np.cos(__Theta), np.sin(__Theta)), axis=1)

    def __init__(self, data: pd.DataFrame, kwargs: dict) -> None:
        grouped = data.groupby("Frequency")[[kwargs["x"], kwargs["y"]]]
        means = grouped.mean()

        # Freqs & base points
        self.__freqs = means.index.to_numpy()
        self.__positions = means.to_numpy()

        # Data point numbers
        self.__n_freqs = len(self.__freqs)
        self.__n_samples = data["Sample Name"].nunique()

        if self.__n_samples > 1:
            covs = grouped.cov().to_numpy().reshape(self.__n_freqs, 2, 2)  # pyright: ignore

            # Scale w/ appropriate factor to represent desired CI
            self.__covs = covs * self.__cov_scaling_factor_from_errorbar_spec(kwargs)
        else:
            self.__covs = np.full((self.__n_freqs, 2, 2), np.nan)

        self.__hull = None
        self.__ellipses = None

    def __cov_scaling_factor_from_errorbar_spec(self, kwargs: dict) -> float:
        errorbar = kwargs["errorbar"]

        measure, scale = ("se", 95)

        if isinstance(errorbar, tuple) and len(errorbar) == 2:
            measure, scale = errorbar
        elif isinstance(errorbar, (int, float)):
            scale = errorbar
        elif isinstance(errorbar, str):
            measure = errorbar
        else:
            raise ValueError(
                "Unsupported errorbar specification for parametric uncertainty (covariance) visualization."
            )

        factor = float(chi2.ppf(scale / 100, df=2))

        if measure == "sd":
            return factor
        elif measure == "se":
            return factor / self.__n_samples
        else:
            raise ValueError(
                "Unsupported errorbar specification for parametric uncertainty (covariance) visualization."
            )

    def __interp_cov(self, alpha: np.ndarray) -> np.ndarray:
        logCovs = [logm(cov) for cov in self.__covs]
        ipLogCovs = interp1d(np.arange(self.__n_freqs), logCovs, axis=0)(alpha)
        return np.array([expm(cov) for cov in ipLogCovs])

    def _hull(self) -> np.ndarray:
        alpha = np.linspace(
            0,
            self.__n_freqs - 1,
            (self.__n_freqs - 1) * COVVIS_INTERPOLATION_POINTS + 1,
        )

        # Interpolate the covariance matrices
        pos = interp1d(np.arange(self.__n_freqs), self.__positions, axis=0)(alpha)
        covs = self.__interp_cov(alpha)

        # Sample the border of the ellipse:
        # Cholesky: LL^T = Σ
        # Ellipse border: (x-c)^T Σ^-1 (x-c) = 1
        # --> <z, z> = 1    with   z = L^-1(x-c)
        # --> x = Lz + c
        L = np.linalg.cholesky(covs)
        Lz = np.einsum("nij,zj->nzi", L, self.__Z)
        x = Lz + pos[:, None, :]

        # Add neighbouring ellipses
        pairs = [
            MultiPoint(np.vstack((x[i], x[i + 1]))).convex_hull
            for i in range(len(x) - 1)
        ]

        # Unite all together
        if (union := unary_union(pairs)).geom_type != "Polygon":
            raise RuntimeError(
                "The union of covariance pairs must always be connected!"
            )

        return np.array(union.exterior.coords)  # pyright: ignore

    def _ellipse_parameters(self) -> list:
        ellipses = []
        for pos, cov in zip(self.__positions, self.__covs):
            eigvals, eigvecs = np.linalg.eigh(cov)
            axes = 2 * np.sqrt(eigvals)  # -> covs are already scaled to CI
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            ellipses.append((tuple(pos), axes[0], axes[1], angle))
        return ellipses

    def draw_hull(self, ax: Axes, config: dict) -> None:
        if self.__hull is None:
            self.__hull = self._hull()
        ax.fill(self.__hull[:, 0], self.__hull[:, 1], **config)

    def draw_ellipses(self, ax: Axes, config: dict) -> None:
        if self.__ellipses is None:
            self.__ellipses = self._ellipse_parameters()
        for pos, a, b, angle in self.__ellipses:
            ax.add_patch(Ellipse(pos, width=a, height=b, angle=angle, **config))
