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

from ..analysis import parallel
from ..config import COVVIS_ANGLE_STEPS, COVVIS_INTERPOLATION_POINTS


# --- 1. GECACHTE BERECHNUNGSFUNKTION ---
# Signatur exakt abgestimmt auf: method(sample, **arg)
# freqs = sample (aus input)
# x, y, factor, calc_hull = **arg (aus args)
@parallel.Cache.cache
def _cached_covariance_calculation(
    freqs: np.ndarray, x: np.ndarray, y: np.ndarray, factor: float, calc_hull: bool
):
    data = pd.DataFrame({"f": freqs, "x": x, "y": y})
    grouped = data.groupby("f", sort=False)[["x", "y"]]

    poss = grouped.mean().to_numpy()

    nfreqs = data["f"].nunique()
    nsamples = len(data) // nfreqs

    if nsamples <= 1:
        return poss, np.full((nfreqs, 2, 2), np.nan), None

    covs = grouped.cov().to_numpy().reshape(nfreqs, 2, 2)
    covs += 1e-12 * np.eye(2)  # Fallback gegen singuläre Matrizen

    covs *= factor

    visualization = (
        CovarianceVisualization._Hull
        if calc_hull
        else CovarianceVisualization._Ellipse_Parameters
    )
    return poss, covs, visualization(poss, covs)


# --- 2. VISUALISIERUNGS- UND STEUERUNGSKLASSE ---
class CovarianceVisualization:
    _Theta = np.linspace(0, 2 * np.pi, 360 // COVVIS_ANGLE_STEPS)
    _Z = np.stack((np.cos(_Theta), np.sin(_Theta)), axis=1)

    def __init__(
        self, poss: np.ndarray, covs: np.ndarray, vis: np.ndarray | list | None
    ) -> None:
        self.positions = poss
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
    def _Ellipse_Parameters(poss: np.ndarray, covs: np.ndarray) -> list:
        ellipses = []
        for pos, cov in zip(poss, covs):
            eigvals, eigvecs = np.linalg.eigh(cov)
            axes = 2 * np.sqrt(eigvals)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            ellipses.append((tuple(pos), float(axes[0]), float(axes[1]), float(angle)))
        return ellipses

    @staticmethod
    def _Interp_Cov(covs: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="logm result may be inaccurate")
            log_covs = [logm(cov) for cov in covs]
            interp_log_covs = interp1d(np.arange(len(covs)), log_covs, axis=0)(alpha)
            return np.array([expm(cov) for cov in interp_log_covs])

    @staticmethod
    def _Hull(poss: np.ndarray, covs: np.ndarray) -> np.ndarray:
        nfreqs = len(covs)

        alpha = np.linspace(
            0, nfreqs - 1, (nfreqs - 1) * COVVIS_INTERPOLATION_POINTS + 1
        )
        interp_poss = interp1d(np.arange(nfreqs), poss, axis=0)(alpha)
        interp_covs = CovarianceVisualization._Interp_Cov(covs, alpha)

        L = np.linalg.cholesky(interp_covs)
        Lz = np.einsum("nij,zj->nzi", L, CovarianceVisualization._Z)
        x = Lz + interp_poss[:, None, :]

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
    def _cov_scaling_factor(errorbar: Any, n_samples: int) -> float:
        measure, scale = ("se", 95)

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

        factor = float(chi2.ppf(scale / 100, df=2))

        if measure == "sd":
            return factor
        elif measure == "se":
            return factor / n_samples
        else:
            raise ValueError(
                "Unsupported errorbar specification for parametric uncertainty visualization."
            )

    @staticmethod
    def Calculate_Covariances(
        groups: list[pd.DataFrame], kwargs: dict, calc_hull: bool = True
    ):
        x_col = kwargs["x"]
        y_col = kwargs["y"]
        errorbar = kwargs.get("errorbar", ("se", 95))

        # 1. input-Liste: Enthält NUR das erste Argument (freqs)
        mp_inputs = []

        # 2. args-Liste: Enthält die Dictionaries für die restlichen Argumente (**arg)
        mp_args = []

        for group in groups:
            n_samples = group["Sample Name"].nunique()
            factor = CovarianceVisualization._cov_scaling_factor(errorbar, n_samples)

            # Befüllen der exakt getrennten Parameterlisten
            mp_inputs.append(group["Frequency"].to_numpy())

            mp_args.append(
                {
                    "x": group[x_col].to_numpy(),
                    "y": group[y_col].to_numpy(),
                    "factor": factor,
                    "calc_hull": calc_hull,
                }
            )

        # Aufruf exakt passend zur unveränderten 'multiprocess'-Logik
        raw_results = parallel.multiprocess(
            method=_cached_covariance_calculation,
            input=mp_inputs,
            args=mp_args,
            tqdm_note="Calculating covariances...",
        )

        return [CovarianceVisualization(*res) for res in raw_results]