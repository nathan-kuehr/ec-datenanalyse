from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import expm, logm
from shapely.geometry import MultiPoint
from shapely.ops import unary_union

from ...config import COVVIS_ANGLE_STEPS, COVVIS_INTERPOLATION_POINTS


_DEGREES_PER_CIRCLE = 360
_THETA = np.linspace(0, 2 * np.pi, _DEGREES_PER_CIRCLE // COVVIS_ANGLE_STEPS)
UNIT_CIRCLE = np.stack((np.cos(_THETA), np.sin(_THETA)), axis=1)


def ellipse_parameters(positions: np.ndarray, covs: np.ndarray) -> list:
    """Decompose covariance matrices into (center, semi-axis a, semi-axis b, angle°)."""
    ellipses = []
    for pos, cov in zip(positions, covs):
        eigvals, eigvecs = np.linalg.eigh(cov)
        axes = 2 * np.sqrt(eigvals)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ellipses.append((tuple(pos), float(axes[0]), float(axes[1]), float(angle)))
    return ellipses


def interp_cov(covs: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Interpolate covariance matrices in log-space - better"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="logm result may be inaccurate")
        log_covs = [logm(cov) for cov in covs]
        interp_log_covs = interp1d(np.arange(len(covs)), log_covs, axis=0)(alpha)
        return np.array([expm(cov) for cov in interp_log_covs])


def hull(positions: np.ndarray, covs: np.ndarray) -> np.ndarray:
    nfreqs = len(covs)

    alpha = np.linspace(
        0, nfreqs - 1, (nfreqs - 1) * COVVIS_INTERPOLATION_POINTS + 1
    )
    interp_positions = interp1d(np.arange(nfreqs), positions, axis=0)(alpha)
    interp_covs = interp_cov(covs, alpha)

    cholesky = np.linalg.cholesky(interp_covs)
    scaled_unit_circle = np.einsum("nij,zj->nzi", cholesky, UNIT_CIRCLE)
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
