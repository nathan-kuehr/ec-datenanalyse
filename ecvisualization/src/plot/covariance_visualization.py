from itertools import combinations
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from matplotlib.colors import hex2color
from scipy.stats import chi2
from scipy.linalg import logm, expm
from shapely.geometry import LineString

from scipy.interpolate import interp1d
from scipy.spatial import geometric_slerp

from ..config import COVVIS_ANGLE_STEPS, COVVIS_INTERPOLATION_POINTS


class CovarianceVisualization:
    def __init__(
        self, data: pd.DataFrame, errorbar, real: str = "Offset-Corrected Resistance"
    ) -> None:
        self.__nF = int(data["Frequency"].nunique())
        self.__N = int(data["Name"].nunique())

        grouped = data.groupby("Frequency")[[real, "Neg. Reactance"]]

        self.__freqs = np.array(list(grouped.groups.keys()))

        if self.__N == 1:
            self.__covs = np.zeros((self.__nF, 2, 2))
        else:
            self.__covs = (
                (grouped.cov() / self.__N).to_numpy().reshape((self.__nF, 2, 2))  # pyright: ignore
            )  # pyright: ignore

            # Scale covariances to desired confidence interval, such that they represent the CI ellipse
            self.__covs *= chi2.ppf(self.ciFromErrorbarSpec(errorbar), df=2)

        self.__positions = grouped.mean().to_numpy()

    def interpCov(self, points: np.ndarray) -> np.ndarray:
        ipCovs = []
        for p in points:
            if int(p) == p:
                ipCovs.append(self.__covs[int(p), :, :])
            else:
                sigma1 = logm(self.__covs[int(np.floor(p)), :, :])
                sigma2 = logm(self.__covs[int(np.ceil(p)), :, :])
                t = p - np.floor(p)
                ipCovs.append(expm(sigma1 + t * (sigma2 - sigma1)))  # pyright: ignore
        return np.array(ipCovs)

    def hull2(self, ax) -> np.ndarray:
        alpha = np.linspace(
            0, self.__nF - 1, (self.__nF - 1) * COVVIS_INTERPOLATION_POINTS + 1
        )

        # Interpolate positions and covariances
        positions = interp1d(np.arange(self.__nF), self.__positions, axis=0)(alpha)
        covs = self.interpCov(alpha)

        # Gradients and normals
        grad = np.gradient(positions, axis=0)
        normals = (
            np.stack([-grad[:, 1], grad[:, 0]], axis=1)
            / np.linalg.norm(grad, axis=1)[:, None]
        )

        # Angular changes between normals
        normalDotProds = np.clip(
            np.einsum("ij,ij->i", normals[1:], normals[:-1]), -1, 1
        )

        # Needed steps to add for angular resolution
        deltaChanges = np.ceil(
            np.degrees(np.arccos(normalDotProds)) / COVVIS_ANGLE_STEPS
        ).astype(int)
        deltaChanges[deltaChanges < 1] = 1

        # 2D cross products to determine rotation direction
        normalCrossProds = (
            normals[:-1, 0] * normals[1:, 1] - normals[:-1, 1] * normals[1:, 0]
        )

        # Create the base directions in which to evaluate maximal extents
        baseDirectionsLeft = []
        baseDirectionsRight = []

        for i, N in enumerate(deltaChanges):
            n0 = normals[i]

            if N == 1:
                baseDirectionsLeft.append(n0[None, :])
                baseDirectionsRight.append(n0[None, :])
            else:
                n1 = normals[i + 1]
                rotDir = 1 if normalCrossProds[i] < 0 else -1

                slerped = geometric_slerp(n0, n1, np.linspace(0, 1, N, endpoint=False))
                baseDirectionsLeft.append(slerped[::rotDir])
                baseDirectionsRight.append(slerped[::-rotDir])

        baseDirectionsLeft.append(normals[-1][None, :])
        baseDirectionsLeft = np.concatenate(baseDirectionsLeft, axis=0)

        baseDirectionsRight.append(normals[-1][None, :])
        baseDirectionsRight = -np.concatenate(baseDirectionsRight, axis=0)

        deltaChanges = np.append(deltaChanges, 1)  # For last point

        # Prepare base points and covariances
        basePositions = np.repeat(positions, deltaChanges, axis=0)
        baseCovs = np.repeat(covs, deltaChanges, axis=0)

        # Push angular steps to real data points
        baseIsMultiplied = np.repeat(deltaChanges > 1, deltaChanges, axis=0)
        baseIsRealData = np.repeat(
            np.arange(len(positions)) % COVVIS_INTERPOLATION_POINTS == 0,
            deltaChanges,
            axis=0,
        )
        baseNextRDPos = np.repeat(
            np.ceil(np.arange(len(positions)) / COVVIS_INTERPOLATION_POINTS),
            deltaChanges,
            axis=0,
        ).astype(int)

        mask = baseIsMultiplied & ~baseIsRealData
        basePositions[mask] = self.__positions[baseNextRDPos[mask]]
        baseCovs[mask] = self.__covs[baseNextRDPos[mask]]

        sigmanLeft = np.einsum("ijk,ik->ij", baseCovs, baseDirectionsLeft)
        scaleLeft = np.sqrt(np.einsum("ij,ij->i", baseDirectionsLeft, sigmanLeft))[
            :, None
        ]
        sigmanRight = np.einsum("ijk,ik->ij", baseCovs, baseDirectionsRight)
        scaleRight = np.sqrt(np.einsum("ij,ij->i", baseDirectionsRight, sigmanRight))[
            :, None
        ]

        leftHull = basePositions + sigmanLeft / scaleLeft
        rightHull = basePositions + sigmanRight / scaleRight

        # Remove loops
        def cleanLoops(hull):
            mask = np.ones(len(hull), dtype=bool)
            vecs = [
                LineString([hull[i, :], hull[i + 1, :]]) for i in range(len(hull) - 1)
            ]
            for (i, vecA), (j, vecB) in combinations(enumerate(vecs), 2):
                if vecA.intersects(vecB) and i != j - 1:
                    mask[i + 1 : j + 1] = False
            return hull[mask]

        return np.vstack([cleanLoops(leftHull), cleanLoops(rightHull)[::-1]])

    @property
    def ellipseParameters(self) -> np.ndarray:
        ellipses = []
        for cov in self.__covs:
            eigvals, eigvecs = np.linalg.eigh(cov)
            axes = np.sqrt(eigvals)  # -> covs are already scaled to CI
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            ellipses.append((axes[0], axes[1], angle))
        return np.array(ellipses)

    @classmethod
    def ciFromErrorbarSpec(cls, errorbar) -> float:
        if isinstance(errorbar, tuple) and errorbar[0] == "ci":
            return errorbar[1] / 100
        elif isinstance(errorbar, float):
            return errorbar / 100
        elif errorbar == "ci":
            return 0.95
        else:
            raise ValueError(
                "Unsupported errorbar specification for covariance visualization"
            )

    def draw(self, ax: Axes, color="#808080") -> None:
        color = list(hex2color(color))
        edgeColor = color + [0.5]
        faceColor = color + [0.2]

        for pos, (a, b, angle) in zip(self.__positions, self.ellipseParameters):
            ax.add_patch(
                Ellipse(
                    tuple(pos),
                    width=2 * a,
                    height=2 * b,
                    angle=angle,
                    edgecolor=edgeColor,
                    facecolor=faceColor,
                    linestyle="-.",
                )
            )
