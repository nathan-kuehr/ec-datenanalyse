import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from matplotlib.colors import hex2color
from scipy.stats import chi2
from scipy.linalg import logm, expm

from scipy.interpolate import interp1d

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
            self.__covs = np.ones((self.__nF, 2, 2)) * np.nan
        else:
            self.__covs = (
                (grouped.cov() / self.__N).to_numpy().reshape((self.__nF, 2, 2))  # pyright: ignore
            )  # pyright: ignore

            # Scale covariances to desired confidence interval, such that they represent the CI ellipse
            self.__covs *= chi2.ppf(self.ciFromErrorbarSpec(errorbar), df=2)

        self.__positions = grouped.mean().to_numpy()

    @property
    def N(self) -> int:
        return self.__N

    def interpCov(self, points: np.ndarray) -> np.ndarray:
        logCovs = [logm(cov) for cov in self.__covs]
        ipLogCovs = interp1d(np.arange(self.__nF), logCovs, axis=0)(points)
        return np.array([expm(cov) for cov in ipLogCovs])

    def cross(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        return p1[..., 0] * p2[..., 1] - p1[..., 1] * p2[..., 0]

    def slerp(self, n0, n1, t):
        alpha = np.arctan2(n0[1], n0[0])
        angle = np.arctan2(n1[1], n1[0]) - alpha
        if np.abs(angle + np.pi) < 1e-9:
            angle = np.pi

        theta = alpha + t * angle
        return np.stack((np.cos(theta), np.sin(theta)), axis=1)

    def findIntersections(self, hullpoints: np.ndarray) -> np.ndarray:
        N = len(hullpoints)

        # Idea:
        # Intersection if A->B crosses C->D, that is iff
        # A and B are on different sides of CD and
        # C and D are on different sides of AB
        # --> det(AB, AC) * det(AB, AD) < 0 and det(CD, CA) * det(CD, CB) < 0

        # i = vec index
        A = hullpoints[:-1, None, :]
        B = hullpoints[1:, None, :]

        # j = vec index
        C = hullpoints[None, :-1, :]
        D = hullpoints[None, 1:, :]

        AB = B - A  # -> shape (vecs, copies, 2)
        CD = D - C  # -> shape (copies, vecs, 2)

        # is vector i splitting vector j?
        isCDsplit = self.cross(AB, C - A) * self.cross(AB, D - A) < 0
        # is vector j splitting vector i?
        isABsplit = self.cross(CD, A - C) * self.cross(CD, B - C) < 0

        intersecMask = isCDsplit & isABsplit

        # Find all intersec indices
        rows, cols = np.where(intersecMask)
        farthest = np.full(N - 1, -1, dtype=int)
        np.maximum.at(farthest, rows, cols)
        farthest[farthest <= np.arange(N - 1)] = -1  # avoid backward intersections

        # Find largest intersection for each row
        from_ = np.arange(N - 1)[farthest > -1]
        to_ = farthest[farthest > -1]

        mask = np.zeros(N)
        np.add.at(mask, from_ + 1, 1)
        np.add.at(mask, to_ + 1, -1)

        mask = np.cumsum(mask) == 0

        return mask

    def hull(self, ax) -> np.ndarray:
        alpha = np.linspace(
            0, self.__nF - 1, (self.__nF - 1) * COVVIS_INTERPOLATION_POINTS + 1
        )

        # Interpolate positions and covariances
        positions = interp1d(np.arange(self.__nF), self.__positions, axis=0)(alpha)
        covs = self.interpCov(alpha)

        # Do the round trip for a closed hull, i.e. right -> left -> back
        positions = np.vstack([positions, positions[-2::-1]])
        covs = np.vstack([covs, covs[-2::-1]])

        # Gradients and normals
        grad = np.gradient(positions, axis=0)

        # Gradient vanishes at turning point so set it 90° turned
        turningPoint = (self.__nF - 1) * COVVIS_INTERPOLATION_POINTS
        prevGrad = grad[turningPoint - 1]
        grad[turningPoint] = [-prevGrad[1], prevGrad[0]]

        normals = (
            np.stack([grad[:, 1], -grad[:, 0]], axis=1)
            / np.linalg.norm(grad, axis=1)[:, None]
        )

        # Attribute the normals of the interpol points to the real data points
        # -> easier angular interpol
        realDataPoints = np.arange(2 * self.__nF - 1) * COVVIS_INTERPOLATION_POINTS
        normals[realDataPoints] = normals[realDataPoints - 1]

        normalDotProds = np.clip(
            np.einsum("ij,ij->i", normals[1:], normals[:-1]), -1, 1
        )
        normalCrossProds = self.cross(normals[:-1], normals[1:])

        rotdirs = np.where(normalCrossProds >= 0, 1, -1)
        angles = np.arccos(normalDotProds)

        # How many support vectors are at each point?
        multiplicity = np.ceil(angles / np.radians(COVVIS_ANGLE_STEPS)).astype(int)
        multiplicity[multiplicity < 1] = 1
        multiplicity[multiplicity > 1] += 1  # account for endpoints in angular sweep

        # Prepapre direction vectors
        dirvecs = []
        for i, N in enumerate(multiplicity):
            n0 = normals[i]

            if N == 1:
                dirvecs.append(n0[None, :])
            elif rotdirs[i] == 1:
                dirvecs.append(self.slerp(n0, normals[i + 1], np.linspace(0, 1, N)))
            else:
                dirvecs.append(np.repeat(n0[None, :], N, axis=0))
        dirvecs = np.concatenate(dirvecs, axis=0)

        # Prepare base positions and covariances
        basepos = np.repeat(positions[:-1], multiplicity, axis=0)
        basecovs = np.repeat(covs[:-1], multiplicity, axis=0)

        # Calculate hull points
        sigman = np.einsum("ijk,ik->ij", basecovs, dirvecs)
        scale = np.sqrt(np.einsum("ij,ij->i", dirvecs, sigman))[:, None]
        hullPoints = basepos + sigman / scale

        # Remove intersections
        turningPoint = np.sum(multiplicity[:turningPoint])
        mask = np.concatenate(
            [
                self.findIntersections(hullPoints[:turningPoint]),
                self.findIntersections(hullPoints[turningPoint:]),
            ]
        )

        # ax.plot(hullPoints[:, 0], hullPoints[:, 1], color="k", linewidth=1, marker='o', markersize=1)
        # ax.quiver(
        #     basepos[:, 0],
        #     basepos[:, 1],
        #     dirvecs[:, 0],
        #     dirvecs[:, 1],
        #     angles="xy",
        #     scale_units="xy",
        #     scale=10,
        #     width=0.002,
        #     color="k",
        #     alpha=1,
        # )
        # ax.quiver(
        #     basepos[:, 0],
        #     basepos[:, 1],
        #     (sigman / scale)[:, 0],
        #     (sigman / scale)[:, 1],
        #     angles="xy",
        #     scale_units="xy",
        #     scale=1,
        #     width=0.002,
        #     color="b",
        #     alpha=0.3,
        # )

        return hullPoints[mask]

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
