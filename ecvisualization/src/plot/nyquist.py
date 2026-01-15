import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import singledispatch

from scipy.interpolate import make_interp_spline
from scipy.stats import chi2
from scipy.linalg import logm, expm

from matplotlib.patches import Ellipse
from matplotlib.axes import Axes

from .plotresult import PlotResult
from .plot import plot, combineDataFrames
from ..eis import EIS

from matplotlib.colors import hex2color, rgb2hex

class CovarianceVisualization:
    @classmethod
    def calculate(cls, data: pd.DataFrame, real: str = "Offset-Corrected Resistance", imag: str = "Neg. Reactance"):
        nF = data["Frequency"].nunique()
        N = data["Name"].nunique()

        grouped = data.groupby("Frequency")[[real, imag]]

        f = np.array(list(grouped.groups.keys()))

        if N == 1:
            covs = np.zeros((nF, 4))
        else:
            covs = (grouped.cov() / N).to_numpy().reshape(nF, 4)

        pos = grouped.mean().to_numpy()

        return np.column_stack((f[:, None], pos, covs))
    
    @classmethod
    def ellipseParameters(cls, covData: np.ndarray, ci: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        chi2Val = chi2.ppf(ci, df=2)

        positions = []
        ellipses = []
        for row in covData:
            freq = row[0]
            pos = row[1:3]
            cov = row[3:].reshape(2, 2)
            eigvals, eigvecs = np.linalg.eigh(cov)
            axes = np.sqrt(eigvals * chi2Val)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            ellipses.append((freq, axes[0], axes[1], angle))
            positions.append(pos)
        
        return np.array(positions), np.array(ellipses)
    
    @classmethod
    def draw(cls, data: pd.DataFrame, ax: Axes, real: str, imag: str, errorbar, color = "#808080") -> None:

        if isinstance(errorbar, tuple) and errorbar[0] == 'ci':
            ci = errorbar[1]/100
        elif isinstance(errorbar, float):
            ci = errorbar/100
        elif errorbar == "ci":
                ci = 0.95
        else:
            raise ValueError(f"Unsupported errorbar specification for covariance visualization")
        

        covData = CovarianceVisualization.calculate(data, real=real, imag=imag)
        positions, ellipses = CovarianceVisualization.ellipseParameters(covData, ci=ci)

        color = list(hex2color(color))
        edgeColor = color + [0.5]
        faceColor = color + [0.2]

        for pos, (freq, a, b, angle) in zip(positions, ellipses):
            # Find the center point (mean) for this frequency
            freq_data = data[data["Frequency"] == freq]
            if len(freq_data) > 0:
                cx = pos[0]
                cy = pos[1]
                ax.add_patch(Ellipse((cx, cy), width=2*a, height=2*b, angle=angle, edgecolor=edgeColor, facecolor=faceColor, linestyle='-.'))
    
    @classmethod
    def interpolate(cls, sigma):
        pass

def covInterpolation(sigma1: np.ndarray, sigma2: np.ndarray, t: float) -> np.ndarray:
    """ 
    Log Euclidean interpolation between two covariance matrices
    """
    log_sigma1 = logm(sigma1)[0]  # logm returns (matrix, info)
    log_sigma2 = logm(sigma2)[0]
    interpolated = (1-t) * log_sigma1 + t * log_sigma2
    return expm(interpolated)[0]  # expm also returns (matrix, info)

@singledispatch
def nyquist(data: EIS, title: str | None = None, Rmin: float = 60, Rspan: float = 50, offsetCorrect: bool = True, **kwargs) -> PlotResult:
    return __nyquist_data(data.data, title=title, Rmin=Rmin, Rspan=Rspan, offsetCorrect=offsetCorrect, **kwargs)

@nyquist.register(pd.DataFrame)
def __nyquist_data(data: pd.DataFrame, title: str | None = None, Rmin: float = 60, Rspan: float = 50, offsetCorrect: bool = True, **kwargs) -> PlotResult:
    config = {
        "x": "Resistance",
        "y": "Neg. Reactance"
    }

    kwargs.setdefault("errorbar", ('ci', 95))

    # See if offset correction is desired
    if offsetCorrect:
        config["x"] = "Offset-Corrected Resistance"

    # Prepare data for mean & covs
    hueGroup = kwargs.get("hue", "Experiment Group")

    GroupingArgs = {"hue", "style", "size"}
    grouping = ["Frequency"] + [kwargs.get(arg) for arg in GroupingArgs if arg in kwargs]
    grouped = data.groupby(grouping)

    meanData = grouped.agg({config["x"]: "mean", config["y"]: "mean", "Palette": "first"}).reset_index()


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
                CovarianceVisualization.draw(group, ax, real=config["x"], imag=config["y"], errorbar=kwargs.get("errorbar"), color=color)
                line.set_zorder(2)  # Bring lines to front

    return PlotResult(title, fig, **kwargs)

@nyquist.register(list)
def __nyquist_multiple(data: list[EIS|pd.DataFrame], title: str | None = None, Rmin: float = 60, Rspan: float = 50, offsetCorrect: bool = True, **kwargs) -> PlotResult:
    if len(data) == 0:
        raise ValueError("Data list is empty.")
    
    combinedData = combineDataFrames(data, **kwargs)
    kwargs["hue"] = "Experiment Group"

    return nyquist(combinedData, title, Rmin=Rmin, Rspan=Rspan, offsetCorrect=offsetCorrect, **kwargs)