import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import numpy as np
import pandas as pd
import os
from scipy.linalg import logm, expm
from scipy.stats import chi2

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from functools import singledispatch
from typing import Iterable

from .eis import EIS

from .config import FIGURE_SETTINGS, DEFAULT_FIGURE_SIZE, DEFAULT_LINEWIDTH, DEFAULT_MARKER_SIZE, DEFAULT_MARKER

from .plotresult import PlotResult
from .settings import Settings


@singledispatch
def plot(data, x: str, y: str, title: str | None = None, noShow: bool = False, **kwargs) -> PlotResult:
    raise TypeError(f"Unsupported data type: {type(data).__name__}. Expected ImpedanceSpectrumExperiment.")
    

@plot.register(EIS)
def __plot_single_eis(data: EIS, x: str, y: str, title: str | None = None, **kwargs) -> PlotResult:
    config = {
        "data": data.data,
        "x": x,
        "y": y,
    }

    kwargs.setdefault("linewidth", DEFAULT_LINEWIDTH)
    kwargs.setdefault("errorbar", "ci")
    kwargs.setdefault("marker", DEFAULT_MARKER)
    kwargs.setdefault("markersize", DEFAULT_MARKER_SIZE)
    kwargs.setdefault("markeredgewidth", 0)

    if kwargs.get("hue") is None:
        kwargs.setdefault("color", data.palette.color)
    else:
        kwargs.setdefault("palette", data.palette.shade(len(data.data[kwargs["hue"]].unique())))
    
    with plt.rc_context(FIGURE_SETTINGS):
        ax = kwargs.get("ax") or plt.figure(figsize=DEFAULT_FIGURE_SIZE).gca()

        fig = ax.figure

        cleanedKwargs = Settings.cleanKwargs(kwargs, otherKeysToRemove={"noSave"})
        
        sns.lineplot(**config, **cleanedKwargs)  

        ax.set_xlabel(f"{x} {data.SeriesInfo[x].symbol} [{data.SeriesInfo[x].unit}]")
        ax.set_ylabel(f"{y} {data.SeriesInfo[y].symbol} [{data.SeriesInfo[y].unit}]")

        ax.set_xscale(data.SeriesInfo[x].scale)
        ax.set_yscale(data.SeriesInfo[y].scale)

        if title is not None:
            ax.set_title(title, fontsize=FIGURE_SETTINGS["axes.titlesize" if len(fig.axes) == 1 else "legend.title_fontsize"],)

        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        
        return PlotResult(title, fig, **kwargs)

@plot.register(list)
def __plot_multiple_eis(data: list[EIS], x: str, y: str, title: str | None = None, **kwargs) -> PlotResult:
    if len(data) == 0:
        raise ValueError("Data list is empty.")
    
    ax = kwargs.get("ax")

    for i, d in enumerate(data):
        with plot(d, x, y, title, ax=ax, noSave=True, **kwargs) as (fig, ax):
            pass
    
    return PlotResult(title, fig)

@singledispatch
def bode(data: EIS, title: str | None = None, **kwargs) -> PlotResult:
    if (ax := kwargs.pop("ax", None)) is None:
        with plt.ioff():
            ax = plt.subplots(2, 1, figsize=DEFAULT_FIGURE_SIZE, sharex=True)[1]
    
    if len(ax) != 2:
        raise ValueError("axs must be a list of two Axes for Bode plot.")
    

    with plt.rc_context(FIGURE_SETTINGS):
        if title is not None:
            ax[0].figure.suptitle(title, fontsize=FIGURE_SETTINGS["axes.titlesize"], fontweight=FIGURE_SETTINGS["axes.titleweight"], y=0.98)

        plot(data, x="Frequency", ax=ax[0], y="Impedance", title="Magnitude", noSave=True, **kwargs)
        plot(data, x="Frequency", ax=ax[1], y="Phase", title="Phase", noSave=True, **kwargs)

    return PlotResult(title, ax[0].figure)

@bode.register(list)
def __bode_multiple(data: list[EIS], title: str | None = None, **kwargs) -> PlotResult:
    if len(data) == 0:
        raise ValueError("Data list is empty.")
    
    ax = kwargs.get("ax")

    for i, d in enumerate(data):
        with bode(d, title, ax=ax, noSave=True, **kwargs) as (fig, axes):
            pass
    
    return PlotResult(title, fig)

@singledispatch
def fresponse(data: EIS, y: str, title: str | None = None, **kwargs) -> PlotResult:
    return plot(data, x="Frequency", y=y, title=title, **kwargs)

@fresponse.register(list)
def __fresponse_multiple(data: list[EIS], y: str, title: str | None = None, **kwargs) -> PlotResult:
    if len(data) == 0:
        raise ValueError("Data list is empty.")
    
    ax = kwargs.get("ax")

    for i, d in enumerate(data):
        with fresponse(d, y, title, ax=ax, noSave=True, **kwargs) as (fig, ax):
            pass
    
    return PlotResult(title, fig)


def covInterpolation(sigma1: np.ndarray, sigma2: np.ndarray, t: float) -> np.ndarray:
    """ 
    Log Euclidean interpolation between two covariance matrices
    """
    return expm((1-t)*logm(sigma1)+ t*logm(sigma2))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.stats import chi2

def __nyquistCI_SmoothNormals(data: pd.DataFrame, ci: float = 0.95, show_envelope: bool = True) -> None:
    # 1. Datenvorbereitung & Sortierung
    data = data.sort_values("Frequency")
    N = data["Name"].nunique()
    nF = data["Frequency"].nunique()
    
    grouped = data.groupby("Frequency")[["Offset-Corrected Resistance", "Neg. Reactance"]]
    
    # Originale Stützstellen (Nodes)
    x_nodes = grouped.mean()["Offset-Corrected Resistance"].to_numpy()
    y_nodes = grouped.mean()["Neg. Reactance"].to_numpy()
    nodes = np.column_stack((x_nodes, y_nodes))
    
    # Kovarianzen an den Stützstellen
    covsDf = grouped.cov() / N 
    covs = covsDf.to_numpy().reshape((nF, 2, 2))

    # --- SCHRITT A: Vorbereitung der Interpolation ---
    t_nodes = np.linspace(0, 1, nF)
    t_fine = np.linspace(0, 1, 1000) # Hohe Auflösung für den Plot

    # 1. Trajektorie interpolieren (Spline)
    spl_x = make_interp_spline(t_nodes, x_nodes, k=2) # k=2 (quadratisch) oft stabiler als kubisch
    spl_y = make_interp_spline(t_nodes, y_nodes, k=2)
    traj_x = spl_x(t_fine)
    traj_y = spl_y(t_fine)
    trajInterpol = np.column_stack((traj_x, traj_y))

    # --- SCHRITT B: Explizite Interpolation der Normalenvektoren ---
    
    # 1. Tangenten an den Original-Knoten berechnen (nicht am Spline!)
    # np.gradient nutzt Zentraldifferenzen, das glättet Ausreißer etwas
    grads_nodes = np.gradient(nodes, axis=0)
    
    # 2. Normalen an den Knoten (90 Grad Rotation)
    normals_nodes = np.zeros_like(grads_nodes)
    normals_nodes[:, 0] = -grads_nodes[:, 1]
    normals_nodes[:, 1] = grads_nodes[:, 0]
    
    # 3. Winkel der Normalen berechnen
    normal_angles = np.arctan2(normals_nodes[:, 1], normals_nodes[:, 0])
    
    # WICHTIG: Unwrap, damit wir keine Sprünge bei 180/-180 Grad haben
    normal_angles = np.unwrap(normal_angles)
    
    # 4. Winkel interpolieren (Das ist der Trick für stabile Schläuche!)
    spl_angle = make_interp_spline(t_nodes, normal_angles, k=2)
    angles_fine = spl_angle(t_fine)
    
    # Zurückrechnen in Vektoren für jeden feinen Punkt
    normals_fine = np.column_stack((np.cos(angles_fine), np.sin(angles_fine)))

    # --- SCHRITT C: Interpolation der Kovarianz-Parameter (Breite/Höhe/Rotation) ---
    widths, heights, cov_angles = [], [], []

    for cov in covs:
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        
        widths.append(np.sqrt(vals[0]))
        heights.append(np.sqrt(vals[1]))
        cov_angles.append(np.arctan2(vecs[1, 0], vecs[0, 0]))

    # Parameter interpolieren
    cov_angles = np.unwrap(np.array(cov_angles), period=np.pi) # Period pi für Ellipsen-Symmetrie
    
    spl_w = make_interp_spline(t_nodes, widths, k=2)
    spl_h = make_interp_spline(t_nodes, heights, k=2)
    spl_a = make_interp_spline(t_nodes, cov_angles, k=2)
    
    w_fine = spl_w(t_fine)
    h_fine = spl_h(t_fine)
    a_fine = spl_a(t_fine)

    # --- SCHRITT D: Envelope Berechnung ---
    scale_factor = np.sqrt(chi2.ppf(ci, df=2))
    upper_bound = []
    lower_bound = []

    for i in range(len(t_fine)):
        # Aktueller interpolierter Normalenvektor
        n = normals_fine[i] 
        
        # Kovarianzmatrix an diesem Punkt rekonstruieren
        theta = a_fine[i]
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        # Rekonstruktion
        current_cov = R @ np.diag([w_fine[i]**2, h_fine[i]**2]) @ R.T
        
        # Projektion auf den interpolierten Normalenvektor
        sigma_sq = n @ current_cov @ n
        
        # Fallback für numerische Ungenauigkeit (negative Varianz verhindern)
        if sigma_sq < 0: sigma_sq = 0
            
        sigma = np.sqrt(sigma_sq)
        offset = scale_factor * sigma * n
        
        upper_bound.append(trajInterpol[i] + offset)
        lower_bound.append(trajInterpol[i] - offset)

    upper_bound = np.array(upper_bound)
    lower_bound = np.array(lower_bound)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.plot(data["Offset-Corrected Resistance"], data["Neg. Reactance"], 
             'o', color='blue', alpha=0.15, markersize=3)
    
    plt.plot(trajInterpol[:,0], trajInterpol[:,1], '-', color='crimson', linewidth=2, label="Mean Trajectory")

    if show_envelope:
        # Füllen
        envelope_x = np.concatenate([upper_bound[:, 0], lower_bound[::-1, 0]])
        envelope_y = np.concatenate([upper_bound[:, 1], lower_bound[::-1, 1]])
        plt.fill(envelope_x, envelope_y, color='red', alpha=0.2, label=f"{int(ci*100)}% CI")
        
        # Ränder zeichnen (zur visuellen Kontrolle der Glätte)
        plt.plot(upper_bound[:,0], upper_bound[:,1], '-', color='red', linewidth=0.8, alpha=0.5)
        plt.plot(lower_bound[:,0], lower_bound[:,1], '-', color='red', linewidth=0.8, alpha=0.5)

    plt.xlabel("Z' (Ohm)")
    plt.ylabel("-Z'' (Ohm)")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.title("Nyquist CI: Interpolated Normals & Parameters")
    plt.show()

@singledispatch
def nyquist(data: EIS, title: str | None = None, Rmin: float = 60, Rspan: float = 50, offsetCorrect: bool = True, **kwargs) -> PlotResult:
    config = {
        "x": "Resistance",
        "y": "Neg. Reactance"
    }

    if offsetCorrect:
        Rmin = 0
        config["x"] = "Offset-Corrected Resistance"

    tempData = None

    if "hue" in kwargs:
        grouped = data.data.groupby(["Frequency", kwargs["hue"]])[[config["x"], config["y"]]]
        meanDf = grouped.mean().reset_index()
        meanDf.columns = ["Frequency", kwargs["hue"], config["x"], config["y"]]
        
        # Ersetze data mit den gemittelten Werten
        tempData = data.data
        data._data = meanDf

    fig, ax = plot(data, **config, title=title, **kwargs).handle()

    ax.set_xscale("linear")
    ax.set_yscale("linear")

    ax.set_xlim(left=Rmin, right=Rmin + Rspan)
    ax.set_ylim(bottom=0, top=Rspan)

    if tempData is not None:
        data._data = tempData

    return PlotResult(title, fig)

@nyquist.register(list)
def __nyquist_multiple(data: list[EIS], title: str | None = None, Rmin: float = 60, Rspan: float = 50, offsetCorrect: bool = True, **kwargs) -> PlotResult:
    ax = kwargs.get("ax")

    for i, d in enumerate(data):
        with nyquist(d, title, Rmin, Rspan, offsetCorrect, ax=ax, **kwargs) as (fig, ax):
            pass
    
    return PlotResult(title, fig)