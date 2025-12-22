import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from functools import singledispatch
from typing import Iterable

from .eis import EIS

from .config import FIGURE_SETTINGS, DEFAULT_FIGURE_SIZE, DEFAULT_LINEWIDTH

@singledispatch
def plot(data, x: str, y: str, title: str | None = None, noShow: bool = False, **kwargs) -> tuple[Figure, Axes]:
    raise TypeError(f"Unsupported data type: {type(data).__name__}. Expected ImpedanceSpectrumExperiment.")
    

@plot.register(EIS)
def __plot_single_eis(data: EIS, x: str, y: str, title: str | None = None, noShow: bool = False, **kwargs) -> tuple[Figure, Axes]:
    config = {
        "data": data.data,
        "x": x,
        "y": y,
    }

    kwargs.setdefault("linewidth", DEFAULT_LINEWIDTH)
    kwargs.setdefault("errorbar", "sd")
    if kwargs.get("hue") is None:
        kwargs.setdefault("color", data.palette.color)
    else:
        assert data.groups is not None, "No grouping information available for plotting."
        kwargs.setdefault("palette", data.palette.shade(len(data.groups[kwargs["hue"]])))
    
    with plt.rc_context(FIGURE_SETTINGS):
        ax = kwargs.get("ax") or plt.figure(figsize=DEFAULT_FIGURE_SIZE).gca()

        fig = ax.figure
        
        sns.lineplot(**config, **kwargs)  

        ax.set_xlabel(f"{x} {data.SeriesInfo[x].symbol} [{data.SeriesInfo[x].unit}]")
        ax.set_ylabel(f"{y} {data.SeriesInfo[y].symbol} [{data.SeriesInfo[y].unit}]")

        ax.set_xscale(data.SeriesInfo[x].scale)
        ax.set_yscale(data.SeriesInfo[y].scale)

        if title is not None:
            ax.set_title(title, fontsize=FIGURE_SETTINGS["axes.titlesize" if len(fig.axes) == 1 else "legend.title_fontsize"],)

        ax.grid(True, which='both', linestyle='--', alpha=0.4)

        if not noShow:
            plt.tight_layout()
            plt.show()
        

        return (fig, ax)

@plot.register(list)
def __plot_multiple_eis(data: list[EIS], x: str, y: str, title: str | None = None, noShow: bool = False, **kwargs) -> tuple[Figure, Axes]:
    ax = kwargs.get("ax")

    for i, d in enumerate(data):
        fig, ax = plot(d, x, y, title, noShow=((i != len(data) - 1) or noShow), ax=ax, **kwargs)
    
    return (fig, ax)

@singledispatch
def bode(data: EIS, title: str | None = None, noShow: bool = False, **kwargs) -> tuple[Figure, list[Axes]]:
    if (ax := kwargs.pop("ax", None)) is None:
        ax = plt.subplots(2, 1, figsize=DEFAULT_FIGURE_SIZE, sharex=True)[1]
    
    if len(ax) != 2:
        raise ValueError("axs must be a list of two Axes for Bode plot.")
    
    if title is not None:
        ax[0].figure.suptitle(title, fontsize=FIGURE_SETTINGS["axes.titlesize"], fontweight=FIGURE_SETTINGS["axes.titleweight"], y=0.98)

    with plt.rc_context(FIGURE_SETTINGS):
        fig, ax[0] = plot(data, x="Frequency", ax=ax[0], y="Impedance", title="Magnitude", noShow=True, **kwargs)
        _, ax[1] = plot(data, x="Frequency", ax=ax[1], y="Phase", title="Phase", noShow=noShow, **kwargs)

        if not noShow:
            plt.show()

    return (fig, ax)

@bode.register(list)
def __bode_multiple(data: list[EIS], title: str | None = None, noShow: bool = False, **kwargs) -> tuple[Figure, list[Axes]]:
    ax = kwargs.get("ax")

    for i, d in enumerate(data):
        fig, ax = bode(d, title, noShow=((i != len(data) - 1) or noShow), ax=ax, **kwargs)
    
    return (fig, ax)