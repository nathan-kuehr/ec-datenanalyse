import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from functools import singledispatch

from ..eis import EIS

from ..config import (
    FIGURE_SETTINGS,
    DEFAULT_FIGURE_SIZE,
    DEFAULT_LINEWIDTH,
    DEFAULT_MARKER_SIZE,
    DEFAULT_MARKER,
)

from .plotresult import PlotResult
from ..settings import Settings


@singledispatch
def plot(
    data, x: str, y: str, title: str | None = None, noShow: bool = False, **kwargs
) -> PlotResult:
    raise TypeError(
        f"Unsupported data type: {type(data).__name__}. Expected ImpedanceSpectrumExperiment."
    )


@plot.register(EIS)
def __plot_single_eis(
    data: EIS, x: str, y: str, title: str | None = None, **kwargs
) -> PlotResult:
    return plot(data.data, x, y, title, **kwargs)


@plot.register(pd.DataFrame)
def __plot_single_eis_data(
    data: pd.DataFrame, x: str, y: str, title: str | None = None, **kwargs
) -> PlotResult:
    config = {
        "data": data,
        "x": x,
        "y": y,
    }

    kwargs.setdefault("linewidth", DEFAULT_LINEWIDTH)
    kwargs.setdefault("errorbar", ("ci", 95))
    kwargs.setdefault("marker", DEFAULT_MARKER)
    kwargs.setdefault("markersize", DEFAULT_MARKER_SIZE)
    kwargs.setdefault("markeredgewidth", 0)

    palettes = data["Palette"].unique()

    if kwargs.get("hue") is None:
        kwargs.setdefault("color", palettes[0].color)
    else:
        Npalettes = len(palettes)
        Nhue = len(data[kwargs["hue"]].unique())

        if len(palettes) > 1 and Npalettes != Nhue:
            raise ValueError(
                f"Number of palettes ({Npalettes}) does not match number of hue categories ({Nhue})."
            )
        elif len(palettes) == 1:
            kwargs.setdefault("palette", palettes[0].shade(Nhue))
        else:
            kwargs.setdefault("palette", [p.color for p in palettes])

    with plt.rc_context(FIGURE_SETTINGS):
        ax = kwargs.get("ax") or plt.figure(figsize=DEFAULT_FIGURE_SIZE).gca()

        fig = ax.figure

        cleanedKwargs = Settings.cleanKwargs(kwargs, otherKeysToRemove={"noSave"})

        sns.lineplot(**config, **cleanedKwargs)

        ax.set_xlabel(f"{x} {EIS.SeriesInfo[x].symbol} [{EIS.SeriesInfo[x].unit}]")
        ax.set_ylabel(f"{y} {EIS.SeriesInfo[y].symbol} [{EIS.SeriesInfo[y].unit}]")

        ax.set_xscale(EIS.SeriesInfo[x].scale)
        ax.set_yscale(EIS.SeriesInfo[y].scale)

        if title is not None:
            ax.set_title(
                title,
                fontsize=FIGURE_SETTINGS[
                    "axes.titlesize" if len(fig.axes) == 1 else "legend.title_fontsize"
                ],
            )

        ax.grid(True, which="both", linestyle="--", alpha=0.4)

        return PlotResult(title, fig, **kwargs)


@plot.register(list)
def __plot_multiple_eis(
    data: list[EIS | pd.DataFrame], x: str, y: str, title: str | None = None, **kwargs
) -> PlotResult:
    if len(data) == 0:
        raise ValueError("Data list is empty.")

    combinedData = combineDataFrames(data, **kwargs)

    kwargs["hue"] = "Experiment Group"
    return plot(combinedData, x, y, title, **kwargs)


@singledispatch
def bode(data: EIS, title: str | None = None, **kwargs) -> PlotResult:
    return bode(data.data, title, **kwargs)


@bode.register(pd.DataFrame)
def __bode_data(data: pd.DataFrame, title: str | None = None, **kwargs) -> PlotResult:
    if (ax := kwargs.pop("ax", None)) is None:
        ax = plt.subplots(2, 1, figsize=DEFAULT_FIGURE_SIZE, sharex=True)[1]

    if len(ax) != 2:
        raise ValueError("axs must be a list of two Axes for Bode plot.")

    fig = ax[0].figure

    kwargsIntermed = kwargs.copy()
    kwargsIntermed["noSave"] = True

    with plt.rc_context(FIGURE_SETTINGS):
        if title is not None:
            fig.suptitle(
                title,
                fontsize=FIGURE_SETTINGS["axes.titlesize"],
                fontweight=FIGURE_SETTINGS["axes.titleweight"],
                y=0.98,
            )

        plot(
            data,
            x="Frequency",
            ax=ax[0],
            y="Impedance",
            title="Magnitude",
            **kwargsIntermed,
        )
        pr = plot(data, x="Frequency", ax=ax[1], y="Phase", title="Phase", **kwargs)

        pr.title(title)
    return pr


@bode.register(list)
def __bode_multiple(
    data: list[EIS | pd.DataFrame], title: str | None = None, **kwargs
) -> PlotResult:
    if len(data) == 0:
        raise ValueError("Data list is empty.")

    combinedData = combineDataFrames(data, **kwargs)

    kwargs["hue"] = "Experiment Group"
    return bode(combinedData, title, **kwargs)


@singledispatch
def fresponse(data: EIS, y: str, title: str | None = None, **kwargs) -> PlotResult:
    return plot(data.data, x="Frequency", y=y, title=title, **kwargs)


@fresponse.register(pd.DataFrame)
def __fresponse_data(
    data: pd.DataFrame, y: str, title: str | None = None, **kwargs
) -> PlotResult:
    return plot(data, x="Frequency", y=y, title=title, **kwargs)


@fresponse.register(list)
def __fresponse_multiple(
    data: list[EIS | pd.DataFrame], y: str, title: str | None = None, **kwargs
) -> PlotResult:
    if len(data) == 0:
        raise ValueError("Data list is empty.")

    combinedData = combineDataFrames(data, **kwargs)

    kwargs["hue"] = "Experiment Group"
    return fresponse(combinedData, y, title, **kwargs)


def combineDataFrames(data: list[EIS | pd.DataFrame], **kwargs) -> pd.DataFrame:
    if "hue" in kwargs:
        raise ValueError("Combining DataFrames with 'hue' is not supported.")

    dataFrames = [d.data if isinstance(d, EIS) else d for d in data]

    return pd.concat(dataFrames, ignore_index=True)
