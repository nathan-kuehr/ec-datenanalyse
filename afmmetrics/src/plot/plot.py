import seaborn as sns
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from functools import singledispatch

from ecanalytics.src.plot.plotresult import PlotResult
from ecanalytics.src.plot.plot import (
    __prepare_palette,
    __make_axes_label,
    __plot_clean_kwargs,
)
from ecanalytics.src.config import (
    FIGURE_SETTINGS,
    RESIDUAL_PLOT_DEFAULT_FIGSIZE,
    SNS_LINEPLOT_DEFAULT_SETTINGS,
)

from ..data.afm import AFMImage
from ..data.microgel_afm import MicrogelImage, MicrogelStats
from ..config import DEFAULT_MG_AUTO_THRESHOLD_ARGS


def __combine_data_frames(
    data: list[MicrogelStats], kwargs: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(data) == 0:
        raise ValueError("Data list is empty.")

    macro = pd.concat([d.macro for d in data], ignore_index=True)
    micro = pd.concat([d.micro for d in data], ignore_index=True)

    group_name = f"{MicrogelStats._Container_Name_Prefix} Name"

    if (hue_group := kwargs.get("hue", None)) is not None:
        # Multiple data sets and hue differentiation
        macro[hue_group] = macro[group_name] + " - " + macro[hue_group]
        micro[hue_group] = micro[group_name] + " - " + micro[hue_group]
    else:
        kwargs["hue"] = group_name  # Use container name as hue if not already specified

    return macro, micro


def auto_threshold_vis(
    mgimg: MicrogelImage, title: str | None = None, **kwargs
) -> PlotResult:
    mgimg.preprocess()  # Make sure preprocessing routine has been run
    img = mgimg.get_oimg("top-hat")

    args = DEFAULT_MG_AUTO_THRESHOLD_ARGS | kwargs
    thr, (peak_intensities, kde, slope) = mgimg._Peak_Based_Auto_Threshold(img, **args)

    x = np.linspace(peak_intensities.min(), peak_intensities.max(), 1000)
    linear_push = np.linspace(0, kde.max() * slope, 1000)

    with plt.rc_context(FIGURE_SETTINGS):
        ax = kwargs.get("ax") or plt.figure(figsize=RESIDUAL_PLOT_DEFAULT_FIGSIZE).gca()
        fig = ax.get_figure()

        if title is not None:
            ax.set_title(
                title,
                fontsize=FIGURE_SETTINGS["axes.titlesize"],
            )

        sns.histplot(
            peak_intensities,
            bins=50,
            kde=False,
            color="lightgray",
            stat="density",
            ax=ax,
            label="Peak Distribution",
        )
        sns.lineplot(x=x, y=kde, color="blue", label="Pure KDE")
        sns.lineplot(
            x=x,
            y=kde + linear_push,
            color="red",
            linestyle="--",
            label=f"KDE + Linear Push ({slope:.2f} slope)",
            linewidth=2,
        )

        ax.axvline(
            thr, color="green", linestyle="-.", label=f"Calculated Threshold: {thr:.2f}"
        )
        ax.legend()
        ax.set(ylabel="Density / Weight", xlabel="Peak Intensity")

    return PlotResult(title, fig, **kwargs)


def workflow(
    afm: AFMImage, title: str | None = None, ncols: int = 4, **kwargs
) -> PlotResult:
    hist = list(afm._history.values())
    nhist = len(hist)

    nrows = 1 + (nhist - 1) // ncols

    if len(hist) > nrows * ncols:
        raise ValueError(
            "Shape needs to be sufficiently large for the AFM image's modification history."
        )

    with plt.rc_context(FIGURE_SETTINGS):
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 4, nrows * 4), squeeze=False
        )

        if title is not None:
            fig.suptitle(
                title,
                fontsize=FIGURE_SETTINGS["axes.titlesize"],
                fontweight=FIGURE_SETTINGS["axes.titleweight"],
            )
            fig.subplots_adjust(top=0.94)

        for i, ax in enumerate(axes.flatten()):
            # Check if we have history items left for this subplot
            if i < len(hist):
                img, desc = hist[i]
                if (dims := len(img.shape)) == 3:
                    ax.imshow(img)
                elif dims == 2:
                    ax.imshow(img, cmap="gray")
                else:
                    raise RuntimeError(
                        f"Unexpected image dimensions: {dims}. Expected 2 or 3."
                    )

                ax.set_title(desc)

            # Hide axis for all
            ax.axis("off")

        fig.tight_layout()
    return PlotResult(title, fig, **kwargs)


@singledispatch
def plot(
    mgstats: MicrogelStats, x: str, y: str, title: str | None = None, **kwargs
) -> PlotResult:
    return __plot_data(mgstats.macro, mgstats.micro, x, y, title, **kwargs)


@plot.register(pd.DataFrame)
def __plot_data(
    macro: pd.DataFrame,
    micro: pd.DataFrame,
    x: str,
    y: str,
    title: str | None = None,
    **kwargs,
) -> PlotResult:
    is_macro = MicrogelStats.Is_Macro_Stat(y)

    config = {"data": macro if is_macro else micro, "x": x, "y": y}

    palette = __prepare_palette(config["data"], kwargs)
    sns_args = SNS_LINEPLOT_DEFAULT_SETTINGS | kwargs | palette | config

    with plt.rc_context(FIGURE_SETTINGS):
        # Create new figure if necessary
        ax: Axes = sns_args.pop("ax", None) or plt.figure().gca()
        fig = ax.figure

        # Plot
        sns.lineplot(**sns_args, sort=True, ax=ax)

        ax.set(
            xlabel=__make_axes_label(x, MicrogelStats.Series_Info[x]),
            ylabel=__make_axes_label(y, MicrogelStats.Series_Info[y]),
            xscale=MicrogelStats.Series_Info[x].scale,
            yscale=MicrogelStats.Series_Info[y].scale,
        )

        # Set title or super title
        if title is not None:
            ax.set_title(
                title,
                fontsize=FIGURE_SETTINGS[
                    "axes.titlesize" if len(fig.axes) == 1 else "legend.title_fontsize"
                ],
            )

        # Turn on grid
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

        return PlotResult(title, fig, **__plot_clean_kwargs(kwargs))  # pyright: ignore


@plot.register(list)
def __plot_multiple(
    data: list[MicrogelStats], x: str, y: str, title: str | None = None, **kwargs
) -> PlotResult:
    return __plot_data(*__combine_data_frames(data, kwargs), x, y, title, **kwargs)
