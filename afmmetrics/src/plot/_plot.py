import numpy as np
import pandas as pd

from functools import singledispatch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ecanalytics.src.config import FIGURE_SETTINGS, DEFAULT_LINEPLOT_SETTINGS
from ecanalytics.src.plot.plotresult import PlotResult
from ecanalytics.src.plot.basics import (
    __prepare_groupby,
    __active_groupby_cols,
    plot,
    __plot_clean_kwargs,
)


from . import colormap
from ..data import ImageWorkflow, AFMImage, MicrogelImage, MicrogelStats
from ..config import SCALEBAR_SETTINGS, SCALEBAR_COLOR_THRESHOLD, PROFILE_PLOT_FIGSIZE


def __draw_in_axis(
    axis: Axes, image: AFMImage, show_scalebar: bool = True, show_height: bool = True
) -> None:
    colormap._register()
    im = axis.imshow(image.data, cmap=(None if image.channels == 3 else "gwyddion"))

    if show_scalebar:
        bar_width = image.scan_size[0] / 5
        bar_width_px = image.um_to_px(bar_width)

        bar_height = image.scan_size[1] / 250
        bar_height_px = image.um_to_px(bar_height)

        axis.add_artist(
            AnchoredSizeBar(
                transform=axis.transData,
                size=bar_width_px,
                size_vertical=bar_height_px,
                label=f"{bar_width:.1f} µm",
                color=__scalebar_color(image.data),
                **SCALEBAR_SETTINGS,
            )
        )

    if show_height:
        # Add new axis for colorbar
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Add space for labels
        phantom = divider.append_axes("right", size="25%", pad=0.0)
        phantom.axis("off")

        # Add colorbar
        cbar = axis.figure.colorbar(im, cax=cax)

        # Sometimes the extrema are not shown
        # --> Add them to the cbar
        vmin, vmax = im.get_clim()
        ticks = cbar.get_ticks()

        # Filter out ticks that are too close -> avoid overlap
        margin = (vmax - vmin) * 0.05
        ticks = [t for t in ticks if (vmin + margin) < t < (vmax - margin)]
        cbar.set_ticks(np.sort(ticks + [vmin, vmax]))

        # Add label
        cbar.set_label("Height [nm]", rotation=270, labelpad=15)

        # Hide if RGB image
        cbar.ax.set_visible(image.channels == 1)

    axis.axis("off")


def __scalebar_color(data: np.ndarray) -> str:
    if len(data.shape) == 3:
        data = np.mean(data, axis=2)

    slicing = tuple(slice(int(s * 0.85), s) for s in data.shape)
    roi = data[slicing]

    mean_intensity = (roi.mean() - data.min()) / (
        data.max() - data.min() + 1e-12
    )  # Avoid division by zero

    return "black" if mean_intensity > SCALEBAR_COLOR_THRESHOLD else "white"


@singledispatch
def show_workflow(
    workflow: ImageWorkflow, title: str | None = None, ncols: int = 4, **kwargs
) -> PlotResult:
    # Calculate number of rows needed for the given number of columns
    nrows = 1 + (len(workflow) - 1) // ncols

    with plt.rc_context(FIGURE_SETTINGS):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 4, nrows * 4),
            squeeze=False,
            layout="constrained",
        )

        # Add title if provided
        if title is not None:
            fig.suptitle(
                title,
                fontsize=FIGURE_SETTINGS["axes.titlesize"],
                fontweight=FIGURE_SETTINGS["axes.titleweight"],
            )

        # Draw each workflow step in its own axis
        for ax, wf_step in zip(axes.flatten(), workflow):
            image, description = wf_step
            __draw_in_axis(ax, image)
            ax.set_title(description, fontsize=FIGURE_SETTINGS["legend.title_fontsize"])

    return PlotResult(title, fig, **kwargs)


@show_workflow.register(MicrogelImage)
def __show_workflow_mg_image(
    mg_image: MicrogelImage, title: str | None = None, ncols: int = 4, **kwargs
) -> PlotResult:
    return show_workflow(mg_image._workflow, title, ncols, **kwargs)


def show(
    image: AFMImage,
    title: str | None = None,
    show_scalebar: bool = True,
    show_height: bool = True,
    **kwargs,
) -> PlotResult:
    with plt.rc_context(FIGURE_SETTINGS):
        ax = kwargs.get("ax") or plt.figure().gca()
        fig = ax.figure

        __draw_in_axis(ax, image, show_scalebar, show_height)

        if title is not None:
            ax.set_title(
                title,
                fontsize=FIGURE_SETTINGS["axes.titlesize"],
                fontweight=FIGURE_SETTINGS["axes.titleweight"],
            )

    return PlotResult(title, fig, **kwargs)


def microgel_profile(
    image: MicrogelStats, title: str | None = None, angle=0.0, **kwargs
) -> PlotResult:
    dfs = []

    for names, group in __prepare_groupby(image.micro_stats, kwargs, ["Palette"]):
        mask = image.micro_stats.index.isin(group.index)
        profile = image.microgel_profile(mask, angle=angle, peak_pivot=True)

        width = len(profile)
        x_coords = 1e3 * image._reference_image.px_to_um(
            np.arange(width) - np.argmax(profile)
        )

        df = pd.DataFrame({"Distance": x_coords, "Height": profile})

        for col, val in zip(__active_groupby_cols(kwargs, ["Palette"]), names):
            df[col] = val

        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    config = {
        "x": "Distance",
        "y": "Height",
        "title": title,
        "no_save": True,
        "series_info": MicrogelImage.Series_Info,
    }
    kwargs = DEFAULT_LINEPLOT_SETTINGS | kwargs | config
    with plot(data, **kwargs) as (fig, ax):
        fig.set_size_inches(PROFILE_PLOT_FIGSIZE)

    return PlotResult(title, fig, **__plot_clean_kwargs(kwargs))
