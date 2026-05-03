import numpy as np

from matplotlib import colors, pyplot as plt
from matplotlib.axes import Axes
from pandas.core.groupby.generic import DataFrameGroupBy

# import matplotlib.ticker as ticker

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from . import core
from .covariance_visualization import CovarianceVisualization
from .basics import _combine_data_frames
from .plotresult import PlotResult
from ..data.experiment import Experiment
from ..config import (
    EIS_EXPERIMENT_FREQUENCY_TOLERANCE,
    FIGURE_SETTINGS,
    DEFAULT_LINEPLOT_SETTINGS,
)


def _clean_nyquist_args(kwargs: dict) -> dict:
    return core._clean_args(
        kwargs, ["data", "title", "Rmin", "Rspan", "offset_correct"]
    )


def __draw_nyquist_errorbars(
    ax: Axes, lines, grouped: DataFrameGroupBy, kwargs: dict
) -> None:
    draw_style = "style" in kwargs
    err_is_bars = kwargs.get("err_style") == "bars"

    for line, (_, group) in zip(lines, grouped):
        if group["Sample Name"].nunique() == 1:
            continue  # No covariance to plot

        c = line.get_color()
        s = line.get_linestyle()

        # Define styling
        if draw_style:
            style_config = {
                "facecolor": colors.to_rgba(c, 0.1),
                "edgecolor": colors.to_rgba(c, 1),
                "linewidth": 0.5,
                "linestyle": s,
                "zorder": 1,
            }
        else:
            style_config = {
                "color": c,
                "alpha": 0.1,
                "zorder": 1,
            }

        # Draw uncertainties
        cov_vis = CovarianceVisualization(group, kwargs)
        if err_is_bars:
            cov_vis.draw_ellipses(ax, style_config)
            # if add_inset:
            #     covvis.draw_ellipses(inset, fill_config)
        else:
            cov_vis.draw_hull(ax, style_config)
            # if add_inset:
            #     covvis.draw_hull(inset, fill_config)

        line.set_zorder(2)


def nyquist(
    data: Experiment | list[Experiment],
    title: str | None = None,
    Rmin: float = 60,
    Rspan: float = 50,
    offset_correct: bool = True,
    add_inset: bool = True,
    add_frequency_labels: bool = False,
    **kwargs,
) -> PlotResult:
    if isinstance(data, Experiment):
        df = data.data
    elif isinstance(data, list):
        df = _combine_data_frames(data, kwargs)
    else:
        raise TypeError("Unsupported data type passed!")

    # Choose correct x-axis
    x_axis = ("Offset-Corrected " if offset_correct else "") + "Resistance"
    config = {
        "x": x_axis,
        "y": "Neg. Reactance",
        "title": title,
        "no_save": True,
        "series_info": Experiment.Series_Info,
    }

    # Subdivide in diff. plot groups
    grouped = core._prepare_groupby(df, kwargs, additional_groups=["Frequency"])

    # Aggregate means
    mean_data = grouped.agg(
        {config["x"]: "mean", config["y"]: "mean", "Palette": "first"}
    ).reset_index()

    kwargs = DEFAULT_LINEPLOT_SETTINGS | kwargs | config
    with core.lineplot(mean_data, **kwargs) as (fig, ax):
        assert isinstance(ax, Axes)

        # Setup axes correctly
        ax.set(
            xscale="linear", yscale="linear", xlim=(Rmin, Rmin + Rspan), ylim=(0, Rspan)
        )

        # Draw uncertainty hulls
        if kwargs.get("errorbar") is not None:
            # Do not group over freqs this time
            grouped = core._prepare_groupby(df, kwargs)
            __draw_nyquist_errorbars(ax, ax.lines, grouped, kwargs)

        if add_frequency_labels:
            freqs = mean_data["Frequency"].unique()
            decades = 10.0 ** np.round(np.log10(freqs))

            target_freqs = set(
                freqs[
                    np.isclose(freqs, decades, rtol=EIS_EXPERIMENT_FREQUENCY_TOLERANCE)
                ]
            )
            if not target_freqs:
                raise ValueError("No frequencies found near decade values!")

            def freq_label_formatter(f):
                if f >= 1e6:
                    return f"{f / 1e6:.0f} MHz"
                if f >= 1e3:
                    return f"{f / 1e3:.0f} kHz"
                if f >= 1:
                    return f"{f:.0f} Hz"
                else:
                    return f"{f:.2f} Hz"

            with plt.rc_context(FIGURE_SETTINGS):
                grouped = core._prepare_groupby(mean_data, kwargs)
                for line, (_, group) in zip(ax.lines, grouped):
                    c = line.get_color()
                    scatter_color = tuple(c * 0.5 for c in colors.to_rgb(c))

                    mask = group["Frequency"].isin(target_freqs)

                    freqs = group["Frequency"][mask].to_numpy()
                    pos = group[[kwargs["x"], kwargs["y"]]][mask].to_numpy()

                    # Plot scatter points at target frequencies
                    ax.scatter(
                        pos[:, 0],
                        pos[:, 1],
                        s=DEFAULT_LINEPLOT_SETTINGS["markersize"] ** 2,
                        color=scatter_color,
                        linewidths=0,
                        zorder=10,
                    )

                # Add annotation for each target frequency
                for f, p in zip(freqs, pos):
                    ax.annotate(
                        freq_label_formatter(f),
                        xy=p,
                        fontsize=plt.rcParams[
                            "legend.fontsize"
                        ],  # Greift die aktuelle Legenden-Schriftgröße ab
                        ha="center",  # Horizontal zentriert
                        va="center",
                    )

        # if add_inset:
        #     inset: Axes = inset_axes(
        #         ax, width="30%", height="30%", loc="upper left", borderpad=2
        #     )
        #     plot(mean_data, **(kwargs | {"ax": inset, "title": None}))

        #     inset.set(xscale="linear", yscale="linear", xlabel=None, ylabel=None)
        #     if (legend := inset.get_legend()) is not None:
        #         legend.remove()

        #     x_min, x_max = inset.get_xlim()
        #     y_min, y_max = inset.get_ylim()

        #     x_mean = (x_min + x_max) / 2
        #     y_mean = (y_min + y_max) / 2

        #     span = max(x_max - x_min, y_max - y_min) / 2

        #     inset.set(
        #         xlim=(x_mean - span, x_mean + span), ylim=(y_mean - span, y_mean + span)
        #     )
        #     inset.tick_params(axis="both", which="major", labelsize=7)

        #     inset.xaxis.set_major_formatter(
        #         ticker.FuncFormatter(lambda x, pos: "{:,.1f}".format(x / 1000) + "K")
        #     )
        #     inset.yaxis.set_major_formatter(
        #         ticker.FuncFormatter(lambda x, pos: "{:,.0f}".format(x / 1000) + "K")
        #     )

    kwargs = _clean_nyquist_args(kwargs)
    kwargs = core._clean_plot_args(kwargs)

    return PlotResult(title, fig, **kwargs)
