import pandas as pd
from functools import singledispatch
import matplotlib.ticker as ticker

from matplotlib.colors import to_rgba
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .plotresult import PlotResult
from .plot import plot, __combine_data_frames, __prepare_groupby, __plot_clean_kwargs
from ..data.experiment import Experiment
from .covariance_visualization import CovarianceVisualization
from ..config import SNS_LINEPLOT_DEFAULT_SETTINGS


@singledispatch
def nyquist(
    data: Experiment,
    title: str | None = None,
    Rmin: float = 60,
    Rspan: float = 50,
    offset_correct: bool = True,
    **kwargs,
) -> PlotResult:
    return __nyquist_data(
        data.data,
        title=title,
        Rmin=Rmin,
        Rspan=Rspan,
        offset_correct=offset_correct,
        **kwargs,
    )


@nyquist.register(pd.DataFrame)
def __nyquist_data(
    data: pd.DataFrame,
    title: str | None = None,
    Rmin: float = 60,
    Rspan: float = 50,
    offset_correct: bool = True,
    add_inset: bool = True,
    **kwargs,
) -> PlotResult:
    # Choose correct x-axis
    x_axis = ("Offset-Corrected " if offset_correct else "") + "Resistance"
    config = {"x": x_axis, "y": "Neg. Reactance", "title": title, "noSave": True}

    # Subdivide in diff. plot groups
    grouped = __prepare_groupby(data, kwargs, include_frequency=True)

    # Aggregate means
    mean_data = grouped.agg(
        {config["x"]: "mean", config["y"]: "mean", "Palette": "first"}
    ).reset_index()

    kwargs = SNS_LINEPLOT_DEFAULT_SETTINGS | kwargs | config
    with plot(mean_data, **kwargs) as (fig, ax):
        assert isinstance(ax, Axes)

        # Linear scale
        ax.set_xscale("linear")
        ax.set_yscale("linear")

        # Equally scaled axes
        ax.set_xlim(left=Rmin, right=Rmin + Rspan)
        ax.set_ylim(bottom=0, top=Rspan)

        if add_inset:
            inset: Axes = inset_axes(
                ax, width="30%", height="30%", loc="upper left", borderpad=2
            )
            plot(mean_data, **(kwargs | {"ax": inset, "title": None}))

            inset.set(xscale="linear", yscale="linear", xlabel=None, ylabel=None)
            if (legend := inset.get_legend()) is not None:
                legend.remove()

            x_min, x_max = inset.get_xlim()
            y_min, y_max = inset.get_ylim()

            x_mean = (x_min + x_max) / 2
            y_mean = (y_min + y_max) / 2

            span = max(x_max - x_min, y_max - y_min) / 2

            inset.set(
                xlim=(x_mean - span, x_mean + span), ylim=(y_mean - span, y_mean + span)
            )
            inset.tick_params(axis="both", which="major", labelsize=7)

            inset.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: "{:,.1f}".format(x / 1000) + "K")
            )
            inset.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: "{:,.0f}".format(x / 1000) + "K")
            )

        # Draw uncertainty hulls
        if kwargs.get("errorbar") is not None:
            # Do not group over freqs this time
            grouped = __prepare_groupby(data, kwargs)

            for line, (_, group) in zip(ax.lines, grouped):
                if group["Sample Name"].nunique() == 1:
                    continue  # No covariance to plot

                # Define styling
                if kwargs.get("style") is not None:
                    fill_config = {
                        "facecolor": to_rgba(line.get_color(), 0.1),
                        "edgecolor": to_rgba(line.get_color(), 1),
                        "linewidth": 0.5,
                        "linestyle": line.get_linestyle(),
                        "zorder": 1,
                    }
                else:
                    fill_config = {
                        "color": line.get_color(),
                        "alpha": 0.1,
                        "zorder": 1,
                    }

                # Draw uncertainties
                covvis = CovarianceVisualization(group, kwargs)
                if kwargs.get("err_style") == "bars":
                    covvis.draw_ellipses(ax, fill_config)
                    if add_inset:
                        covvis.draw_ellipses(inset, fill_config)
                else:
                    covvis.draw_hull(ax, fill_config)
                    if add_inset:
                        covvis.draw_hull(inset, fill_config)

                line.set_zorder(2)

    return PlotResult(title, fig, **__plot_clean_kwargs(kwargs))


@nyquist.register(list)
def __nyquist_multiple(
    data: list[Experiment | pd.DataFrame],
    title: str | None = None,
    Rmin: float = 60,
    Rspan: float = 50,
    offset_correct: bool = True,
    **kwargs,
) -> PlotResult:
    return nyquist(
        __combine_data_frames(data, kwargs),
        title,
        Rmin=Rmin,
        Rspan=Rspan,
        offset_correct=offset_correct,
        **kwargs,
    )
