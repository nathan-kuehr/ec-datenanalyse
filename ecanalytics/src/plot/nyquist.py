import pandas as pd
from functools import singledispatch

from .plotresult import PlotResult
from .plot import plot, combine_data_frames
from ..eis import EIS
from .covariance_visualization import CovarianceVisualization


@singledispatch
def nyquist(
    data: EIS,
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
    **kwargs,
) -> PlotResult:
    config = {"x": "Resistance", "y": "Neg. Reactance"}

    kwargs.setdefault("errorbar", ("ci", 95))
    kwargs.setdefault("err_style", "band")

    # See if offset correction is desired
    if offset_correct:
        config["x"] = "Offset-Corrected Resistance"

    # Prepare data for mean & covs
    hue_group = kwargs.get("hue", "Experiment Group")

    Grouping_Args = {"hue", "style", "size"}
    grouping = ["Frequency"] + [
        kwargs.get(arg) for arg in Grouping_Args if arg in kwargs
    ]
    grouped = data.groupby(grouping)

    mean_data = grouped.agg(
        {config["x"]: "mean", config["y"]: "mean", "Palette": "first"}
    ).reset_index()

    kwargs_intermed = kwargs.copy()
    kwargs_intermed["noSave"] = True

    with plot(mean_data, **config, title=title, **kwargs_intermed) as (fig, axes):
        ax = axes[0]
        ax.set_xscale("linear")
        ax.set_yscale("linear")

        ax.set_xlim(left=Rmin, right=Rmin + Rspan)
        ax.set_ylim(bottom=0, top=Rspan)

        if kwargs.get("errorbar") is not None:
            grouped = data.groupby(hue_group)
            for (_, group), line in zip(data.groupby(hue_group), ax.lines):
                color = line.get_color()
                cov_vis = CovarianceVisualization(
                    group, kwargs.get("errorbar"), config["x"]
                )

                if cov_vis.N == 1:
                    continue  # No covariance to plot

                if kwargs.get("err_style") == "band":
                    hull = cov_vis.hull(ax)
                    ax.fill(
                        hull[:, 0],
                        hull[:, 1],
                        color=color,
                        alpha=0.1,
                        label="Hüllkurve",
                        zorder=1,
                    )
                elif kwargs.get("err_style") == "bars":
                    cov_vis.draw(ax, color=color)
                else:
                    raise ValueError(
                        f"Unknown err_style '{kwargs.get('err_style')}'. Supported styles are 'band' and 'bars'."
                    )

                line.set_zorder(2)  # Bring lines to front

    return PlotResult(title, fig, **kwargs)


@nyquist.register(list)
def __nyquist_multiple(
    data: list[EIS | pd.DataFrame],
    title: str | None = None,
    Rmin: float = 60,
    Rspan: float = 50,
    offset_correct: bool = True,
    **kwargs,
) -> PlotResult:
    if len(data) == 0:
        raise ValueError("Data list is empty.")

    combined_data = combine_data_frames(data, **kwargs)
    kwargs["hue"] = "Experiment Group"

    return nyquist(
        combined_data,
        title,
        Rmin=Rmin,
        Rspan=Rspan,
        offset_correct=offset_correct,
        **kwargs,
    )
