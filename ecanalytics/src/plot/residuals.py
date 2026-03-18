import numpy as np
import pandas as pd
import seaborn as sns


from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from scipy.stats import norm

from ..analysis.data_quality import DataQuality
from ..data.experiment import Experiment

from .plot import *
from .plotresult import PlotResult

from ..config import (
    RESIDUAL_PLOT_DEFAULT_FIGSIZE,
    FIGURE_SETTINGS,
)


@singledispatch
def residuals(data: DataQuality, title: str | None = None, **kwargs) -> PlotResult:
    return residuals(data.residuals, title, **kwargs)


@residuals.register(pd.DataFrame)
def __residuals_data(
    data: pd.DataFrame, title: str | None = None, **kwargs
) -> PlotResult:
    config = {
        "x": "Frequency",
        "y": "Residual",
        "hue": "Sample Name",
        "style": "Component",
        "title": title,
    }

    # Fuse real and imaginary residuals together
    residual_data = data.rename(
        columns={"Real Residual": "Real", "Imag. Residual": "Imaginary"}
    ).melt(
        ["Frequency", "Sample Name", "Palette"],
        ["Real", "Imaginary"],
        "Component",
        "Residual",
    )

    # Plot with standard plotter
    pr = plot(residual_data, **(kwargs | config))

    with pr as (fig, ax):
        fig.set_size_inches(RESIDUAL_PLOT_DEFAULT_FIGSIZE)

        # Equilibrated y axis
        m = max(ax.get_ylim(), key=abs)
        ax.set_ylim((-abs(m), abs(m)))

        # Good data borders
        ax.axhline(-1, linewidth=0.75, zorder=0, linestyle="-.")
        ax.axhline(+1, linewidth=0.75, zorder=0, linestyle="-.")

        # Make 2 column legend
        sns.move_legend(ax, "best", ncol=2)

    return pr


@singledispatch
def residual_distr(
    data: DataQuality, title: str | None = None, min_bound: float = 1.0, **kwargs
) -> PlotResult:
    return residual_distr(data.residuals, title, min_bound, **kwargs)


@residual_distr.register(pd.DataFrame)
def __residual_distr_data(
    data: pd.DataFrame, title: str | None = None, min_bound: float = 1.0, **kwargs
) -> PlotResult:
    color = data["Palette"].iloc[0].shade(1)[0]

    config = {"x": "Real Residual", "y": "Imag. Residual"}
    config["joint_kws"] = {"color": color} | kwargs.get("joint_kws", {})
    config["marginal_kws"] = {
        "facecolor": to_rgba(color, 0.5),
        "edgecolor": color,
        "stat": "density",
    } | kwargs.get("marginal_kws", {})

    residuals = data[["Real Residual", "Imag. Residual"]]

    with plt.rc_context(FIGURE_SETTINGS):
        joint = sns.jointplot(data, **(kwargs | config))

        fig = joint.figure
        joint_ax = fig.axes[0]
        marginal_axes = fig.axes[1:]

        if title is not None:
            fig.suptitle(
                title,
                fontsize=FIGURE_SETTINGS["axes.titlesize"],
                fontweight=FIGURE_SETTINGS["axes.titleweight"],
            )
            fig.subplots_adjust(top=0.94)

        # fig.subplots_adjust(top=0.94)
        bound = float(residuals.abs().max(axis=None))
        bound = max(bound, min_bound)

        lims = (-1.05 * bound, 1.05 * bound)

        # Temporary Solution
        Series_Info = Experiment.Series_Info | DataQuality.Series_Info

        joint_ax.set(
            xlabel=f"Real Residual {Series_Info['Real Residual'].symbol} [{Series_Info['Real Residual'].unit}]",
            ylabel=f"Imag. Residual {Series_Info['Imag. Residual'].symbol} [{Series_Info['Imag. Residual'].unit}]",
            xlim=lims,
            ylim=lims,
        )
        joint_ax.grid(True, which="both", linestyle="--", alpha=0.4)
        joint.refline(x=0, y=0, linewidth=0.75, marginal=True, zorder=0, linestyle="-.")

        # Note -> these are 0 centered stds
        stds = np.sqrt(np.mean(np.square(residuals.to_numpy()), axis=0))

        for ax, std, transpose in zip(marginal_axes, stds, [False, True]):
            x = np.linspace(-3 * std, 3 * std, 100)
            y = norm.pdf(x, loc=0, scale=std)

            gauss_data = (y, x) if transpose else (x, y)

            ax.plot(*gauss_data, color=color, linestyle="--")

        rms = np.sqrt(2 * np.mean(np.square(residuals.to_numpy())))
        rho = residuals.corr("pearson").iloc[0, 1]

        phantom = Rectangle((0, 0), 1, 1, visible=False)
        joint_ax.legend(
            [phantom] * 2,
            ["$\\Delta_{rms}$" + f" = {rms:.2f} %", f"$\\rho$ = {rho:.4f}"],
            loc="best",
            frameon=False,
        )

    return PlotResult(title, fig, **kwargs)
