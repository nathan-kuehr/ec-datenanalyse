import numpy as np
import pandas as pd
import seaborn as sns

from seaborn import JointGrid


from matplotlib.patches import Rectangle


from matplotlib.axes import Axes
from ..analysis.analysis import Analysis
from ..data.experiment import Experiment


from . import core
from .plotresult import PlotResult
from ..analysis.kkt import KKT
from ..config import LARGE_FIGURE_SIZE, RESIDUAL_PLOT_SETTINGS


def _add_stats_overview(ax: Axes, data: pd.DataFrame, kwargs: dict):
    if len(core._active_groupby_cols(kwargs)) > 0:
        grouped = core._prepare_groupby(data, kwargs)
        legend = core._iterate_legend(ax, dummy=False)

        for (_, text), (_, group) in zip(legend, grouped):
            stats = np.squeeze(KKT.Compile_Stats_Data(group, pool=True))
            text.set_text(text.get_text() + f" [{_make_stat_label(stats)}]")
    else:
        # No legend drawn yet
        phantom = Rectangle((0, 0), 1, 1, visible=False)
        stats = np.squeeze(KKT.Compile_Stats_Data(data, pool=True))

        ax.legend(
            handles=[phantom],
            labels=[_make_stat_label(stats)],
            loc="best",
            frameon=False,
        )


def _combine_residual_data_frames(data: list[Experiment], kwargs: dict) -> pd.DataFrame:
    if len(data) == 0:
        raise ValueError("Data list is empty.")

    combined = pd.concat([exp.analysis.kkt.data for exp in data], ignore_index=True)

    if (hue_group := kwargs.get("hue", None)) is not None:
        # Multiple data sets and hue differentiation
        combined[hue_group] = combined["Experiment Name"] + " - " + combined[hue_group]
    else:
        kwargs["hue"] = "Experiment Name"

    return combined


def _listify(obj):
    if not isinstance(obj, list):
        return [obj]
    return obj


def _make_stat_label(stat_row: np.ndarray) -> str:
    rms, _, _, rho = stat_row
    return rf"$\Delta_{{\mathrm{{rms}}}}$ = {rms:.2f} %, $\rho$ = {rho:.4f}"


def residuals(
    exp: Experiment | list[Experiment], title: str | None = None, **kwargs
) -> PlotResult:
    if isinstance(exp, Experiment):
        data = exp.analysis.kkt.data
    elif isinstance(exp, list):
        data = _combine_residual_data_frames(exp, kwargs)
    else:
        raise TypeError("Unsupported data type passed!")

    # Get component representation
    data = KKT.As_Component_Data(data)

    config = RESIDUAL_PLOT_SETTINGS | {
        "title": title,
        "series_info": Analysis.Series_Info,
    }

    res = core.lineplot(data, **(kwargs | config))
    with res as (fig, ax):
        assert isinstance(ax, Axes)
        fig.set_size_inches(LARGE_FIGURE_SIZE)

        # Equilibrated y axis
        m = max(ax.get_ylim(), key=abs)
        ax.set_ylim((-abs(m), abs(m)))

        # Good data borders
        ax.axhline(-1, linewidth=0.75, zorder=0, linestyle="-.", color="k")
        ax.axhline(+1, linewidth=0.75, zorder=0, linestyle="-.", color="k")

        # Make 2 column legend
        sns.move_legend(ax, "best", ncol=2)

    return res


def residual_distribution(
    exp: Experiment | list[Experiment],
    title: str | None = None,
    min_bound: float = 1.0,
    include_stats: bool = True,
    **kwargs,
) -> PlotResult:
    if isinstance(exp, Experiment):
        data = exp.analysis.kkt.data
    elif isinstance(exp, list):
        data = _combine_residual_data_frames(exp, kwargs)
    else:
        raise TypeError("Unsupported data type passed!")

    config = {
        "x": "Real Residual",
        "y": "Imag. Residual",
        "data": data,
        "title": title,
        "series_info": Analysis.Series_Info,
    }

    # Plot limits
    bound = 1.05 * np.max(
        np.abs(data[["Real Residual", "Imag. Residual"]].to_numpy()), initial=min_bound
    )

    # If only one group: bar plot
    if len(core._active_groupby_cols(kwargs)) == 0:
        config["marginal_kws"] = kwargs.pop("marginal_kws", {}) | {
            "binwidth": bound / 35
        }

    res = core.joint_distribution_plot(**config, **kwargs)
    with res as (fig, axes):
        assert isinstance(axes, list) and len(axes) == 3

        # Get handles
        joint_ax = axes[0]

        # Draw line cross through the origin
        joint: JointGrid = res.get_meta("jointplot")
        joint.refline(x=0, y=0, linewidth=0.75, marginal=True, zorder=0, linestyle="-.")

        # Add the statistics
        if include_stats:
            _add_stats_overview(joint_ax, data, kwargs)

        # Make equilibrated size
        joint_ax.set(xlim=(-bound, bound), ylim=(-bound, bound))

    return PlotResult(title, fig, **kwargs)
