import pandas as pd

from matplotlib.figure import Figure
from matplotlib import pyplot as plt

from . import core
from ..data.experiment import Experiment
from .plotresult import PlotResult
from ..config import FIGURE_SETTINGS


def _combine_data_frames(data: list[Experiment], kwargs: dict) -> pd.DataFrame:
    if len(data) == 0:
        raise ValueError("Data list is empty.")

    combined = pd.concat([exp.data for exp in data], ignore_index=True)

    if len({exp.name for exp in data}) > 1:
        if (hue_group := kwargs.get("hue", None)) is not None:
            # Multiple data sets and hue differentiation
            combined[hue_group] = (
                combined["Experiment Name"] + " - " + combined[hue_group]
            )
        else:
            kwargs["hue"] = "Experiment Name"

    return combined


def plot(
    data: Experiment | list[Experiment],
    x: str,
    y: str,
    title: str | None = None,
    **kwargs,
) -> PlotResult:
    if isinstance(data, Experiment):
        return core.lineplot(data.data, x, y, title, Experiment.Series_Info, **kwargs)
    elif isinstance(data, list):
        return core.lineplot(
            _combine_data_frames(data, kwargs),
            x,
            y,
            title,
            Experiment.Series_Info,
            **kwargs,
        )
    else:
        raise TypeError("Unsupported data type passed!")


def bode(
    data: Experiment | list[Experiment], title: str | None = None, **kwargs
) -> PlotResult:
    if isinstance(data, Experiment):
        df = data.data
    elif isinstance(data, list):
        df = _combine_data_frames(data, kwargs)
    else:
        raise TypeError("Unsupported data type passed!")

    config = {"data": df, "x": "Frequency", "no_save": True}

    with plt.rc_context(FIGURE_SETTINGS):
        axes = kwargs.pop("ax", None) or plt.subplots(2, 1, sharex=True)[1]

        if len(axes) != 2:
            raise ValueError("ax must be a list of two axes for Bode plot.")

        assert isinstance(fig := axes[0].figure, Figure)

        # Set title beforhand because buggy otherwise
        if title is not None:
            fig.suptitle(title)

        plot_args = kwargs | config

        for ax, y, t in zip(axes, ["Impedance", "Phase"], ["Magnitude", "Phase"]):
            plot(**plot_args, ax=ax, y=y, title=t)

    return PlotResult(title, fig, **core._clean_plot_args(kwargs))


def fresponse(
    data: Experiment | list[Experiment], y: str, title: str | None = None, **kwargs
) -> PlotResult:
    return plot(data, x="Frequency", y=y, title=title, **kwargs)
