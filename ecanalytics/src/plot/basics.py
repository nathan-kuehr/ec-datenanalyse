import pandas as pd

from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from typing import Callable

from . import core
from ..data.experiment import Experiment
from .plotresult import PlotResult
from ..config import FIGURE_SETTINGS

def _combine_experiment_data(exps: Experiment | list[Experiment], *extrs: Callable[[Experiment], pd.DataFrame], kwargs: dict) -> pd.DataFrame | list[pd.DataFrame]:
    # Listify
    if isinstance(exps, Experiment):
        exps = [exps]
    elif isinstance(exps, list):
        if not exps:
            raise ValueError("No experimental data passed!")
    else:
        raise TypeError("Unsupported data type passed!")
    
    # Prepare the transform
    def transform(data: pd.DataFrame):
        data[diff_col] = data["Experiment Name"] + " - " + data[diff_col].astype(str)

    # Add differentiators 
    do_transform = False
    if len(exps) > 1:
        diff_col = kwargs.get("hue") or kwargs.get("tile")
        if diff_col is None:
            kwargs["hue"] = "Experiment Name"
        else:
            do_transform = diff_col != "Experiment Name"
    
    combined = []
    for extr in extrs:
        df = pd.concat([extr(exp) for exp in exps], axis=0, ignore_index=True)
        
        if do_transform:
            transform(df)

        combined.append(df)
    
    return combined[0] if len(extrs) == 1 else combined

def plot(
    data: Experiment | list[Experiment],
    x: str,
    y: str,
    title: str | None = None,
    **kwargs,
) -> PlotResult:
    df = _combine_experiment_data(data, lambda x: x.data, kwargs=kwargs)
    return core.lineplot(df, x, y, title, Experiment.Series_Info, **kwargs)


def bode(
    data: Experiment | list[Experiment], title: str | None = None, **kwargs
) -> PlotResult:
    df = _combine_experiment_data(data, lambda x: x.data, kwargs=kwargs)

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
