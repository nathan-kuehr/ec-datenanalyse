import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib import colors

from typing import Callable
from seaborn import FacetGrid



from . import core
from .nyquist_plot import nyquist
from .basics import _combine_experiment_data
from ..config import DataSeriesInfo
from ..data.experiment import Experiment
from ..analysis.fitting import Fit

def _mix_colors(a, b = (0, 0, 0), factor: float = .5):
    argb = colors.to_rgb(a)
    brgb = colors.to_rgb(b)

    return tuple((1-factor) * a + factor * b for a, b in zip(argb, brgb))

def _combine_fitted_params_data_frames(data: list[Experiment], kwargs):
    if len(data) == 0:
        raise ValueError("Data list is empty.")

    combined = pd.concat([exp.analysis.fit.params_long for exp in data], ignore_index=True)

    if (hue_group := kwargs.get("hue", None)) is not None:
        # Multiple data sets and hue differentiation
        combined[hue_group] = combined["Experiment Name"] + " - " + combined[hue_group]

    return combined

def fitted_parameters(exp: Experiment | list[Experiment], title: str | None = None, series_info: dict[str, DataSeriesInfo] = {}, **kwargs):
    if isinstance(exp, Experiment):
        data = exp.analysis.fit.params_long
    elif isinstance(exp, list):
        data = _combine_fitted_params_data_frames(exp, kwargs)
    else:
        raise TypeError("Unsupported data type passed!")
    
    return core.parameter_plot(data, kwargs.pop("hue", "Experiment Name"), data["Parameter"].unique(), title, Fit.Series_Info | series_info, **kwargs)
    
def show_fit(exps: Experiment | list[Experiment], kind: Callable, title: str | None = None, series_info: dict[str, DataSeriesInfo] = {}, **kwargs):
    # Listify the input
    if isinstance(exps, Experiment):
        exps = [exps]
    
    if kind is nyquist:
        # Make frequency grid
        freqs = np.vstack([exp.data["Frequency"].unique() for exp in exps])
        fmax, fmin = freqs.max(), freqs.min()
        freq_grid = np.logspace(np.log10(fmin), np.log10(fmax), 1000)
        
        # Simulate the fitted circuits
        sims = [ exp.analysis.fit.simulate_experiment(freq_grid)  for exp in exps]
        res = nyquist(exps + sims, title, series_info=series_info, legend=True, **kwargs)

        with res as (fig, axes):
            if isinstance(axes, Axes):
                axes = [axes]
            
            ax: Axes
            for ax in axes:
                lines = ax.lines
                if len(ax.child_axes) > 0:
                    lines += ax.child_axes[0].lines
                for line in lines:
                    if len(line.get_xdata()) == len(freq_grid):
                        line.set_markersize(0)
                        print(colors.to_rgb(line.get_color()))
                        print(c := _mix_colors(line.get_color()))
                        line.set_color(c)

        
    else:   
        pass

    return res

            


        
