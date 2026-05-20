import pandas as pd

from matplotlib.axes import Axes
from typing import Iterable

from . import core

def _draw_markers(axes: Iterable[Axes], data: pd.DataFrame, region_data: pd.DataFrame, show_regions: bool | Iterable[str], x: str, kwargs: dict):
    if not show_regions:
        return
    elif isinstance(show_regions, bool):
        show_regions = ["Kink Frequency"]

    grouped = core._prepare_groupby(region_data, {"tile": kwargs.get("tile")})

    ax: Axes
    for ax, (_, group) in zip(axes, grouped):
        view = data[data["Sample Name"].isin(group["Sample Name"].unique())]

        for region in show_regions:
            freqs = group[region].unique()
            mask = view["Frequency"].isin(freqs)

            vals = view[x][mask].unique()

            for v in vals:
                ax.axvline(v, linewidth=0.75, zorder=0, linestyle="-.", color="k")