import matplotlib as mpl

from matplotlib.colors import LinearSegmentedColormap


# This code uses the exact colormap from Gwyddion, David Necas (Yeti) and others
# Original code:
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/data/gradients/Gwyddion.net
_GWYDDION_COLORS = [
    (0.000000, (0.000000, 0.000000, 0.000000)),  # Black (start)
    (0.344671, (0.658824, 0.156863, 0.0588235)),  # Dark red
    (0.687075, (0.953506, 0.759686, 0.363821)),  # Yellow-orange
    (1.000000, (1.000000, 1.000000, 1.000000)),  # White (end)
]
_GWYDDION_CMAP = LinearSegmentedColormap.from_list("gwyddion", _GWYDDION_COLORS)

_colormap_is_registered = False


def _register() -> None:
    global _colormap_is_registered
    if not _colormap_is_registered:
        mpl.colormaps.register(cmap=_GWYDDION_CMAP)
        _colormap_is_registered = True
