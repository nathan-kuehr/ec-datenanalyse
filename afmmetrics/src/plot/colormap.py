import matplotlib as mpl

from matplotlib.colors import LinearSegmentedColormap

# This code uses the exact colormap from Gwyddion, David Necas (Yeti) and others
# Original code:
# https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/data/gradients/Gwyddion.net
__gwyddion_colors = [
    (0.000000, (0.000000, 0.000000, 0.000000)),  # Schwarz (Start)
    (0.344671, (0.658824, 0.156863, 0.0588235)),  # Dunkelrot
    (0.687075, (0.953506, 0.759686, 0.363821)),  # Gelb-Orange
    (1.000000, (1.000000, 1.000000, 1.000000)),  # Weiß (Ende)
]
__gwyddion_cmap = LinearSegmentedColormap.from_list("gwyddion", __gwyddion_colors)

mpl.colormaps.register(cmap=__gwyddion_cmap)
