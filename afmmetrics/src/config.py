MUGEL_DIAMETER_RANGE = (0.12, 0.22)

DEFAULT_PADDING_FACTOR = 2

DEFAULT_CAMBER_CUTOFF_WAVELENGTH = 1.302  # μm
DEFAULT_DENOISE_H_PARAMETER = 3
DEFAULT_DENOISE_TEMPLATE_WINDOW_SIZE = (
    0.08  # μm --> Defaults to 4x4px in 10x10 μm @ 512x512 px
)
DEFAULT_DENOISE_SEARCH_WINDOW_SIZE = (
    0.6  # μm --> Defaults to 30x30px in 10x10 μm @ 512x512 px
)

DEFAULT_TOPHAT_DIM_MARGIN = 1.5

FIGURE_SETTINGS = {
    "font.family": "Arial",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "medium",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.figsize": (7, 6),
}

DEFAULT_PEAK_DISTR_PRIOR = (140, 30)
