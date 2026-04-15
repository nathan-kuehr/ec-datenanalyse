from ecanalytics.src.config import DataSeriesInfo


# Config for AFM Image

READIN_HEIGHT_BLOCK_REGEX = r"# Channel: Height\n# Width: (\d*) (.*)\n# Height: (\d*) (.*)\n# Value units: (.*)([^#]*)"
IMAGE_PADDING_FACTOR = 2

# Config for Microgel Image

ESTIMATED_MICROGEL_DIAMETER_RANGE = (0.12, 0.22)  # μm

DEFAULT_PREPROCESS_SETTINGS = {
    "scan-line-align": {},
    "outlier-removal": {
        "multiplier": 1.75,
        "dilation": 0.1,  # μm
    },
    "hpf": {
        "cutoff": 1.302,  # μm --> proved to be good in gwyddion
        "width": None,  # auto-choose the same as the cutoff
    },
    "denoise": {
        "h": 3,
        "patch_size": 0.08,  # μm --> defaults to 4x4px in 10x10 μm @ 512x512 px
        "patch_distance": 0.6,  # μm --> defaults to 30x30px in 10x10 μm @ 512x512 px
        "fast_mode": True,
    },
    "top-hat": {
        "scale": 1.75 * max(ESTIMATED_MICROGEL_DIAMETER_RANGE)  # μm
    },
    "gaussian": {
        "cutoff": 0.1  # μm
    },
}

TARGET_REGION_PROPS = {
    "area",
    "axis_major_length",
    "axis_minor_length",
    "centroid_weighted",
    "eccentricity",
    "equivalent_diameter_area",
    "label",
    "orientation",
    "perimeter",
}

REGION_PROPS_RENAMING = {
    "area": "Area",
    "axis_major_length": "Length",
    "axis_minor_length": "Width",
    "eccentricity": "Eccentricity",
    "equivalent_diameter_area": "Equivalent Diameter",
    "label": "Label",
    "orientation": "Orientation",
    "perimeter": "Perimeter",
    "_Masked_Intensity_Patch": "Microgel Patch",
    "_Masked_Max_Intensity": "Height",
}

# == PLOTTING SETTINGS ==

SCALEBAR_SETTINGS = {"sep": 3, "loc": "lower right", "borderpad": 0.5, "frameon": False}
SCALEBAR_COLOR_THRESHOLD = 0.6
PROFILE_PLOT_FIGSIZE = (13, 6)


MICROGEL_SERIES_INFO = {
    "Height": DataSeriesInfo("$h$", "nm", "linear"),
    "Distance": DataSeriesInfo("$d$", "nm", "linear"),
}


MG_STATS_SERIES_INFO = {
    "Concentration": DataSeriesInfo(
        "$c_\\mathrm{MG}$", "$\\frac{\\mathrm{mg}}{\\mathrm{ml}}$", "linear"
    ),
    "Eccentricity": DataSeriesInfo("$\\epsilon$", None, "linear"),
    "Circularity": DataSeriesInfo("$C$", None, "linear"),
    "Equiv. Diameter": DataSeriesInfo("$\\Phi_{\\mathrm{eq}}$", "μm", "linear"),
    "Density": DataSeriesInfo("$\\sigma$", "$\\mu\\mathrm{m}^{-2}$", "linear"),
    "Coverage": DataSeriesInfo("$\\theta$", None, "linear"),
    "Count": DataSeriesInfo("$N_\\mathrm{MG}$", None, "linear"),
}
