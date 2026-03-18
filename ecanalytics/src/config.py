from typing import NamedTuple


class DataSeriesInfo(NamedTuple):
    symbol: str
    unit: str
    scale: str


EIS_SAMPLE_REQUIRED_DATA_SERIES = {
    "Frequency",
    "Impedance",
    "Resistance",
    "Neg. Reactance",
    "Phase",
}

EIS_EXPERIMENT_SERIES_INFO = {
    "Frequency": DataSeriesInfo("$F$", "Hz", "log"),
    "Impedance": DataSeriesInfo("$Z$", "$\\Omega$", "log"),
    "Capacitance": DataSeriesInfo("$C$", "F", "log"),
    "Phase": DataSeriesInfo("$\\varphi$", "°", "linear"),
    "Resistance": DataSeriesInfo("$R$", "$\\Omega$", "log"),
    "Offset-Corrected Resistance": DataSeriesInfo("$R_{oc}$", "$\\Omega$", "log"),
    "Neg. Reactance": DataSeriesInfo("-$X$", "$\\Omega$", "log"),
}

EIS_EXPERIMENT_FREQUENCY_TOLERANCE = (
    1e-3  # Tolerance for frequency matching in EIS experiments
)

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

RESIDUAL_PLOT_DEFAULT_FIGSIZE = (13, 6)

SNS_LINEPLOT_DEFAULT_SETTINGS = {
    "linewidth": 2.5,
    "marker": "o",
    "markersize": 5,
    "markeredgewidth": 0,
    "errorbar": ("se", 95),
    "err_style": "band",
}

COVVIS_ANGLE_STEPS = 1
COVVIS_INTERPOLATION_POINTS = 30

DATA_QUALITY_SERIES_INFO = {
    "Frequency": DataSeriesInfo("$F$", "Hz", "log"),
    "Real Residual": DataSeriesInfo("$\\Delta_{re}$", "% of |Z|", "linear"),
    "Imag. Residual": DataSeriesInfo("$\\Delta_{im}$", "% of |Z|", "linear"),
    "Residual": DataSeriesInfo("$\\Delta$", "% of |Z|", "linear"),
}
