from typing import NamedTuple


class DataSeriesInfo(NamedTuple):
    symbol: str
    unit: str | None
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

SMALL_FIGURE_SIZE = (7, 6)
LARGE_FIGURE_SIZE = (13, 6)

FIGURE_SETTINGS = {
    "axes.grid": True,
    "axes.grid.which": "both",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titlesize": 14,
    "axes.titleweight": "medium",
    "figure.constrained_layout.use": True,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "figure.figsize": SMALL_FIGURE_SIZE,
    "font.family": "Arial",
    "font.size": 12,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

RESIDUAL_PLOT_SETTINGS = {
    "x": "Frequency",
    "y": "Residual",
    "hue": "Sample Name",
    "style": "Component",
}

DEFAULT_LINEPLOT_SETTINGS = {
    "linewidth": 2.5,
    "marker": "o",
    "markersize": 5,
    "markeredgewidth": 0,
    "errorbar": ("se", 95),
    "err_style": "band",
    "sort": False,
}

DEFAULT_JOINT_DISTRIBUTION_PLOT_SETTINGS = {"marginal_kws": {"common_norm": False}}

DENSITY_SERIES_INFO = {"Density": DataSeriesInfo("", None, "linear")}

COVVIS_ANGLE_STEPS = 1
COVVIS_INTERPOLATION_POINTS = 30

DATA_QUALITY_SERIES_INFO = {
    "Frequency": DataSeriesInfo("$F$", "Hz", "log"),
    "Real Residual": DataSeriesInfo("$\\Delta_{re}$", "% of |Z|", "linear"),
    "Imag. Residual": DataSeriesInfo("$\\Delta_{im}$", "% of |Z|", "linear"),
    "Residual": DataSeriesInfo("$\\Delta$", "% of |Z|", "linear"),
    "Time Constant": DataSeriesInfo("$\\tau$", "s", "log"),
    "Polarization Density": DataSeriesInfo(
        "$\\gamma_{\\mathrm{log}}$", "$\\Omega$", "linear"
    ),
}
