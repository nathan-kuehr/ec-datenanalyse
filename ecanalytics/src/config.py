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
    "Frequency": DataSeriesInfo("F", "Hz", "log"),
    "Impedance": DataSeriesInfo("Z", "$\\Omega$", "log"),
    "Capacitance": DataSeriesInfo("C", "F", "log"),
    "Phase": DataSeriesInfo("varphi", "°", "linear"),
    "Resistance": DataSeriesInfo("R", "$\\Omega$", "log"),
    "Offset-Corrected Resistance": DataSeriesInfo("R_oc", "$\\Omega$", "log"),
    "Neg. Reactance": DataSeriesInfo("-X", "$\\Omega$", "log"),
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
        "$\\gamma_{\\mathrm{log}}$", r"$\frac{\Omega}{\mathrm{s}}$", "linear"
    ),
}

# FITTING

PARAMETER_PLOT_FIGURE_SETTINGS = {
    "figure.constrained_layout.use": False,
    "axes.titlesize": FIGURE_SETTINGS["legend.title_fontsize"]
}

DEFAULT_PARAMETER_PLOT_SETTINGS = {
    "catplot_kws": {
        "col_wrap": 3,
        "sharey": False,
        "showfliers": False,
        "boxprops": {"alpha": .9},
        "legend": True,
        "height": 3.5,
        "aspect": 1.2,
    },
    "stripplot_kws": {
        "size": 4,
        "jitter": True
    }
}

FITTING_RANDLES_SERIES_INFO = {
    "Charge Transfer Resistance": DataSeriesInfo("R_ct", r"$\Omega$", "linear"),
    "Solution Resistance": DataSeriesInfo("R_s", r"$\Omega$", "linear"),
    "Double-Layer Admittance": DataSeriesInfo("Y_dl", r"$\mu\mathrm{S}\mathrm{s}^n$", "linear"),
    "Double-Layer Dispersion Factor": DataSeriesInfo("n_dl", None, "linear"),
    "Warburg Admittance": DataSeriesInfo("Y_diff", r"$\mu\mathrm{S}$", "linear"),
    "Warburg Time Constant": DataSeriesInfo("B_diff", r"$10^{-6}\cdot\mathrm{s}^n$", "linear"),
    "Warburg Exponent": DataSeriesInfo("n_diff", None, "linear"),
}

FITTING_DEFAULT_INITIAL_VALUES = {
    "randles": {
        "n_dl": 0.97,
        "B_diff": 5e-5,
        "Y_diff": 2.5e-4
    },
    "zarc": {
        "n_zarc": 0.97
    }
}

FITTING_TO_UNIT_CONVERSION_MULTIPLIERS = {
    "Y_dl": 1e6,
    "B_diff": 1e6,
    "Y_diff": 1e6,
}

FITTING_DRT_POLARIZATION_CORRECTION_FACTOR = 1.1


DEFAULT_LINEPLOT_GRID_SETTINGS = {
    "kind": "line",
    "facet_kws": {"sharex": True, "sharey": True},
}