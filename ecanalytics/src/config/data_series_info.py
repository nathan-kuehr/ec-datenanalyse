from typing import NamedTuple


class DataSeriesInfo(NamedTuple):
    symbol: str
    unit: str | None
    scale: str


EIS_EXPERIMENT_SERIES_INFO = {
    "Frequency": DataSeriesInfo("F", "Hz", "log"),
    "Impedance": DataSeriesInfo("Z", "$\\Omega$", "log"),
    "Capacitance": DataSeriesInfo("C", "F", "log"),
    "Phase": DataSeriesInfo("varphi", "°", "linear"),
    "Resistance": DataSeriesInfo("R", "$\\Omega$", "log"),
    "Offset-Corrected Resistance": DataSeriesInfo("R_oc", "$\\Omega$", "log"),
    "Neg. Reactance": DataSeriesInfo("-X", "$\\Omega$", "log"),
}

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

DENSITY_SERIES_INFO = {"Density": DataSeriesInfo("", None, "linear")}

FITTING_RANDLES_SERIES_INFO = {
    "Charge Transfer Resistance": DataSeriesInfo("R_ct", r"$\Omega$", "linear"),
    "Solution Resistance": DataSeriesInfo("R_s", r"$\Omega$", "linear"),
    "Double-Layer Admittance": DataSeriesInfo("Y_dl", r"$\mathrm{S}\cdot\mathrm{s}^n$", "linear"),
    "Double-Layer Dispersion Factor": DataSeriesInfo("n_dl", None, "linear"),
    "Warburg Admittance": DataSeriesInfo("Y_diff", r"$\mathrm{S}$", "linear"),
    "Warburg Time Constant": DataSeriesInfo("B_diff", r"$\mathrm{s}^n$", "linear"),
    "Warburg Exponent": DataSeriesInfo("n_diff", None, "linear"),
    "Randles Time Constant": DataSeriesInfo("tau_r", "s", "linear"),
    "Double Layer Equivalent Capacitance": DataSeriesInfo("C_dl, eq", r"$\mathrm{F}$", "linear"),
}
