from __future__ import annotations

import numpy as np
import pandas as pd

from pyimpspec import (
    Circuit,
    Series,
    Parallel,
    Resistor,
    WarburgOpen,
    ConstantPhaseElement,
)
from pyimpspec.circuit.registry import (
    register_element,
    ElementDefinition,
    ParameterDefinition
)
from scipy import stats

from ..model import Model
from ..stage import FittingProcedure, FittingStage
from ..reparam import ReparametrizedElement
from ....config import (
    FITTING_DEFAULT_INITIAL_VALUES,
    FITTING_DRT_POLARIZATION_CORRECTION_FACTOR,
    FITTING_TAU_DIFF_CORRECTION_FACTOR
)
from ....data.experiment import Experiment

def _estimate_r_s(data: pd.DataFrame, regions_data: pd.DataFrame):
    # Guess r_s as the x-axis offset for capacitive frequencies. Fall back to the
    # highest measured frequency when no inductive Limit was detected.
    return regions_data.fillna(data["Frequency"].max()).merge(
        data,
        left_on=["Sample Name", "Inductive Limit"],
        right_on=["Sample Name", "Frequency"],
        how="left",
    )["Resistance"].to_numpy()

def _estimate_r_ct(peak_data: pd.DataFrame):
    # Estimate r_ct from the DRT polarization
    return peak_data["Polarisation"].to_numpy() * FITTING_DRT_POLARIZATION_CORRECTION_FACTOR

def _estimate_n_dl(r_ct):
    # No estimation for n_dl yet 
    return np.ones_like(r_ct) * FITTING_DEFAULT_INITIAL_VALUES["randles"]["n_dl"]

def _estimate_tau_dl(peak_data: pd.DataFrame):
    return 10 ** peak_data["Log. Position"].to_numpy()

def _estimate_tau_diff(regions_data: pd.DataFrame, tau_dl):
    # Estimate tau_diff from the kink 45°->90°. If no kink is found, it is shadowed by semicircle
    # -> use tau_dl as estimation
    mt_resolvable = regions_data["Mass Transport Resolvable"].to_numpy()
    diff_cap_onset = regions_data["Diffusive-Capacitive Onset"].to_numpy()

    return FITTING_TAU_DIFF_CORRECTION_FACTOR * np.where(mt_resolvable, 1 / (2 * np.pi * diff_cap_onset), tau_dl)

def _estimate_n_r_diff(exp: Experiment, tau_diff):
    diff_cap_masks = exp.analysis.regions.make_mask(region="diffusive:capacitive", overlay_valid=True)
    data_by_sample = exp.data.groupby("Sample Name", sort=False)

    n_diff, q_diff = [], []

    for (_, group), mask in zip(data_by_sample, diff_cap_masks):
        group = group[mask]

        freqs = group["Frequency"].to_numpy()
        neg_x = group["Neg. Reactance"].to_numpy()
        r = group["Resistance"].to_numpy()

        # Fit ΔR / ΔX rather than ΔX / ΔR for stability as we approach 90°
        m_inv = stats.linregress(x=neg_x, y=r).slope
        n = np.arctan(1 / m_inv) / np.pi if m_inv != 0.0 else 0.5
        q = np.mean(np.sin(n*np.pi) / (neg_x * np.power(2 * np.pi * freqs, 2*n)))

        n_diff.append(n)
        q_diff.append(q)

    n_diff = np.array(n_diff)
    r_diff = tau_diff ** (2 * n_diff) / (3 * np.array(q_diff))

    return n_diff, r_diff


class OrthoRandles(ReparametrizedElement):
    @classmethod
    def _build_wrapped(cls) -> Circuit:
        r_s = Resistor().set_label("s")
        r_ct = Resistor().set_label("ct")
        q_dl = ConstantPhaseElement().set_label("dl")
        w_diff = (
            WarburgOpen()
            .set_label("diff")
            .set_fixed(Y=False, n=False, B=False)
            .set_upper_limits(n=0.5)
        )
        return Circuit(Series([r_s, Parallel([Series([r_ct, w_diff]), q_dl])]))

    @classmethod
    def _to_wrapped(
        cls, 
        log_R_s: float, 
        log_R_ct: float, 
        log_tau_dl: float, 
        n_dl: float, 
        log_R_diff: float, 
        log_tau_diff: float, 
        n_diff: float
    ) -> dict[str, float]:
        R_s = 10**log_R_s
        R_ct = 10**log_R_ct
        tau_dl = 10**log_tau_dl
        R_diff = 10**log_R_diff
        B_diff = 10**log_tau_diff

        return dict(
            R_s=R_s,
            R_ct=R_ct,
            Y_dl=(tau_dl**n_dl) / R_ct,
            n_dl=n_dl,
            B_diff=B_diff,
            Y_diff=B_diff / np.power(3 * R_diff, 1 / n_diff),
            n_diff=n_diff
        )

class Randles(Model):
    def __init__(self, *, charac_peak_tau: float | None = None) -> None:
        super().__init__()
        # If None is given, don't pre init the values
        self._charac_peak_tau = charac_peak_tau

    # Override abstract
    def _build(self) -> Circuit:
        return Circuit(Series([OrthoRandles()]))

    def _initial_values(self, exp: Experiment) -> list[dict] | None:
        # If None is given, don't pre init the values
        if self._charac_peak_tau is None:
            return None
        
        # Get data frames
        data = exp.data
        peak_data = exp.analysis.drt.peak_select([self._charac_peak_tau])
        regions_data = exp.analysis.regions.data

        r_s = _estimate_r_s(data, regions_data)
        r_ct = _estimate_r_ct(peak_data)
        n_dl = _estimate_n_dl(r_ct)
        tau_dl = _estimate_tau_dl(peak_data)
        tau_diff = _estimate_tau_diff(regions_data, tau_dl)
        n_diff, r_diff = _estimate_n_r_diff(exp, tau_diff)

        return pd.DataFrame({
            "log_R_s": np.log10(r_s),
            "log_R_ct": np.log10(r_ct),
            "log_tau_dl": np.log10(tau_dl),
            "n_dl": n_dl,
            "log_R_diff": np.log10(r_diff),
            "log_tau_diff": np.log10(tau_diff),
            "n_diff": n_diff,
        }).to_dict(orient="records")

    def _stages(self, exp: Experiment) -> list[FittingProcedure]:
        # Find the masks 
        kinetic_masks = exp.analysis.regions.make_mask(region="kinetic", overlay_valid=True)
        valid_masks = exp.analysis.regions.make_mask(region="valid")

        procedures = []
        for kinetic, valid in zip(kinetic_masks, valid_masks):
            procedures.append([
                FittingStage(
                    # vary values are +/- percent bands around the parameter value.
                    vary={} if self._charac_peak_tau is None else {"log_tau_dl": 20}, # +/- 20 % around DRT estimate
                    fix=["log_R_diff", "log_tau_diff", "n_diff"],
                    mask=kinetic,
                ),
                FittingStage(
                    fix=["log_R_s"],
                    vary={
                        "log_R_ct": 10, # +/- 10 %
                        "log_tau_dl": 10, # +/- 10 %
                        "n_dl": 5, # +/- 5 %
                    },
                    mask=valid,
                ),
                FittingStage(
                    fix=["log_R_s"], # Sometimes, the solver sacrifices R_s for a weird diffusive region. Prevent
                    mask=valid,
                ),
            ])

        return procedures
    
    # def _post_process_parameters(self, parameters: pd.DataFrame) -> pd.DataFrame:
    #     # Add CT derived parameters
    #     parameters["tau_r"] = (parameters["Y_dl"] * parameters["R_ct"]) ** (1 / parameters["n_dl"])
    #     parameters["C_dl, eq"] = parameters["tau_r"] / parameters["R_ct"]

    #     # Add diffusion derived parameters
    #     parameters["Q_diff"] = (parameters["B_diff"] * parameters["Y_diff"]) ** parameters["n_diff"]
        
    #     exp = 1 / (2 * parameters["n_diff"]) - 1
    #     parameters["C_diff"] = np.sqrt(parameters["B_diff"] * parameters["Y_diff"]) * ((parameters["R_ct"] + parameters["R_s"]) ** exp) # Apply Hsu Mansfeld

    #     return parameters
    
register_element(
    ElementDefinition(
        Class=OrthoRandles,
        symbol="Randles",
        name="Randles (orthogonal)",
        description="Randles circuit but in reparametrized base.",
        equation=(
            "10**log_R_s + 1/("
                "((10**log_tau_dl)**n_dl / 10**log_R_ct) * (2*I*pi*f)**n_dl"
                " + 1/("
                    "10**log_R_ct"
                    " + coth(("
                        "((10**log_tau_diff)**n_diff) * I*2*pi*f"
                    ")**n_diff) / ("
                        "(((10**log_tau_diff)**n_diff) / (3*10**log_R_diff)**(1/n_diff)) * I*2*pi*f"
                    ")**n_diff"
                ")"
            ")"
        ),
        parameters=[
            ParameterDefinition(
                symbol="log_R_s",
                unit="",
                description="Log. Solution Resistance",
                value=float(np.log10(50)),
                lower_limit=-20,
                upper_limit=20,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="log_R_ct",
                unit="",
                description="Log. Charge-Transfer Resistance",
                value=float(np.log10(20)),
                lower_limit=-20,
                upper_limit=20,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="log_tau_dl",
                unit="",
                description="Log. Charge-Transfer Relaxation Time",
                value=float(np.log10(2e-4)),
                lower_limit=-20,
                upper_limit=20,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="n_dl",
                unit="",
                description="",
                value=0.95,
                lower_limit=0,
                upper_limit=1,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="log_R_diff",
                unit="",
                description="Log. Diffusion Resistance",
                value=0,
                lower_limit=-20,
                upper_limit=20,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="log_tau_diff",
                unit="",
                description="Log. Diffusion Time Constant",
                value=0,
                lower_limit=-20,
                upper_limit=20,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="n_diff",
                unit="",
                description="",
                value=0.95/2,
                lower_limit=0,
                upper_limit=1,
                fixed=False,
            ),
        ],
    ),
)



