from __future__ import annotations

import logging

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
from scipy import stats

from .model import Model
from .stage import FittingStage, FittingProcedure
from ...config import (
    FITTING_DEFAULT_INITIAL_VALUES,
    FITTING_DRT_POLARIZATION_CORRECTION_FACTOR,
)
from ...data.experiment import Experiment

_logger = logging.getLogger(__name__)


_WARBURG_N_UPPER_LIMIT = 0.5
_TAU_PEAK_TOLERANCE = 10 ** 0.2 # Two deci-decades


class Randles(Model):
    def __init__(self, *, charac_peak_tau: float | None = None) -> None:
        super().__init__()
        # If None is given, don't pre init the values
        self._charac_peak_tau = charac_peak_tau

    # Override abstract
    def _build(self) -> Circuit:
        r_s = Resistor().set_label("s")
        r_ct = Resistor().set_label("ct")
        q_dl = ConstantPhaseElement().set_label("dl")
        w_diff = (
            WarburgOpen()
            .set_label("diff")
            .set_fixed(Y=False, n=False, B=False)
            .set_upper_limits(n=_WARBURG_N_UPPER_LIMIT)
        )
        return self._set_lmfit_params(
            circuit=Circuit(Series([r_s, Parallel([Series([r_ct, w_diff]), q_dl])]))
        )

    def _initial_values(self, exp: Experiment) -> list[dict] | None:
        # If None is given, don't pre init the values
        if self._charac_peak_tau is None:
            return None
        
        # Get data frames
        data = exp.data
        peak_data = exp.analysis.drt.peak_select([self._charac_peak_tau])
        regions_data = exp.analysis.regions.data

        # Get only the diffusive-capacitive region
        diffusive_masks = exp.analysis.regions.make_mask(region="diffusive:capacitive", overlay_valid=True)

        # Guess R_s as the x-axis offset for capacitive frequencies. Fall back to the
        # highest measured frequency when no Inductive Limit was detected.
        r_s_freq = regions_data["Inductive Limit"].fillna(data["Frequency"].max())
        r_s = regions_data.assign(_r_s_freq=r_s_freq).merge(
            data,
            left_on=["Sample Name", "_r_s_freq"],
            right_on=["Sample Name", "Frequency"],
            how="left",
        )["Resistance"].to_numpy()

        # Estimate R_ct from the DRT polarization
        r_ct = (
            peak_data["Polarisation"].to_numpy()
            * FITTING_DRT_POLARIZATION_CORRECTION_FACTOR
        )

        # No estimation for n_dl yet 
        n_dl = np.ones_like(r_ct) * FITTING_DEFAULT_INITIAL_VALUES["randles"]["n_dl"]

        # Calculate Y_dl from the relaxation time and the polarization
        taus = 10 ** peak_data["Log. Position"].to_numpy()
        y_dl = (taus ** n_dl) / r_ct
        
        # Estimate B_diff from the kink 45°->90°. If no kink is found, it is shadowed by semicircle
        # -> use semicircle tau as estimation
        mass_transport_resolvable = regions_data["Mass Transport Resolvable"].to_numpy(dtype=bool)
        secondary_kink = regions_data["Diffusive-Capacitive Onset"].to_numpy(dtype=float)
        secondary_kink = np.where(
            mass_transport_resolvable, secondary_kink, 1 / (2 * np.pi * taus)
        )
        b_diff = 4 / (2 * np.pi * secondary_kink)

        n_diff, q_diff = [], []
        for (_, group), mask in zip(data.groupby("Sample Name", sort=False), diffusive_masks):
            masked_group = group[mask]

            freqs = masked_group["Frequency"].to_numpy()
            neg_reactance = masked_group["Neg. Reactance"].to_numpy()
            resistance = masked_group["Resistance"].to_numpy()

            # Fit ΔR / ΔX rather than ΔX / ΔR for stability as we approach 90°
            m_inv = stats.linregress(
                x=neg_reactance,
                y=resistance,
            ).slope
            n = np.arctan(1 / m_inv) / np.pi if m_inv != 0.0 else 0.5

            q = np.mean(np.sin(n*np.pi) / (neg_reactance * np.power(2 * np.pi * freqs, 2*n)))

            n_diff.append(n)
            q_diff.append(q)

        y_diff = (np.array(q_diff) ** (1 / np.array(n_diff))) / b_diff

        return pd.DataFrame({
            "R_s": r_s,
            "R_ct": r_ct,
            "Y_dl": y_dl,
            "n_dl": n_dl,
            "B_diff": b_diff,
            "Y_diff": y_diff,
            "n_diff": n_diff,
        }).to_dict(orient="records")

    def _stages(self, exp: Experiment) -> list[FittingProcedure]:
        nsamples = exp.data["Sample Name"].nunique()

        # Find the masks 
        kinetic_masks = exp.analysis.regions.make_mask(region="kinetic", overlay_valid=True)
        valid_masks = exp.analysis.regions.make_mask(region="valid")

        # If we got a tau, constrain over it 
        if self._charac_peak_tau is not None:
            peak_data = exp.analysis.drt.peak_select([self._charac_peak_tau])
            taus = 10 ** peak_data["Log. Position"].to_numpy()

            y_dl = self._lmfit_params["Y_dl"]
            n_dl = self._lmfit_params["n_dl"]
            r_ct = self._lmfit_params["R_ct"]
            ces = {y_dl: f"tau_r**{n_dl} / {r_ct}"}

            constraints = []
            for tau in taus:
                constraints.append(dict(
                    constraint_expressions=ces,
                    constraint_variables={
                        "tau_r": {
                            "value": tau,
                            "min": tau / _TAU_PEAK_TOLERANCE,
                            "max": _TAU_PEAK_TOLERANCE * tau,
                            "vary": True,
                        },
                    },
                ))
        else:
            constraints = [{}] * nsamples

        procedures = []
        for kinetic, valid, constraint in zip(kinetic_masks, valid_masks, constraints):
            procedures.append([
                FittingStage(
                    fix=["B_diff", "Y_diff", "n_diff"],
                    mask=kinetic,
                    **constraint
                ),
                FittingStage(
                    fix=["R_s"],
                    vary={
                        "R_ct": 10, # ±10 %
                        "n_dl": 5,
                    },
                    mask=valid,
                    **constraint
                ),
                FittingStage(
                    fix=["R_s"], # Sometimes, the solver sacrifices R_s for a weird diffusive region. Prevent
                    mask=valid,
                    **constraint,
                ),
            ])

        return procedures
    
    def _post_process_parameters(self, parameters: pd.DataFrame) -> pd.DataFrame:
        # Add CT derived parameters
        parameters["tau_r"] = (parameters["Y_dl"] * parameters["R_ct"]) ** (1 / parameters["n_dl"])
        parameters["C_dl, eq"] = parameters["tau_r"] / parameters["R_ct"]

        # Add diffusion derived parameters
        parameters["Q_diff"] = (parameters["B_diff"] * parameters["Y_diff"]) ** parameters["n_diff"]
        
        exp = 1 / (2 * parameters["n_diff"]) - 1
        parameters["C_diff"] = np.sqrt(parameters["B_diff"] * parameters["Y_diff"]) * ((parameters["R_ct"] + parameters["R_s"]) ** exp) # Apply Hsu Mansfeld

        return parameters