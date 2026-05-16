from __future__ import annotations

import logging
import pyimpspec

import numpy as np
import pandas as pd

from pyimpspec import (
    Circuit,
    Series,
    Parallel,
    Resistor,
    WarburgOpen,
    ConstantPhaseElement,
    generate_fit_identifiers
)
from scipy import stats, signal as sig
from typing import TYPE_CHECKING

from .model import Model, _iterate_elements
from .stage import FittingStage, FittingProcedure
from ...config import (
    FITTING_DEFAULT_INITIAL_VALUES,
    FITTING_DRT_POLARIZATION_CORRECTION_FACTOR,
)
from ...data.experiment import Experiment

_logger = logging.getLogger(__name__)


_WARBURG_N_UPPER_LIMIT = 0.5


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
        
        # Get lengths
        data = exp.data
        nsamples = data["Sample Name"].nunique()
        nfreqs = data["Frequency"].nunique()

        # Get data frames
        peak_data = exp.analysis.drt.peak_select([self._charac_peak_tau])
        region_data = exp.analysis.regions.masks

        diffusive_masks = region_data["Diffusive Mask"].to_numpy().reshape((nsamples, nfreqs))

        r_ct = (
            peak_data["Polarisation"].to_numpy()
            * FITTING_DRT_POLARIZATION_CORRECTION_FACTOR
        )

        n_dl = np.ones_like(r_ct) * FITTING_DEFAULT_INITIAL_VALUES["randles"]["n_dl"]

        taus = 10 ** peak_data["Log. Position"].to_numpy()
        y_dl = np.power(taus, n_dl) / r_ct

        b_diff = np.ones_like(r_ct) * FITTING_DEFAULT_INITIAL_VALUES["randles"]["B_diff"]
        y_diff = np.ones_like(r_ct) * FITTING_DEFAULT_INITIAL_VALUES["randles"]["Y_diff"]

        r_s, n_diff = [], []
        for (_, group), mask in zip(data.groupby("Sample Name", sort=False), diffusive_masks):
            impedance_data = group[["Resistance", "Neg. Reactance"]].to_numpy()

            # Fit ΔR / ΔX rather than ΔX / ΔR for stability as we approach 90°
            m_inv = stats.linregress(
                x=impedance_data[mask, 1],
                y=impedance_data[mask, 0],
            ).slope
            n_diff.append((np.arctan(1 / m_inv) / np.pi) if m_inv != 0.0 else 0.5)

            r_s.append(np.min(impedance_data[:, 0]))

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
        nfreqs = exp.data["Frequency"].nunique()

        region_data = exp.analysis.regions.masks

        # Find the masks 
        kinetic_masks = region_data["Kinetic Mask"].to_numpy().reshape((nsamples, nfreqs))
        valid_masks = region_data["Valid Mask"].to_numpy().reshape((nsamples, nfreqs))

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
                    constraint_variables={"tau_r": {"value": tau,"min":   0.9 * tau,"max":   1.1 * tau,"vary":  True},
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
                        "R_ct": 5, # ±5 %
                        "n_dl": 5,
                    },
                    mask=valid,
                    **constraint
                ),
                FittingStage(
                    mask=valid,
                    **constraint,
                ),
            ])

        return procedures
    
    def _derived_parameters(self, parameters: pd.DataFrame) -> pd.DataFrame:
        parameters["tau_r"] = (parameters["Y_dl"] * parameters["R_ct"]) ** (1 / parameters["n_dl"])
        parameters["C_dl, eq"] = parameters["tau_r"] / parameters["R_ct"]

        return parameters