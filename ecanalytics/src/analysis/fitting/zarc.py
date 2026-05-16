"""Single-ZARC equivalent-circuit model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pyimpspec import Circuit, Series, ZARC as ZARCElement

from .model import Model
from ...config import (
    FITTING_DEFAULT_INITIAL_VALUES,
    FITTING_DRT_POLARIZATION_CORRECTION_FACTOR,
)

if TYPE_CHECKING:
    from ...data.experiment import Experiment


class Zarc(Model):
    """Single ZARC element in series.

    Pass `charac_peak_tau` to enable DRT-driven per-sample pre-initialisation.
    """

    def __init__(self, *, charac_peak_tau: float | None = None) -> None:
        self._charac_peak_tau = charac_peak_tau

    def _build(self) -> Circuit:
        rq = ZARCElement().set_label("zarc")
        return Circuit(Series([rq]))

    def _initial_values(self, experiment: "Experiment") -> list[dict] | None:
        if self._charac_peak_tau is None:
            return None

        peak_data = experiment.analysis.drt.peak_select([self._charac_peak_tau])

        r = (
            peak_data["Polarisation"].to_numpy()
            * FITTING_DRT_POLARIZATION_CORRECTION_FACTOR
        )
        tau = 10 ** peak_data["Log. Position"].to_numpy()
        n = np.ones_like(r) * FITTING_DEFAULT_INITIAL_VALUES["zarc"]["n_zarc"]

        return pd.DataFrame({
            "R_zarc": r,
            "tau_zarc": tau,
            "n_zarc": n,
        }).to_dict(orient="records")
