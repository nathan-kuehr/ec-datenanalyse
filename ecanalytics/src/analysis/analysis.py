
import numpy as np

from copy import deepcopy


from ..data.experiment import Experiment
from ..config import DATA_QUALITY_SERIES_INFO
from .api_adapter import APIAdapter, _API, Circuit

from .drt import DRT
from .kkt import KKT


class Analysis:
    Series_Info = DATA_QUALITY_SERIES_INFO

    def __init__(self, root: Experiment) -> None:
        self._root = root

        self.kkt = KKT(root)
        self.drt = DRT(root)

    def fit_circuit(self, circuit: Circuit):
        data = self._root.data

        # Find frequencies w/ inductive behaviour
        inductive_freqs = data[data["Neg. Reactance"] < 0]["Frequency"].unique()
        inductive_freqs_mask = np.isin(data["Frequency"].unique(), inductive_freqs)

        results = self._Multiprocess(
            method=APIAdapter.Fit_Circuit,
            input=APIAdapter.As_Impedance_Payload(
                self._root.data, ~inductive_freqs_mask
            ),
            tqdm_note=f"Fitting circuit '{circuit}' to EIS Experiment '{self._root.name}'",
            args={"circuit": circuit, "method": "auto", "weight": "boukamp"},
        )

        return APIAdapter.From_Fit_Result_Payload(results, self._sample_names())

    def simulate(self, fcr) -> Experiment:
        self._root.data["Data Kind"] = "Original"

        sim = deepcopy(self._root)

        df = fcr[0]

        df["Palette"] = sim._palette
        df["Experiment Name"] = sim.name

        df["Impedance"] = np.hypot(df["Resistance"], df["Neg. Reactance"])
        df["Phase"] = np.degrees(np.arctan2(df["Neg. Reactance"], df["Resistance"]))

        omega = 2 * np.pi * df["Frequency"].to_numpy()
        df["Capacitance"] = np.abs(1 / (omega * df["Neg. Reactance"]))

        resistance_offset = float(df["Resistance"][df["Neg. Reactance"] > 0].min())  # type: ignore
        df["Offset-Corrected Resistance"] = df["Resistance"] + (
            self._root.data["Offset-Corrected Resistance"]
            - self._root.data["Resistance"]
        )
        df["Data Kind"] = "Fitted"

        sim._data = df
        return sim

    def _sample_names(self) -> list[str]:
        return list(self._root.data["Sample Name"].unique())

    @staticmethod
    def Nonideal_Randles_Circuit() -> Circuit:
        R_s = (
            _API.Resistor(R=50)
            .set_label("s")
            .set_lower_limits(R=0)
            .set_upper_limits(200)
        )
        R_ct = (
            _API.Resistor(R=30)
            .set_label("ct")
            .set_lower_limits(R=0)
            .set_upper_limits(R=100)
        )

        # Usually, tau is at 2e-4, n roughly 0.75
        tau, n_dl = 2e-4, 0.75
        Y_dl = np.power(tau, n_dl) / R_ct.get_value(
            "R"
        )  # Initial guess for Y based on Brug's formula ~> 10^-5

        Q_dl = (
            _API.ConstantPhaseElement(Y=Y_dl, n=n_dl)
            .set_label("dl")
            .set_lower_limits(Y=1e-10, n=0.5)
            .set_upper_limits(Y=1e-2, n=1.0)
        )

        W_diff = (
            _API.WarburgOpen(Y=1e-4, n=0.5, B=0.01)
            .set_label("diff")
            .set_lower_limits(Y=1e-10, n=0.3, B=1e-4)
            .set_upper_limits(Y=1e-2, n=1, B=3)
            .set_fixed(Y=False, n=False, B=False)
        )

        return Circuit(
            _API.Series([R_s, _API.Parallel([Q_dl, _API.Series([R_ct, W_diff])])])
        )
