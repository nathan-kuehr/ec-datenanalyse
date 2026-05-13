
import numpy as np

from copy import deepcopy


from ..data.experiment import Experiment
from ..config import DATA_QUALITY_SERIES_INFO

from .drt import DRT
from .kkt import KKT
from .fitting import Fit


class Analysis:
    Series_Info = DATA_QUALITY_SERIES_INFO

    def __init__(self, root: Experiment) -> None:
        self._root = root

        self.kkt = KKT(root)
        self.drt = DRT(root)
        self.fit = Fit(root)

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
