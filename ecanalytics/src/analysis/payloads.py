import numpy as np
import pandas as pd

from dataclasses import dataclass


@dataclass
class ImpedancePayload:
    frequencies: np.ndarray
    impedances: np.ndarray
    sample_name: str

    @staticmethod
    def From_Data(
        data: pd.DataFrame, mask: list[np.ndarray] | np.ndarray | None = None
    ) -> list[ImpedancePayload]:
        freqs = data["Frequency"].unique()

        # If no mask is provided, use all frequencies
        if mask is None:
            mask = np.ones_like(freqs, dtype=bool)

        # Listify
        if not isinstance(mask, list):
            mask = [mask] * data["Sample Name"].nunique()

        def complex_imp(data: pd.DataFrame) -> np.ndarray:
            return (data["Resistance"] - 1j * data["Neg. Reactance"]).to_numpy()

        return [
            ImpedancePayload(freqs[m], complex_imp(group)[m], str(name))
            for m, (name, group) in zip(mask, data.groupby("Sample Name", sort=False))
        ]


@dataclass
class KramersKronigPayload:
    frequencies: np.ndarray
    residuals: np.ndarray
    sample_name: str


@dataclass
class DRTPayload:
    taus: np.ndarray
    gammas: np.ndarray
    peak_infos: list[dict]
    sample_name: str


@dataclass
class FitResultPayload:
    frequencies: np.ndarray
    impedances: np.ndarray
    residuals: np.ndarray
    parameters: dict[str, float]
    cdc: str
    sample_name: str