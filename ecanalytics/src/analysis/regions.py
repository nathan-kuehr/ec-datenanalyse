import numpy as np
import pandas as pd

from scipy import signal as sig

from ._args import call_argument_parser
from ..data.experiment import Experiment


_DEFAULT_SAVGOL_WINDOW = 11
_SAVGOL_POLYORDER = 3


class Regions:
    _DEFAULT_CALCULATION_ARGS = {"kink_min_phase": -60, "artefact_min_phase": -10}

    def __init__(self, root: Experiment) -> None:
        self._root = root

        self._data: pd.DataFrame | None = None
        self._masks: pd.DataFrame | None = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self.analyze_phase()
            assert self._data is not None
        return self._data

    @property
    def masks(self) -> pd.DataFrame:
        if self._masks is None:
            self()
        assert self._masks is not None
        return self._masks

    def __call__(self, *args, **kwargs):
        args_list = call_argument_parser(
            args, kwargs, self._DEFAULT_CALCULATION_ARGS, self._root.sample_names, "Regions"
        )

        freqs = self._root.data["Frequency"].unique()

        grouped = self._root.data.groupby("Sample Name", sort=False)
        resistance = np.column_stack(
            [g["Offset-Corrected Resistance"].to_numpy() for _, g in grouped]
        )
        reactance = -np.column_stack(
            [g["Neg. Reactance"].to_numpy() for _, g in grouped]
        )

        phases = np.arctan2(reactance, resistance)
        smooth_phases = self._phase_smoothing(phases)

        kink_indices, artefact_indices = [], []
        for arg, smooth in zip(args_list, smooth_phases.T):
            peak_indices, _ = sig.find_peaks(smooth)

            kink_indices.append(self._find_kink_idx(peak_indices, smooth, arg["kink_min_phase"]))
            artefact_indices.append(self._find_artefact_idx(peak_indices, smooth, arg["artefact_min_phase"]))

        kink_indices = np.array(kink_indices)
        artefact_indices = np.array(artefact_indices)

        kink_freqs = freqs[kink_indices]
        artefact_freqs = freqs[artefact_indices]

        # Masks based on the smoothed versions
        valid_mask = freqs[:, None] <= artefact_freqs[None, :]
        diffusive_mask = freqs[:, None] <= kink_freqs[None, :]
        kinetic_mask = ~diffusive_mask

        diffusive_mask &= valid_mask
        kinetic_mask &= valid_mask

        # Mask based on the real data points
        inductive_mask = reactance > 0

        mask_dfs = []
        for i, name in enumerate(self._root.sample_names):
            mask_dfs.append(pd.DataFrame({
                "Frequency": freqs,
                "Diffusive Mask": diffusive_mask[:, i],
                "Kinetic Mask": kinetic_mask[:, i],
                "Valid Mask": valid_mask[:, i],
                "Inductive Mask": inductive_mask[:, i],
                "Sample Name": name,
            }))

        self._data = pd.DataFrame({
            "Kink Frequency": kink_freqs,
            "HF Artefact Threshold Frequency": artefact_freqs,
            "Sample Name": self._root.sample_names,
        })
        self._masks = pd.concat(mask_dfs, axis=0, ignore_index=True)

    @staticmethod
    def _find_artefact_idx(
        peak_indices: np.ndarray, smooth: np.ndarray, min_phase: float
    ) -> int:
        # Mask where the phase enters the region close to zero
        mask = smooth > np.radians(min_phase)

        # Find the peaks in that region -> are delimiters for the artefact region
        delims = list(peak_indices[mask[peak_indices]])
        if len(delims) > 0:
            delims = [delims[0]]

        # Always include inductive frequencies
        delims += list(np.where(smooth > 0)[0])

        return np.max(delims, initial=0)

    @staticmethod
    def _find_kink_idx(
        peak_indices: np.ndarray, smooth: np.ndarray, min_phase: float
    ) -> int:
        mask = smooth > np.radians(min_phase)

        peak_indices = peak_indices[mask[peak_indices]]

        if len(peak_indices) == 0:
            raise ValueError("Could not find kink")
        return peak_indices[-1]

    @staticmethod
    def _phase_smoothing(phase: np.ndarray, window_length: int = _DEFAULT_SAVGOL_WINDOW) -> np.ndarray:
        return sig.savgol_filter(
            phase, window_length=window_length, polyorder=_SAVGOL_POLYORDER, axis=0
        )
