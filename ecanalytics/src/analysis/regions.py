import numpy as np
import pandas as pd

from scipy import signal as sig

from ..data.experiment import Experiment

class Regions:
    def __init__(self, root: Experiment) -> None:
        self._root = root

        self._data = None
        self._masks = None
    
    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self.analyze_phase()
            assert self._data is not None
        return self._data
    
    @property
    def masks(self) -> pd.DataFrame:
        if self._masks is None:
            self.analyze_phase()
            assert self._masks is not None
        return self._masks

    def analyze_phase(self, kink_min_phase: float = -60, artefact_min_phase: float = -10):
        freqs = self._root.data["Frequency"].unique()

        grouped = self._root.data.groupby("Sample Name", sort=False)
        R = np.column_stack([g["Offset-Corrected Resistance"].to_numpy() for _, g in grouped])
        X = -np.column_stack([g["Neg. Reactance"].to_numpy() for _, g in grouped])

        phases = np.arctan2(X, R)
        smooth_phases = self._phase_smoothing(phases)

        kink_idc, artefact_idc = [], []
        for smooth in smooth_phases.T:
            peak_idc, _ = sig.find_peaks(smooth)

            kink_idc.append(self._find_kink_idx(peak_idc, smooth, kink_min_phase))
            artefact_idc.append(self._find_artefact_idx(peak_idc, smooth, artefact_min_phase))
        
        # Convert to numpy for broadcasting
        kink_idc, artefact_idc = np.array(kink_idc), np.array(artefact_idc)

        # Get associated frequencies
        kink_freqs = freqs[kink_idc]
        artefact_freqs = freqs[artefact_idc]

        # Get masks based on the smoothed versions
        valid_mask = freqs[:, None] <= artefact_freqs[None, :]
        diffusive_mask = freqs[:, None] <= kink_freqs[None, :]
        kinetic_mask = ~diffusive_mask

        diffusive_mask &= valid_mask
        kinetic_mask &= valid_mask

        # Masks based on the real data points
        inductive_mask = X > 0
        
        mask_dfs = []
        for i, name in enumerate(self._root.sample_names):
            mask_dfs.append(pd.DataFrame({
                "Frequency": freqs,
                "Diffusive Mask": diffusive_mask[:, i],
                "Kinetic Mask": kinetic_mask[:, i],
                "Valid Mask": valid_mask[:, i],
                "Inductive Mask":  inductive_mask[:, i],
                "Sample Name": name
            }))

        self._data = pd.DataFrame({
            "Kink Frequency": kink_freqs,
            "HF Artefact Threshold Frequency": artefact_freqs,
            "Sample Name": self._root.sample_names
        })
        self._masks = pd.concat(mask_dfs, axis=0, ignore_index=True)

    @staticmethod
    def _find_artefact_idx(peak_idc, smooth, min_phase: float = -10) -> int:
        # Mask where the phase enters the region close to zero
        mask = smooth > np.radians(min_phase)

        # Find the peaks in that region -> are delimiters for the artefact region
        delims = list(peak_idc[mask[peak_idc]])
        # Add inductive freqs for sure
        delims += list(np.where(smooth > 0)[0])

        return np.max(delims, initial=0)
    
    @staticmethod
    def _find_kink_idx(peak_idc, smooth, min_phase: float = -60):
        # Mask where the phase enters the region close to zero
        mask = smooth > np.radians(min_phase)

        peak_idc = peak_idc[mask[peak_idc]]

        if len(peak_idc) == 0:
            raise ValueError("Could not find ")
        else:
            return peak_idc[0]
        
    @staticmethod
    def _phase_smoothing(phase: np.ndarray, wlen: int = 11) -> np.ndarray:
        return sig.savgol_filter(phase, window_length=wlen, polyorder=3, axis=0)