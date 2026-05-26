import logging

import numpy as np
import pandas as pd

from scipy import signal as sig

from ._args import call_argument_parser
from ..data.experiment import Experiment

_logger = logging.getLogger(__name__)


_DEFAULT_SAVGOL_WINDOW = 11
_SAVGOL_POLYORDER = 3
_HF_ARTEFACT_SEARCH_SPREAD = 3


class Regions:
    _DEFAULT_CALCULATION_ARGS = {"kink_min_phase": -60, "artefact_min_phase": -10}

    def __init__(self, root: Experiment) -> None:
        self._root = root

        self._data: pd.DataFrame | None = None
        self._masks: pd.DataFrame | None = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self()
            assert self._data is not None
        return self._data

    @property
    def masks(self) -> pd.DataFrame:
        if self._masks is None:
            self()
        assert self._masks is not None
        return self._masks
    
    def make_mask(self, region: str, overlay_valid: bool = True):
        freqs = self._root.data["Frequency"].unique()

        if region == "diffusive":
            kink_freqs = self.data["Kink Frequency"].to_numpy()

            mask = freqs[None, :] <= kink_freqs[:, None]
        elif region == "kinetic":
            kink_freqs = self.data["Kink Frequency"].to_numpy()

            mask = freqs[None, :] > kink_freqs[:, None]
        elif region == "LF artefact":
            lf_artefact_freqs = self.data["LF Artefact Threshold Frequency"].to_numpy()

            mask = (lf_artefact_freqs[:, None] < freqs[None, :])
            overlay_valid = False
        elif region == "HF artefact":
            hf_artefact_freqs = self.data["HF Artefact Threshold Frequency"].to_numpy()

            mask = (hf_artefact_freqs[:, None] >= freqs[None, :])
            overlay_valid = False
        elif region == "valid":
            mask = np.ones((len(self.data), len(freqs)), dtype=bool)
            overlay_valid = True
        else:
            raise ValueError("Invalid Mask Name")
        
        if overlay_valid:
            lf_artefact_freqs = self.data["LF Artefact Threshold Frequency"].to_numpy()
            hf_artefact_freqs = self.data["HF Artefact Threshold Frequency"].to_numpy()

            valid_mask = (lf_artefact_freqs[:, None] < freqs[None, :]) & (freqs[None, :] <= hf_artefact_freqs[:, None])

            return mask & valid_mask
        else:
            return mask


    def __call__(self, *args, **kwargs):
        # Get the args for each sample
        args_list = call_argument_parser(
            args, kwargs, self._DEFAULT_CALCULATION_ARGS, self._root.sample_names, "Regions"
        )

        freqs = self._root.data["Frequency"].unique()

        if not np.all(np.diff(freqs) < 0):
            raise ValueError("Frequencies must be ordered in descending order!")

        # Get lengths
        nfreqs = len(freqs)
        nsamples = self._root.data["Sample Name"].nunique()

        raw_magnitude = self._root.data["Offset-Corrected Impedance"].to_numpy().reshape(nsamples, nfreqs)
        raw_phase = self._root.data["Offset-Corrected Phase"].to_numpy().reshape(nsamples, nfreqs)

        # Smoothing
        smooth_magnitude = self._smooth(raw_magnitude)
        smooth_phase = self._smooth(raw_phase)

        # Indices
        kink_idc, hf_artefact_idc, lf_artefact_idc = [], [], []
        for args, smagn, sphase, rphase in zip(args_list, smooth_magnitude, smooth_phase, raw_phase):
            kink_idx = self._find_kink_from_phase(sphase, args["kink_min_phase"])
            hf_idx = self._find_hf_artefact_from_both(smagn, rphase)
            lf_idx = self._find_lf_artefact_from_phase(sphase, kink_idx)

            kink_idc.append(kink_idx)
            hf_artefact_idc.append(hf_idx)
            lf_artefact_idc.append(lf_idx)

        kink_freqs = freqs[kink_idc]
        hf_artefact_freqs = freqs[hf_artefact_idc]
        lf_artefact_freqs = freqs[lf_artefact_idc]


        # Raw Phase > 0 <=> inductive behaviour
        # Logical or accumulation to the left (::-1) to mask out all freqs higher than the lowest inductive frequency
        inductive_mask = np.logical_or.accumulate((raw_phase > 0)[::-1])[::-1]

        # Valid mask is cutting out the 
        valid_mask = (lf_artefact_freqs[:, None] < freqs[None, :]) & (freqs[None, :] <= hf_artefact_freqs[:, None])
        valid_mask &= ~inductive_mask

        # Diffusive & kinetic masks
        diffusive_mask = freqs[None, :] <= kink_freqs[:, None]
        kinetic_mask = ~diffusive_mask

        mask_dfs = []
        for i, name in enumerate(self._root.sample_names):
            mask_dfs.append(pd.DataFrame({
                "Frequency": freqs,
                "Diffusive Mask": diffusive_mask[i],
                "Kinetic Mask": kinetic_mask[i],
                "Valid Mask": valid_mask[i],
                "Inductive Mask": inductive_mask[i],
                "Sample Name": name,
            }))

        masks = pd.concat(mask_dfs, axis=0, ignore_index=True)
        data = pd.DataFrame({
            "Kink Frequency": kink_freqs,
            "HF Artefact Threshold Frequency": hf_artefact_freqs,
            "LF Artefact Threshold Frequency": lf_artefact_freqs,
            "Sample Name": self._root.sample_names,
        })

        self._data = self._root._add_metadata_to_data(data)
        self._masks = self._root._add_metadata_to_data(masks)

    @staticmethod
    def _find_kink_from_phase(smooth_phase: np.ndarray, min_phase: float):
        mask = smooth_phase > min_phase

        peak_idc, _ = sig.find_peaks(smooth_phase)
        peak_idc = peak_idc[mask[peak_idc]]

        if not len(peak_idc):
            raise ValueError("Could not find kink")
        return peak_idc[-1]
    
    @staticmethod
    def _find_hf_artefact_from_both(smooth_magnitude: np.ndarray, raw_phase: np.ndarray):
        slope = np.gradient(smooth_magnitude)

        pos_idc = np.where(slope < 0)[0]

        if not len(pos_idc):
            _logger.warning("Could not find the HF frequency artefact!")
            return 0
        else:
            target_idx = pos_idc[-1] + 1
            b, e = max(0, target_idx - _HF_ARTEFACT_SEARCH_SPREAD), min(len(smooth_magnitude), target_idx + _HF_ARTEFACT_SEARCH_SPREAD)
            return b + np.argmax(raw_phase[b:e]) 

    @staticmethod
    def _find_lf_artefact_from_phase(smooth_phase: np.ndarray, kink_idx: int):
        peak_idc, _ = sig.find_peaks(-smooth_phase[kink_idx:])

        if not len(peak_idc):
            _logger.warning("Could not find the LF frequency artefact!")
            return len(smooth_phase) - 1
        return peak_idc[0] + kink_idx
        
    @staticmethod
    def _smooth(phase: np.ndarray, window_length: int = _DEFAULT_SAVGOL_WINDOW) -> np.ndarray:
        return sig.savgol_filter(
            phase, window_length=window_length, polyorder=_SAVGOL_POLYORDER, axis=1
        )
