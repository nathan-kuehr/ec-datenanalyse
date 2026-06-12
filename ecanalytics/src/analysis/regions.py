import numpy as np
import pandas as pd
import logging

from scipy import signal as sig

import matplotlib.pyplot as plt

from ._args import call_argument_parser
from ..data.experiment import Experiment

logger = logging.getLogger(__name__)

_DEFAULT_SAVGOL_WINDOW = 11
_SAVGOL_POLYORDER = 3

def curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.gradient(x, axis=-1)
    dy = np.gradient(y, axis=-1)
    
    ddx = np.gradient(dx, axis=-1)
    ddy = np.gradient(dy, axis=-1)
    
    num = np.abs(dx * ddy - dy * ddx)
    denom = (dx**2 + dy**2)**(1.5)
    
    return np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)

def smooth(x: np.ndarray) -> np.ndarray:
    return sig.savgol_filter(
        x, window_length=_DEFAULT_SAVGOL_WINDOW, polyorder=_SAVGOL_POLYORDER, axis=-1
    )

def descend(vec: np.ndarray, i: int) -> int:
    while True:
        j = min((k for k in (i-1, i, i+1) if 0 <= k < len(vec)), key=vec.__getitem__)
        
        if j == i:
            return j
        else:
            i = j

class Regions:
    _DEFAULT_CALCULATION_ARGS = {"a_minprom": 25.0, "a_minw": 2.0, "cap_angle": 70.0, "d_minprom": 0.15, "tail_angle_tol": 3}

    def __init__(self, root: Experiment) -> None:
        self._root = root

        self._data: pd.DataFrame | None = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self()
            assert self._data is not None
        return self._data

    def make_mask(self, region: str, overlay_valid: bool = True) -> np.ndarray:
        f = self._root.data["Frequency"].unique()[None, :]

        def col(name: str, fill: float) -> np.ndarray:
            return self.data[name].fillna(fill).to_numpy()[:, None]

        # Exclusion regions define what "valid" removes -> never overlay them
        if region == "artefact:lf":
            return f < col("LF Artefact Onset", -np.inf)
        if region == "artefact:hf":
            return f > col("HF Artefact Limit", np.inf)
        if region == "inductive":
            return f >= col("Inductive Limit", np.inf)

        valid = ~(
            self.make_mask("artefact:lf")
            | self.make_mask("artefact:hf")
            | self.make_mask("inductive")
        )
        if region == "valid":
            return valid

        if region == "kinetic":
            mask = f >= col("Kinetic Limit", np.inf)
        elif region == "diffusive":
            mask = f <= col("Diffusive Onset", -np.inf)
        elif region == "diffusive:mass-transport":
            mask = col("Mass Transport Resolvable", 0.0).astype(bool)
            mask &= f <= col("Diffusive Onset", -np.inf)
            mask &= f > col("Diffusive-Capacitive Onset", np.inf)
        elif region == "diffusive:capacitive":
            mask = f <= col("Diffusive-Capacitive Onset", np.inf)
        else:
            raise ValueError(f"Unknown region mask: {region!r}")

        return mask & valid if overlay_valid else mask


    def __call__(self, *args, **kwargs):
        # Get the args for each sample
        args_list = call_argument_parser(
            args, kwargs, self._DEFAULT_CALCULATION_ARGS, self._root.sample_names, "Regions"
        )

        sample_names = self._root.sample_names
        freqs = self._root.data["Frequency"].unique()
        if not np.all(np.diff(freqs) < 0):
            raise ValueError("Frequencies must be ordered in descending order!")

        # Get lengths
        nsamples, nfreqs = len(sample_names), len(freqs)

        # Get data
        raw_rs = self._root.data["Offset-Corrected Resistance"].to_numpy().reshape(nsamples, nfreqs)
        raw_xs = -self._root.data["Neg. Reactance"].to_numpy().reshape(nsamples, nfreqs)

        # Smooth
        smooth_rs, smooth_xs = smooth(raw_rs), smooth(raw_xs)

        # Use tangent angle to find transition between semicircle and diffusion part
        phis = -np.degrees(np.arctan2(np.gradient(smooth_xs, axis=1), np.gradient(smooth_rs, axis=1)))
        dphis = np.gradient(phis, axis=1)

        ## Point A: 
        # -> transistion between semicircle and diffusion part
        # A curvature based approach was used before but didn't work out as well
        peaks = [
            sig.find_peaks(-phi, prominence=args["a_minprom"], width=args["a_minw"])[0] 
            for phi, args in zip(phis, args_list)
        ]
        CCs = np.array([peak[-1] if len(peak) else -1 for peak in peaks], dtype=int)

        ## Point B:
        # -> beginning of capacitive tail
        # A first analysis was done using curvature peaks but this is easier
        cap_angles = np.array([args["cap_angle"] for args in args_list])
        cap_tail_mask = (phis >= cap_angles[:, None]) & (np.arange(nfreqs) > CCs[:, None])
        EEs = np.where(np.any(cap_tail_mask, axis=1), np.argmax(cap_tail_mask, axis=1), -1).astype(int)

        ## Point C:
        # -> sometimes A is not perfectly well chosen but a bit too early.
        # Take minimum between A and B
        freq_indices = np.arange(nfreqs)
        dsegment = (freq_indices[None, :] >= CCs[:, None]) & (freq_indices[None, :] < EEs[:, None])
        DDs = np.argmax(np.where(dsegment, smooth_xs, -np.inf), axis=1)

        ## Mass transport ? 
        # -> if we see that there is a plateau in φ (<=> two peaks in dφ <=> a valley in -dφ), there is diffusive:mass transport region here
        norm_dphis = dphis / np.max(np.where(dsegment, dphis, 1e-10), axis=1, keepdims=True)
        nvalleys = np.array([
            len(sig.find_peaks(-ndphi, prominence=args["d_minprom"])[0]) 
            for ndphi, args in zip(norm_dphis, args_list)
        ])
        has_mass_transport = nvalleys > 0

        ## Point D:
        # -> sometimes we have weird HF artefacts
        # Offset-corrected impedance (calculated directly here) shows local minumum
        dmagns = np.gradient(np.hypot(smooth_rs, smooth_xs), axis=1)
        phases = np.atan2(smooth_xs, smooth_rs)
        
        # First iteration: find last negative dmagn
        neg_dmagn_mask = (dmagns < 0) & (freq_indices < np.where(CCs < 0, nfreqs, CCs)[:, None])
        BBs = np.where(neg_dmagn_mask, freq_indices, -1).max(axis=1)

        # Second iteration: ascend into phase peak
        BBs = np.array([e if e < 0 else descend(-phase, e) for phase, e in zip(phases, BBs)])

        ## Point E:
        # -> inductive artefacts
        AAs = np.where(raw_xs > 0, freq_indices, -1).max(axis=1)

        ## Point F:
        tail_angle_tols = np.array([args["tail_angle_tol"] for args in args_list])

        max_phis_idc = np.maximum(EEs, np.argmax(phis, axis=1))
        max_phis = phis[np.arange(nsamples), max_phis_idc]

        tail_drop_mask = ((phis < (max_phis - tail_angle_tols)[:, None])) & (freq_indices[None, :] > max_phis_idc[:, None])

        Fs = np.where(tail_drop_mask, freq_indices, nfreqs).min(axis=1)

        # Put everything together
        freqs_series = pd.Series(freqs)

        region_df = pd.DataFrame({
            "Inductive Limit": AAs,
            "HF Artefact Limit": BBs,
            "Kinetic Limit": CCs, 
            "Diffusive Onset": np.maximum(CCs, DDs),
            "Diffusive-Capacitive Onset": EEs,
            "LF Artefact Onset": Fs,
        }, dtype=pd.Int64Dtype()).replace({-1: pd.NA, nfreqs: pd.NA}).apply(lambda col: col.map(freqs_series))

        region_df["Mass Transport Resolvable"] = has_mass_transport
        region_df["Sample Name"] = sample_names
        self._data = self._root._add_metadata_to_data(region_df)