import numpy as np
import pandas as pd
import logging

from scipy import signal as sig

import matplotlib.pyplot as plt

from typing import Iterable
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
    _DEFAULT_CALCULATION_ARGS = {"a_minprom": 25.0, "a_minw": 2.0, "cap_angle": 70.0, "d_minprom": 0.05, "tail_angle_tol": 3}

    def __init__(self, root: Experiment) -> None:
        self._root = root

        self._data: pd.DataFrame | None = None
        self._curve_data: pd.DataFrame | None = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self()
            assert self._data is not None
        return self._data
    
    @property
    def curve_data(self) -> pd.DataFrame:
        if self._curve_data is None:
            self()
            assert self._curve_data is not None
        return self._curve_data

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

        # Use tangent angle (in Nyquist plot w/ inverted Im part) to find transition between semicircle and diffusion part
        phis = -np.degrees(np.arctan2(np.gradient(smooth_xs, axis=1), np.gradient(smooth_rs, axis=1)))
        dphis = np.gradient(phis, axis=1)

        ## Kinetic Limit
        # -> transistion between semicircle and diffusion part
        # A curvature based approach was used before but didn't work out as well
        peaks = [
            sig.find_peaks(-phi, prominence=args["a_minprom"], width=args["a_minw"])[0] 
            for phi, args in zip(phis, args_list)
        ]
        kin_lims = np.array([peak[-1] if len(peak) else nfreqs for peak in peaks], dtype=int)

        ## Diffusive-Capacitive Onset:
        # -> beginning of capacitive tail
        # A first analysis was done using curvature peaks but this is easier
        cap_angles = np.array([args["cap_angle"] for args in args_list])
        cap_tail_mask = (phis >= cap_angles[:, None]) & (np.arange(nfreqs) > kin_lims[:, None])
        diff_cap_onsets = np.where(np.any(cap_tail_mask, axis=1), np.argmax(cap_tail_mask, axis=1), -1).astype(int)

        ## Diffusive-Onset:
        # -> sometimes A is not perfectly well chosen but a bit too early.
        # Take minimum between A and B
        freq_indices = np.arange(nfreqs)
        segment = (freq_indices[None, :] >= kin_lims[:, None]) & (freq_indices[None, :] < diff_cap_onsets[:, None])
        diff_onsets = np.maximum(kin_lims, np.argmax(np.where(segment, smooth_xs, -np.inf), axis=1))

        ## Mass transport ? 
        # -> if we see that there is a plateau in φ (<=> two peaks in dφ <=> a valley in -dφ), there is diffusive:mass transport region here
        norm_dphis = dphis / np.max(np.where(segment, dphis, 1e-10), axis=1, keepdims=True)
        valleys = np.array([
            sig.find_peaks(-ndphi, prominence=args["d_minprom"])[0] 
            for ndphi, args in zip(norm_dphis, args_list)
        ])

        has_mass_transport = np.array([np.sum(s[p]) > 0 for s, p in zip(segment, valleys)])

        ## HF Artefact Limit:
        valleys = [sig.find_peaks(x[:lim])[0] for x, lim in zip(smooth_xs, kin_lims)]
        hf_artefact_lims = np.array([valley[-1] if len(valley) else nfreqs for valley in valleys], dtype=int)

        ## Inductive artefacts
        inductive_lims = np.where(raw_xs > 0, freq_indices, -1).max(axis=1)

        ## Point F:
        tail_angle_tols = np.array([args["tail_angle_tol"] for args in args_list])

        max_phis_idc = np.maximum(diff_cap_onsets, np.argmax(phis, axis=1))
        max_phis = phis[np.arange(nsamples), max_phis_idc]

        tail_drop_mask = ((phis < (max_phis - tail_angle_tols)[:, None])) & (freq_indices[None, :] > max_phis_idc[:, None])

        lf_artefact_lims = np.where(tail_drop_mask, freq_indices, nfreqs).min(axis=1)

        # Put everything together
        freqs_series = pd.Series(freqs)

        region_df = pd.DataFrame({
            "Inductive Limit": inductive_lims,
            "HF Artefact Limit": hf_artefact_lims,
            "Kinetic Limit": kin_lims, 
            "Diffusive Onset": diff_onsets,
            "Diffusive-Capacitive Onset": diff_cap_onsets,
            "LF Artefact Onset": lf_artefact_lims,
        }, dtype=pd.Int64Dtype()).replace({-1: pd.NA, nfreqs: pd.NA}).apply(lambda col: col.map(freqs_series))

        region_df["Mass Transport Resolvable"] = has_mass_transport
        region_df["Sample Name"] = sample_names
        self._data = self._root._add_metadata_to_data(region_df)

        curve_df = pd.DataFrame({
            "Frequency": np.tile(freqs, nsamples),
            "Tangent Angle": phis.reshape(-1, 1).squeeze(),
            "Tangent Angle Derivative": dphis.reshape(-1, 1).squeeze(),
            "Sample Name": np.repeat(sample_names, nfreqs)
        })
        self._curve_data = self._root._add_metadata_to_data(curve_df)

    @staticmethod
    def select_in(data: pd.DataFrame, region: pd.DataFrame, rpoints: str | Iterable[str] | None = None) -> pd.DataFrame:
        add_point_col = not isinstance(rpoints, str)

        if rpoints is None:
            rpoints = ["Inductive Limit", "HF Artefact Limit", "Kinetic Limit", "Diffusive Onset", "Diffusive-Capacitive Onset", "LF Artefact Onset"]
        elif isinstance(rpoints, str):
            rpoints = [rpoints]

        regions_long = region.melt(
            id_vars=["Sample Name"],
            value_vars=rpoints,
            var_name="Point",
            value_name="Frequency",
        )

        selected = pd.merge(data, regions_long, how="inner", on=["Frequency", "Sample Name"])

        return selected if add_point_col else selected.drop("Point", axis=1)