import numpy as np
import pandas as pd

from scipy import signal as sig

import matplotlib.pyplot as plt

from ._args import call_argument_parser
from ..data.experiment import Experiment

_DEFAULT_SAVGOL_WINDOW = 11
_SAVGOL_POLYORDER = 3
_HF_ARTEFACT_SEARCH_SPREAD = 3

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


class Regions:
    _DEFAULT_CALCULATION_ARGS = {"min_peak_prominence": 0.25}

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
            return f < col("LF Artefact", -np.inf)
        if region == "artefact:hf":
            return f >= col("HF Artefact", np.inf)
        if region == "inductive":
            return f > col("Capacitive Limit", np.inf)

        valid = ~(
            self.make_mask("artefact:lf")
            | self.make_mask("artefact:hf")
            | self.make_mask("inductive")
        )
        if region == "valid":
            return valid

        main, secondary = col("Main Kink", np.inf), col("Secondary Kink", np.inf)
        if region == "kinetic":
            mask = f > col("Main Kink", -np.inf)
        elif region == "diffusive":
            mask = f <= main
        elif region == "diffusive:45":
            mask = (f <= main) & (f > secondary)
        elif region == "diffusive:90":
            mask = (f <= main) & (f <= secondary)
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
        raw_magnitudes = self._root.data["Offset-Corrected Impedance"].to_numpy().reshape(nsamples, nfreqs)
        raw_phases = self._root.data["Offset-Corrected Phase"].to_numpy().reshape(nsamples, nfreqs)


        dfs = []
        for sample_args, magn, phase, in zip(args_list, raw_magnitudes, raw_phases):
            region: dict = {}
            self._find_hf_artefact(region, magn=magn, phase=phase)
            self._find_kink(region, magn=magn, phase=phase, min_peak_prominence=sample_args["min_peak_prominence"])
            self._find_lf_artefact(region, phase=phase)
            self._find_capacitive_limit(region, phase=phase)

            dfs.append(pd.Series(region, dtype=pd.Int64Dtype()))
        
        freqs_series = pd.Series(freqs)
        region_idc_df = pd.DataFrame(dfs, dtype=pd.Int64Dtype())
        region_df = region_idc_df.apply(lambda col: col.map(freqs_series))

        region_df["Sample Name"] = sample_names

        self._data = self._root._add_metadata_to_data(region_df)

    
    @staticmethod
    def _find_kink(region: dict, *, magn: np.ndarray, phase: np.ndarray, min_peak_prominence: float = 0.25) -> None:
        x = smooth(magn * np.cos(np.radians(phase)))
        y = smooth(magn * np.sin(np.radians(phase)))

        # Restrict search space to under HF artefact
        hf_artefact_idx = region.get("HF Artefact") or 0
        
        curv = curvature(x, y)[hf_artefact_idx:]

        # We go from HF to LF, so we want positive curvature peaks
        kink_idc, _ = sig.find_peaks(curv)
        if not len(kink_idc):
            region |= {"Main Kink": None, "Secondary Kink": None}
            return

        # Normalize against the strongest peak
        curv = curv / np.max(curv[kink_idc])
        kink_idc = kink_idc[curv[kink_idc] >= min_peak_prominence]
        kink_idc = kink_idc[np.argsort(curv[kink_idc])[::-1]]

        
        main_kink_idx = kink_idc[0]
        later = kink_idc[kink_idc > main_kink_idx]
        second_kink_idx = later[0] if len(later) else None

        region |= {
            "Main Kink": main_kink_idx + hf_artefact_idx,
            "Secondary Kink": None if second_kink_idx is None else second_kink_idx + hf_artefact_idx,
        }

    @staticmethod
    def _find_hf_artefact(region: dict, *, magn: np.ndarray, phase: np.ndarray) -> None:
        slope = np.gradient(smooth(magn))

        pos_idc = np.where(slope < 0)[0]

        if len(pos_idc):
            target_idx = pos_idc[-1] + 1
            b, e = max(0, target_idx - _HF_ARTEFACT_SEARCH_SPREAD), min(len(magn), target_idx + _HF_ARTEFACT_SEARCH_SPREAD)
            hf_artefact_idx = b + np.argmax(phase[b:e])
        else:
            hf_artefact_idx = None

        region |= {"HF Artefact": hf_artefact_idx}

    @staticmethod
    def _find_lf_artefact(region: dict, *, phase: np.ndarray) -> None:
        kink_idx = region.get("Main Kink") or 0

        peak_idc, _ = sig.find_peaks(-smooth(phase)[kink_idx:])

        region |= {"LF Artefact": peak_idc[0] + kink_idx if len(peak_idc) else None}

    @staticmethod
    def _find_capacitive_limit(region: dict, *, phase: np.ndarray) -> None:
        inductive_mask = np.logical_or.accumulate((phase > 0)[::-1])[::-1]
        idc = np.where(~inductive_mask)[0]

        region |= {"Capacitive Limit": idc[0] if len(idc) else None}