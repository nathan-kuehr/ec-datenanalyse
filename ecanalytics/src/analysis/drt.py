from __future__ import annotations

import pyimpspec

import numpy as np
import pandas as pd

from . import parallel
from .kkt import _call_argument_parser
from .payloads import ImpedancePayload, DRTPayload
from ..data.experiment import Experiment


_PEAK_SIGMA_MAX = 10
_PEAK_INFO_COLUMN_COUNT = 5


@parallel.CACHE.cache
def _cached_drt(payload: ImpedancePayload, **kwargs) -> DRTPayload:
    drt = pyimpspec.calculate_drt(
        data=pyimpspec.DataSet(payload.frequencies, payload.impedances), **kwargs
    )

    # Get calculated log polarization density
    taus, gammas = drt.get_drt_data()

    api_peaks = drt.analyze_peaks()
    peak_infos = [_peak_info(peak, taus) for peak in api_peaks if peak.sigma < _PEAK_SIGMA_MAX]

    if len(api_peaks.peaks) != len(peak_infos):
        print(
            f"Warning: ill-fitted peaks detected for {payload.sample_name}. Consider adapting the DRT cutoff frequency."
        )

    return DRTPayload(taus, gammas, peak_infos, payload.sample_name)


def _peak_info(peak: pyimpspec.DRTPeak, tau_grid: np.ndarray) -> dict:
    return {
        "Log. Position": peak.position * peak.x_scale + peak.x_offset,
        "Log. Sigma": peak.sigma * peak.x_scale,
        "Max. Gamma": peak.height * peak.y_scale + peak.y_offset,
        "Alpha": peak.alpha,
        "Polarisation": peak.get_area(tau_grid),
    }


class DRT:
    _DEFAULT_CALCULATION_ARGS = {
        "cutoff_frequency": 15,  # Hz
        "method": "tr-nnls",
    }

    def __init__(self, root: Experiment) -> None:
        self._root = root
        self._data: pd.DataFrame | None = None
        self._peak_data: pd.DataFrame | None = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self.__call__()
            assert self._data is not None
        return self._data

    @property
    def peak_data(self) -> pd.DataFrame:
        if self._peak_data is None:
            self.__call__()
            assert self._peak_data is not None
        return self._peak_data

    def __call__(self, *args, **kwargs) -> None:
        data = self._root.data
        sample_names = list(data["Sample Name"].unique())

        args_list = _call_argument_parser(
            args, kwargs, self._DEFAULT_CALCULATION_ARGS, sample_names, "DRT"
        )

        freqs = data["Frequency"].unique()
        masks = [freqs > args.pop("cutoff_frequency") for args in args_list]
        input_list = ImpedancePayload.from_data(data, masks)

        payloads: list[DRTPayload] = parallel.multiprocess(
            method=_cached_drt,
            inputs=input_list,
            tqdm_note=f"Calculating the DRT for EIS Experiment '{self._root.name}'",
            args=args_list,
        )

        drt_dfs, peak_dfs = [], []
        for payload in payloads:
            drt_dfs.append(
                pd.DataFrame(
                    {
                        "Time Constant": payload.taus,
                        "Polarization Density": payload.gammas,
                        "Sample Name": payload.sample_name,
                    }
                )
            )
            peak_dfs.append(
                pd.DataFrame(payload.peak_infos).assign(
                    **{"Sample Name": payload.sample_name}
                )
            )
        drt_df = pd.concat(drt_dfs, axis=0, ignore_index=True)
        peak_df = pd.concat(peak_dfs, axis=0, ignore_index=True)

        self._data = self._root._add_metadata_to_data(drt_df)
        self._peak_data = self._root._add_metadata_to_data(peak_df)

    def manual_peak_correct(self, name: str, tau: float, peak_info: list[float]) -> DRT:
        mask = self.peak_data["Sample Name"] == name
        target = (self.peak_data.loc[mask, "Log. Position"] - np.log10(tau)).abs().idxmin()

        assert len(peak_info) == _PEAK_INFO_COLUMN_COUNT
        self.peak_data.iloc[target, 0:_PEAK_INFO_COLUMN_COUNT] = peak_info

        return self

    def peak_select(self, target_tau: list[float]) -> pd.DataFrame:
        nselect = len(target_tau)

        selected_peak_rows = []

        target_log_tau = np.log10(target_tau)

        grouped = self.peak_data.groupby("Sample Name", sort=False)
        for name, peaks in grouped:
            log_pos = peaks["Log. Position"].to_numpy().astype(float)

            if len(log_pos) < nselect:
                raise ValueError(
                    f"Not enough peaks found for sample '{name}' to select {nselect} peaks. Found only {len(log_pos)} peaks."
                )

            for target in target_log_tau:
                peak_idx = np.argmin(np.abs(log_pos - target))

                selected_peak_rows.append(peaks.iloc[peak_idx])

                # Forbid multiple selection of the same peak
                log_pos[peak_idx] = np.inf

        return pd.DataFrame(selected_peak_rows).reset_index(drop=True)

    @staticmethod
    def sample_peak_data(peak_data: pd.DataFrame, tau_grid: np.ndarray) -> np.ndarray:
        nsamples = peak_data["Sample Name"].nunique()
        npeaks = len(peak_data) // nsamples
        ntau = len(tau_grid)

        if npeaks * nsamples != len(peak_data):
            raise ValueError("The passed peak data frame is not valid!")

        cols = ["Log. Position", "Log. Sigma", "Alpha", "Max. Gamma"]
        pos, sigma, alpha, gamma = peak_data[cols].to_numpy().T[:, :, None]

        pos_grid = np.log10(tau_grid)[None, :]

        # Skew normal evaluation
        dpos = pos_grid - pos
        denom = 2 * sigma**2
        num = (dpos * (1 + alpha * np.sign(dpos))) ** 2

        sampled_peaks = gamma * np.exp(-num / denom)
        return sampled_peaks.reshape((nsamples, npeaks, ntau))

    def sample_peaks(self, target_tau: list[float], tau_grid: np.ndarray) -> np.ndarray:
        return self.sample_peak_data(self.peak_select(target_tau), tau_grid)
