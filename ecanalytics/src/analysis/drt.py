import pyimpspec

import numpy as np
import pandas as pd

from . import parallel
from .payloads import ImpedancePayload, DRTPayload
from ..data.experiment import Experiment


@parallel.Cache.cache
def _cached_drt(payload: ImpedancePayload, kwargs: dict) -> DRTPayload:
    drt = pyimpspec.calculate_drt(
        data=pyimpspec.DataSet(payload.frequencies, payload.impedances), **kwargs
    )

    # Get calculated log polarization density
    taus, gammas = drt.get_drt_data()

    api_peaks = drt.analyze_peaks()
    peak_infos = [_peak_info(peak, taus) for peak in api_peaks if peak.sigma < 10]

    if len(api_peaks.peaks) != len(peak_infos):
        print(
            f"Warning: ill-fitted peaks detected for {payload.sample_name}. Consider adapting the DRT cutoff frequency."
        )

    return DRTPayload(taus, gammas, peak_infos, payload.sample_name)


def _peak_info(peak: pyimpspec.DRTPeak, tau_grid: np.ndarray) -> dict:
    info = dict()

    info["Log. Position"] = peak.position * peak.x_scale + peak.x_offset
    info["Log. Sigma"] = peak.sigma * peak.x_scale
    info["Max. Gamma"] = peak.height * peak.y_scale + peak.y_offset
    info["Alpha"] = peak.alpha
    info["Polarisation"] = peak.get_area(tau_grid)

    return info


class DRT:
    def __init__(self, root: Experiment):
        self._root = root
        self._data = None
        self._peak_data = None

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

        sample_names = data["Sample Name"].unique()

        #
        Default_Calculation_Args = {
            "cutoff_frequency": 15,  # Hz
            "method": "tr-nnls",
        }

        # Count of arguments
        nargs, nkwargs = len(args), len(kwargs)

        # Prepare the args for each sample
        if nargs > 0 and nkwargs == 0:
            if nargs > 1:
                raise ValueError(
                    f"Only one positional argument allowed, but {nargs} were given."
                )
            elif not isinstance(args[0], dict):
                raise ValueError(
                    "Please provide a dictionary mapping the sample names to the parameters to apply for the DRT calculation."
                )
            argument_list = [
                Default_Calculation_Args | args[0].get(name, {})
                for name in sample_names
            ]
        elif nkwargs >= 0 and nargs == 0:
            argument_list = [(Default_Calculation_Args | kwargs) for _ in sample_names]
        else:
            raise ValueError(
                "Please provide either only positional or only keyword arguments, not both."
            )

        # Prepare the input
        freqs = data["Frequency"].unique()
        masks = [freqs > args.pop("cutoff_frequency") for args in argument_list]
        input_list = ImpedancePayload.From_Data(data, masks)

        # Run the parallelized DRT calculation
        payloads: list[DRTPayload] = parallel.multiprocess(
            method=DRT._Calculate_Single_DRT,
            input=input_list,
            tqdm_note=f"Calculating the DRT for EIS Experiment '{self._root.name}'",
            args=argument_list,
        )

        # Post process the payloads into data frames
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

        # Add metadata
        self._data = self._root._add_metadata_to_data(drt_df)
        self._peak_data = self._root._add_metadata_to_data(peak_df)

    def peak_select(self, target_tau: list[float]) -> pd.DataFrame:
        # Length
        nselect = len(target_tau)

        selected_peak_rows = []

        # Transform tau to log tau space
        target_log_tau = np.log10(target_tau)

        grouped = self.peak_data.groupby("Sample Name", sort=False)
        for name, peaks in grouped:
            log_pos = peaks["Log. Position"].to_numpy().astype(float)

            # We need at least npeaks peaks found for that sample!
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
    def Sample_Peak_Data(peak_data: pd.DataFrame, tau_grid: np.ndarray) -> np.ndarray:
        nsamples = peak_data["Sample Name"].nunique()
        npeaks = len(peak_data) // nsamples
        ntau = len(tau_grid)

        if npeaks * nsamples != len(peak_data):
            raise ValueError("The passed peak data frame is not valid!")

        # Get peak data on 1. dim
        cols = ["Log. Position", "Log. Sigma", "Alpha", "Max. Gamma"]
        pos, sig, alpha, gamma = peak_data[cols].to_numpy().T[:, :, None]

        # Get taus on 2. dim
        pos_grid = np.log10(tau_grid)[None, :]

        # Calculate skew normals
        dpos = pos_grid - pos
        denom = 2 * sig**2
        num = (dpos * (1 + alpha * np.sign(dpos))) ** 2

        sampled_peaks = gamma * np.exp(-num / denom)
        return sampled_peaks.reshape((nsamples, npeaks, ntau))

    def sample_peaks(self, target_tau: list[float], tau_grid: np.ndarray) -> np.ndarray:
        return self.Sample_Peak_Data(self.peak_select(target_tau), tau_grid)

    @staticmethod
    def _Calculate_Single_DRT(payload: ImpedancePayload, **kwargs):
        return _cached_drt(payload, kwargs)
