from __future__ import annotations

import logging

import pyimpspec
import pyimpspec.analysis.drt.peak_analysis as pa

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

from .. import parallel
from ._args import call_argument_parser
from .payloads import ImpedancePayload, DRTPayload
from ..data.experiment import Experiment, SimulatedExperiment


_PEAK_SIGMA_MAX = 10
_PEAK_INFO_COLUMN_COUNT = 5

# Monkey-patch the original parameter generation function to fight vanishing gradients
_original_generate_parameters = pa._generate_parameters

def _make_generate_parameters_wrapper(taus: np.ndarray, sigma_multiplier: float = 3.0):
    # Transform to logged x axis as in analyze_peaks()
    x = np.log(taus)
    x -= x.min()
    x /= x.max()

    def patch(peaks, disallow_skew):
        parameters, num_variables = _original_generate_parameters(
            peaks, disallow_skew
        )

        for i, (px, _) in enumerate(peaks):
            neighbour_dist_left = (px - x[x < px][-1]) if np.any(x < px) else x[1] - x[0]
            neighbour_dist_right = (x[x > px][0] - px) if np.any(x > px) else x[-1] - x[-2]

            sigma_init = parameters[f"sigma_{i}"].value
            # Alpha is 0 initialized, so ignore

            widthl_init = max(sigma_multiplier * sigma_init, neighbour_dist_left)
            widthr_init = max(sigma_multiplier * sigma_init, neighbour_dist_right)

            if disallow_skew:
                parameters.add(f"widthl_{i}", value=widthl_init, min=max(neighbour_dist_left, neighbour_dist_right))
                parameters.add(f"widthr_{i}", expr=f"widthl_{i}") # symmetric, so follow left width
            else:
                parameters.add(f"widthl_{i}", value=widthl_init, min=neighbour_dist_left)
                parameters.add(f"widthr_{i}", value=widthr_init, min=neighbour_dist_right)

            # Deactivate alpha and sigma
            parameters[f"sigma_{i}"].set(
                vary=False,
                expr=f"(2.0/{sigma_multiplier}) * (widthl_{i} * widthr_{i}) / (widthl_{i} + widthr_{i})",
            )
            parameters[f"alpha_{i}"].set(
                vary=False,
                expr=f"(widthl_{i} - widthr_{i}) / (widthl_{i} + widthr_{i})",
            )

        return parameters, num_variables

    return patch


@parallel.CACHE.cache
def _cached_drt(payload: ImpedancePayload, **kwargs) -> DRTPayload:
    drt = pyimpspec.calculate_drt(
        data=pyimpspec.DataSet(payload.frequencies, payload.impedances), **kwargs
    )

    # Get calculated log polarization density
    taus, gammas = drt.get_drt_data()

    # Monkey patch the parameter generation to fight vanishing gradients
    pa._generate_parameters = _make_generate_parameters_wrapper(taus)
    api_peaks = drt.analyze_peaks()
    pa._generate_parameters = _original_generate_parameters

    peak_infos = [_peak_info(peak, taus) for peak in api_peaks if peak.sigma < _PEAK_SIGMA_MAX]

    if len(api_peaks.peaks) != len(peak_infos):
        _logger.warning(
            "ill-fitted peaks detected for %s. Consider adapting the DRT cutoff frequency.",
            payload.sample_name,
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


def evaluate_peak_curves(peak_data: pd.DataFrame, tau_grid: np.ndarray) -> np.ndarray:
    nsamples = peak_data["Sample Name"].nunique()
    npeaks = len(peak_data) // nsamples
    ntau = len(tau_grid)

    if npeaks * nsamples != len(peak_data):
        raise ValueError("The passed peak data frame is not valid!")

    cols = ["Log. Position", "Log. Sigma", "Alpha", "Max. Gamma"]
    pos, sigma, alpha, gamma = peak_data[cols].to_numpy().T[:, :, None]

    pos_grid = np.log10(tau_grid)[None, :]

    # Skew-normal evaluation
    dpos = pos_grid - pos
    denom = 2 * sigma**2
    num = (dpos * (1 + alpha * np.sign(dpos))) ** 2

    sampled_peaks = gamma * np.exp(-num / denom)
    return sampled_peaks.reshape((nsamples, npeaks, ntau))


class DRT:
    _DEFAULT_CALCULATION_ARGS = {
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
            self()
        assert self._peak_data is not None
        return self._peak_data

    def __call__(self, *args, **kwargs) -> None:
        assert not isinstance(self._root, SimulatedExperiment)
        data = self._root.data

        args_list = call_argument_parser(
            args, kwargs, self._DEFAULT_CALCULATION_ARGS, self._root.sample_names, "DRT"
        )

        # Get masks
        masks = [m for m in ~self._root.analysis.regions.make_mask("diffusive:capacitive", overlay_valid=False)]
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
        """Return rows of peak_data closest (in log-tau) to each value in target_tau."""
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
