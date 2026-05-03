import os

import numpy as np
import pandas as pd

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import pyimpspec as _API
from pyimpspec import Circuit

from dataclasses import dataclass

from joblib import Memory


_cache = Memory("./lab-analytics-cache", verbose=0)


@_cache.cache
def _cached_kramers_kronig_test(
    data: ImpedancePayload, kwargs: dict
) -> KramersKronigTest:
    ds = _API.DataSet(data.frequencies, data.impedances)
    kkt_result = _API.perform_kramers_kronig_test(ds, **kwargs)
    return KramersKronigTest(
        kkt_result.frequencies, kkt_result.residuals * 100, data.sample_name
    )


@_cache.cache
def _cached_drt(payload: ImpedancePayload, kwargs: dict) -> DRT:
    drt = _API.calculate_drt(
        data=_API.DataSet(payload.frequencies, payload.impedances), **kwargs
    )

    # Get calculated log polarization density
    taus, gammas = drt.get_drt_data()

    api_peaks = drt.analyze_peaks()

    peak_infos = [_peak_info(peak, taus) for peak in api_peaks if peak.sigma < 10]

    if len(api_peaks.peaks) != len(peak_infos):
        print(
            "Warning: ill-fitted peaks detected. Consider adapting the DRT cutoff frequency."
        )

    return DRT(taus, gammas, peak_infos, payload.sample_name)


@_cache.cache
def _cached_fit_circuit(
    data: APIAdapter.ImpedancePayload, cdc: str, kwargs
) -> APIAdapter.FitResultPayload:
    ds = _API.DataSet(data.frequencies, data.impedances)
    circuit = _API.parse_cdc(cdc)
    fit_result = _API.fit_circuit(circuit=circuit, data=ds, **kwargs)

    params = {}
    for elem, fit in fit_result.parameters.items():
        label = elem.split("_", maxsplit=1)[1] if "_" in elem else ""

        for p, fit_p in fit.items():
            name = f"{p}_{label}" if label != "" else p
            if name in params:
                raise ValueError(
                    f"Parameter name conflict: '{name}' already exists in params dict."
                )
            params[name] = {"Value": fit_p.value, "Std. Error": fit_p.stderr}

    return APIAdapter.FitResultPayload(
        fit_result.frequencies, fit_result.impedances, fit_result.residuals, params
    )


def _data_exp_name(data: pd.DataFrame) -> str:
    return str(data["Experiment Name"].iloc[0])


def _multiprocess(
    method: Callable, input: list, tqdm_note: str = "", args: dict = {}
) -> list:
    # Prepare list which stores the results in correct order
    results = [None] * len(input)

    with ProcessPoolExecutor(max_workers=(3 * (os.cpu_count() or 2) // 4)) as executor:
        # Get futures of the processed data
        futures = {
            executor.submit(method, sample, **args): i for i, sample in enumerate(input)
        }

        # Update progress bar as futures are filled
        for future in tqdm(
            as_completed(futures),
            total=len(input),
            desc=tqdm_note,
        ):
            results[futures[future]] = future.result()

    return results


def _peak_info(peak: _API.DRTPeak, tau_grid: np.ndarray) -> dict:
    info = dict()

    info["Log. Position"] = peak.position * peak.x_scale + peak.x_offset
    info["Log. Sigma"] = peak.sigma * peak.x_scale
    info["Max. Gamma"] = peak.height * peak.y_scale + peak.y_offset
    info["Alpha"] = peak.alpha
    info["Polarisation"] = peak.get_area(tau_grid)

    return info


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
class KramersKronigTest:
    frequencies: np.ndarray
    residuals: np.ndarray
    sample_name: str

    @staticmethod
    def Run(
        data: pd.DataFrame, mask: np.ndarray | None = None, **kwargs
    ) -> pd.DataFrame:
        payloads = _multiprocess(
            method=KramersKronigTest._From_Single_Payload,
            input=ImpedancePayload.From_Data(data, mask),
            tqdm_note=f"Calculating Kramers-Kronig Tests for EIS Experiment '{_data_exp_name(data)}'",
            args=kwargs,
        )

        dfs = []
        for payload in payloads:
            dfs.append(
                pd.DataFrame(
                    {
                        "Frequency": payload.frequencies,
                        "Real Residual": np.real(payload.residuals),
                        "Imag. Residual": np.imag(payload.residuals),
                        "Sample Name": payload.sample_name,
                    }
                )
            )

        return pd.concat(dfs, axis=0, ignore_index=True)

    @staticmethod
    def _From_Single_Payload(payload: ImpedancePayload, **kwargs):
        return _cached_kramers_kronig_test(payload, kwargs)


@dataclass
class DRT:
    taus: np.ndarray
    gammas: np.ndarray
    peak_infos: list[dict]
    sample_name: str

    @staticmethod
    def Calculate(
        data: pd.DataFrame, mask: np.ndarray | None = None, **kwargs
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        payloads = _multiprocess(
            method=DRT._From_Single_Payload,
            input=ImpedancePayload.From_Data(data, mask),
            tqdm_note=f"Calculating the DRT for EIS Experiment '{_data_exp_name(data)}'",
            args=kwargs,
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
        return pd.concat(drt_dfs, axis=0, ignore_index=True), pd.concat(
            peak_dfs, axis=0, ignore_index=True
        )

    @staticmethod
    def _From_Single_Payload(payload: ImpedancePayload, **kwargs):
        return _cached_drt(payload, kwargs)


class APIAdapter:
    @staticmethod
    def As_Impedance_Payload(
        data: pd.DataFrame, mask: np.ndarray | None = None
    ) -> list[ImpedancePayload]:
        freqs = data["Frequency"].unique()

        if mask is None:
            mask = np.ones_like(freqs, dtype=bool)

        def complex_imp(data: pd.DataFrame) -> np.ndarray:
            return (data["Resistance"] - 1j * data["Neg. Reactance"]).to_numpy()

        return [
            APIAdapter.ImpedancePayload(freqs[mask], complex_imp(group)[mask])
            for _, group in data.groupby("Sample Name", sort=False)
        ]

    @staticmethod
    def From_KKT_Payload(
        payloads: list[APIAdapter.KKTPayload], sample_names: list[str]
    ) -> pd.DataFrame:
        dfs = []
        for payload, name in zip(payloads, sample_names):
            dfs.append(
                pd.DataFrame(
                    {
                        "Frequency": payload.frequencies,
                        "Real Residual": np.real(payload.residuals),
                        "Imag. Residual": np.imag(payload.residuals),
                        "Sample Name": name,
                    }
                )
            )

        return pd.concat(dfs, axis=0, ignore_index=True)

    @staticmethod
    def From_DRT_Payload(
        payloads: list[APIAdapter.DRTPayload], sample_names: list[str]
    ) -> pd.DataFrame:
        dfs = []
        for payload, name in zip(payloads, sample_names):
            dfs.append(
                pd.DataFrame(
                    {
                        "Time Constant": payload.taus,
                        "Polarization Density": payload.gammas,
                        "Sample Name": name,
                    }
                )
            )
        return pd.concat(dfs, axis=0, ignore_index=True)

    @staticmethod
    def From_Fit_Result_Payload(
        payloads: list[APIAdapter.FitResultPayload], sample_names: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        dfs_freq = []
        dfs_p = []

        for payload, name in zip(payloads, sample_names):
            dfs_freq.append(
                pd.DataFrame(
                    {
                        "Frequency": payload.frequencies,
                        "Resistance": np.real(payload.impedances),
                        "Neg. Reactance": -np.imag(payload.impedances),
                        "Real Residual": np.real(payload.residuals),
                        "Imag. Residual": np.imag(payload.residuals),
                        "Sample Name": name,
                    }
                )
            )
            dfs_p.append(
                pd.DataFrame(payload.parameters | {"Sample Name": name})
                .reset_index()
                .rename(columns={"index": "Kind"})
            )

        return pd.concat(dfs_freq, axis=0, ignore_index=True), pd.concat(
            dfs_p, axis=0, ignore_index=True
        )

    @staticmethod
    def Kramers_Kronig_Test(
        data: APIAdapter.ImpedancePayload, **kwargs
    ) -> APIAdapter.KKTPayload:
        return _cached_kramers_kronig_test(data, kwargs)

    @staticmethod
    def Distribution_Of_Relaxation_Times(
        data: APIAdapter.ImpedancePayload, **kwargs
    ) -> APIAdapter.DRTPayload:
        return _cached_drt(data, kwargs)

    @staticmethod
    def Fit_Circuit(
        data: APIAdapter.ImpedancePayload, circuit: Circuit | str, **kwargs
    ) -> APIAdapter.FitResultPayload:
        return _cached_fit_circuit(
            data, circuit if isinstance(circuit, str) else circuit.to_string(10), kwargs
        )

    @dataclass
    class ImpedancePayload:
        frequencies: np.ndarray
        impedances: np.ndarray

    @dataclass
    class KKTPayload:
        frequencies: np.ndarray
        residuals: np.ndarray

    @dataclass
    class DRTPayload:
        taus: np.ndarray
        gammas: np.ndarray

    @dataclass
    class FitResultPayload:
        frequencies: np.ndarray
        impedances: np.ndarray
        residuals: np.ndarray
        parameters: dict[str, dict]
