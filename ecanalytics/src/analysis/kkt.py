import pyimpspec

import numpy as np
import pandas as pd

from .. import parallel
from ._args import call_argument_parser
from .payloads import ImpedancePayload, KramersKronigPayload
from ..data.experiment import Experiment, SimulatedExperiment


@parallel.CACHE.cache
def _cached_kramers_kronig_test(
    payload: ImpedancePayload, kwargs: dict
) -> KramersKronigPayload:
    kkt = pyimpspec.perform_kramers_kronig_test(
        data=pyimpspec.DataSet(payload.frequencies, payload.impedances), **kwargs
    )
    return KramersKronigPayload(
        kkt.frequencies, kkt.residuals * 100, payload.sample_name
    )


def compile_residual_stats(
    data: pd.DataFrame, zero_centered: bool = False, pool: bool = False
) -> np.ndarray:
    """Computes RMS / std / correlation for residual data.

    Returns an (nsamples, 4) array of (rms, std_real, std_imag, rho) — with nsamples=1 if pool=True.
    """
    flat_residuals = data[["Real Residual", "Imag. Residual"]].to_numpy()

    nsamples = data["Sample Name"].nunique()
    nfreqs = len(data) // nsamples

    # If pooling, append the samples on the frequency axis to each other
    shape = (1, nsamples * nfreqs, 2) if pool else (nsamples, nfreqs, 2)
    residuals = np.reshape(flat_residuals, shape)

    rms = np.sqrt(2 * np.mean(np.square(residuals), axis=(1, 2)))[:, None]

    if not zero_centered:
        delta_res = residuals - np.mean(residuals, axis=1, keepdims=True)
    else:
        delta_res = residuals

    std = np.sqrt(np.mean(np.square(delta_res), axis=1))
    cov = np.mean(np.prod(delta_res, axis=2), axis=1)

    std_prod = std[:, 0] * std[:, 1]
    rho = np.zeros_like(cov)
    np.divide(cov, std_prod, out=rho, where=(std_prod != 0))

    return np.hstack((rms, std, rho[:, None]))


def as_component_data(data: pd.DataFrame) -> pd.DataFrame:
    melted = data.rename(
        columns={"Real Residual": "Real", "Imag. Residual": "Imaginary"}
    ).melt(
        id_vars=["Frequency", "Sample Name"],
        value_vars=["Real", "Imaginary"],
        var_name="Component",
        value_name="Residual",
    )[["Frequency", "Residual", "Component", "Sample Name"]]
    return melted


class KKT:
    _DEFAULT_CALCULATION_ARGS = {"test": "complex"}

    def __init__(self, root: Experiment) -> None:
        self._root = root
        self._data: pd.DataFrame | None = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self.__call__()
            assert self._data is not None
        return self._data
    
    @property
    def data_long(self) -> pd.DataFrame:
        data = as_component_data(self.data)
        return self._root._add_metadata_to_data(data)


    def __call__(self, *args, **kwargs) -> None:
        assert not isinstance(self._root, SimulatedExperiment)
        data = self._root.data
        sample_names = self._root.sample_names

        args_list = call_argument_parser(
            args, kwargs, self._DEFAULT_CALCULATION_ARGS, sample_names, "KKT"
        )

        # Find frequencies with inductive behaviour & mask them
        inductive_freqs = data[data["Neg. Reactance"] < 0]["Frequency"].unique()
        masks = [~np.isin(data["Frequency"].unique(), inductive_freqs)] * len(sample_names)
        input_list = ImpedancePayload.from_data(data, masks)

        payloads: list[KramersKronigPayload] = parallel.multiprocess(
            method=_cached_kramers_kronig_test,
            inputs=input_list,
            tqdm_note=f"Calculating Kramers-Kronig Tests for EIS Experiment '{self._root.name}'",
            args=args_list,
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

        df = pd.concat(dfs, axis=0, ignore_index=True)
        self._data = self._root._add_metadata_to_data(df)

    def compile_stats(
        self, zero_centered: bool = True, pool: bool = False
    ) -> np.ndarray:
        return compile_residual_stats(self.data, zero_centered, pool)