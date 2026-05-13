import pyimpspec

import numpy as np
import pandas as pd

from . import parallel
from .payloads import ImpedancePayload, KramersKronigPayload
from ..data.experiment import Experiment


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


def _call_argument_parser(
    args: tuple,
    kwargs: dict,
    default: dict,
    sample_names: list[str],
    name: str = "",
) -> list:
    nargs, nkwargs = len(args), len(kwargs)

    if nargs > 0 and nkwargs == 0:
        if nargs > 1:
            raise ValueError(
                f"Only one positional argument allowed, but {nargs} were given."
            )
        elif not isinstance(args[0], dict):
            raise ValueError(
                f"Please provide a dictionary mapping the sample names to the parameters to apply for the {name} calculation."
            )
        return [default | args[0].get(name, {}) for name in sample_names]
    elif nkwargs >= 0 and nargs == 0:
        return [(default | kwargs) for _ in sample_names]
    else:
        raise ValueError(
            "Please provide either only positional or only keyword arguments, not both."
        )


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

    def __call__(self, *args, **kwargs) -> None:
        data = self._root.data
        sample_names = list(data["Sample Name"].unique())

        args_list = _call_argument_parser(
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
        return self.compile_stats_data(self.data, zero_centered, pool)

    @staticmethod
    def compile_stats_data(
        data: pd.DataFrame, zero_centered: bool = False, pool: bool = False
    ) -> np.ndarray:
        flat_residuals = data[["Real Residual", "Imag. Residual"]].to_numpy()

        nsamples = data["Sample Name"].nunique()
        nfreqs = len(data) // nsamples

        # Reshape. If pooling, append the samples on the frequency axis to each other
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

    @staticmethod
    def as_component_data(data: pd.DataFrame) -> pd.DataFrame:
        return data.rename(
            columns={"Real Residual": "Real", "Imag. Residual": "Imaginary"}
        ).melt(
            id_vars=["Frequency", "Sample Name", "Palette"],
            value_vars=["Real", "Imaginary"],
            var_name="Component",
            value_name="Residual",
        )
