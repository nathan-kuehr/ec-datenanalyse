import pyimpspec

import numpy as np
import pandas as pd

from . import parallel
from .payloads import ImpedancePayload, KramersKronigPayload
from ..data.experiment import Experiment


@parallel.Cache.cache
def _cached_kramers_kronig_test(
    data: ImpedancePayload, kwargs: dict
) -> KramersKronigPayload:
    ds = pyimpspec.DataSet(data.frequencies, data.impedances)
    kkt_result = pyimpspec.perform_kramers_kronig_test(ds, **kwargs)
    return KramersKronigPayload(
        kkt_result.frequencies, kkt_result.residuals * 100, data.sample_name
    )


class KKT:
    def __init__(self, root: Experiment) -> None:
        self._root = root

        self._data = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self.__call__()
            assert self._data is not None
        return self._data

    def __call__(self, *args, **kwargs):
        data = self._root.data

        sample_names = data["Sample Name"].unique()

        #
        Default_Calculation_Args = {"test": "complex"}

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
                    "Please provide a dictionary mapping the sample names to the parameters to apply for the KKT calculation."
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

        # Find frequencies w/ inductive behaviour & mask them
        ind_freqs = data[data["Neg. Reactance"] < 0]["Frequency"].unique()
        masks = [~np.isin(data["Frequency"].unique(), ind_freqs)] * len(sample_names)
        input_list = ImpedancePayload.From_Data(data, masks)

        # Run the parallelized KKT calculation
        payloads = parallel.multiprocess(
            method=self._Calculate_Single_KKT,
            input=input_list,
            tqdm_note=f"Calculating Kramers-Kronig Tests for EIS Experiment '{self._root.name}'",
            args=argument_list,
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
        return self.Compile_Stats_Data(self.data, zero_centered, pool)

    @staticmethod
    def Compile_Stats_Data(
        data: pd.DataFrame, zero_centered: bool = False, pool: bool = False
    ) -> np.ndarray:
        flat_residuals = data[["Real Residual", "Imag. Residual"]].to_numpy()

        nsamples = data["Sample Name"].nunique()
        nfreqs = len(data) // nsamples

        # Reshape. If we do pooling, just append the samples on the frequency axis to each other
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
    def As_Component_Data(data: pd.DataFrame) -> pd.DataFrame:
        return data.rename(
            columns={"Real Residual": "Real", "Imag. Residual": "Imaginary"}
        ).melt(
            id_vars=["Frequency", "Sample Name", "Palette"],
            value_vars=["Real", "Imaginary"],
            var_name="Component",
            value_name="Residual",
        )

    @staticmethod
    def _Calculate_Single_KKT(payload: ImpedancePayload, **kwargs):
        return _cached_kramers_kronig_test(payload, kwargs)
