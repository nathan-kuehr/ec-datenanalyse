import numpy as np
import pandas as pd

from pyimpspec import DataSet as APIDataSet
from pyimpspec import KramersKronigResult as APIKramersKronigResult


class PyimpspecAdapter:
    @staticmethod
    def To_API_Dataset(data: pd.DataFrame) -> list[APIDataSet]:
        freqs = data["Frequency"].unique()
        inductive_freqs = data[data["Neg. Reactance"] < 0]["Frequency"].unique()

        mask = {i: (f in inductive_freqs) for i, f in enumerate(freqs)}

        def complex_imp(data: pd.DataFrame) -> np.ndarray:
            return (data["Resistance"] - 1j * data["Neg. Reactance"]).to_numpy()

        return [
            APIDataSet(freqs, complex_imp(group), mask=mask, label=name)
            for name, group in data.groupby("Sample Name", sort=False)
        ]

    @staticmethod
    def To_Residuals_Dataframe(
        kk_results: list[APIKramersKronigResult], sample_names: list[str]
    ) -> pd.DataFrame:
        residuals = list()

        for result, name in zip(kk_results, sample_names):
            df = pd.DataFrame(result.get_residuals_data()).transpose()
            df["Sample Name"] = name
            residuals.append(df)

        df = pd.concat(residuals, axis=0, ignore_index=True)
        df.columns = ["Frequency", "Real Residual", "Imag. Residual", "Sample Name"]

        return df
