import os

import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from pyimpspec import KramersKronigResult, perform_kramers_kronig_test
from tqdm import tqdm

from ..data.experiment import Experiment
from .pyimpspec_adapter import PyimpspecAdapter
from ..config import DATA_QUALITY_SERIES_INFO


class DataQuality:
    Series_Info = DATA_QUALITY_SERIES_INFO

    # __Stat_Labels = ["Log pseudo chi-squared",
    #                  "Estimated SD of Gaussian noise (% of |Z|)",
    #                  "Mean of residuals, real (% of |Z|)",
    #                  "Mean of residuals, imag. (% of |Z|)",
    #                  "SD of residuals, real (% of |Z|)",
    #                  "SD of residuals, imag. (% of |Z|)"]

    def __init__(self, root: Experiment) -> None:
        self.__root = root

        self.__residuals = None

        # Calculate the Kramers Kronig Test Results
        self.__api_kk_result = self.__calculate_kramers_kronig()

    @property
    def residuals(self) -> pd.DataFrame:
        if self.__residuals is None:
            self.__residuals = pd.merge(
                left=PyimpspecAdapter.To_Residuals_Dataframe(
                    self.__api_kk_result, self.__root.data["Sample Name"].unique()
                ),
                right=self.__root.data[["Frequency", "Sample Name", "Palette"]],
                on=["Frequency", "Sample Name"],
                how="left",
            )

        return self.__residuals

    def __calculate_kramers_kronig(self) -> list[KramersKronigResult]:
        N = self.__root.data["Sample Name"].nunique()

        # Prepare list which stores the results in correct order
        results: list[None | KramersKronigResult] = [None] * N

        # Perform Kramers-Kronig tests in parallel, with progress bar
        with ProcessPoolExecutor(
            max_workers=(3 * (os.cpu_count() or 2) // 4)
        ) as executor:
            futures = {
                executor.submit(perform_kramers_kronig_test, exp, "complex"): i
                for i, exp in enumerate(
                    PyimpspecAdapter.To_API_Dataset(self.__root.data)
                )
            }
            for future in tqdm(
                as_completed(futures),
                total=N,
                desc=f"Calculating Kramers-Kronig Tests for EIS Experiment '{self.__root.name}'",
            ):
                results[futures[future]] = future.result()

        assert all(r is not None for r in results), (
            "Error: Some Kramers-Kronig tests did not complete successfully."
        )
        return results

    def __extract_stats(self, results: list[KramersKronigResult]) -> pd.DataFrame:
        stats = pd.concat(
            [
                r.to_statistics_dataframe()
                .set_index("Label")["Value"]
                .reindex(self.__Stat_Labels)
                for r in results
            ],
            ignore_index=True,
            axis=1,
        )

        peaks = pd.DataFrame(
            {
                "Max. Real Residual [%]]": [
                    max(residual[1], key=abs) for residual in self.__residuals
                ],
                "Max. Imag. Residual [%]": [
                    max(residual[2], key=abs) for residual in self.__residuals
                ],
            }
        )

        stats = pd.concat([stats, peaks], axis=0).transpose()

        # #stats.rename(["Decadic Logarithm of Pseudo Chi-Squared",
        #               "Estimated SD of Gaussian Noise [%]",
        #               "Real Residual Mean [%]",
        #               "Imag. Residual Mean [%]",
        #               "Real Residual SD [%]",
        #               "Imag. Residual SD [%]",
        #               "Max. Real Residual [%]",
        #               "Max. Imag. Residual [%]"])

        print(stats)

    @classmethod
    def __Report_Separator(cls) -> str:
        return (
            ("-" * cls.__W_Chi2)
            + "-+-"
            + ("-" * cls.__W_Noise)
            + "-+-"
            + ("-" * cls.__W_Res_Component)
            + "-+-"
            + ("-" * cls.__W_Res_Component)
            + "-+"
        )

    @classmethod
    def __Report_Header(cls) -> str:
        h1 = str()
        h2 = str()

        BOLD = "\033[1m"
        RESET = "\033[0m"

        # 1. Log χ2
        h1 += f"{BOLD}{'Decadic Logarithm of':<{cls.__W_Chi2}}{RESET} | "
        h2 += f"{BOLD}{'Pseudo Chi-Squared':<{cls.__W_Chi2}}{RESET} | "

        # 2. Gaussian Noise
        h1 += f"{BOLD}{'Estimated Gaussian':<{cls.__W_Noise}}{RESET} | "
        h2 += f"{BOLD}{'Noise SD [%]':<{cls.__W_Noise}}{RESET} | "

        # 3. Residual Statistics (Gruppierter Header)
        h1 += f"{BOLD}{'Residual Statistics [%]':<{2 * cls.__W_Res_Component + 3}}{RESET} |"
        h2 += f"{'Real: Mean ± SD : Max.':<{cls.__W_Res_Component}} | {'Imag.: Mean ± SD : Max.':<{cls.__W_Res_Component}} |"

        return f"{h1}\n{h2}"

    @classmethod
    def __Report_Row(cls, stats: pd.DataFrame) -> str:
        BAD = "\033[1;31m"
        RESET = "\033[0m"

        def format_val(
            val: float, width: int, threshold: float, check_abs: bool = True
        ) -> str:
            val_str = f"{val: <{width}.3f}"
            # Check if bad value
            bad = abs(val) > threshold if check_abs else val > threshold
            return f"{BAD}{val_str}{RESET}" if bad else val_str

        # Extract important values
        try:
            vals = stats["Value"].iloc
            chi2 = vals[0]
            noise = vals[20]
            m_re, m_im = vals[6], vals[7]
            s_re, s_im = vals[8], vals[9]
            max_re, max_im = vals[23], vals[24]
        except:
            return "Error reading stats."

        row = str()

        # Log Chi-Squared
        row += f"{format_val(chi2, cls.__W_Chi2, -2.0, check_abs=False)} | "

        # Noise
        row += f"{format_val(noise, cls.__W_Noise, 1.0)} | "

        # Residuals
        row += f"{format_val(m_re, cls.__W_Res_Field, 1.0)} ± {format_val(s_re, cls.__W_Res_Field, 1.0)}: {format_val(max_re, cls.__W_Res_Field, 1.0)} | "
        row += f"{format_val(m_im, cls.__W_Res_Field, 1.0)} ± {format_val(s_im, cls.__W_Res_Field, 1.0)}: {format_val(max_im, cls.__W_Res_Field, 1.0)} |"

        return row

    def to_str(self) -> str:
        overview = str()
        overview += self.__Report_Separator() + "\n"
        overview += self.__Report_Header() + "\n"
        overview += self.__Report_Separator() + "\n"

        for stats in self.__data["Statistics"]:
            overview += self.__Report_Row(stats) + "\n"
        overview += self.__Report_Separator() + "\n"

        return overview
