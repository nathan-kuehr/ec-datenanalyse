import os
import io

import pandas as pd
import numpy as np

from collections import defaultdict
from itertools import groupby

from ..config import EIS_SAMPLE_REQUIRED_DATA_SERIES
from .device_profile import DeviceProfile, BIOLOGIC, PALMSENS
from . import files


class Sample:
    _REQUIRED_DATA_SERIES = EIS_SAMPLE_REQUIRED_DATA_SERIES

    def __init__(self, file_path: str, content: str, profile: DeviceProfile) -> None:
        self._profile = profile
        self._content = self._extract_data_field(content, profile)
        self._data: pd.DataFrame | None = None

        if (parsed_file_name := files.parse_file_name(file_path)) is None:
            raise ValueError(
                f"File name '{os.path.basename(file_path)}' does not follow the required naming convention."
            )

        self._sample_number, self._name_parts = parsed_file_name
        self._name = "_".join(self._name_parts)
        self._filename = os.path.splitext(os.path.basename(file_path))[0]

        # Placeholder; will be calculated when data is loaded
        self._resistance_offset = -1.0

    @property
    def data(self) -> pd.DataFrame:
        """Returns the EIS data as a pandas DataFrame. Lazy-loads on first access."""
        if self._data is None:
            self._data = self._load_data()
        if self._data is None:
            raise ValueError("Data could not be loaded.")
        return self._data

    @property
    def resistance_offset(self) -> float:
        """Returns the calculated resistance offset value."""
        self.data  # Ensure data is loaded and resistance offset is calculated
        return self._resistance_offset

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(
            io.StringIO(self._content),
            sep=self._profile.delimiter,
            decimal=self._profile.decimal,
        )
        df.rename(columns=self._profile.series_naming, inplace=True)
        df = pd.DataFrame(df[list(self._REQUIRED_DATA_SERIES)])

        if df.shape[1] != len(self._REQUIRED_DATA_SERIES):
            missing_columns = self._REQUIRED_DATA_SERIES - set(df.columns)
            raise ValueError(
                f"Dataframe is missing required columns: {missing_columns}"
            )

        if self._profile.negative_phase:
            df.loc[:, "Phase"] *= -1

        # Calculate total capacitance
        with np.errstate(divide="ignore"):
            omega = 2 * np.pi * df["Frequency"]
            df["Capacitance"] = np.abs(1 / (omega * df["Neg. Reactance"]))

        self._resistance_offset = float(
            df["Resistance"][df["Neg. Reactance"] > 0].min()
        )  # type: ignore
        df["Offset-Corrected Resistance"] = df["Resistance"] - self._resistance_offset

        df["Sample Name"] = self._name

        return df

    @staticmethod
    def _extract_data_field(content: str, profile: DeviceProfile) -> str:
        delim = profile.delimiter

        split_content = [line.rstrip(delim) for line in content.splitlines()]
        column_counts = [len(line.split(delim)) for line in split_content]

        max_columns = max(column_counts)

        for match, group in groupby(
            enumerate(column_counts), lambda x: x[1] == max_columns
        ):
            if match:
                group_list = list(group)
                data_lines = split_content[group_list[0][0] : group_list[-1][0] + 1]
                return "\n".join(data_lines)

        raise ValueError("No consistent data field found in file.")

    @classmethod
    def load_factory(cls, file_path: str) -> "Sample":
        """Opens and parses the impedance spectrum file, detecting the device profile.

        Args:
            file_path: Path to the EIS data file (.txt or .csv)
        """
        if not files.is_allowed_file(file_path):
            raise FileNotFoundError(f"Unsupported/Nonexistent file: {file_path}")

        # Vote-based device profile detection: encoding + header marker
        votes: defaultdict[DeviceProfile, int] = defaultdict(int)

        try:
            with open(file_path, "r", encoding="utf-16") as file:
                content = file.read()
                votes[PALMSENS] += 1
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read()
                votes[BIOLOGIC] += 1

        if "Impedance Spectroscopy" not in content:
            raise ValueError(
                f"File '{file_path}' does not contain Impedance Spectroscopy data."
            )

        votes[BIOLOGIC if content.startswith("EC-Lab ASCII FILE") else PALMSENS] += 1

        profile = max(votes, key=lambda k: votes[k])

        return cls(file_path, content, profile)

    @classmethod
    def batch_load_factory(cls, folder_path: str) -> list["Sample"]:
        """Loads all EIS files from a folder.

        Args:
            folder_path: Path to folder containing EIS data files
        """
        file_paths = files.files_from_folder(folder_path)
        return [cls.load_factory(file_path) for file_path in file_paths]
