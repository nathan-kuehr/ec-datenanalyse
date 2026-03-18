import os
import io

import pandas as pd
import numpy as np

from collections import defaultdict
from itertools import groupby

from ..config import EIS_SAMPLE_REQUIRED_DATA_SERIES
from .importer import Importer


class Sample:
    __Required_Data_Series = EIS_SAMPLE_REQUIRED_DATA_SERIES

    _Device_Details: dict | None = None

    def __init__(self, file_path: str, content: str) -> None:
        self.__content = self.__Extract_Data_Field(content)
        self._data: pd.DataFrame | None = None

        if (parsed_file_name := Importer.Parse_File_Name(file_path)) is None:
            raise ValueError(
                f"File name '{os.path.basename(file_path)}' does not follow the required naming convention."
            )

        self._sample_number, self._name_parts = parsed_file_name
        self._name = "_".join(self._name_parts)
        self._filename = os.path.splitext(os.path.basename(file_path))[0]

        self._resistance_offset = (
            -1.0
        )  # Placeholder value, will be calculated when data is loaded

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if cls._Device_Details is None or not isinstance(cls._Device_Details, dict):
            raise NotImplementedError(
                "Subclasses of EISSample must define _Device_Details class variable as a dict."
            )
        if "Delimiter" not in cls._Device_Details or not isinstance(
            cls._Device_Details["Delimiter"], str
        ):
            raise NotImplementedError(
                "_Device_Details must contain a 'Delimiter' key with a string value."
            )
        if "Decimal" not in cls._Device_Details or not isinstance(
            cls._Device_Details["Decimal"], str
        ):
            raise NotImplementedError(
                "_Device_Details must contain a 'Decimal' key with a string value."
            )
        if "Series Naming" not in cls._Device_Details or not isinstance(
            cls._Device_Details["Series Naming"], dict
        ):
            raise NotImplementedError(
                "_Device_Details must contain a 'Series Naming' key with a dict value."
            )
        if "Negative Phase" not in cls._Device_Details or not isinstance(
            cls._Device_Details["Negative Phase"], bool
        ):
            raise NotImplementedError(
                "_Device_Details must contain a 'Negative Phase' key with a boolean value."
            )

    @property
    def data(self) -> pd.DataFrame:
        """Returns the EIS data as a pandas DataFrame. Lazy-loads on first access."""
        if self._data is None:
            self._data = self.__load_data()
        if self._data is None:
            raise ValueError("Data could not be loaded.")
        return self._data

    @property
    def resistance_offset(self) -> float:
        """Returns the calculated resistance offset value."""
        self.data  # Ensure data is loaded and resistance offset is calculated
        return self._resistance_offset

    def __load_data(self) -> pd.DataFrame:
        assert self._Device_Details is not None, (
            "Subclass does not initialize Device Details!"
        )
        details: dict = self._Device_Details

        df = pd.read_csv(
            io.StringIO(self.__content),
            sep=details["Delimiter"],
            decimal=details["Decimal"],
        )
        df.rename(columns=details["Series Naming"], inplace=True)
        df = pd.DataFrame(df[list(self.__Required_Data_Series)])

        if df.shape[1] != len(self.__Required_Data_Series):
            missing_columns = self.__Required_Data_Series - set(df.columns)
            raise ValueError(
                f"Dataframe is missing required columns: {missing_columns}"
            )

        # Adjust phase sign if necessary
        if details["Negative Phase"]:
            df.loc[:, "Phase"] *= -1

        # Calculate total capacitance
        with np.errstate(divide="ignore"):
            omega = 2 * np.pi * df["Frequency"]
            df["Capacitance"] = np.abs(1 / (omega * df["Neg. Reactance"]))

        self._resistance_offset = float(
            df["Resistance"][df["Neg. Reactance"] > 0].min()
        )  # type: ignore
        df["Offset-Corrected Resistance"] = df["Resistance"] - self._resistance_offset

        # Add general attributes
        df["Sample Name"] = self._name

        return df

    @classmethod
    def __Extract_Data_Field(cls, content: str) -> str:
        assert cls._Device_Details is not None, (
            "Subclass does not initialize Device Details!"
        )
        delim = cls._Device_Details["Delimiter"]

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
    def Load_Factory(cls, file_path: str) -> "Sample":
        """
        Factory function. Opens and parses the impedance spectrum file, returning an
        instance of the appropriate subclass.

        Args:
            file_path: Path to the EIS data file (.txt or .csv)
        """
        from .specializations import EISSampleBioLogic, EISSamplePalmSens

        if not Importer.Is_Allowed_File(file_path):
            raise FileNotFoundError(f"Unsupported/Nonexistent file: {file_path}")

        # Prepare to guess device type
        device_guess: dict = defaultdict(int)

        try:
            with open(file_path, "r", encoding="utf-16") as file:
                content = file.read()
                device_guess[EISSamplePalmSens] += 1
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read()
                device_guess[EISSampleBioLogic] += 1

        if "Impedance Spectroscopy" not in content:
            raise ValueError(
                f"File '{file_path}' does not contain Impedance Spectroscopy data."
            )

        device_guess[
            EISSampleBioLogic
            if content.startswith("EC-Lab ASCII FILE")
            else EISSamplePalmSens
        ] += 1

        guessed_device = max(device_guess, key=lambda k: device_guess[k])

        return guessed_device(file_path, content)

    @classmethod
    def Batch_Load_Factory(cls, folder_path: str) -> list["Sample"]:
        """Factory function to load all EIS files from a folder.

        Args:
            folder_path: Path to folder containing EIS data files
        """
        file_paths = Importer.Files_From_Folder(folder_path)
        return [cls.Load_Factory(file_path) for file_path in file_paths]
