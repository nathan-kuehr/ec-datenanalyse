import os
import io

import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import NamedTuple
from collections.abc import Callable
from itertools import groupby

from .utils import Import
from .palette import ColoredObject
from .config import FREQUENCY_TOLERANCE


class SingleExpEIS(ABC):
    Required_Data_Series = {
        "Frequency",
        "Impedance",
        "Resistance",
        "Neg. Reactance",
        "Phase",
    }

    @classmethod
    def load(cls, file_path: str) -> "SingleExpEIS":
        """
        Factory function. Opens and parses the impedance spectrum file, returning an
        instance of the appropriate subclass.

        Args:
            filePath: Path to the EIS data file (.txt or .csv)
        """
        from .specializations import SingleExpEISBioLogic, SingleExpEISPalmSens

        if not Import.Is_Allowed_File(file_path):
            raise FileNotFoundError(f"Unsupported/Nonexistent file: {file_path}")

        # Prepare to guess device type
        device_guess: dict = defaultdict(int)

        try:
            with open(file_path, "r", encoding="utf-16") as file:
                content = file.read()
                device_guess[SingleExpEISPalmSens] += 1
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read()
                device_guess[SingleExpEISBioLogic] += 1

        if "Impedance Spectroscopy" not in content:
            raise ValueError(
                f"File '{file_path}' does not contain Impedance Spectroscopy data."
            )

        device_guess[
            SingleExpEISBioLogic
            if content.startswith("EC-Lab ASCII FILE")
            else SingleExpEISPalmSens
        ] += 1

        guessed_device = max(device_guess, key=lambda k: device_guess[k])

        return guessed_device(file_path, content)

    def __init__(self, file_path: str, content: str) -> None:
        self._content = content
        self._split_content = [
            line.rstrip(self._delimiter) for line in content.splitlines()
        ]
        self._data: pd.DataFrame | None = None

        if (parsed_file_name := Import.Parse_File_Name(file_path)) is None:
            raise ValueError(
                f"File name '{os.path.basename(file_path)}' does not follow the required naming convention."
            )

        self._resistance_offset = 0
        self._sample_number, self._name_parts = parsed_file_name
        self._name = "_".join(self._name_parts)
        self._filename = os.path.splitext(os.path.basename(file_path))[0]

    @property
    @abstractmethod
    def _delimiter(self) -> str:
        pass

    @property
    @abstractmethod
    def _decimal(self) -> str:
        pass

    @property
    @abstractmethod
    def _series_naming(self) -> dict[str, str]:
        pass

    @property
    @abstractmethod
    def _negative_phase(self) -> bool:
        pass

    def _extract_data_field(self) -> str:
        column_counts = [
            len(line.split(self._delimiter)) for line in self._split_content
        ]
        max_columns = max(column_counts)

        for match, group in groupby(
            enumerate(column_counts), lambda x: x[1] == max_columns
        ):
            if match:
                group_list = list(group)
                data_lines = self._split_content[
                    group_list[0][0] : group_list[-1][0] + 1
                ]
                return "\n".join(data_lines)

        raise ValueError("No consistent data field found in file.")

    def _load_data(self) -> None:
        """Load and parse the data from the file content."""
        data = self._extract_data_field()

        df = pd.read_csv(io.StringIO(data), sep=self._delimiter, decimal=self._decimal)

        df.rename(columns=self._series_naming, inplace=True)
        df = df[list(self.Required_Data_Series)]

        if df.shape[1] != len(self.Required_Data_Series):
            missing_columns = self.Required_Data_Series - set(df.columns)
            raise ValueError(
                f"Dataframe is missing required columns: {missing_columns}"
            )

        # Adjust phase sign if necessary
        if self._negative_phase:
            df["Phase"] = -df["Phase"]

        # Calculate total capacitance
        with np.errstate(divide="ignore"):
            omega = 2 * np.pi * df["Frequency"]
            df["Capacitance"] = np.abs(1 / (omega * df["Neg. Reactance"]))
            self._resistance_offset = float(df["Resistance"].min())  # type: ignore
            df["Offset-Corrected Resistance"] = (
                df["Resistance"] - self._resistance_offset
            )

        self._data = pd.DataFrame(df)

    @property
    def data(self) -> pd.DataFrame:
        """Lazy-load the dataframe on first access."""
        if self._data is None:
            self._load_data()
            if self._data is None:
                raise ValueError("Data could not be loaded.")
        return self._data

    @property
    def resistance_offset(self) -> float:
        """Get the resistance offset value."""
        if self._data is None:
            self._load_data()
        return self._resistance_offset

    @classmethod
    def load_folder(cls, folderPath: str) -> list["SingleExpEIS"]:
        """Load all EIS files from a folder.

        Args:
            folderPath: Path to folder containing EIS files
        """
        files = Import.Files_From_Folder(folderPath)
        return [cls.load(filePath) for filePath in files]


class DataSeriesInfo(NamedTuple):
    symbol: str
    unit: str
    scale: str


class EIS(ColoredObject):
    Series_Info = {
        "Frequency": DataSeriesInfo("$F$", "Hz", "log"),
        "Impedance": DataSeriesInfo("$Z$", "$\\Omega$", "log"),
        "Capacitance": DataSeriesInfo("$C$", "F", "log"),
        "Phase": DataSeriesInfo("$\\varphi$", "°", "linear"),
        "Resistance": DataSeriesInfo("$R$", "$\\Omega$", "log"),
        "Offset-Corrected Resistance": DataSeriesInfo("$R_{oc}$", "$\\Omega$", "log"),
        "Neg. Reactance": DataSeriesInfo("-$X$", "$\\Omega$", "log"),
    }

    Experiment_Groups_In_Use = set()

    @classmethod
    def Reset_Tracked_Objects(cls) -> None:
        super().reset_color()
        cls.Experiment_Groups_In_Use = set()

    def __init__(self, group: str | None = None, reset_color: bool = False) -> None:
        """Initialize a new experiment container.

        Args:
            resetColor: Reset the global color palette index
        """
        super().__init__(reset_color)

        self.__experiments: list[SingleExpEIS] = []
        self.__freq_data: list[float] = []
        self.__data: pd.DataFrame | None = None

        group = group or f"ExpGroup_{id(self)}"
        if group in EIS.Experiment_Groups_In_Use:
            raise ValueError(f"Experiment group '{group}' is already in use.")

        self.__experiment_group = group
        EIS.Experiment_Groups_In_Use.add(group)

        self.__offset_shift = 0.0

    def __group(
        self, exp: SingleExpEIS, grouping: dict[str, str | Callable[[list[str]], str]]
    ) -> None:
        """Apply grouping labels to experiment data.

        Args:
            exp: Experiment to group
            grouping: Dict mapping group names to values or callables
        """
        for group_key, g in grouping.items():
            if isinstance(g, str):
                exp.data[group_key] = g
            elif callable(g):
                exp.data[group_key] = g(exp._name_parts)
            else:
                raise ValueError(
                    "Grouping values must be either strings or callable functions."
                )

    def load(
        self,
        folder_path: str,
        grouping: dict[str, str | Callable[[list[str]], str]] | None = None,
    ) -> "EIS":
        """Load EIS experiments from a folder with optional grouping.

        Args:
            folderPath: Path to folder containing EIS files
            grouping: Dict mapping group names to values or callables applied to filename parts
        """
        if not (new_exps := SingleExpEIS.load_folder(folder_path)):
            return self

        grouping = grouping or {}

        # Set up frequency data if first addition
        self.__freq_data = self.__freq_data or new_exps[0].data["Frequency"].tolist()

        # Check frequency data matches - no interpolation impl. yet
        for exp in new_exps:
            # Check if frequency data matches
            exp_freq_data = exp.data["Frequency"].tolist()
            for f_exp, f_ref in zip(exp_freq_data, self.__freq_data):
                if abs(f_exp - f_ref) > FREQUENCY_TOLERANCE * f_ref:
                    raise ValueError(
                        f"Frequency data does not match. Difference: {abs(f_exp - f_ref)} "
                        f"exceeds tolerance: {FREQUENCY_TOLERANCE * f_ref}"
                    )

            exp.data["Name"] = exp._name
            exp.data["Frequency"] = self.__freq_data
            exp.data["Experiment Group"] = self.__experiment_group
            exp.data["Palette"] = self._palette

            # Grouping
            self.__group(exp, grouping)

            self.__experiments.append(exp)

        return self

    def remove(self, to_remove: str | set[str]) -> None:
        if isinstance(to_remove, str):
            to_remove = {to_remove}

        self.__experiments = [
            exp
            for exp in self.__experiments
            if (exp._name not in to_remove and exp._filename not in to_remove)
        ]
        self.__data = None  # Invalidate cached data

    @property
    def data(self) -> pd.DataFrame:
        if self.__data is None:
            self.__data = pd.concat(
                [exp.data for exp in self.__experiments], ignore_index=True
            )
            self.__data["Offset-Corrected Resistance"] = (
                self.__data["Offset-Corrected Resistance"] + self.mean_resistance_offset
            )
        return self.__data

    @property
    def experiment_group(self) -> str:
        return self.__experiment_group

    @property
    def __mean_resistance_offset(self) -> float:
        """Get the mean resistance offset across all experiments."""
        if not self.__experiments:
            return 0.0
        return float(np.mean([exp.resistance_offset for exp in self.__experiments]))

    @property
    def mean_resistance_offset(self) -> float:
        """Get the mean resistance offset across all experiments."""
        return self.__mean_resistance_offset + self.__offset_shift

    @mean_resistance_offset.setter
    def mean_resistance_offset(self, newMeanResistance) -> None:
        self.__offset_shift = newMeanResistance - self.__mean_resistance_offset
        self.__data = None
