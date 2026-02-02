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
    RequiredDataSeries = {
        "Frequency",
        "Impedance",
        "Resistance",
        "Neg. Reactance",
        "Phase",
    }

    @classmethod
    def load(cls, filePath: str) -> "SingleExpEIS":
        """
        Factory function. Opens and parses the impedance spectrum file, returning an
        instance of the appropriate subclass.

        Args:
            filePath: Path to the EIS data file (.txt or .csv)
        """
        from .specializations import SingleExpEISBioLogic, SingleExpEISPalmSens

        if not Import.isAllowedFile(filePath):
            raise FileNotFoundError(f"Unsupported/Nonexistent file: {filePath}")

        # Prepare to guess device type
        deviceGuess: dict = defaultdict(int)

        try:
            with open(filePath, "r", encoding="utf-16") as file:
                content = file.read()
                deviceGuess[SingleExpEISPalmSens] += 1
        except UnicodeDecodeError:
            with open(filePath, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read()
                deviceGuess[SingleExpEISBioLogic] += 1

        if "Impedance Spectroscopy" not in content:
            raise ValueError(
                f"File '{filePath}' does not contain Impedance Spectroscopy data."
            )

        deviceGuess[
            SingleExpEISBioLogic
            if content.startswith("EC-Lab ASCII FILE")
            else SingleExpEISPalmSens
        ] += 1

        guessedDevice = max(deviceGuess, key=lambda k: deviceGuess[k])

        return guessedDevice(filePath, content)

    def __init__(self, filePath: str, content: str) -> None:
        self._content = content
        self._splitContent = [
            line.rstrip(self._delimiter) for line in content.splitlines()
        ]
        self._data: pd.DataFrame | None = None

        if (parsedFileName := Import.parseFileName(filePath)) is None:
            raise ValueError(
                f"File name '{os.path.basename(filePath)}' does not follow the required naming convention."
            )

        self._resistanceOffset = 0
        self._sampleNumber, self._nameParts = parsedFileName
        self._name = "_".join(self._nameParts)
        self._filename = os.path.splitext(os.path.basename(filePath))[0]

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
    def _seriesNaming(self) -> dict[str, str]:
        pass

    @property
    @abstractmethod
    def _negativePhase(self) -> bool:
        pass

    def _extractDataField(self) -> str:
        columnCounts = [len(line.split(self._delimiter)) for line in self._splitContent]
        maxColumns = max(columnCounts)

        for match, group in groupby(
            enumerate(columnCounts), lambda x: x[1] == maxColumns
        ):
            if match:
                groupList = list(group)
                dataLines = self._splitContent[groupList[0][0] : groupList[-1][0] + 1]
                return "\n".join(dataLines)

        raise ValueError("No consistent data field found in file.")

    def _loadData(self) -> None:
        """Load and parse the data from the file content."""
        data = self._extractDataField()

        df = pd.read_csv(io.StringIO(data), sep=self._delimiter, decimal=self._decimal)

        df.rename(columns=self._seriesNaming, inplace=True)
        df = df[list(self.RequiredDataSeries)]

        if df.shape[1] != len(self.RequiredDataSeries):
            missingColumns = self.RequiredDataSeries - set(df.columns)
            raise ValueError(f"Dataframe is missing required columns: {missingColumns}")

        # Adjust phase sign if necessary
        if self._negativePhase:
            df["Phase"] = -df["Phase"]

        # Calculate total capacitance
        with np.errstate(divide="ignore"):
            omega = 2 * np.pi * df["Frequency"]
            df["Capacitance"] = np.abs(1 / (omega * df["Neg. Reactance"]))
            self._resistanceOffset = float(df["Resistance"].min())  # type: ignore
            df["Offset-Corrected Resistance"] = (
                df["Resistance"] - self._resistanceOffset
            )

        self._data = pd.DataFrame(df)

    @property
    def data(self) -> pd.DataFrame:
        """Lazy-load the dataframe on first access."""
        if self._data is None:
            self._loadData()
            if self._data is None:
                raise ValueError("Data could not be loaded.")
        return self._data

    @property
    def resistanceOffset(self) -> float:
        """Get the resistance offset value."""
        if self._data is None:
            self._loadData()
        return self._resistanceOffset

    @classmethod
    def loadFolder(cls, folderPath: str) -> list["SingleExpEIS"]:
        """Load all EIS files from a folder.

        Args:
            folderPath: Path to folder containing EIS files
        """
        files = Import.filesFromFolder(folderPath)
        return [cls.load(filePath) for filePath in files]


class DataSeriesInfo(NamedTuple):
    symbol: str
    unit: str
    scale: str


class EIS(ColoredObject):
    SeriesInfo = {
        "Frequency": DataSeriesInfo("$F$", "Hz", "log"),
        "Impedance": DataSeriesInfo("$Z$", "$\\Omega$", "log"),
        "Capacitance": DataSeriesInfo("$C$", "F", "log"),
        "Phase": DataSeriesInfo("$\\varphi$", "°", "linear"),
        "Resistance": DataSeriesInfo("$R$", "$\\Omega$", "log"),
        "Offset-Corrected Resistance": DataSeriesInfo("$R_{oc}$", "$\\Omega$", "log"),
        "Neg. Reactance": DataSeriesInfo("-$X$", "$\\Omega$", "log"),
    }

    ExperimentGroupsInUse = set()

    @classmethod
    def resetTrackedObjects(cls) -> None:
        super().resetColor()
        cls.ExperimentGroupsInUse = set()

    def __init__(self, group: str | None = None, resetColor: bool = False) -> None:
        """Initialize a new experiment container.

        Args:
            resetColor: Reset the global color palette index
        """
        super().__init__(resetColor)

        self.__experiments: list[SingleExpEIS] = []
        self.__freqData: list[float] = []
        self.__data: pd.DataFrame | None = None

        group = group or f"ExpGroup_{id(self)}"
        if group in EIS.ExperimentGroupsInUse:
            raise ValueError(f"Experiment group '{group}' is already in use.")

        self.__experimentGroup = group
        EIS.ExperimentGroupsInUse.add(group)

        self.__offsetShift = 0.0

    def __group(
        self, exp: SingleExpEIS, grouping: dict[str, str | Callable[[list[str]], str]]
    ) -> None:
        """Apply grouping labels to experiment data.

        Args:
            exp: Experiment to group
            grouping: Dict mapping group names to values or callables
        """
        for groupKey, g in grouping.items():
            if isinstance(g, str):
                exp.data[groupKey] = g
            elif callable(g):
                exp.data[groupKey] = g(exp._nameParts)
            else:
                raise ValueError(
                    "Grouping values must be either strings or callable functions."
                )

    def load(
        self,
        folderPath: str,
        grouping: dict[str, str | Callable[[list[str]], str]] | None = None,
    ) -> None:
        """Load EIS experiments from a folder with optional grouping.

        Args:
            folderPath: Path to folder containing EIS files
            grouping: Dict mapping group names to values or callables applied to filename parts
        """
        if not (newExps := SingleExpEIS.loadFolder(folderPath)):
            return

        grouping = grouping or {}

        # Set up frequency data if first addition
        self.__freqData = self.__freqData or newExps[0].data["Frequency"].tolist()

        # Check frequency data matches - no interpolation impl. yet
        for exp in newExps:
            # Check if frequency data matches
            expFreqData = exp.data["Frequency"].tolist()
            for fExp, fRef in zip(expFreqData, self.__freqData):
                if abs(fExp - fRef) > FREQUENCY_TOLERANCE * fRef:
                    raise ValueError(
                        f"Frequency data does not match. Difference: {abs(fExp - fRef)} "
                        f"exceeds tolerance: {FREQUENCY_TOLERANCE * fRef}"
                    )

            exp.data["Name"] = exp._name
            exp.data["Frequency"] = self.__freqData
            exp.data["Experiment Group"] = self.__experimentGroup
            exp.data["Palette"] = self._palette

            # Grouping
            self.__group(exp, grouping)

            self.__experiments.append(exp)

    def remove(self, toRemove: str | set[str]) -> None:
        if isinstance(toRemove, str):
            toRemove = {toRemove}

        self.__experiments = [
            exp
            for exp in self.__experiments
            if (exp._name not in toRemove and exp._filename not in toRemove)
        ]
        self.__data = None  # Invalidate cached data

    @property
    def data(self) -> pd.DataFrame:
        if self.__data is None:
            self.__data = pd.concat(
                [exp.data for exp in self.__experiments], ignore_index=True
            )
            self.__data["Offset-Corrected Resistance"] = (
                self.__data["Offset-Corrected Resistance"] + self.meanResistanceOffset
            )
        return self.__data

    @property
    def experimentGroup(self) -> str:
        return self.__experimentGroup

    @property
    def __meanResistanceOffset(self) -> float:
        """Get the mean resistance offset across all experiments."""
        if not self.__experiments:
            return 0.0
        return float(np.mean([exp.resistanceOffset for exp in self.__experiments]))

    @property
    def meanResistanceOffset(self) -> float:
        """Get the mean resistance offset across all experiments."""
        return self.__meanResistanceOffset + self.__offsetShift

    @meanResistanceOffset.setter
    def meanResistanceOffset(self, newMeanResistance) -> None:
        self.__offsetShift = newMeanResistance - self.__meanResistanceOffset
        self.__data = None
