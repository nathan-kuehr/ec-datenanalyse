import copy

import pandas as pd
import numpy as np

from typing import TYPE_CHECKING
from collections.abc import Callable

from .sample import Sample
from .sample_container import SampleContainer

from ..config import EIS_EXPERIMENT_SERIES_INFO, EIS_EXPERIMENT_FREQUENCY_TOLERANCE

if TYPE_CHECKING:
    from ..analysis.analysis import Analysis


class Experiment(SampleContainer):
    Series_Info = EIS_EXPERIMENT_SERIES_INFO

    _Container_Name_Prefix = "Experiment"

    def __init__(self, name: str, color: None | str = None) -> None:
        super().__init__(name, color)

        # Main data storage
        self.__samples: list[Sample] = []
        self.__freqs: list[float] = []
        self._data: pd.DataFrame | None = None

        # Resisitve Shift Correction
        self.__resistive_shift = 0.0

        # Analysis related attributes
        # self.__analysis: Analysis | None = None  # Lazy creation
        self.__analysis: Analysis | None = None

    def _add_metadata_to_data(self, data: pd.DataFrame) -> pd.DataFrame:
        metadata = self.data[self.data.loc[:, "Sample Name":].columns].drop_duplicates()
        extended_data = pd.merge(
            left=data, right=metadata, how="left", on="Sample Name"
        )

        if len(extended_data) != len(data):
            raise ValueError(
                "Error during adding of metadata - must be not unique per sample!"
            )
        return extended_data

    @property
    def analysis(self) -> Analysis:
        from ..analysis.analysis import Analysis  # Avoid circular import

        self.__analysis = self.__analysis or Analysis(self)
        return self.__analysis

    @property
    def mean_resistance_offset(self) -> float:
        """Get the mean resistance offset across all samples."""
        if not self.__samples:
            return 0.0
        return float(np.mean([sample.resistance_offset for sample in self.__samples]))

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self.__reload_data()
        if self._data is None:
            raise ValueError("Data could not be loaded.")
        return self._data

    def load(
        self,
        folder_path: str,
        grouping: dict[str, str | Callable] | None = None,
    ) -> "Experiment":
        """Load EIS experiments from a folder with optional grouping.

        Args:
            folderPath: Path to folder containing EIS files
            grouping: Dict mapping group names to values or callables applied to filename parts
        """
        if not (new_samples := Sample.Batch_Load_Factory(folder_path)):
            return self

        # Set up frequency data if first addition
        self.__freqs = self.__freqs or new_samples[0].data["Frequency"].tolist()

        for sample in new_samples:
            # Verify all data has same frequency points within tolerance
            if not np.allclose(
                sample.data["Frequency"].to_numpy(),
                self.__freqs,
                rtol=EIS_EXPERIMENT_FREQUENCY_TOLERANCE,
            ):
                raise ValueError(
                    "Frequency data does not match within the specified tolerance."
                )

            # Align frequencies
            sample.data["Frequency"] = self.__freqs

            self._apply_container_groups(sample.data, sample._name_parts, grouping)
            self.__samples.append(sample)

        # Reload new data into main DataFrame
        self.__reload_data()

        return self

    def extract_subexp(
        self, group: str, value: str, name: str | None = None, color: str | None = None
    ) -> "Experiment":
        if len(self.__samples):
            raise ValueError("No samples have been loaded yet!")
        elif group not in set(self.__samples[0].data.columns):
            raise KeyError(f"No group with the name '{group}' exists!")

        subexp = Experiment(name or (self.__name + " - " + value), color)

        subexp.__samples = [
            copy.deepcopy(s)
            for s in self.__samples
            if s.data[group].unique()[0] == value
        ]

        for sample in subexp.__samples:
            sample.data["Experiment Name"] = subexp.name
            sample.data["Palette"] = subexp._palette

        return subexp

    def remove_samples(self, to_remove: str | set[str]) -> "Experiment":
        """Remove samples from the experiment by name or filename."""
        if isinstance(to_remove, str):
            to_remove = {to_remove}

        def keep(sample: Sample) -> bool:
            return sample._name not in to_remove and sample._filename not in to_remove

        self.__samples = [sample for sample in self.__samples if keep(sample)]
        self.__reload_data()
        return self

    def shift(self, shift_value: float) -> "Experiment":
        """Apply a resistive shift correction to the experiment data."""
        self.__resistive_shift = shift_value
        return self

    def __reload_data(self) -> None:
        self._data = pd.concat(
            [sample.data for sample in self.__samples], ignore_index=True
        )
        self._data["Offset-Corrected Resistance"] = (
            self._data["Offset-Corrected Resistance"]
            + self.mean_resistance_offset
            + self.__resistive_shift
        )
