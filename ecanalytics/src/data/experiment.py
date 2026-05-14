from __future__ import annotations

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
    SERIES_INFO = EIS_EXPERIMENT_SERIES_INFO

    _CONTAINER_NAME_PREFIX = "Experiment"

    def __init__(self, name: str, color: None | str = None) -> None:
        super().__init__(name, color)

        # Main data storage
        self._samples: list[Sample] = []
        self._freqs: list[float] = []
        self._data: pd.DataFrame | None = None

        # Analysis related attributes
        self._analysis: Analysis | None = None

    @property
    def mean_resistance_offset(self) -> float:
        """Get the mean resistance offset across all samples."""
        if not self._samples:
            return 0.0
        return float(np.mean([sample.resistance_offset for sample in self._samples]))

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._reload_data()
        if self._data is None:
            raise ValueError("Data could not be loaded.")
        return self._data

    @property
    def sample_names(self) -> np.ndarray:
        return self.data["Sample Name"].unique()

    @property
    def analysis(self) -> Analysis:
        from ..analysis.analysis import Analysis
        if self._analysis is None:
            self._analysis = Analysis(self)
        return self._analysis

    def load(
        self,
        folder_path: str,
        grouping: dict[str, str | Callable] | None = None,
    ) -> "Experiment":
        """Load EIS experiments from a folder with optional grouping.

        Args:
            folder_path: Path to folder containing EIS files
            grouping: Dict mapping group names to values or callables applied to filename parts
        """
        if not (new_samples := Sample.batch_load_factory(folder_path)):
            return self

        # Set up frequency data if first addition
        self._freqs = self._freqs or new_samples[0].data["Frequency"].tolist()

        for sample in new_samples:
            # Verify all data has same frequency points within tolerance
            if not np.allclose(
                sample.data["Frequency"].to_numpy(),
                self._freqs,
                rtol=EIS_EXPERIMENT_FREQUENCY_TOLERANCE,
            ):
                raise ValueError(
                    "Frequency data does not match within the specified tolerance."
                )

            # Align frequencies
            sample.data["Frequency"] = self._freqs

            self._apply_container_groups(sample.data, sample._name_parts, grouping)
            self._samples.append(sample)

        self._reload_data()

        return self

    def extract_subexp(
        self, group: str, value: str, name: str | None = None, color: str | None = None
    ) -> "Experiment":
        if not self._samples:
            raise ValueError("No samples have been loaded yet!")
        elif group not in set(self._samples[0].data.columns):
            raise KeyError(f"No group with the name '{group}' exists!")

        subexp = Experiment(name or (self._name + " - " + value), color)

        subexp._samples = [
            copy.deepcopy(s)
            for s in self._samples
            if s.data[group].unique()[0] == value
        ]

        for sample in subexp._samples:
            sample.data["Experiment Name"] = subexp.name
            sample.data["Palette"] = subexp._palette

        return subexp

    def remove_samples(self, to_remove: str | set[str]) -> "Experiment":
        """Remove samples from the experiment by name or filename."""
        if isinstance(to_remove, str):
            to_remove = {to_remove}

        def keep(sample: Sample) -> bool:
            return sample._name not in to_remove and sample._filename not in to_remove

        self._samples = [sample for sample in self._samples if keep(sample)]
        self._reload_data()
        return self

    def phantom(self, new_data: None | pd.DataFrame) -> "SimulatedExperiment":
        return SimulatedExperiment.from_source(self, new_data)

    def _reload_data(self) -> None:
        self._data = pd.concat(
            [sample.data for sample in self._samples], ignore_index=True
        )
        self._data["Offset-Corrected Resistance"] = (
            self._data["Offset-Corrected Resistance"]
            + self.mean_resistance_offset
        )

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


class SimulatedExperiment(Experiment):
    def __init__(self, name: str, color: str | None = None) -> None:
        super().__init__(name, color)

    @classmethod
    def from_source(
        cls, source: Experiment, new_data: pd.DataFrame | None
    ) -> "SimulatedExperiment":
        sim = cls.__new__(cls)
        # Copy state from the source experiment, then overwrite some slots
        sim.__dict__.update(copy.deepcopy(source.__dict__))
        sim._samples = []
        sim._freqs = []
        if new_data is not None:
            sim._data = new_data
            
        # Completely disable analysis
        sim.__setattr__("analysis", None)
        return sim

    def _reload_data(self) -> None:  # Impossible
        return
