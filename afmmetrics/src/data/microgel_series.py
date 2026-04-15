import pandas as pd

from collections.abc import Callable

from ecanalytics.src.data.sample_container import SampleContainer
from ecanalytics.src.data.importer import Importer

from .afm_image import AFMImage
from .microgel_image import MicrogelImage
from .microgel_stats import MicrogelStatsMixin


class MicrogelSeries(SampleContainer, MicrogelStatsMixin):
    _Container_Name_Prefix = "Microgel"

    # Must override for the mixin to work
    @property
    def _reference_image(self) -> AFMImage:
        return self._mg_images[0]._reference_image

    # Must override for the mixin to work
    def _calculate_macro(self) -> pd.DataFrame:
        return pd.concat([s.macro_stats for s in self._mg_images], ignore_index=True)

    # Must override for the mixin to work
    def _calculate_micro(self) -> pd.DataFrame:
        return pd.concat([s.micro_stats for s in self._mg_images], ignore_index=True)

    def __init__(self, name: str, color: None | str = None) -> None:
        SampleContainer.__init__(self, name, color)
        MicrogelStatsMixin.__init__(self)

        # Samples
        self._mg_images = list[MicrogelImage]()

    def load(
        self, folder_path: str, grouping: dict[str, str | Callable] | None = None
    ) -> "MicrogelSeries":
        for image in MicrogelImage.Batch_Load_Factory(folder_path):
            if (
                parsed := Importer.Parse_File_Name(image.path, require_sample_no=False)
            ) is None:
                raise ValueError(
                    f"File name '{image.name}' does not follow the required naming convention."
                )
            else:
                name_parts = parsed[1]

            self._apply_container_groups(image.macro_stats, name_parts, grouping)
            self._apply_container_groups(image.micro_stats, name_parts, grouping)

            self._mg_images.append(image)

        # Reload new data into main data frames
        self._reset_stats()
        return self
