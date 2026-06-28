import pandas as pd
from tqdm import tqdm

from collections.abc import Callable

from ecanalytics.src.data.sample_container import SampleContainer
from ecanalytics.src.data import files as ec_files

from .afm_image import AFMImage
from .microgel_image import MicrogelImage
from .microgel_stats import MicrogelStatsMixin


class MicrogelSeries(SampleContainer, MicrogelStatsMixin):
    _CONTAINER_NAME_PREFIX = "Microgel"

    # Must override for the mixin to work
    @property
    def _reference_image(self) -> AFMImage:
        return self._microgel_images[0]._reference_image

    def _calculate_macro(self) -> pd.DataFrame:
        return pd.concat(
            [image.macro_stats for image in self._microgel_images], ignore_index=True
        )

    def _calculate_micro(self) -> pd.DataFrame:
        return pd.concat(
            [image.micro_stats for image in self._microgel_images], ignore_index=True
        )

    def __init__(self, name: str, color: None | str = None) -> None:
        SampleContainer.__init__(self, name, color)
        MicrogelStatsMixin.__init__(self)

        self._microgel_images: list[MicrogelImage] = []

    def load(
        self, folder_path: str, grouping: dict[str, str | Callable] | None = None
    ) -> "MicrogelSeries":
        images = MicrogelImage.batch_load_factory(folder_path)

        for image in tqdm(images,
                total=len(images),
                desc=f"Loading images of series '{self._name}'",
            ):
            if (
                parsed := ec_files.parse_file_name(image.path, require_sample_no=False)
            ) is None:
                raise ValueError(
                    f"File name '{image.name}' does not follow the required naming convention."
                )
            name_parts = parsed[1]

            self._apply_container_groups(image.macro_stats, name_parts, grouping)
            self._apply_container_groups(image.micro_stats, name_parts, grouping)

            self._microgel_images.append(image)

        # Reload new data into main data frames
        self._reset_stats()
        return self
