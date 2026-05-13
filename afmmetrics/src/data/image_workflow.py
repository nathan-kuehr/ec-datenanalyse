import numpy as np

from collections import OrderedDict
from itertools import islice
from typing import Iterator, NamedTuple

from .afm_image import AFMImage


_RESOLUTION_TOLERANCE = 1e-12


class WorkflowStep(NamedTuple):
    image: AFMImage
    description: str


class ImageWorkflow:
    def __init__(self) -> None:
        self._workflow: OrderedDict[str, WorkflowStep] = OrderedDict()

    def push(self, image: AFMImage, key: str, description: str) -> None:
        if key in self._workflow:
            raise ValueError(f"Key '{key}' already exists in workflow! Must be unique.")

        if not self._is_allowed_image(image):
            raise ValueError(
                "Image dimensions do not match the previously added images in the workflow!"
            )

        self._workflow[key] = WorkflowStep(image, description)

    def get(self, key: str | int = -1) -> AFMImage:
        if isinstance(key, int):
            values = self._workflow.values()
            return next(
                islice(*((reversed(values), -key - 1) if key < 0 else (values, key)), None)
            ).image
        return self._workflow[key].image

    def keys(self) -> set[str]:
        return set(self._workflow.keys())

    def _is_allowed_image(self, image: AFMImage) -> bool:
        if len(self._workflow) == 0:
            return True

        base = self._base_image()

        return np.all(
            (image.shape == base.shape)
            & (image.spatial_resolution - base.spatial_resolution < _RESOLUTION_TOLERANCE)
        )

    def _base_image(self) -> AFMImage:
        return next(iter(self._workflow.values())).image

    def __len__(self) -> int:
        return len(self._workflow)

    def __iter__(self) -> Iterator[WorkflowStep]:
        return iter(self._workflow.values())
