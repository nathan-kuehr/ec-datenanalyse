import numpy as np

from collections import OrderedDict
from itertools import islice
from typing import Iterator, NamedTuple

from .afm_image import AFMImage


class WorkflowStep(NamedTuple):
    image: AFMImage
    description: str


class ImageWorkflow:
    def __init__(self) -> None:
        self.__workflow = OrderedDict[str, WorkflowStep]()

    def push(self, image: AFMImage, key: str, description: str) -> None:
        if key in self.__workflow:
            raise ValueError(f"Key '{key}' already exists in workflow! Must be unique.")

        if not self._is_allowed_image(image):
            raise ValueError(
                "Image dimensions do not match the previously added images in the workflow!"
            )

        self.__workflow[key] = WorkflowStep(image, description)

    def get(self, id: str | int = -1) -> AFMImage:
        if isinstance(id, int):
            vs = self.__workflow.values()
            return next(
                islice(*((reversed(vs), -id - 1) if id < 0 else (vs, id)), None)
            ).image
        else:
            return self.__workflow[id].image

    def keys(self) -> set[str]:
        return set(self.__workflow.keys())

    def _is_allowed_image(self, image: AFMImage) -> bool:
        if len(self.__workflow) == 0:
            return True
        else:
            base = self._base_image()

            return np.all(
                (image.shape == base.shape)
                & (image.spatial_resolution - base.spatial_resolution < 1e-12)
            )

    def _base_image(self) -> AFMImage:
        return next(iter(self.__workflow.values())).image

    def __len__(self) -> int:
        return len(self.__workflow)

    def __iter__(self) -> Iterator[WorkflowStep]:
        return iter(self.__workflow.values())
