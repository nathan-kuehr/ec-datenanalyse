from __future__ import annotations

import re
import os
import numpy as np

from abc import ABC, abstractmethod
from copy import deepcopy
from io import StringIO

from ..config import READIN_HEIGHT_BLOCK_REGEX, IMAGE_PADDING_FACTOR


class AFMImage:
    _Height_Block_Regex = re.compile(READIN_HEIGHT_BLOCK_REGEX)

    def __init__(self, path: str) -> None:
        self._path = path
        self._name = os.path.splitext(os.path.basename(path))[0]

        # Load data
        self._scan_size, self._data = self._Load_From_File(path)

        # Prepare lazy loaded metadata
        self._shape = None
        self._channels = None
        self._center = None
        self._spatial_resolution = None
        self._spectral_resolution = None
        self._area = None

        # Current padding
        self._applied_padding = np.zeros(2)

        # Mask
        self._mask: np.ndarray | None = None

    @property
    def scan_size(self) -> np.ndarray:
        return self._scan_size

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = new_data
        self._reset_metadata()

    @property
    def shape(self) -> np.ndarray:
        if self._shape is None:
            self._shape = np.array(self._data.shape)[0:2]
        return self._shape

    @property
    def channels(self) -> int:
        if self._channels is None:
            self._channels = 1 if self._data.ndim == 2 else self._data.shape[2]
        return self._channels

    @property
    def center(self) -> np.ndarray:
        if self._center is None:
            self._center = self.shape // 2
        return self._center

    @property
    def spatial_resolution(self) -> np.ndarray:
        if self._spatial_resolution is None:
            self._spatial_resolution = self._scan_size / self.shape  # in μm/px
        return self._spatial_resolution

    @property
    def spectral_resolution(self) -> np.ndarray:
        if self._spectral_resolution is None:
            self._spectral_resolution = 1 / self._scan_size  # in 1/(μm*bin)
        return self._spectral_resolution

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = float(np.prod(self.shape * self.spatial_resolution))
        return self._area

    @property
    def mask(self) -> np.ndarray | None:
        return self._mask

    @mask.setter
    def mask(self, new_mask: np.ndarray | None) -> None:
        if new_mask is not None:
            if np.any(new_mask.shape != self.shape[0:2]):
                raise ValueError(
                    f"Invalid mask! Expected mask of shape {self.shape} but got {new_mask.shape}."
                )
        self._mask = new_mask

    def copy(self) -> AFMImage:
        return deepcopy(self)

    def pad(self, padding_factor: float = IMAGE_PADDING_FACTOR) -> AFMImage:
        padded = self.copy()

        if padding_factor <= 1:
            raise ValueError("Padding factor must be larger than 1!")

        # Insert now padded image & adapt scan size
        px, py = ((padding_factor - 1) * self.shape / 2).astype(int)

        padded.data = np.pad(self.data, pad_width=((px, px), (py, py)), mode="reflect")
        padded._scan_size = padded.shape * self.spatial_resolution

        padded._applied_padding = np.array([px, py])

        return padded

    def unpad(self) -> AFMImage:
        if np.all(self._applied_padding == np.zeros((2, 1))):
            return self

        unpadded = self.copy()

        # Insert cropped image & adapt scan size
        px, py = self._applied_padding

        unpadded.data = self.data[px:-px, py:-py]
        unpadded._scan_size = unpadded.shape * self.spatial_resolution

        unpadded._applied_padding = np.zeros(2)

        return unpadded

    def um_to_px(
        self,
        ums: float | np.ndarray,
        min_px: int | None = None,
        max_px: int | None = None,
    ) -> int | np.ndarray:
        return np.clip(
            np.round(ums / self.spatial_resolution[0]), min_px, max_px
        ).astype(int)

    def px_to_um(
        self,
        px: int | np.ndarray,
        min_um: float | None = None,
        max_um: float | None = None,
    ) -> float | np.ndarray:
        return np.clip(px * self.spatial_resolution[0], min_um, max_um).astype(float)

    def _reset_metadata(self) -> None:
        self._shape = None
        self._channels = None
        self._center = None
        self._spatial_resolution = None
        self._spectral_resolution = None
        self._area = None

    @classmethod
    def _Load_From_File(cls, path: str) -> tuple[np.ndarray, np.ndarray]:
        with open(path, "r") as f:
            content = f.read()

        if (match := re.search(cls._Height_Block_Regex, content)) is not None:
            w, wu, h, hu, vu, raw_data = match.groups()

            # Currently, only μm sidelengths and m for the height are implemented
            if (wu != "µm") or (hu != "µm") or (vu != "m"):
                raise ValueError(
                    f"Unsupported units: width unit '{wu}', height unit '{hu}', value unit '{vu}'!"
                )

            data = np.loadtxt(StringIO(raw_data), np.float32)

            return np.array([int(w), int(h)]), data * 1e9  # for conversion to μm / nm

        raise ValueError("Unsupported format: no height block found!")


class ImageMixinBase(ABC):
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def _reference_image(self) -> AFMImage:
        pass

    @property
    def scan_size(self) -> np.ndarray:
        return self._reference_image.scan_size

    @property
    def shape(self) -> np.ndarray:
        return self._reference_image.shape

    @property
    def center(self) -> np.ndarray:
        return self._reference_image.center

    @property
    def spatial_resolution(self) -> np.ndarray:
        return self._reference_image.spatial_resolution

    @property
    def spectral_resolution(self) -> np.ndarray:
        return self._reference_image.spectral_resolution

    @property
    def area(self) -> float:
        return self._reference_image.area

    def um_to_px(
        self,
        ums: float | np.ndarray,
        min_px: int | None = None,
        max_px: int | None = None,
    ) -> int | np.ndarray:
        return self._reference_image.um_to_px(ums, min_px, max_px)

    def px_to_um(
        self,
        px: int | np.ndarray,
        min_um: float | None = None,
        max_um: float | None = None,
    ) -> float | np.ndarray:
        return self._reference_image.px_to_um(px, min_um, max_um)
