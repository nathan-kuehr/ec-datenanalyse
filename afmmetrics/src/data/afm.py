import os
import re

import numpy as np
import cv2 as cv  # pyright: ignore


from collections import OrderedDict
from itertools import islice
from typing import Callable
from scipy.special import erf
import skimage.morphology as morph

from ..config import DEFAULT_PADDING_FACTOR


def _array_within_interval(img: np.ndarray, interval: tuple) -> np.ndarray:
    low, high = interval
    return (low <= img) & (img <= high)


class AFMImage:
    Default_Dimensions = None

    @classmethod
    def __Extract_Scan_Size(cls, path: str) -> np.ndarray:
        """
        Extracts the AFM picture's scan size based on the naming of the file.
        Needs to be of the following format, e.g. "..._10x10μm2...".

        Args:
            path: The path to the file.
        """
        filename = os.path.splitext(os.path.basename(path))[0]

        matches = re.findall(r"_(\d+)x(\d+)μm2", filename)

        if len(matches) == 1:
            return np.array(matches[0], dtype=float)
        elif len(matches) == 0 and cls.Default_Dimensions is not None:
            return cls.Default_Dimensions
        else:
            raise NameError("Unclear dimensions specified in filename of '{path}'!")

    def __init__(self, path: str) -> None:
        im3c = cv.imread(path)
        if im3c is None:
            raise ValueError(f"The image file '{path}' cannot be read!")

        # Convert to single channel grayscale image
        img = cv.cvtColor(im3c, cv.COLOR_BGR2GRAY)

        self._scan_size = self.__Extract_Scan_Size(path)

        # Shapes
        self._oshape = np.array(img.shape)  # original shape
        self._padding = self._oshape // DEFAULT_PADDING_FACTOR
        self._shape = self._oshape + 2 * self._padding  # padded shape

        padding_increase = self._shape / self._oshape

        # Resolutions
        self._spatial_resolution = self._scan_size / self._oshape  # in μm/px
        self._ospectral_resolution = 1 / self._scan_size  # in 1/(μm*bin)
        self._spectral_resolution = 1 / (
            padding_increase * self._scan_size
        )  # in 1/(μm*bin)

        # Image Centers
        self._ocenter = self._oshape // 2
        self._center = self._shape // 2

        # Tracking: key -> (img, descr)
        self._history = OrderedDict[str, tuple[np.ndarray, str]]()
        self.push(img, "original", "Original Image")  # Add first image to history

        # Require squared sizes for the afm image
        if not (np.diff(self._oshape)[0] == np.diff(self._scan_size)[0] == 0):
            raise ValueError(
                "Non-square pictures (in pixels and scan size) are currently not supported."
            )

    def push(self, img: np.ndarray, key: str, desc: str = "") -> None:
        shape = np.array(img.shape)

        if key in self._history:
            raise KeyError("Image key is not unique!")

        if np.all(shape[0:1] == self._oshape):
            self._history[key] = img, desc
        elif np.all(shape[0:1] == self._shape):
            px, py = tuple(self._padding)
            self._history[key] = (img[px:-px, py:-py], desc)
        else:
            raise ValueError("Incompatible sizes!")

    def get_oimg(self, id: str | int = -1) -> np.ndarray:
        if isinstance(id, int):
            if id < 0:
                return next(islice(reversed(self._history.items()), -id - 1, None))[1][
                    0
                ]
            else:
                return next(islice(self._history.items(), id, None))[1][0]
        else:
            return self._history[id][0]

    def get_img(self, id: str | int = -1) -> np.ndarray:
        return np.pad(self.get_oimg(id), self._padding, "reflect")

    @classmethod
    def Top_Hat(cls, afm: "AFMImage", scale: float) -> np.ndarray:
        """
        Uses the top hat transform to remove larger structures than the μGels, especially parts of the background.
        Structures that fit into the structuring element are preserved.

        Args:
            afm: AFM image to apply the filter on
            scale: scale of the particles
        """
        # Create structuring element
        radius_px = scale / (2 * afm._spatial_resolution[0])
        se = morph.disk(radius_px)

        return cv.morphologyEx(afm.get_oimg(), cv.MORPH_TOPHAT, se)

    @classmethod
    def Denoise(
        cls,
        afm: "AFMImage",
        h: float,
        tws: float,
        sws: float,
    ) -> np.ndarray:
        """
        Denoises the AFM image w/ the Non-Local Means Algorithm.

        Args:
            afm: AFM image to apply the filter on
            h: Determines how strongly template patches that are different should be sanctioned in the algorithm, i.e.
                have lower weights. Defaults to 3 (play around).
            tws: Determines the size of the template window in μm. Should be smaller than the particle size
            sws: Determines the size of the search region. Should be (much) larger than the particle size
        """
        # Transfer to sizes in px
        tws_px = int(tws / afm._spatial_resolution[0])
        sws_px = int(sws / afm._spatial_resolution[0])

        # Work with non-padded img, and uint8 (required for Non-Local Means)
        img = cv.normalize(afm.get_oimg(), None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        return cv.fastNlMeansDenoising(
            img, h=3, templateWindowSize=tws_px, searchWindowSize=sws_px
        )

    @classmethod
    def Remove_Camber(
        cls,
        afm: "AFMImage",
        cutoff: float,
        width: float | None = None,
    ) -> np.ndarray:
        """
        Removes the low frequent background (camber or curvature) from an AFM image.

        Args:
            afm: AFM image to apply the filter on
            cutoff: Sets the cutoff wavelength in μm. Structures with larger wavelengths shall be filtered out. Defaults
                to a value for which good results have been perceived, preserving the μGels.
            width: Determines the width of the transition region around the cutoff of the filter. Determined in a way, such that
                in the interval [cutoff - width/2; cutoff + width/2], the filter decreases from 92,14% -> 7,87%.
        """
        # -> The use of the error function as an anti-ringing filter is inspired by freq_split.c, gwyddion
        # Original Implementation in C: David Necas (Yeti) and others
        # https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/process/freq_split.c

        # If width is None, use cutoff as standard width
        width = width or cutoff

        # One can derive that the two limits from the wavelength domain scale with the factor
        # factor 1/(mean^2 - 0.25*delta^2) to the frequency domain. The factor is calculated the same for the inverse
        conversion_factor = cutoff**2 - 0.25 * width**2

        if np.abs(conversion_factor) < 1e-12:
            raise ValueError(
                "Cannot choose cutoff ≈ 2 * width, as this would result in an infinite frequency boundary!"
            )

        fcutoff = cutoff / conversion_factor
        fwidth = width / conversion_factor

        # Radial filter function
        def radial_map(f):
            if fwidth < 1e-12:
                return 1.0 * (f > fcutoff)
            else:
                return 0.5 * (erf(2 * (f - fcutoff) / fwidth) + 1)

        # Prepare 2D filter
        sx, sy = tuple(afm._shape)
        cx, cy = tuple(afm._center)

        x, y = np.ogrid[0:sx, 0:sy]

        freqs = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) * afm._spectral_resolution[0]
        filter = np.fft.ifftshift(radial_map(freqs))

        # Get image
        dft = np.fft.fft2(afm.get_img())

        # Filter
        return np.fft.ifft2(dft * filter).real

    @classmethod
    def Row_Align(cls, afm: "AFMImage") -> np.ndarray:
        """ """
        # -> Inspired by the 'median' row alignment algorithm of gwyddion, linematch.c
        # Original Implementation in C: David Necas (Yeti) and others
        # https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/process/linematch.c#l438
        img = afm.get_oimg().astype(np.float32)
        return img - np.median(img, axis=1, keepdims=True) + np.median(img)

    @classmethod
    def Interdecile_Outlier_Detect(
        cls, data: np.ndarray, multiplier: float
    ) -> np.ndarray:
        p10, p90 = np.percentile(data, (10, 90))
        idr = p90 - p10
        dev = multiplier * idr
        return ~_array_within_interval(data, (p10 - dev, p90 + dev))

    @classmethod
    def Remove_Outliers(
        cls,
        afm: "AFMImage",
        multiplier: float = 1.75,
        mask_dest: np.ndarray | None = None,
        id: str | int = -1,
    ) -> np.ndarray:
        img = afm.get_oimg(id).astype(np.float32)
        background = cv.GaussianBlur(img, (0, 0), 20)

        outlier = cls.Interdecile_Outlier_Detect(img, multiplier)

        if mask_dest is not None:
            mask_dest[:] = outlier

        return np.where(outlier, background, img)

    def apply_filter(
        self, filter: Callable[["AFMImage"], np.ndarray], key: str, desc: str = ""
    ) -> None:
        self.push(filter(self), key, desc)

    def apply_procedure(
        self, steps: OrderedDict[str, tuple[Callable[["AFMImage"], np.ndarray], str]]
    ) -> None:
        for key, (filter, desc) in steps.items():
            self.apply_filter(filter, key, desc)
