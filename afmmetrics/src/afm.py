import numpy as np
import cv2 as cv  # pyright: ignore


from collections import OrderedDict
from itertools import islice
from functools import partial
from typing import Callable


import os
import re
from scipy.special import erf
import copy

import skimage.feature as feat
import skimage.morphology as morph
import skimage.segmentation as segm
import skimage.measure as meas
import skimage.color as color

import pandas as pd


MUGEL_DIAMETER_RANGE = (0.12, 0.22)

DEFAULT_CAMBER_CUTOFF_WAVELENGTH = 1.302  # μm
MEAN_EST_MUGEL_DIAMETER = 0.17  # μm

DEFAULT_DENOISE_H_PARAMETER = 3
DEFAULT_DENOISE_TEMPLATE_WINDOW_SIZE = 0.08  # μm
DEFAULT_DENOISE_SEARCH_WINDOW_SIZE = 0.6  # μm

DEFAULT_PADDING_FACTOR = 2

DEFAULT_TOPHAT_DIM_MARGIN = 1.5


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

        # Tracking
        self.__history = OrderedDict[str, np.ndarray]()
        self.__push(img, "Original Image")  # Add first image to history

    def __push(self, img: np.ndarray, desc: str) -> None:
        shape = np.array(img.shape)

        if desc in self.__history:
            raise KeyError("Image description is not unique!")

        if np.all(shape[0:1] == self._oshape):
            self.__history[desc] = img
        elif np.all(shape[0:1] == self._shape):
            px, py = tuple(self._padding)
            self.__history[desc] = img[px:-px, py:-py]

    def get_oimg(self, id: str | int = -1) -> np.ndarray:
        if isinstance(id, int):
            if id < 0:
                return next(islice(reversed(self.__history.items()), -id - 1, None))[1]
            else:
                return next(islice(self.__history.items(), id, None))[1]
        else:
            return self.__history[id]

    def get_img(self, id: str | int = -1):
        return np.pad(self.get_oimg(id), self._padding, "reflect")

    def apply_filter(self, filter: Callable[["AFMImage"], np.ndarray], desc: str):
        self.__push(filter(self), desc)

    def apply_procedure(
        self, steps: OrderedDict[str, Callable[["AFMImage"], np.ndarray]]
    ):
        for desc, filter in steps.items():
            self.apply_filter(filter, desc)

    def history(self):
        return copy.deepcopy(self.__history)


class MicrogelImage(AFMImage):
    def __init__(self, path: str):
        super().__init__(path)

        # Require squared sizes for the μGel image
        if not (np.diff(self._oshape)[0] == np.diff(self._scan_size)[0] == 0):
            raise ValueError(
                "Non-square microgel pictures (in pixels and scan size) are not supported."
            )

    def apply_standard_procedure(self):
        steps = OrderedDict()

        remove_camber = partial(
            self.remove_camber, cutoff=DEFAULT_CAMBER_CUTOFF_WAVELENGTH, width=None
        )
        denoise = partial(
            self.denoise,
            h=DEFAULT_DENOISE_H_PARAMETER,
            tws=DEFAULT_DENOISE_TEMPLATE_WINDOW_SIZE,
            sws=DEFAULT_DENOISE_SEARCH_WINDOW_SIZE,
        )
        top_hat = partial(self.top_hat, scale_margin=DEFAULT_TOPHAT_DIM_MARGIN)

        steps[
            f"Camber Removal with HP Filter (λ_co: {DEFAULT_CAMBER_CUTOFF_WAVELENGTH} μm, Δλ: {DEFAULT_CAMBER_CUTOFF_WAVELENGTH} μm)"
        ] = remove_camber
        steps[
            f"Denoising with Non-Local Means Filter (h: {1}, Template Size: {1}, search: {1}"
        ] = denoise
        steps[
            f"Segmentation Preparation with Top-Hat Transform Means Filter (SE-Size: {1})"
        ] = top_hat

        self.apply_procedure(steps)

    @classmethod
    def top_hat(cls, afm: AFMImage, scale_margin: float = DEFAULT_TOPHAT_DIM_MARGIN):
        """
        Uses the top hat transform to remove larger structures than the μGels, especially parts of the background.
        Structures that fit into the structuring element are preserved.

        Args:
            afm: AFM image to apply the filter on
            scale_margin: Used to scale the structuring element of the top hat transform. Usually +50% compared
                to larger bound of the μGel diameter.
        """

        # Create structuring element
        radius = (scale_margin * np.max(MUGEL_DIAMETER_RANGE)) / (
            2 * afm._spatial_resolution[0]
        )
        se = morph.disk(radius)

        return cv.morphologyEx(afm.get_oimg(), cv.MORPH_TOPHAT, se)

    @classmethod
    def denoise(
        cls,
        afm: AFMImage,
        h: float = DEFAULT_DENOISE_H_PARAMETER,
        tws: float = DEFAULT_DENOISE_TEMPLATE_WINDOW_SIZE,
        sws: float = DEFAULT_DENOISE_SEARCH_WINDOW_SIZE,
    ):
        """
        Denoises the AFM image w/ the Non-Local Means Algorithm.

        Args:
            afm: AFM image to apply the filter on
            h: Determines how strongly template patches that are different should be sanctioned in the algorithm, i.e.
                have lower weights. Defaults to 3 (play around).
            tws: Determines the size of the template window in μm. Should be smaller than the minimal μGel diameter.
                Defaults to a value such that the resulting template is 4x4 px for 10x10μm image @ 512x512px.
            sws: Determines the size of the search region. Should be (much) larger than the maximal μGel diameter.
                Defaults to a value such that the resulting template is 30x30 px for 10x10μm image @ 512x512px.
        """

        # Check if parameters are okay
        if tws >= np.min(MUGEL_DIAMETER_RANGE):
            raise ValueError(
                "Not recommended to have the template window larger than the μGel."
            )
        if sws < np.max(MUGEL_DIAMETER_RANGE):
            raise ValueError(
                "Search window should be at least bigger than μGel diameter."
            )

        # Transfer to sizes in px
        tws = int(tws / afm._spatial_resolution[0])
        sws = int(sws / afm._spatial_resolution[0])

        # Work with non-padded img, and uin8 (required for Non-Local Means)
        img = cv.normalize(afm.get_oimg(), None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        return cv.fastNlMeansDenoising(
            img, h=3, templateWindowSize=tws, searchWindowSize=sws
        )

    @classmethod
    def remove_camber(
        cls,
        afm: AFMImage,
        cutoff: float = DEFAULT_CAMBER_CUTOFF_WAVELENGTH,
        width: float | None = None,
    ):
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
        # https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/process/freq_split.c#l559

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

    def segment(self, overlayImg: np.ndarray | None = None, overlayAlpha: float = 0.1):
        img = self.get_oimg()

        _, thresholded = cv.threshold(img, 0.1 * np.max(img), 255, cv.THRESH_TOZERO)
        inverted = 255 - thresholded

        peaks = feat.peak_local_max(
            img, min_distance=3, threshold_rel=0.1, exclude_border=False
        )

        markers = np.zeros_like(img, dtype=np.int32)
        markers[tuple(peaks.T)] = np.arange(2, len(peaks) + 2)  # Label peaks
        markers[img == 0] = 1  # Background

        seg = segm.watershed(inverted, markers, compactness=1)

        if overlayImg is None:
            return seg
        else:
            overlay = color.label2rgb(
                seg, image=overlayImg, bg_label=1, alpha=overlayAlpha, kind="overlay"
            )
            return seg, overlay

    def analyzeMicrogels(self, seg: np.ndarray, denoised: np.ndarray):
        xprops = {
            "area",
            "bbox",
            "equivalent_diameter_area",
            "perimeter",
            "centroid",
            "centroid_weighted",
            "image",
            "image_intensity",
            "axis_major_length",
            "axis_minor_length",
        }

        res = dict()

        props = meas.regionprops_table(
            seg, denoised, properties=xprops, spacing=self._spatial_resolution
        )

        df = (
            pd.DataFrame(props).iloc[1:].reset_index(drop=True)
        )  # Drop first line which is background

        # Cleaning up data frame
        df["Bounding Box"] = list(
            zip(df["bbox-0"], df["bbox-1"], df["bbox-2"], df["bbox-3"])
        )
        df["Centroid"] = list(zip(df["centroid-0"], df["centroid-1"]))
        df["Weighted Centroid"] = list(
            zip(df["centroid_weighted-0"], df["centroid_weighted-1"])
        )

        df = df.drop(
            columns=[
                "bbox-0",
                "bbox-1",
                "bbox-2",
                "bbox-3",
                "centroid-0",
                "centroid-1",
                "centroid_weighted-0",
                "centroid_weighted-1",
            ]
        )
        df = df.rename(
            columns={
                "perimeter": "Perimeter",
                "equivalent_diameter_area": "Eq. Diameter",
                "area": "Area",
                "image": "Image Mask",
                "image_intensity": "Image Patch",
                "axis_major_length": "Length",
                "axis_minor_length": "Width",
            }
        )

        df["Circularity"] = (4 * np.pi * df["Area"]) / (df["Perimeter"] ** 2)
        df["Aspect Ratio"] = df["Length"] / df["Width"]

        res["Details"] = df

        res["Count"] = len(df)
        res["Coverage"] = np.sum(df["Area"]) / self.area
        res["Density"] = res["Count"] / self.area
        res["Mean Eq. Diameter"] = np.mean(df["Eq. Diameter"])
        res["Std Eq. Diameter"] = np.std(df["Eq. Diameter"])
        res["Mean Circularity"] = np.mean(df["Circularity"])
        res["Std Circularity"] = np.std(df["Circularity"])
        res["Mean Apect Ratio"] = np.mean(df["Aspect Ratio"])
        res["Std Apect Ratio"] = np.std(df["Aspect Ratio"])

        return res

    @property
    def area(self):
        return (self._oshape[0] * self._spatial_resolution[0]) ** 2
