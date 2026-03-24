import numpy as np
import pandas as pd
import cv2 as cv  # pyright: ignore


from collections import OrderedDict
from functools import partial

from scipy import stats, signal
from skimage import feature as feat, segmentation as segm, measure as meas, color


from .afm import AFMImage, _array_within_interval
from ..config import (
    DEFAULT_CAMBER_CUTOFF_WAVELENGTH,
    DEFAULT_DENOISE_H_PARAMETER,
    DEFAULT_DENOISE_SEARCH_WINDOW_SIZE,
    DEFAULT_DENOISE_TEMPLATE_WINDOW_SIZE,
    MUGEL_DIAMETER_RANGE,
    DEFAULT_TOPHAT_DIM_MARGIN,
)


class MicrogelStats:
    def __init__(self, df: pd.DataFrame, sample_area: float) -> None:
        self.individual = df
        self.__area = sample_area

    @property
    def count(self) -> int:
        return len(self.individual)

    @property
    def density(self) -> float:
        return self.count / self.__area

    @property
    def coverage(self) -> float:
        return float(self.individual["Area"].sum()) / self.__area  # pyright: ignore


class MicrogelImage(AFMImage):
    __Region_Properties = {
        "area",
        "axis_major_length",
        "axis_minor_length",
        "eccentricity",
        "equivalent_diameter_area",
        "image",
        "image_intensity",
        "intensity_max",
        "intensity_min",
        "perimeter",
        "label",
    }
    __MG_Diameter_Size = np.array(MUGEL_DIAMETER_RANGE) * np.array([0.75, 1.25])

    def __init__(self, path: str):
        super().__init__(path)

        self.__outlier_mask = np.zeros_like(self.get_oimg(), dtype=bool)

    def pre_process(self):
        steps = OrderedDict()

        # 1. Scan line artifact removal
        steps["scan-line-align"] = (self.Row_Align, "Scan Lign Artefact Removal")

        steps["outlier-removal"] = (
            partial(self.Remove_Outliers, mask_dest=self.__outlier_mask),
            "Outlier Removal",
        )

        # 2. Remove Camber
        remove_camber = partial(
            self.Remove_Camber, cutoff=DEFAULT_CAMBER_CUTOFF_WAVELENGTH, width=None
        )
        steps["remove-camber"] = (remove_camber, "Camber Removal")

        # 3. NL Means Denoising
        if DEFAULT_DENOISE_TEMPLATE_WINDOW_SIZE >= np.min(MUGEL_DIAMETER_RANGE):
            raise ValueError(
                "Not recommended to have the template window larger than the μGel."
            )
        if DEFAULT_DENOISE_SEARCH_WINDOW_SIZE < np.max(MUGEL_DIAMETER_RANGE):
            raise ValueError(
                "Search window should be at least bigger than μGel diameter."
            )
        denoise = partial(
            self.Denoise,
            h=DEFAULT_DENOISE_H_PARAMETER,
            tws=DEFAULT_DENOISE_TEMPLATE_WINDOW_SIZE,
            sws=DEFAULT_DENOISE_SEARCH_WINDOW_SIZE,
        )
        steps["denoise"] = (denoise, "Non-Local Means Denoising")

        # 4. Top hat to remove large structures
        top_hat = partial(
            self.Top_Hat, scale=DEFAULT_TOPHAT_DIM_MARGIN * np.max(MUGEL_DIAMETER_RANGE)
        )
        steps["top-hat"] = (top_hat, "Top-Hat Transform")

        self.apply_procedure(steps)

    @classmethod
    def __Peak_Based_Auto_Threshold(cls, img: np.ndarray) -> float:
        # Get peak intensity distribution
        peaks = feat.peak_local_max(img, min_distance=3, exclude_border=False)
        peak_intensities = img[peaks[:, 0], peaks[:, 1]]

        # Continuous approximate w/ gaussians
        x = np.linspace(peak_intensities.min(), peak_intensities.max(), 1000)
        kde = stats.gaussian_kde(peak_intensities, 0.1)(x)

        # Add a small linear curve to push the threshold to smaller values
        # whenever the local minimum is very flat
        kde += np.linspace(0, kde.max() * 0.3, 1000)

        minima = signal.find_peaks(-kde)[0]
        return 20 if len(minima) == 0 else x[minima[0]]

    @classmethod
    def __Make_Markers(
        cls, shape: tuple[int, int], seeds: np.ndarray, bkg_mask: np.ndarray
    ) -> tuple[np.ndarray, int]:
        seed_map = np.zeros(shape, dtype=bool)
        seed_map[tuple(seeds.T)] = True

        markers: np.ndarray = meas.label(seed_map)  # pyright: ignore

        bkg_label = markers.max() + 1
        markers[bkg_mask] = bkg_label

        return markers, bkg_label

    def __get_dilated_outlier_mask(self) -> np.ndarray:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        return cv.dilate(self.__outlier_mask.astype(np.uint8), kernel) > 0

    def segment(
        self, overlay: tuple[np.ndarray, float] | None = None
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        img = self.get_oimg()

        # Calculate threshold
        thr = self.__Peak_Based_Auto_Threshold(img)
        bkg_mask = img <= thr

        # Find original set of seeds
        seeds = feat.peak_local_max(
            img, min_distance=1, threshold_abs=thr, exclude_border=False
        )
        seed_outlier_touch = self.__get_dilated_outlier_mask()[tuple(seeds.T)]

        bkg_seeds = seeds[seed_outlier_touch]
        seeds = seeds[~seed_outlier_touch]

        while True:
            bkg_mask[tuple(bkg_seeds.T)] = True

            markers, bkg_label = self.__Make_Markers(img.shape, seeds, bkg_mask)

            seg = segm.watershed(-img, markers, compactness=2)
            seg = np.where(seg == bkg_label, 0, seg)

            #
            props = meas.regionprops_table(
                seg,
                properties=("label", "equivalent_diameter_area"),
                spacing=self._spatial_resolution,
            )

            labels = props["label"]
            diameters = props["equivalent_diameter_area"]

            outlier = self.Interdecile_Outlier_Detect(
                diameters, 1.0
            ) | ~_array_within_interval(diameters, tuple(self.__MG_Diameter_Size))

            if np.any(outlier):
                discard_mask = np.isin(seg[tuple(seeds.T)], labels[outlier])
                bkg_seeds = np.vstack([bkg_seeds, seeds[discard_mask]])
                seeds = seeds[~discard_mask]
            else:
                break

            # Remove seeds that touch outlier mask
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            dil_outlier_mask = cv.dilate(self.__outlier_mask.astype(np.uint8), kernel)
            seeds = seeds[dil_outlier_mask[tuple(seeds.T)] == 0]

        return (
            seg
            if overlay is None
            else (
                seg,
                color.label2rgb(seg, overlay[0], alpha=overlay[1], kind="overlay"),
            )
        )

    def microgel_stats(self, seg: np.ndarray, denoised: np.ndarray) -> MicrogelStats:
        stats = meas.regionprops_table(
            seg, denoised, self.__Region_Properties, spacing=self._spatial_resolution
        )

        # Drop first line which is background
        df = pd.DataFrame(stats).iloc[1:].reset_index(drop=True)
        df = df.rename(
            columns={
                "area": "Area",
                "axis_major_length": "Length",
                "axis_minor_length": "Width",
                "eccentricity": "Eccentricity",
                "equivalent_diameter_area": "Equiv. Diameter",
                "image": "Image Mask",
                "image_intensity": "Image Patch",
                "intensity_max": "Max. Intensity",
                "intensity_min": "Min. Intensity",
                "perimeter": "Perimeter",
                "label": "Label",
            }
        )
        df["Circularity"] = (4 * np.pi * df["Area"]) / (df["Perimeter"] ** 2)
        df["Aspect Ratio"] = df["Length"] / df["Width"]

        return MicrogelStats(df, self.area)

    @property
    def area(self) -> float:
        return (self._oshape[0] * self._spatial_resolution[0]) ** 2
