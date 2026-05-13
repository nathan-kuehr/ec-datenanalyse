import numpy as np
import pandas as pd

from collections import OrderedDict
from scipy import ndimage as ndi, spatial
from skimage import (
    color,
    feature as feat,
    measure as meas,
    morphology as morph,
    segmentation as segm,
)

from ecanalytics.src.data.importer import Importer

from .afm_image import AFMImage, ImageMixinBase
from .image_workflow import ImageWorkflow
from .microgel_stats import MicrogelStatsMixin
from .. import processing as proc
from ..config import (
    DEFAULT_PREPROCESS_SETTINGS,
    ESTIMATED_MICROGEL_DIAMETER_RANGE,
    MICROGEL_SERIES_INFO,
    REGION_PROPS_RENAMING,
    TARGET_REGION_PROPS,
    AUTOTHRESHOLD_MULTIPLIER,
)


# 1 / Phi^(-1)(0.75) = 1.4826 — robust MAD-to-standard-deviation conversion factor under normality
_MAD_TO_STD_FACTOR = 1.4826
# Avoids division by zero when normalizing intensity ranges
_INTENSITY_NORM_EPSILON = 1e-12
_OVERLAY_ALPHA = 0.3
_WATERSHED_COMPACTNESS = 2
_OUTLIER_FOOTPRINT = (3, 3)
# Range factor applied to ESTIMATED_MICROGEL_DIAMETER_RANGE to define the tolerance band
_DIAMETER_TOLERANCE_FACTORS = (1 / 3, 3)
_BACKGROUND_PERCENTILE = 25


class MicrogelImage(ImageMixinBase, MicrogelStatsMixin):
    SERIES_INFO = MICROGEL_SERIES_INFO

    _PREPROCESSING_STEPS = OrderedDict(
        [
            ("scan-line-align", (proc.row_align, "Scan Line Artifact Removal")),
            ("outlier-removal", (proc.remove_outlier_particles, "Outlier Removal")),
            ("hpf", (proc.high_pass_filter, "Background Curvature Removal")),
            ("denoise", (proc.non_local_means, "Non-Local Means Denoising")),
            ("top-hat", (proc.top_hat, "Large Structure Removal")),
            ("gaussian", (proc.gaussian_blur, "Gaussian Smoothing")),
        ]
    )

    _TOLERATED_MICROGEL_DIAMETER = ESTIMATED_MICROGEL_DIAMETER_RANGE * np.array(
        _DIAMETER_TOLERANCE_FACTORS
    )

    # Must override for the mixin to work
    @property
    def _reference_image(self) -> AFMImage:
        return self._original_image

    def _calculate_macro(self) -> pd.DataFrame:
        if "segment" not in self._workflow.keys():
            self.segment()

        if (seg := self._workflow.get("segment").mask) is None:
            raise ValueError("Segmentation mask does not exist!")

        return pd.DataFrame(
            {
                "Count": [seg.max()],
                "Coverage": [np.mean((seg != 0).astype(float))],
                "Density": [seg.max() / self.area],
                "Sample Name": [self.name],
                "Spatial Resolution": [self.spatial_resolution],
            }
        )

    def _calculate_micro(self) -> pd.DataFrame:
        if not {"segment", "denoise"}.issubset(self._workflow.keys()):
            self.segment()

        denoise = self._workflow.get("denoise")

        if (seg := self._workflow.get("segment").mask) is None:
            raise ValueError("Segmentation mask does not exist!")

        # Don't include border microgels as they are deformed
        seg = segm.clear_border(seg, bgval=0)

        stats = meas.regionprops_table(
            label_image=seg,
            intensity_image=denoise.data,
            properties=TARGET_REGION_PROPS,
            extra_properties=(self._masked_intensity_patch, self._masked_max_intensity),
            spacing=self.spatial_resolution,
        )

        # Drop first line which is background
        micro = pd.DataFrame(stats).iloc[1:].reset_index(drop=True)
        micro = micro.rename(columns=REGION_PROPS_RENAMING)

        # Small scale dimensions in nm
        micro[["Length", "Width"]] *= 1e3
        # Use degrees for orientation
        micro["Orientation"] = np.degrees(micro["Orientation"])

        micro["Aspect Ratio"] = micro["Length"] / micro["Width"]
        micro["Sample Name"] = denoise._name

        # Adjust centroids (add as tuple)
        centroids = micro[["centroid_weighted-0", "centroid_weighted-1"]].to_numpy()
        micro["Centroid"] = list(zip(centroids[:, 0], centroids[:, 1]))

        # Remove all old centroid columns
        micro = micro.drop(columns=["centroid_weighted-0", "centroid_weighted-1"])

        # Get nearest neighbor distance for each microgel
        centroids_um = denoise.px_to_um(centroids)

        tree = spatial.KDTree(centroids_um)
        dists, _ = tree.query(centroids_um, k=2)

        micro["Nearest Neighbor Distance"] = dists[:, 1] * 1e3

        return micro

    def __init__(self, path: str) -> None:
        ImageMixinBase.__init__(self)
        MicrogelStatsMixin.__init__(self)

        self._original_image = AFMImage(path)

        self._workflow = ImageWorkflow()

        self._workflow.push(self._original_image, "original", "Original Image")

        self._preprocess_args: dict = {}

    @property
    def name(self) -> str:
        return self._original_image._name

    @property
    def path(self) -> str:
        return self._original_image._path

    def preprocess(self) -> None:
        if set(self._PREPROCESSING_STEPS.keys()).issubset(self._workflow.keys()):
            return

        for key, (op, desc) in self._PREPROCESSING_STEPS.items():
            args = DEFAULT_PREPROCESS_SETTINGS[key] | self._preprocess_args.get(key, {})
            self._workflow.push(
                image=op(self._workflow.get(), **args), key=key, description=desc
            )

    def segment(self, threshold: float | None = None) -> None:
        if "segment" in self._workflow.keys():
            return
        elif not {"denoise", "gaussian"}.issubset(self._workflow.keys()):
            self.preprocess()

        topology = self._workflow.get("denoise")
        smoothed = self._workflow.get("gaussian")

        if (outlier_mask := self._workflow.get("outlier-removal").mask) is None:
            raise ValueError("Outlier removal mask does not exist!")

        # Calculate threshold if none is passed (usual)
        threshold = threshold or self._autothreshold()
        bg_mask = (smoothed.data < threshold) | outlier_mask

        # Find initial set of peaks
        seeds = feat.peak_local_max(
            smoothed.data, min_distance=1, threshold_abs=threshold, exclude_border=False
        )

        init_seeds = seeds

        # Invalidate seeds touching directly the outlier mask
        footprint = morph.footprint_rectangle(_OUTLIER_FOOTPRINT)
        invalid_seeds_mask = morph.dilation(outlier_mask, footprint)[tuple(seeds.T)]

        bg_seeds = seeds[invalid_seeds_mask]
        seeds = seeds[~invalid_seeds_mask]

        while True:
            bg_mask[tuple(bg_seeds.T)] = True

            markers, bg_label = self._make_markers(seeds, bg_mask)

            seg = segm.watershed(-topology.data, markers, compactness=_WATERSHED_COMPACTNESS)
            seg[seg == bg_label] = 0

            props = meas.regionprops_table(
                seg,
                properties=("label", "equivalent_diameter_area"),
                spacing=self.spatial_resolution,
            )

            # Find microgels that are unnaturally big / small
            seed_labels = seg[tuple(seeds.T)]
            outlier_mask = proc._interdecile_outliers(
                props["equivalent_diameter_area"], 1.0
            ) | ~proc._in_interval(
                props["equivalent_diameter_area"], self._TOLERATED_MICROGEL_DIAMETER
            )

            if not np.any(outlier_mask):
                break

            bad_labels = props["label"][outlier_mask]

            invalid_seeds_mask = np.isin(seed_labels, bad_labels)
            bg_seeds = np.vstack([bg_seeds, seeds[invalid_seeds_mask]])
            seeds = seeds[~invalid_seeds_mask]

        segment = topology.copy()
        segment.data = color.label2rgb(
            label=seg,
            image=(
                (topology.data - topology.data.min())
                / (topology.data.max() - topology.data.min() + _INTENSITY_NORM_EPSILON)
            ),
            alpha=_OVERLAY_ALPHA,
            kind="overlay",
        )
        segment.mask = seg

        # Mark initial seeds
        segment.data[tuple(init_seeds.T)] = (1, 1, 1)

        self._workflow.push(segment, "segment", "Segmentation Result")

    def _autothreshold(self) -> float:
        data = self._workflow.get("gaussian").data

        grad = np.hypot(ndi.sobel(data, 0), ndi.sobel(data, 1))

        # First approximate background by low gradient and low height
        height_limit = np.percentile(data, _BACKGROUND_PERCENTILE)
        grad_limit = np.percentile(grad, _BACKGROUND_PERCENTILE)

        bg_approx = data[(data < height_limit) & (grad < grad_limit)]

        # Robust threshold estimation: _MAD_TO_STD_FACTOR * MAD ≈ STD
        median = np.median(bg_approx)
        mad = np.median(np.abs(bg_approx - median))
        return median + (AUTOTHRESHOLD_MULTIPLIER * _MAD_TO_STD_FACTOR * mad)

    def _make_markers(
        self, seeds: np.ndarray, bg_mask: np.ndarray
    ) -> tuple[np.ndarray, int]:
        seed_map = np.zeros(self._workflow.get().shape, dtype=bool)
        seed_map[tuple(seeds.T)] = True

        markers, num_markers = ndi.label(seed_map)

        bg_label = num_markers + 1
        markers[bg_mask] = bg_label

        return markers, bg_label

    @classmethod
    def batch_load_factory(cls, folder_path: str) -> list["MicrogelImage"]:
        """Factory function to load all MG files from a folder.

        Args:
            folder_path: Path to folder containing microgel data files
        """
        file_paths = Importer.files_from_folder(folder_path, {".txt"})
        return [MicrogelImage(file_path) for file_path in file_paths]

    @staticmethod
    def _masked_intensity_patch(mask: np.ndarray, data: np.ndarray) -> np.ndarray:
        return (data - np.min(data[mask])) * mask

    @staticmethod
    def _masked_max_intensity(mask: np.ndarray, data: np.ndarray) -> np.ndarray:
        return np.max(data[mask]) - np.min(data[mask])
