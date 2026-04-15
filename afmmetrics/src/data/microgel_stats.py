import numpy as np
import pandas as pd

from abc import abstractmethod
from scipy import ndimage as ndi, stats
from skimage.registration import phase_cross_correlation

from .afm_image import AFMImage


class MicrogelStatsMixin:
    def __init__(self) -> None:
        self._macro: pd.DataFrame | None = None
        self._micro: pd.DataFrame | None = None

    @abstractmethod
    def _calculate_macro(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _calculate_micro(self) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def _reference_image(self) -> AFMImage:
        pass

    @property
    def macro_stats(self) -> pd.DataFrame:
        if self._macro is None:
            self._macro = self._calculate_macro()
        return self._macro

    @property
    def micro_stats(self) -> pd.DataFrame:
        if self._micro is None:
            self._micro = self._calculate_micro()
        return self._micro

    def _reset_stats(self) -> None:
        self._macro = None
        self._micro = None

    def robust_microgel_patch(
        self,
        mask: np.ndarray | None = None,
        size: float | None = None,
        trim_proportion: float = 0.2,
    ) -> AFMImage:
        # Get selected data
        data = self.micro_stats if mask is None else self.micro_stats[mask]
        if data.empty or not {"Orientation", "Microgel Patch"}.issubset(data.columns):
            raise ValueError("No microgels were available / selected!")

        # Automatically determine needed canvas size
        if size is None:
            size_px = 2 * max((max(patch.shape) for patch in data["Microgel Patch"]))
        else:
            size_px = self._reference_image.um_to_px(size)
        canvas_shape = np.array((size_px, size_px), dtype=int)

        # First loop: align all microgels along the x-axis and pad to equal size
        prep_patches = []
        for angle, patch in zip(
            data["Orientation"].to_numpy(), data["Microgel Patch"].to_numpy()
        ):
            aligned = ndi.rotate(
                patch - patch.min(), -angle - 90, reshape=True, order=1, prefilter=False
            )

            if np.any(aligned.shape > canvas_shape):
                raise ValueError(
                    "Aligned patch is larger than target size. Consider increasing the target size."
                )

            start = (canvas_shape - aligned.shape) // 2
            end = start + aligned.shape

            canvas = np.zeros(canvas_shape, dtype=np.float64)
            canvas[tuple(slice(s, e) for s, e in zip(start, end))] = aligned

            prep_patches.append(canvas)

        # Make reference for the registration
        ref = np.mean(prep_patches, axis=0)

        # Second loop: register all the patches, i.e. align them using cross correlation
        reg_patches = []
        for patch in prep_patches:
            shift, _, _ = phase_cross_correlation(ref, patch)
            reg_patches.append(ndi.shift(patch, shift, mode="constant", cval=0.0))

        n_mean_effective = int(np.ceil(len(reg_patches) * (1 - trim_proportion)))

        robust_patch = self._reference_image.copy()
        robust_patch.data = stats.trim_mean(reg_patches, trim_proportion, axis=0)
        robust_patch._name = f"Averaged over {n_mean_effective} microgels."
        return robust_patch

    def microgel_profile(
        self,
        mask: np.ndarray | None = None,
        angle: float = 0.0,
        peak_pivot: bool = False,
    ) -> np.ndarray:
        mean_patch = self.robust_microgel_patch(mask)

        aligned = ndi.rotate(
            mean_patch.data,
            -angle,
            reshape=True,
            order=1,
            prefilter=False,
            mode="nearest",
        )

        if peak_pivot:
            row, _ = np.unravel_index(np.argmax(aligned), aligned.shape)
        else:
            row = aligned.shape[0] // 2

        profile = aligned[row, :]
        nonzeros = np.nonzero(np.abs(profile) > 1e-10)[0]

        if len(nonzeros) > 0:
            if (trim := min(nonzeros[0], len(profile) - 1 - nonzeros[-1]) - 2) > 0:
                return profile[trim:-trim]

        return profile

    @classmethod
    def Is_Macro_Stat(cls, key: str) -> bool:
        return key in {"Count", "Density", "Coverage"}
