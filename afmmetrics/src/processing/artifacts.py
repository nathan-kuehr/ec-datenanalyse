import numpy as np

from skimage import morphology as morph, restoration as rest

from ..data.afm_image import AFMImage


def _in_interval(data: np.ndarray, vint: tuple[float, float]) -> np.ndarray:
    vmin, vmax = vint
    return (data >= vmin) & (data <= vmax)


def _interdecile_outliers(data: np.ndarray, multiplier: float) -> np.ndarray:
    p10, p90 = np.percentile(data, (10, 90))
    idr = p90 - p10
    dev = multiplier * idr
    return ~_in_interval(data, (p10 - dev, p90 + dev))


def remove_outlier_particles(
    image: AFMImage, multiplier: float = 1.75, dilation: float = 0.1
) -> AFMImage:
    """
    Removes large artifacts such as dust particles from the AFM image.

    Args:
        image: The AFM image to process.
        multiplier: Sensitivity for detecting the outlier particle peaks (default: 1.75).
        dilation: How many µm the mask should be expanded around the particles to capture the complete edges/slopes.
    """

    # Find outliers
    outlier_mask = _interdecile_outliers(image.data, multiplier)
    if not np.any(outlier_mask):
        copy = image.copy()
        copy.mask = outlier_mask
        return copy

    # Dilate if necessary:
    if dilation > 0:
        dilation_px = image.um_to_px(dilation, min_px=1)
        se = morph.disk(dilation_px)
        dilated_outlier_mask = morph.dilation(outlier_mask, footprint=se)
    elif dilation == 0:
        dilated_outlier_mask = outlier_mask
    else:
        raise ValueError("Dilation must be non-negative.")

    cleaned = image.copy()
    cleaned.data = rest.inpaint_biharmonic(image.data, dilated_outlier_mask)

    cleaned.mask = dilated_outlier_mask

    return cleaned
