import numpy as np
import cv2 as cv

import skimage.restoration as rest

from ..data import AFMImage


_NLM_INTENSITY_SCALE = 255.0


def row_align(image: AFMImage) -> AFMImage:
    # Inspired by the 'median' row alignment algorithm of gwyddion, linematch.c
    # Original Implementation in C: David Necas (Yeti) and others
    # https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/process/linematch.c#l438
    img = (aligned := image.copy()).data
    aligned.data = img - np.median(img, axis=1, keepdims=True) + np.median(img)
    return aligned


def gaussian_blur(image: AFMImage, cutoff: float) -> AFMImage:
    """Works similarly to the high_pass_filter, where cutoff designates 50% point in the filter's amplitude."""
    # Factor relating the cutoff wavelength to sigma
    conversion_factor = np.sqrt(np.log(2) / 2) / np.pi

    sigma = cutoff * conversion_factor / image.spatial_resolution[0]

    img = (blurred := image.copy()).data
    blurred.data = cv.GaussianBlur(
        img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv.BORDER_REFLECT
    )

    return blurred


def non_local_means(
    image: AFMImage,
    h: np.uint8,
    patch_size: float,
    patch_distance: float,
    fast_mode: bool = True,
) -> AFMImage:
    """Denoises the AFM image with the Non-Local Means Algorithm.

    Args:
        image: AFM image to apply the filter on
        h: Filter strength (Based on a 0-255 scale for intuition from 8-bit OpenCV impl.)
        patch_size: Determines the size of the template window in μm. Should be smaller than particle size.
        patch_distance: Determines the size of the search radius in μm.
    """
    patch_size_px = image.um_to_px(patch_size, min_px=1)
    patch_distance_px = image.um_to_px(patch_distance, min_px=1)

    img = (denoised := image.copy()).data

    # Convert h from 0-255 intuition-scale to scikit-image's intensity-relative h
    h_scikit = h / _NLM_INTENSITY_SCALE * (img.max() - img.min())

    denoised.data = rest.denoise_nl_means(
        img,
        h=h_scikit,
        fast_mode=fast_mode,
        patch_size=patch_size_px,
        patch_distance=patch_distance_px,
    )
    return denoised
