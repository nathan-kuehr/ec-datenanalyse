import cv2 as cv
import skimage.morphology as morph

from ..data.afm_image import AFMImage


def top_hat(image: AFMImage, scale: float) -> AFMImage:
    """Applies the top-hat transformation to remove structures larger than 'scale'.

    Args:
        image: The AFM image to apply the filter on.
        scale: The maximum diameter of the particles to be preserved, in µm.

    Returns:
        A new AFMImage instance with the filtered data.
    """
    if scale <= 0:
        raise ValueError("The particle scale must be greater than 0 µm!")

    radius_px = image.um_to_px(scale / 2, min_px=1)
    se = morph.disk(radius_px)

    data = (working_image := image.copy()).data
    working_image.data = cv.morphologyEx(data, cv.MORPH_TOPHAT, se)
    return working_image
