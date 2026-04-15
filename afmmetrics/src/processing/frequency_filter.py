import numpy as np

from scipy.special import erf

from ..data.afm_image import AFMImage


def high_pass_filter(
    image: AFMImage, cutoff: float, width: float | None = None
) -> AFMImage:
    """Removes the low frequent background (camber or curvature) from an AFM image.

    Args:
        image: AFM image to apply the filter on
        cutoff: Sets the cutoff wavelength in μm. Structures with larger wavelengths shall be filtered out.
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
    wimg = image.pad()

    sx, sy = wimg.shape
    cx, cy = wimg.center

    x, y = np.ogrid[0:sx, 0:sy]

    freqs = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) * wimg.spectral_resolution[0]
    filt = np.fft.ifftshift(radial_map(freqs))

    # Get image
    dft = np.fft.fft2(wimg.data)

    # Filter
    wimg.data = np.fft.ifft2(dft * filt).real

    return wimg.unpad()
