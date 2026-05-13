import numpy as np

from scipy.special import erf

from ..data.afm_image import AFMImage


_SINGULARITY_EPSILON = 1e-12


def high_pass_filter(
    image: AFMImage, cutoff: float, width: float | None = None
) -> AFMImage:
    """Removes the low frequent background (camber or curvature) from an AFM image.

    Args:
        image: AFM image to apply the filter on.
        cutoff: Cutoff wavelength in μm. Structures with larger wavelengths shall be filtered out.
        width: Width of the transition region around the cutoff of the filter. Determined in a way that
            in the interval [cutoff - width/2; cutoff + width/2], the filter decreases from 92.14% -> 7.87%.
    """
    # The use of the error function as an anti-ringing filter is inspired by freq_split.c, gwyddion
    # Original Implementation in C: David Necas (Yeti) and others
    # https://sourceforge.net/p/gwyddion/code/HEAD/tree/trunk/gwyddion/modules/process/freq_split.c

    width = width or cutoff

    # The two limits from the wavelength domain scale with the factor
    # 1/(mean^2 - 0.25*delta^2) to the frequency domain.
    conversion_factor = cutoff**2 - 0.25 * width**2

    if np.abs(conversion_factor) < _SINGULARITY_EPSILON:
        raise ValueError(
            "Cannot choose cutoff ≈ 2 * width, as this would result in an infinite frequency boundary!"
        )

    f_cutoff = cutoff / conversion_factor
    f_width = width / conversion_factor

    def radial_map(f: np.ndarray) -> np.ndarray:
        if f_width < _SINGULARITY_EPSILON:
            return 1.0 * (f > f_cutoff)
        return 0.5 * (erf(2 * (f - f_cutoff) / f_width) + 1)

    # Prepare 2D filter
    working_image = image.pad()

    sx, sy = working_image.shape
    cx, cy = working_image.center

    x, y = np.ogrid[0:sx, 0:sy]

    freqs = (
        np.sqrt((x - cx) ** 2 + (y - cy) ** 2) * working_image.spectral_resolution[0]
    )
    filt = np.fft.ifftshift(radial_map(freqs))

    dft = np.fft.fft2(working_image.data)

    working_image.data = np.fft.ifft2(dft * filt).real

    return working_image.unpad()
