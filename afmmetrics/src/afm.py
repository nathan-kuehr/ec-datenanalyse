import cv2 as cv  # pyright: ignore
import os
import re
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

MUGEL_DIAMETER_RANGE = (0.12, 0.22)

DEFAULT_CAMBER_CUTOFF_WAVELENGTH = 1.302  # μm
MEAN_EST_MUGEL_DIAMETER = 0.17  # μm

DEFAULT_DENOISE_H_PARAMETER = 3
DEFAULT_DENOISE_TEMPLATE_WINDOW_SIZE = 0.08  # μm
DEFAULT_DENOISE_SEARCH_WINDOW_SIZE = 0.6  # μm

DEFAULT_PADDING_FACTOR = 2


class AFMImage:
    DefaultDimensions = None

    @classmethod
    def sideLengthFromFilename(cls, path: str) -> float:
        filename = os.path.splitext(os.path.basename(path))[0]

        matches = re.findall(r"_(\d+)x(\d+)μm2", filename)

        if len(matches) == 1:
            dx, dy = matches[0]
            if dx != dy:
                raise ValueError("Non-square AFM pictures are not supported.")
            return float(dx)
        elif len(matches) == 0 and cls.DefaultDimensions is not None:
            return cls.DefaultDimensions

        raise NameError("Unclear dimensions specified in filename of '{path}'!")

    def __init__(self, path: str) -> None:
        imrgb = cv.imread(path)
        assert imrgb is not None, "Ungültiger Pfad"
        img = cv.cvtColor(imrgb, cv.COLOR_BGR2GRAY)

        # Shapes
        self.__oshape = np.array(img.shape)  # original shape
        self.__padding = self.__oshape[0] // DEFAULT_PADDING_FACTOR
        self.__shape = self.__oshape + 2 * self.__padding  # padded shape

        if self.__oshape[0] != self.__oshape[1]:
            raise ValueError("Non-square AFM pictures are not supported.")

        self.__sideLength = self.sideLengthFromFilename(path)

        # Resolutions
        self.__spatialResolution = self.__sideLength / self.__oshape[0]  # in μm/px
        self.__ospectralResolution = 1 / self.__sideLength  # in 1/(μm*bin)
        self.__spectralResolution = self.__oshape[0] / (
            self.__shape[0] * self.__sideLength
        )

        # Keep track of the changes
        self.__history = []
        self.__push(img)

        self.__ocenter = self.__oshape // 2
        self.__center = self.__shape // 2

    @property
    def ocurImg(self) -> np.ndarray:
        return self.__history[-1]

    @property
    def curImg(self) -> np.ndarray:
        return np.pad(self.ocurImg, self.__padding, "reflect")

    def __push(self, img):
        if (img.shape == self.__oshape).all():
            self.__history.append(img)
        elif (img.shape == self.__shape).all():
            p = self.__padding
            self.__history.append(img[p:-p, p:-p])
        else:
            raise ValueError("Image has wrong shape.")

    @property
    def center(self):
        return self.__center[0], self.__center[1]

    @property
    def shape(self):
        return self.__shape[0], self.__shape[1]

    def show(self):
        plt.imshow(self.ocurImg, cmap=plt.cm.gray)  # pyright: ignore

    def denoise(
        self,
        h: float = DEFAULT_DENOISE_H_PARAMETER,
        templateWindowSize: float = DEFAULT_DENOISE_TEMPLATE_WINDOW_SIZE,
        searchWindowSize: float = DEFAULT_DENOISE_SEARCH_WINDOW_SIZE,
    ):
        """
        Denoises the AFM image.
        """

        # Check if parameters are okay
        if templateWindowSize >= np.min(MUGEL_DIAMETER_RANGE):
            raise ValueError(
                "Not recommended to have the template window larger than the μGel."
            )
        if searchWindowSize < np.max(MUGEL_DIAMETER_RANGE):
            raise ValueError(
                "Search window should be at least bigger than μGel diameter."
            )

        # Transfer to sizes in px
        tws = int(templateWindowSize / self.__spatialResolution)
        sws = int(searchWindowSize / self.__spatialResolution)

        img = self.ocurImg

        cv.fastNlMeansDenoising(img, h=3, templateWindowSize=tws, searchWindowSize=sws)

    def removeCamber(
        self,
        cutoff: float = DEFAULT_CAMBER_CUTOFF_WAVELENGTH,
        width: float | None = None,
    ):
        """
        Removes the low frequent background (camber or curvature) from an AFM image.

        Args:
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
        conversionFactor = cutoff**2 - 0.25 * width**2

        if np.abs(conversionFactor) < 1e-12:
            raise ValueError(
                "Cannot choose cutoff ≈ 2 * width, as this would result in an infinite frequency boundary!"
            )

        freqCutoff = cutoff / conversionFactor
        freqWidth = width / conversionFactor

        # Radial filter function
        def radialMap(f):
            if freqWidth < 1e-12:
                return 1.0 * (f > freqCutoff)
            else:
                return 0.5 * (erf(2 * (f - freqCutoff) / freqWidth) + 1)

        # Prepare 2D filter

        # Prepare 2D filter
        sx, sy = self.shape
        cx, cy = self.center

        x, y = np.ogrid[0:sx, 0:sy]

        freqs = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) * self.__spectralResolution
        filter = np.fft.ifftshift(radialMap(freqs))

        # Get image
        dft = np.fft.fft2(self.curImg)

        # Filter
        res = np.fft.ifft2(dft * filter).real
        self.__push(res - np.min(res))
