import numpy as np
from matplotlib.colors import hex2color, rgb2hex


class NEIColorPalette:
    Colors = np.array(
        ["#FF8000", "#DE173C", "#74035C", "#1159A6", "#27C1CF", "#197643"]
    )

    # Mapping from main colors to their shades
    Shades = {
        "#FF8000": np.array(
            ["#ff8000", "#f14400", "#d00000", "#92100b", "#4e001c", "#370617"]
        ),
        "#DE173C": None,  # No shades defined
        "#74035C": np.array(
            ["#fc506e", "#cf3668", "#a11d62", "#74035c", "#580045", "#320428"]
        ),
        "#1159A6": np.array(
            ["#9de9f0", "#5bb9d3", "#4299c4", "#1159a6", "#0e4581", "#082849"]
        ),
        "#27C1CF": None,  # No shades defined
        "#197643": np.array(
            ["#c5eb66", "#89c954", "#48a644", "#197643", "#115535", "#0a3320"]
        ),
    }

    def __init__(self, colorIndex: int = 0) -> None:
        assert colorIndex >= 0 and colorIndex < len(self.Colors), (
            f"colorIndex must be between 0 and {len(self.Colors) - 1}"
        )
        self.__color = self.Colors[colorIndex]
        self.__shades = self.Shades[self.__color]

    @property
    def shadeable(self) -> bool:
        return self.__shades is not None

    @property
    def color(self) -> str:
        return self.__color

    def shade(self, nShades: int) -> list[str]:
        if not self.shadeable:
            raise ValueError(f"Color {self.__color} does not have defined shades.")

        if nShades < 1:
            raise ValueError("nShades must be at least 1")

        if nShades == 1:
            return [self.color]

        # Wenn weniger Farben als verfügbar angefordert: mittlere Farben nehmen
        if nShades <= len(self.__shades):
            start_index = (len(self.__shades) - nShades) // 2
            return self.__shades[start_index : start_index + nShades].tolist()

        # Wenn mehr Farben angefordert: interpolieren zwischen existierenden Farben
        shades = self.__shades.tolist()
        result = []

        for i in range(nShades):
            # Position im Bereich [0, len(shades)-1]
            position = i * (len(shades) - 1) / (nShades - 1)

            # Finde die beiden umgebenden Farben
            lower_idx = int(np.floor(position))
            upper_idx = int(np.ceil(position))

            if lower_idx == upper_idx:
                # Exakte Übereinstimmung
                result.append(shades[lower_idx])
            else:
                # Interpoliere zwischen zwei Farben
                t = position - lower_idx
                color1_rgb = np.array(hex2color(shades[lower_idx]))
                color2_rgb = np.array(hex2color(shades[upper_idx]))

                # Lineare Interpolation im RGB-Raum
                interpolated_rgb = (1 - t) * color1_rgb + t * color2_rgb
                interpolated_hex = rgb2hex(tuple(interpolated_rgb))
                result.append(interpolated_hex)

        return result


class ColoredObject:
    ColorPaletteIndex = 0

    @classmethod
    def resetColor(cls) -> None:
        cls.ColorPaletteIndex = 0

    @classmethod
    def nextColor(cls) -> NEIColorPalette:
        color = NEIColorPalette(cls.ColorPaletteIndex)
        cls.ColorPaletteIndex += 1
        return color

    def __init__(self, resetColor: bool = False) -> None:
        """Initialize a new colored object.

        Args:
            resetColor: Reset the global color palette index
        """
        if resetColor:
            self.resetColor()

        self._palette = self.nextColor()

    @property
    def palette(self) -> NEIColorPalette:
        return self._palette
