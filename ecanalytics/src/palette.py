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

    def __init__(self, color_index: int = 0) -> None:
        assert color_index >= 0 and color_index < len(self.Colors), (
            f"colorIndex must be between 0 and {len(self.Colors) - 1}"
        )
        self.__color = self.Colors[color_index]
        self.__shades = self.Shades[self.__color]

    @property
    def shadeable(self) -> bool:
        return self.__shades is not None

    @property
    def color(self) -> str:
        return self.__color

    def shade(self, n_shades: int) -> list[str]:
        if not self.shadeable:
            raise ValueError(f"Color {self.__color} does not have defined shades.")

        if n_shades < 1:
            raise ValueError("nShades must be at least 1")

        if n_shades == 1:
            return [self.color]

        # Wenn weniger Farben als verfügbar angefordert: mittlere Farben nehmen
        if n_shades <= len(self.__shades):
            start_index = (len(self.__shades) - n_shades) // 2
            return self.__shades[start_index : start_index + n_shades].tolist()

        # Wenn mehr Farben angefordert: interpolieren zwischen existierenden Farben
        shades = self.__shades.tolist()
        result = []

        for i in range(n_shades):
            # Position im Bereich [0, len(shades)-1]
            position = i * (len(shades) - 1) / (n_shades - 1)

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
    Color_Palette_Index = 0

    @classmethod
    def reset_color(cls) -> None:
        cls.Color_Palette_Index = 0

    @classmethod
    def next_color(cls, shadeable: bool) -> NEIColorPalette:
        while not (color := NEIColorPalette(cls.Color_Palette_Index)).shadeable:
            cls.Color_Palette_Index += 1
        cls.Color_Palette_Index += 1
        return color

    def __init__(self, reset_color: bool = False, shadeable: bool = True) -> None:
        """Initialize a new colored object.

        Args:
            resetColor: Reset the global color palette index
        """
        if reset_color:
            self.reset_color()

        self._palette = self.next_color(shadeable)

    @property
    def palette(self) -> NEIColorPalette:
        return self._palette
