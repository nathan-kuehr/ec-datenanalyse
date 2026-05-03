import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.colors import hex2color, rgb2hex
from scipy.interpolate import interp1d

from .config import DEFAULT_LINEPLOT_SETTINGS, FIGURE_SETTINGS


class NEIColorPalette:
    __Colors = np.array(
        ["#FF8000", "#DE173C", "#74035C", "#1159A6", "#27C1CF", "#197643"]
    )

    __Color_Names = {
        "orange": "#FF8000",
        "red": "#DE173C",
        "violett": "#74035C",
        "blue": "#1159A6",
        "cyan": "#27C1CF",
        "green": "#197643",
    }

    # Mapping from main colors to their shades
    __Shades = {
        "#FF8000": np.array(
            ["#ff8000", "#f14400", "#d00000", "#92100b", "#4e001c", "#370617"]
        ),
        "#DE173C": np.array(
            ["#fdd8de", "#f68b9d", "#eb425c", "#de173c", "#9e0a27", "#5c0214"]
        ),
        "#74035C": np.array(
            ["#fc506e", "#cf3668", "#a11d62", "#74035c", "#580045", "#320428"]
        ),
        "#1159A6": np.array(
            ["#9de9f0", "#5bb9d3", "#4299c4", "#1159a6", "#0e4581", "#082849"]
        ),
        "#27C1CF": np.array(
            ["#d4fcf3", "#7bf2df", "#3cdbcc", "#27c1cf", "#168f99", "#07575e"]
        ),
        "#197643": np.array(
            ["#c5eb66", "#89c954", "#48a644", "#197643", "#115535", "#0a3320"]
        ),
    }

    __Colors_In_Use = np.zeros_like(__Colors, dtype=bool)

    def __init__(self, color_name: None | str = None) -> None:
        # Get color
        if color_name is None:
            self.__color = self.__Next_Color()
        else:
            self.__color = self.__Color_Names[color_name]
            self.__Colors_In_Use[np.where(self.__Colors == self.__color)[0]] = True

    def shade(self, n_shades: int) -> list[str]:
        if n_shades < 1:
            raise ValueError("n_shades must be at least 1")
        elif n_shades == 1:
            return [self.__color]

        # Get shades
        shades = self.__Shades[self.__color]

        # If not all shades needed, take out of the middle (avoid too bright / dark)
        if n_shades <= len(shades):
            d = len(shades) / (n_shades + 1)
            vals = ((np.arange(n_shades) + 1) * d).astype(int)
            return shades[vals].tolist()

        # More shades needed -> interpolate
        rgb_shades = np.array([hex2color(s) for s in shades])
        f = interp1d(np.arange(len(shades)), rgb_shades, axis=0, kind="linear")
        interpolated_rgb_shades = f(np.linspace(0, len(shades) - 1, n_shades))
        return [rgb2hex(s) for s in interpolated_rgb_shades]

    @property
    def name(self) -> str:
        for name, hex in self.__Color_Names.items():
            if hex == self.__color:
                return name
        raise ValueError("Color not found in color names mapping.")

    @classmethod
    def Reset_Colors_In_Use(cls) -> None:
        cls.__Colors_In_Use = np.zeros_like(cls.__Colors, dtype=bool)

    @classmethod
    def __Next_Color(cls) -> str:
        available = np.where(~cls.__Colors_In_Use)[0]

        if len(available) == 0:  # Reset
            cls.Reset_Colors_In_Use()
            return cls.__Next_Color()

        idx = int(available[0])

        cls.__Colors_In_Use[idx] = True
        return cls.__Colors[idx]

    @classmethod
    def Showcase(cls):
        # Function to draw
        def parabole(x, k, d):
            return k * (x**2) + d

        # X data
        x = np.linspace(-2, 2, 21)

        data = []
        palette = []

        # Flatten shade dict to (index of color, index of shade, shade) tupels
        shade_iterator = (
            (cidx, sidx, s)
            for cidx, (_, v) in enumerate(cls.__Shades.items())
            for sidx, s in enumerate(v)
        )

        # Create y data and fill palette
        for cidx, sidx, shade in shade_iterator:
            y = parabole(x, 1 + sidx * 0.25, 5 * cidx)
            data.append(pd.DataFrame({"x": x, "y": y, "Shade": shade}))
            palette.append(shade)

        # Concat all dfs together
        df = pd.concat(data, ignore_index=True)

        args = DEFAULT_LINEPLOT_SETTINGS | {
            "data": df,
            "x": "x",
            "y": "y",
            "hue": "Shade",
            "palette": palette,
        }

        with plt.rc_context(FIGURE_SETTINGS):
            ax = plt.figure(figsize=(11, 6)).gca()

            sns.lineplot(**args, ax=ax)

            ax.set_title("Showcase: NEI Colorpalette")
            ax.legend(loc="upper center", ncol=6, fontsize="x-small")

            ax.set_ylim((-2, 40))

        plt.tight_layout()
        plt.show()
