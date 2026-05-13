import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.colors import hex2color, rgb2hex
from scipy.interpolate import interp1d

from .config import DEFAULT_LINEPLOT_SETTINGS, FIGURE_SETTINGS


class NEIColorPalette:
    _COLORS = np.array(
        ["#FF8000", "#DE173C", "#74035C", "#1159A6", "#27C1CF", "#197643"]
    )

    _COLOR_NAMES = {
        "orange": "#FF8000",
        "red": "#DE173C",
        "violett": "#74035C",
        "blue": "#1159A6",
        "cyan": "#27C1CF",
        "green": "#197643",
    }

    # Mapping from main colors to their shades
    _SHADES = {
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

    _colors_in_use = np.zeros_like(_COLORS, dtype=bool)

    def __init__(self, color_name: None | str = None) -> None:
        if color_name is None:
            self._color = self._next_color()
        else:
            self._color = self._COLOR_NAMES[color_name]
            self._colors_in_use[np.where(self._COLORS == self._color)[0]] = True

    def shade(self, nshades: int) -> list[str]:
        if nshades < 1:
            raise ValueError("nshades must be at least 1")
        elif nshades == 1:
            return [self._color]

        shades = self._SHADES[self._color]

        # If not all shades needed, take out of the middle (avoid too bright / dark)
        if nshades <= len(shades):
            d = len(shades) / (nshades + 1)
            indices = ((np.arange(nshades) + 1) * d).astype(int)
            return shades[indices].tolist()

        # More shades needed -> interpolate
        rgb_shades = np.array([hex2color(s) for s in shades])
        f = interp1d(np.arange(len(shades)), rgb_shades, axis=0, kind="linear")
        interpolated_rgb_shades = f(np.linspace(0, len(shades) - 1, nshades))
        return [rgb2hex(s) for s in interpolated_rgb_shades]

    @property
    def name(self) -> str:
        for name, hex_code in self._COLOR_NAMES.items():
            if hex_code == self._color:
                return name
        raise ValueError("Color not found in color names mapping.")

    @classmethod
    def reset_colors_in_use(cls) -> None:
        cls._colors_in_use = np.zeros_like(cls._COLORS, dtype=bool)

    @classmethod
    def _next_color(cls) -> str:
        available = np.where(~cls._colors_in_use)[0]

        if len(available) == 0:  # Reset
            cls.reset_colors_in_use()
            return cls._next_color()

        idx = int(available[0])

        cls._colors_in_use[idx] = True
        return cls._COLORS[idx]

    @classmethod
    def showcase(cls) -> None:
        def parabola(x: np.ndarray, k: float, d: float) -> np.ndarray:
            return k * (x**2) + d

        x = np.linspace(-2, 2, 21)

        data = []
        palette = []

        # Flatten shade dict to (color index, shade index, shade) tuples
        shade_iterator = (
            (color_idx, shade_idx, shade)
            for color_idx, (_, shade_array) in enumerate(cls._SHADES.items())
            for shade_idx, shade in enumerate(shade_array)
        )

        for color_idx, shade_idx, shade in shade_iterator:
            y = parabola(x, 1 + shade_idx * 0.25, 5 * color_idx)
            data.append(pd.DataFrame({"x": x, "y": y, "Shade": shade}))
            palette.append(shade)

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
