import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.colors import hex2color, rgb2hex
from scipy.interpolate import interp1d

from .config import DEFAULT_LINEPLOT_SETTINGS, FIGURE_SETTINGS

class NEIColorPalette:
    _COLOR_NAMES = {
        "orange": "#ff8000",
        "red": "#de173c",
        "purple": "#74035c",
        "blue": "#1159a6",
        "cyan": "#27c1cf",
        "green": "#197643",
        "yellow": "#ffd000", # Added by me -> not in official NEI
        "grey": "#7d7d7d", # Added by me -> not in official NEI
    }

    # Mapping from main colors to their shades
    _SHADES = {
        "#ff8000": (
            "#ff8000", "#f14400", "#d00000", "#92100b", "#4e001c", "#370617",
        ),
        "#de173c": (
            "#ffb3bb", "#f76570", "#ea2c45", "#de173c", "#9e0a27", "#5c0214",
        ),
        "#74035c": (
            "#fc506e", "#cf3668", "#a11d62", "#74035c", "#580045", "#320428",
        ),
        "#1159a6": (
            "#9de9f0", "#5bb9d3", "#4299c4", "#1159a6", "#0e4581", "#082849",
        ),
        "#27c1cf": (
            "#b5f5e8", "#7bf2df", "#3cdbcc", "#27c1cf", "#15848f", "#073f4a",
        ),
        "#197643": (
            "#c5eb66", "#89c954", "#48a644", "#197643", "#115535", "#0a3320",
        ),
        "#ffd000": (
            "#ffd000", "#f5b800", "#d99800", "#a87400", "#735000", "#473100",
        ),
        "#7d7d7d": (
            "#d1d1d1", "#bdbdbd", "#999999", "#7d7d7d", "#4f4f4f", "#262626",
        ),
    }

    _colors_in_use = {name: False for name in _COLOR_NAMES}

    def __init__(self, color_name: str | None = None) -> None:
        if color_name is None:
            self._color = self._next_color()
        else:
            if color_name not in self._COLOR_NAMES:
                raise ValueError(
                    f"Unknown color {color_name!r}. "
                    f"Valid names: {list(self._COLOR_NAMES)}"
                )
            self._color = self._COLOR_NAMES[color_name]
            type(self)._colors_in_use[color_name] = True

    @property
    def color(self) -> str:
        return self._color

    @property
    def name(self) -> str:
        for name, hex_code in self._COLOR_NAMES.items():
            if hex_code == self._color:
                return name
        raise ValueError("Color not found in color names mapping.")

    def shade(self, nshades: int) -> list[str]:
        if nshades < 1:
            raise ValueError("nshades must be at least 1")
        if nshades == 1:
            return [self._color]

        shades = self._SHADES[self._color]
        n = len(shades)
        main_idx = shades.index(self._color) # index of main color inside the shades

        # Select among existing shades
        if nshades <= n:
            d = n / (nshades + 1)
            idc = ((np.arange(nshades) + 1) * d).astype(int)

            if main_idx not in idc:
                nearest = int(np.abs(idc - main_idx).argmin())
                shift = main_idx - idc[nearest]
                idc = np.clip(idc + shift, 0, n - 1)
            return [shades[i] for i in idc]
        
        # More shades than existing -> interpolate
        rgb = np.array([hex2color(s) for s in shades])

        positions = np.linspace(0, n - 1, nshades)
        interpolated = interp1d(np.arange(n), rgb, axis=0)(positions)
        result = [rgb2hex(c) for c in interpolated]

        # Ensure main is included -> snap nearest to it
        if self._color not in result:
            nearest = int(np.abs(positions - main_idx).argmin())
            result[nearest] = self._color
        return result

    @classmethod
    def reset_colors_in_use(cls) -> None:
        cls._colors_in_use = {name: False for name in cls._COLOR_NAMES}

    @classmethod
    def _next_color(cls) -> str:
        for name, in_use in cls._colors_in_use.items():
            if not in_use:
                cls._colors_in_use[name] = True
                return cls._COLOR_NAMES[name]
            
        cls.reset_colors_in_use()
        return cls._next_color()

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
            for color_idx, shade_array in enumerate(cls._SHADES.values())
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