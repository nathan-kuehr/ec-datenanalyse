import numpy as np

class NEIColorPalette:
    Colors = np.array(["#197643", "#1159A6", "#FF8000", "#74035C", "#DE173C", "#27C1CF"])
    
    # Mapping from main colors to their shades
    Shades = {
        "#FF8000": np.array(["#ff8000","#f14400","#d00000","#92100b", "#4e001c","#370617"]),
        "#DE173C": None, # No shades defined
        "#74035C": np.array(["#fc506e","#cf3668","#a11d62","#74035c","#580045","#320428"]),
        "#1159A6": np.array(["#9de9f0","#5bb9d3","#4299c4","#1159a6","#0e4581","#082849"]),
        "#27C1CF": None, # No shades defined
        "#197643": np.array(["#c5eb66","#89c954","#48a644","#197643","#115535","#0a3320"])
    }

    def __init__(self, colorIndex: int = 0) -> None:
        assert colorIndex >= 0 and colorIndex < len(self.Colors), \
            f"colorIndex must be between 0 and {len(self.Colors)-1}"
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
        
        assert nShades > 0 and nShades <= len(self.__shades), \
            f"nShades must be between 1 and {len(self.__shades)} for color {self.__color}"
        
        if nShades == 1:
            return [self.color]
        
        # Return the middle nShades
        start_index = (len(self.__shades) - nShades) // 2
        return self.__shades[start_index:start_index + nShades].tolist()
    
    
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