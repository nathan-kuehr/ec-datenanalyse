from .eis import SingleExpEIS as EIS


class SingleExpEISBioLogic(EIS):
    def __init__(self, filePath: str, content: str):
        super().__init__(filePath, content)

    @property
    def _delimiter(self) -> str:
        return "\t"

    @property
    def _decimal(self) -> str:
        return ","

    @property
    def _series_naming(self) -> dict[str, str]:
        return {
            "freq/Hz": "Frequency",
            "|Z|/Ohm": "Impedance",
            "Re(Z)/Ohm": "Resistance",
            "-Im(Z)/Ohm": "Neg. Reactance",
            "Phase(Z)/deg": "Phase",
        }

    @property
    def _negative_phase(self) -> bool:
        return False


class SingleExpEISPalmSens(EIS):
    def __init__(self, filePath: str, content: str):
        super().__init__(filePath, content)

    @property
    def _delimiter(self) -> str:
        return ","

    @property
    def _decimal(self) -> str:
        return "."

    @property
    def _series_naming(self) -> dict[str, str]:
        return {
            "freq / Hz": "Frequency",
            "Z / Ohm": "Impedance",
            "Z' / Ohm": "Resistance",
            "-Z'' / Ohm": "Neg. Reactance",
            "neg. Phase / °": "Phase",
        }

    @property
    def _negative_phase(self) -> bool:
        return True
