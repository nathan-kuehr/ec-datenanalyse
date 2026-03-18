from .sample import Sample


class EISSampleBioLogic(Sample):
    _Device_Details = {
        "Delimiter": "\t",
        "Decimal": ",",
        "Series Naming": {
            "freq/Hz": "Frequency",
            "|Z|/Ohm": "Impedance",
            "Re(Z)/Ohm": "Resistance",
            "-Im(Z)/Ohm": "Neg. Reactance",
            "Phase(Z)/deg": "Phase",
        },
        "Negative Phase": False,
    }

    def __init__(self, filePath: str, content: str):
        super().__init__(filePath, content)


class EISSamplePalmSens(Sample):
    _Device_Details = {
        "Delimiter": ",",
        "Decimal": ".",
        "Series Naming": {
            "freq / Hz": "Frequency",
            "Z / Ohm": "Impedance",
            "Z' / Ohm": "Resistance",
            "-Z'' / Ohm": "Neg. Reactance",
            "neg. Phase / °": "Phase",
        },
        "Negative Phase": True,
    }

    def __init__(self, filePath: str, content: str):
        super().__init__(filePath, content)
