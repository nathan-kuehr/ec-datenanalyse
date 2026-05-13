from .sample import Sample


class EISSampleBioLogic(Sample):
    _DEVICE_DETAILS = {
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

    def __init__(self, file_path: str, content: str) -> None:
        super().__init__(file_path, content)


class EISSamplePalmSens(Sample):
    _DEVICE_DETAILS = {
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

    def __init__(self, file_path: str, content: str) -> None:
        super().__init__(file_path, content)
