from dataclasses import dataclass, field

# Format details for parsing an EIS measurement file produced by one device
@dataclass(frozen=True)
class DeviceProfile:
    delimiter: str
    decimal: str
    series_naming: dict[str, str]
    negative_phase: bool


BIOLOGIC = DeviceProfile(
    delimiter="\t",
    decimal=",",
    series_naming={
        "freq/Hz": "Frequency",
        "|Z|/Ohm": "Impedance",
        "Re(Z)/Ohm": "Resistance",
        "-Im(Z)/Ohm": "Neg. Reactance",
        "Phase(Z)/deg": "Phase",
    },
    negative_phase=False,
)

PALMSENS = DeviceProfile(
    delimiter=",",
    decimal=".",
    series_naming={
        "freq / Hz": "Frequency",
        "Z / Ohm": "Impedance",
        "Z' / Ohm": "Resistance",
        "-Z'' / Ohm": "Neg. Reactance",
        "neg. Phase / °": "Phase",
    },
    negative_phase=True,
)
