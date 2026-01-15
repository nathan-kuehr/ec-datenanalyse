import ecvisualization as ecv


GlassGroupSelector = ecv.Selector(
    drop=[
        "[0-9]+",
        "EIS",
        "PEDOTPSS",
        "SpinCoated",
        "ITO",
        "S[0-9]+",
        "C[0-9]+",
        "NoMG[0-9]",
        "MG[0-9]",
    ]
)
SampleSelector = ecv.Selector(
    drop=[
        "[0-9]+",
        "EIS",
        "PEDOTPSS",
        "SpinCoated",
        "ITO",
        "S[0-9]+]",
        "C[0-9]+",
        "G[0-9]+",
    ]
)


NoMG = ecv.EIS("Without Microgel")
NoMG.load(
    "/Users/nathan/Library/CloudStorage/OneDrive-ForschungszentrumJülichGmbH/Messungen/2025-12-17/csv/NoMG",
    grouping={
        "Glass Group": GlassGroupSelector,
        "Sample": SampleSelector,
    },
)

MG = ecv.EIS("With Microgel")
MG.load(
    "/Users/nathan/Library/CloudStorage/OneDrive-ForschungszentrumJülichGmbH/Messungen/2025-12-17/csv",
    grouping={
        "Glass Group": GlassGroupSelector,
        "Sample": SampleSelector,
    },
)
MG.remove("20251217_EIS_PEDOTPSS_MG1_G2_S01")

# ecv.fresponse(NoMG, "Capacitance", title="Nyquist Without Microgel", hue="Sample", showOnSave=True)

ecv.nyquist(
    NoMG,
    title="Nyquist Without Microgel",
    Rmin=65,
    hue="Sample",
    showOnSave=False,
    offsetCorrect=False,
)
ecv.nyquist(
    MG,
    title="Nyquist With Microgel",
    Rmin=65,
    hue="Sample",
    showOnSave=True,
    offsetCorrect=False,
)
ecv.nyquist(
    [NoMG, MG], title="Nyquist Comparison", Rmin=65, showOnSave=True, offsetCorrect=True
)
