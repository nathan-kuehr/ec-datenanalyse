import ecvisualization as ecv

selector = ecv.Selector(drop=["[0-9]+", "EIS", "PEDOTPSS", "SpinCoated", "ITO", "G[0-9]", "S[0-9]+]", "C[0-9]+"], rename={r"(\d+)min": r"BioLogic - Sonic. @ \1 min"})


NoMG = ecv.EIS("Without Microgel")
NoMG.load("/Users/nathan/Library/CloudStorage/OneDrive-ForschungszentrumJülichGmbH/Messungen/2025-12-17/csv/NoMG", grouping={"Sample": selector})

MG = ecv.EIS("With Microgel")
MG.load("/Users/nathan/Library/CloudStorage/OneDrive-ForschungszentrumJülichGmbH/Messungen/2025-12-17/csv", grouping={"Sample": selector})

#ecv.fresponse(NoMG, "Capacitance", title="Nyquist Without Microgel", hue="Sample", showOnSave=True)

ecv.nyquist([NoMG, MG], title="Nyquist With Microgel", showOnSave=True)