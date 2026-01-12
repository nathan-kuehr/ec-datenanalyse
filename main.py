import ec_visualization as ecv

selector = ecv.FileNameGroupSelector(drop=["[0-9]+", "EIS", "PEDOTPSS", "SpinCoated", "ITO", "G[0-9]", "S[0-9]+]", "C[0-9]+"], rename={r"(\d+)min": r"BioLogic - Sonic. @ \1 min"})


NoMG = ecv.EIS("Without Microgel")
NoMG.load("/Users/nathan/Library/CloudStorage/OneDrive-ForschungszentrumJülichGmbH/Messungen/2025-12-17/csv/NoMG", grouping={"Sample": selector})

MG = ecv.EIS("With Microgel")
MG.load("/Users/nathan/Library/CloudStorage/OneDrive-ForschungszentrumJülichGmbH/Messungen/2025-12-17/csv", grouping={"Sample": selector})

ecv.nyquist(NoMG, title="Nyquist Without Microgel", hue="Sample").save()
ecv.nyquist(MG, title="Nyquist With Any Kind of Microgel", hue="Sample").save()

plot = ecv.bode([NoMG, MG], title="Bode Plots With and Without Microgel")

for i, ax in enumerate(plot.handle()[1]):
    ax.legend(["Without Microgel", "95% CI", "With Microgel", "95% CI"], loc=["upper right", "lower right"][i])

plot.save()

plot = ecv.fresponse([NoMG, MG], "Capacitance", title="Capacitance Response With and Without Microgel")

ax = plot.handle()[1]
ax.legend(["Without Microgel", "95% CI", "With Microgel", "95% CI"], loc="lower left")
plot.save()

