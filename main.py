import ec_visualization as ecv

selector = ecv.FileNameGroupSelector()

palmsens = ecv.EIS("PalmSens")
palmsens.load("/Users/nathan/Library/CloudStorage/OneDrive-ForschungszentrumJülichGmbH/Messungen/2025-11-27/EIS/PalmSens", {"G1": selector})


biologic = ecv.EIS("BioLogic")
biologic.load("/Users/nathan/Library/CloudStorage/OneDrive-ForschungszentrumJülichGmbH/Messungen/2025-11-27/EIS/BioLogic", {"G1": selector})

ecv.bode([palmsens, biologic], title="PalmSens Bode Plot", hue="Group")